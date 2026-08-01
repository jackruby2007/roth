"""Strategy interface.

A strategy declares named **rules**. Each rule, when evaluated, reports what it
looked at, what it required, and whether that held. A trade is taken only when
every required rule passes.

The engine records the full rule evaluation for *every* candidate signal,
including the ones that failed and why. Two things fall out of that for free:

* Trade explanation. Every entry can be traced to the exact values that
  produced it.
* Visibility into which rule is doing the filtering work. A rule that never
  fails is not a filter; a rule that fails 99% of the time is the strategy.

Adding a hypothesis means writing one of these classes and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from roth.backtest.access import TimeGate
from roth.backtest.fills import Leg, Right
from roth.backtest.selection import ContractSpec


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    SIGNAL_EXIT = "signal_exit"
    EXPIRY_APPROACH = "expiry_approach"
    END_OF_BACKTEST = "end_of_backtest"
    # Reached expiry still holding, with no tradeable quote to exit against.
    # Settled at the contractual payoff rather than an invented price.
    EXPIRED_INTRINSIC = "expired_settled_at_intrinsic"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleResult:
    """The record of one rule evaluation."""

    rule_name: str
    passed: bool
    actual_value: float | str | None
    threshold: float | str | None
    direction: str  # ">=", "<", "==", "in", ...
    required: bool = True

    def explain(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{verdict}  {self.rule_name}: {self.actual_value} "
            f"{self.direction} {self.threshold}"
        )


@dataclass
class Context:
    """What a rule is allowed to see.

    Deliberately narrow. It exposes the current session's features and the
    history behind it, both served through the TimeGate, so a rule physically
    cannot reach a future row.
    """

    day: date
    row: pd.Series
    gate: TimeGate

    def history(self, lookback: int | None = None) -> pd.DataFrame:
        return self.gate.history(lookback)

    def value(self, column: str) -> float | None:
        v = self.row.get(column)
        return None if v is None or pd.isna(v) else v


class Rule(ABC):
    """One named condition."""

    def __init__(self, name: str, required: bool = True) -> None:
        self.name = name
        self.required = required

    @abstractmethod
    def evaluate(self, ctx: Context) -> RuleResult: ...


class ThresholdRule(Rule):
    """Compare a feature column against a fixed threshold."""

    OPS = {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(
        self, name: str, column: str, op: str, threshold: float, required: bool = True
    ) -> None:
        super().__init__(name, required)
        if op not in self.OPS:
            raise ValueError(f"Unsupported operator {op!r}")
        self.column = column
        self.op = op
        self.threshold = threshold

    def evaluate(self, ctx: Context) -> RuleResult:
        actual = ctx.value(self.column)
        # A missing feature is a failed rule, never a passed one. Long-window
        # indicators are NaN early in the sample, and treating that as a pass
        # would trade on absent information.
        passed = False if actual is None else bool(self.OPS[self.op](actual, self.threshold))
        return RuleResult(
            rule_name=self.name,
            passed=passed,
            actual_value=None if actual is None else float(actual),
            threshold=self.threshold,
            direction=self.op,
            required=self.required,
        )


class CategoryRule(Rule):
    """Require a categorical feature to be one of an allowed set."""

    def __init__(
        self, name: str, column: str, allowed: set[str], required: bool = True
    ) -> None:
        super().__init__(name, required)
        self.column = column
        self.allowed = allowed

    def evaluate(self, ctx: Context) -> RuleResult:
        raw = ctx.row.get(self.column)
        actual = None if raw is None or pd.isna(raw) else str(raw)
        return RuleResult(
            rule_name=self.name,
            passed=actual in self.allowed,
            actual_value=actual,
            threshold=",".join(sorted(self.allowed)),
            direction="in",
            required=self.required,
        )


class CallableRule(Rule):
    """Escape hatch for logic that is not a simple comparison."""

    def __init__(self, name: str, fn, required: bool = True) -> None:
        super().__init__(name, required)
        self.fn = fn

    def evaluate(self, ctx: Context) -> RuleResult:
        return self.fn(ctx)


# ---------------------------------------------------------------------------
# Exits and sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExitRules:
    """When to close a position.

    All are optional. `time_stop_days` counts trading sessions held.
    `close_before_expiry_days` exists because holding to expiry means accepting
    assignment mechanics the harness does not model -- better to exit and say so
    than to invent a settlement.
    """

    profit_target_pct: float | None = None  # +0.50 is a 50% gain on debit paid
    stop_loss_pct: float | None = None  # -0.50 is a 50% loss
    time_stop_days: int | None = None
    close_before_expiry_days: int | None = 1
    exit_on_signal_failure: bool = False


@dataclass(frozen=True)
class PositionSizing:
    """How large a position to take.

    `contracts` is the simple fixed-size case. `risk_fraction` sizes against
    account equity, capped by `max_contracts`.
    """

    contracts: int = 1
    risk_fraction: float | None = None
    max_contracts: int = 100


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


@dataclass
class SignalDecision:
    """The full record of one candidate signal, taken or not."""

    day: date
    symbol: str
    rule_results: list[RuleResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.rule_results if r.required)

    @property
    def failed_rules(self) -> list[RuleResult]:
        return [r for r in self.rule_results if r.required and not r.passed]

    def explain(self) -> str:
        head = "SIGNAL" if self.passed else "no signal"
        lines = [f"{self.day} {self.symbol}: {head}"]
        lines += [f"  {r.explain()}" for r in self.rule_results]
        return "\n".join(lines)


class Strategy(ABC):
    """Base class for every hypothesis.

    Subclasses declare rules, what contract to trade, how to exit, and how large
    to go. Everything else -- walking time, filling, journalling, reporting --
    is the engine's job.
    """

    name: str = "unnamed"
    version: str = "0.1.0"
    structure_type: str = "long_call"
    direction: Direction = Direction.LONG

    @abstractmethod
    def rules(self) -> list[Rule]:
        """Named conditions that must hold for an entry."""

    @abstractmethod
    def contract_spec(self, ctx: Context) -> ContractSpec:
        """Which contract to trade when the rules pass."""

    def exits(self) -> ExitRules:
        return ExitRules()

    def sizing(self) -> PositionSizing:
        return PositionSizing()

    def spread_width(self, ctx: Context) -> float:
        """Strike distance for vertical structures. Ignored otherwise."""
        return 5.0

    def build_legs(self, selected, short_strike: float | None) -> list[Leg]:
        """Assemble legs from the selected contract."""
        from roth.backtest.fills import (
            long_call,
            long_put,
            vertical_call_spread,
            vertical_put_spread,
        )

        if self.structure_type == "long_call":
            return long_call(selected.strike, selected.expiration)
        if self.structure_type == "long_put":
            return long_put(selected.strike, selected.expiration)
        if self.structure_type == "vertical_call_spread":
            return vertical_call_spread(selected.strike, short_strike, selected.expiration)
        if self.structure_type == "vertical_put_spread":
            return vertical_put_spread(selected.strike, short_strike, selected.expiration)
        raise ValueError(f"Unknown structure type {self.structure_type!r}")

    @property
    def needs_vertical(self) -> bool:
        return self.structure_type in ("vertical_call_spread", "vertical_put_spread")

    @property
    def right(self) -> Right:
        return Right.PUT if "put" in self.structure_type else Right.CALL

    def evaluate(self, ctx: Context) -> SignalDecision:
        """Run every rule and record the result, pass or fail."""
        decision = SignalDecision(day=ctx.day, symbol=ctx.gate.symbol)
        decision.rule_results = [rule.evaluate(ctx) for rule in self.rules()]
        return decision
