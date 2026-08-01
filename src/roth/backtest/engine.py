"""Event-driven backtest engine.

Walks forward one session at a time. Not vectorised, deliberately: a vectorised
implementation makes lookahead a one-character mistake, and the cost of the loop
is irrelevant next to being able to trust the result.

**The ordering that makes it honest**, on every session D:

1. Mark open positions against session D's quotes and check exits.
2. Evaluate the strategy's rules using session D's features -- everything the
   TimeGate will serve, and nothing beyond it.
3. If the rules pass, record the intent. Do **not** fill it.
4. Advance the clock to D+1. Only now does D+1's chain become readable.
5. Fill the pending intent against D+1's quotes, or reject it.

Step 3 and 4 are the whole point. The signal bar is never the fill bar, and that
is enforced by the gate rather than by remembering to shift an index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from roth.backtest.access import TimeGate
from roth.backtest.fills import Action, FillModel, FillResult, Leg, RejectReason
from roth.backtest.selection import select_contract, select_vertical
from roth.backtest.strategy import (
    Context,
    ExitReason,
    SignalDecision,
    Strategy,
)
from roth.config import COSTS, CostModel


@dataclass
class OpenPosition:
    symbol: str
    strategy_name: str
    strategy_version: str
    structure_type: str
    direction: str
    legs: list[Leg]
    contracts: int

    entry_day: date
    entry_price_per_unit: float  # dollars per structure unit, signed
    entry_commission: float
    entry_fill: FillResult

    signal_day: date
    signal_id: int

    expiration: date
    entry_delta: float | None = None
    entry_iv: float | None = None
    entry_underlying: float | None = None
    regime_at_entry: dict = field(default_factory=dict)

    sessions_held: int = 0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    # Failed exits are logged once per position, not once per session. A
    # position that cannot be closed would otherwise emit one rejection every
    # remaining bar and drown the report.
    exit_failure_logged: bool = False

    @property
    def capital_at_risk(self) -> float:
        """Debit paid. For the long structures supported here that is the most
        that can be lost, which makes it the correct denominator for R."""
        return abs(self.entry_price_per_unit) * self.contracts + self.entry_commission


@dataclass
class ClosedTrade:
    symbol: str
    strategy_name: str
    strategy_version: str
    structure_type: str
    direction: str

    entry_day: date
    exit_day: date
    signal_day: date
    signal_id: int

    strikes: str
    expiration: date
    contracts: int

    entry_price: float
    exit_price: float
    entry_bid: float
    entry_ask: float
    exit_bid: float
    exit_ask: float

    capital_at_risk: float
    gross_pnl: float
    commissions: float
    net_pnl: float
    return_pct: float
    r_multiple: float
    holding_days: int

    max_favorable_excursion: float
    max_adverse_excursion: float
    exit_reason: str

    entry_delta: float | None = None
    entry_iv: float | None = None
    entry_underlying: float | None = None
    trend_direction: str | None = None
    trend_strength: str | None = None
    vol_bucket: str | None = None
    day_type: str | None = None


@dataclass
class UnfillableSignal:
    """A signal the strategy wanted to act on but could not.

    These are counted in every report. A strategy that only works when bad
    quotes are ignored does not work.
    """

    day: date
    symbol: str
    signal_id: int
    action: str
    reason: str
    detail: str


@dataclass
class BacktestResult:
    strategy_name: str
    strategy_version: str
    symbols: tuple[str, ...]
    start: date | None
    end: date | None

    trades: list[ClosedTrade] = field(default_factory=list)
    signals: list[SignalDecision] = field(default_factory=list)
    unfillable: list[UnfillableSignal] = field(default_factory=list)

    sessions_processed: int = 0
    quarantined_sessions_excluded: int = 0
    used_mid_fills: bool = False
    synthetic_data: bool = False

    @property
    def candidate_signals(self) -> int:
        return sum(1 for s in self.signals if s.passed)

    @property
    def unfillable_count(self) -> int:
        return len(self.unfillable)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def rule_evaluations_frame(self) -> pd.DataFrame:
        """Every rule evaluation for every candidate signal, passed or failed."""
        rows = []
        for i, sig in enumerate(self.signals):
            for r in sig.rule_results:
                rows.append(
                    {
                        "signal_id": i,
                        "day": sig.day,
                        "symbol": sig.symbol,
                        "signal_passed": sig.passed,
                        "rule_name": r.rule_name,
                        "passed": r.passed,
                        "actual_value": r.actual_value,
                        "threshold": r.threshold,
                        "direction": r.direction,
                        "required": r.required,
                    }
                )
        return pd.DataFrame(rows)

    def rule_filter_summary(self) -> pd.DataFrame:
        """How often each rule failed. Shows which rule does the filtering."""
        df = self.rule_evaluations_frame()
        if df.empty:
            return df
        return (
            df.groupby("rule_name", as_index=False)
            .agg(evaluations=("passed", "size"), failures=("passed", lambda s: int((~s).sum())))
            .assign(fail_rate=lambda d: d["failures"] / d["evaluations"])
            .sort_values("fail_rate", ascending=False)
        )


class BacktestEngine:
    """Runs one strategy over one symbol's gated data."""

    def __init__(
        self,
        strategy: Strategy,
        costs: CostModel = COSTS,
        use_mid_fills: bool = False,
    ) -> None:
        self.strategy = strategy
        self.fill_model = FillModel(costs=costs, use_mid=use_mid_fills)
        self.use_mid_fills = use_mid_fills

    # -- helpers -----------------------------------------------------------

    def _mark_position(self, pos: OpenPosition, chain: pd.DataFrame) -> float | None:
        """Current value of a position per structure unit, at the bid side.

        Marking at what could actually be received on exit, not at mid. A mark
        at mid would make every excursion statistic optimistic.
        """
        result = self.fill_model.fill(
            chain, pos.legs, Action.CLOSE, pos.entry_day, pos.contracts
        )
        return result.price_per_unit if result.filled else None

    @staticmethod
    def _regime(row: pd.Series) -> dict:
        return {
            k: (None if row.get(k) is None or pd.isna(row.get(k)) else str(row.get(k)))
            for k in ("trend_direction", "trend_strength", "vol_bucket", "day_type")
        }

    def _close(
        self,
        pos: OpenPosition,
        exit_fill: FillResult,
        exit_day: date,
        reason: ExitReason,
    ) -> ClosedTrade:
        exit_value = exit_fill.price_per_unit * pos.contracts
        entry_value = pos.entry_price_per_unit * pos.contracts

        gross = exit_value - entry_value
        commissions = pos.entry_commission + exit_fill.commission
        net = gross - commissions

        risk = pos.capital_at_risk
        entry_legs = pos.entry_fill.legs
        exit_legs = exit_fill.legs

        return ClosedTrade(
            symbol=pos.symbol,
            strategy_name=pos.strategy_name,
            strategy_version=pos.strategy_version,
            structure_type=pos.structure_type,
            direction=pos.direction,
            entry_day=pos.entry_day,
            exit_day=exit_day,
            signal_day=pos.signal_day,
            signal_id=pos.signal_id,
            strikes="/".join(f"{leg.strike:g}" for leg in pos.legs),
            expiration=pos.expiration,
            contracts=pos.contracts,
            entry_price=pos.entry_price_per_unit,
            exit_price=exit_fill.price_per_unit,
            entry_bid=entry_legs[0].quote.bid if entry_legs else float("nan"),
            entry_ask=entry_legs[0].quote.ask if entry_legs else float("nan"),
            exit_bid=exit_legs[0].quote.bid if exit_legs else float("nan"),
            exit_ask=exit_legs[0].quote.ask if exit_legs else float("nan"),
            capital_at_risk=risk,
            gross_pnl=gross,
            commissions=commissions,
            net_pnl=net,
            return_pct=(net / risk) if risk else 0.0,
            r_multiple=(net / risk) if risk else 0.0,
            holding_days=pos.sessions_held,
            max_favorable_excursion=pos.max_favorable,
            max_adverse_excursion=pos.max_adverse,
            exit_reason=reason.value,
            entry_delta=pos.entry_delta,
            entry_iv=pos.entry_iv,
            entry_underlying=pos.entry_underlying,
            **pos.regime_at_entry,
        )

    @staticmethod
    def _intrinsic_per_unit(pos: OpenPosition, underlying: float) -> float:
        """Contractual payoff of the structure at expiry, per unit.

        This is not a substituted quote. At expiration an option is worth its
        intrinsic value by the terms of the contract, so settling here states a
        determinate fact rather than inventing a price the way a mid-fill or an
        interpolated quote would.

        It is still second-best: a real exit would have crossed a spread. Trades
        settled this way carry their own exit reason so they can be counted and,
        if they matter, excluded.
        """
        total = 0.0
        for leg in pos.legs:
            if leg.right.value == "C":
                payoff = max(underlying - leg.strike, 0.0)
            else:
                payoff = max(leg.strike - underlying, 0.0)
            total += leg.quantity * payoff
        return total * 100.0

    def _settle_expired(
        self, pos: OpenPosition, row: pd.Series, day: date
    ) -> ClosedTrade | None:
        """Resolve a position that reached expiry without a tradeable quote."""
        underlying = row.get("close")
        if underlying is None or pd.isna(underlying):
            return None

        per_unit = self._intrinsic_per_unit(pos, float(underlying))
        # Exercise or assignment still costs a contract fee.
        commission = self.fill_model.costs.commission_per_contract * sum(
            abs(leg.quantity) for leg in pos.legs
        ) * pos.contracts

        settle = FillResult(
            filled=True,
            action=Action.CLOSE,
            day=day,
            legs=[],
            price_per_unit=per_unit,
            commission=commission,
            contracts=pos.contracts,
        )
        return self._close(pos, settle, day, ExitReason.EXPIRED_INTRINSIC)

    def _should_exit(
        self, pos: OpenPosition, mark: float | None, day: date, signal_failed: bool
    ) -> ExitReason | None:
        rules = self.strategy.exits()

        if rules.close_before_expiry_days is not None:
            if (pos.expiration - day).days <= rules.close_before_expiry_days:
                return ExitReason.EXPIRY_APPROACH

        if rules.time_stop_days is not None and pos.sessions_held >= rules.time_stop_days:
            return ExitReason.TIME_STOP

        if rules.exit_on_signal_failure and signal_failed:
            return ExitReason.SIGNAL_EXIT

        if mark is not None and pos.entry_price_per_unit:
            pnl_pct = (mark - pos.entry_price_per_unit) / abs(pos.entry_price_per_unit)
            if rules.profit_target_pct is not None and pnl_pct >= rules.profit_target_pct:
                return ExitReason.PROFIT_TARGET
            if rules.stop_loss_pct is not None and pnl_pct <= -abs(rules.stop_loss_pct):
                return ExitReason.STOP_LOSS

        return None

    # -- main loop ---------------------------------------------------------

    def run(self, gate: TimeGate, result: BacktestResult | None = None) -> BacktestResult:
        strategy = self.strategy
        result = result or BacktestResult(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            symbols=(gate.symbol,),
            start=None,
            end=None,
        )
        result.used_mid_fills = self.use_mid_fills

        position: OpenPosition | None = None
        pending: dict | None = None

        while gate.advance():
            day = gate.now
            result.sessions_processed += 1
            if result.start is None:
                result.start = day
            result.end = day

            chain = gate.chain(day)
            row = gate.current()
            ctx = Context(day=day, row=row, gate=gate)

            # 1. Fill anything the previous session asked for. This happens
            #    first, against *this* session's quotes -- never the signal
            #    session's.
            if pending is not None and position is None:
                position = self._try_open(pending, chain, day, result)
                pending = None

            # 2. Mark and manage an open position.
            if position is not None:
                # Only count sessions *after* the entry session. Incrementing on
                # the entry bar meant a one-session time stop exited at the same
                # close it entered -- a zero-length hold that can do nothing but
                # pay the spread, and every longer time stop was off by one.
                if position.entry_day < day:
                    position.sessions_held += 1
                mark = self._mark_position(position, chain)
                if mark is not None:
                    excursion = (mark - position.entry_price_per_unit) * position.contracts
                    position.max_favorable = max(position.max_favorable, excursion)
                    position.max_adverse = min(position.max_adverse, excursion)

            # 3. Evaluate rules on this session's close.
            decision = strategy.evaluate(ctx)
            result.signals.append(decision)
            signal_id = len(result.signals) - 1

            # 4. Exit checks, using this session's quotes.
            if position is not None:
                mark = self._mark_position(position, chain)
                reason = self._should_exit(position, mark, day, not decision.passed)
                at_or_past_expiry = day >= position.expiration

                if reason is not None or at_or_past_expiry:
                    exit_fill = self.fill_model.fill(
                        chain, position.legs, Action.CLOSE, day, position.contracts
                    )
                    if exit_fill.filled:
                        result.trades.append(
                            self._close(
                                position, exit_fill, day, reason or ExitReason.EXPIRY_APPROACH
                            )
                        )
                        position = None
                    elif at_or_past_expiry:
                        # No quote left to trade against, and the contract has
                        # expired. Settle at the contractual payoff so the
                        # position resolves instead of blocking every later
                        # signal for the rest of the run.
                        settled = self._settle_expired(position, row, day)
                        detail = (
                            "settled at intrinsic"
                            if settled is not None
                            else "no underlying close either; position dropped unresolved"
                        )
                        result.unfillable.append(
                            UnfillableSignal(
                                day=day,
                                symbol=gate.symbol,
                                signal_id=position.signal_id,
                                action="exit",
                                reason="expired_without_quote",
                                detail=f"no tradeable quote at expiry; {detail}",
                            )
                        )
                        if settled is not None:
                            result.trades.append(settled)
                        position = None
                    elif not position.exit_failure_logged:
                        position.exit_failure_logged = True
                        result.unfillable.append(
                            UnfillableSignal(
                                day=day,
                                symbol=gate.symbol,
                                signal_id=position.signal_id,
                                action="exit",
                                reason=(exit_fill.reject_reason or RejectReason.NO_CHAIN).value,
                                detail=exit_fill.reject_detail,
                            )
                        )

            # 5. Record entry intent for the *next* session. Nothing is filled
            #    here, because this session's quotes are the signal's own bar.
            if position is None and pending is None and decision.passed:
                intent = self._plan_entry(ctx, chain, signal_id)
                if intent is None:
                    result.unfillable.append(
                        UnfillableSignal(
                            day=day,
                            symbol=gate.symbol,
                            signal_id=signal_id,
                            action="entry",
                            reason="no_contract_matching_spec",
                            detail="selection found no listed contract for the requested spec",
                        )
                    )
                elif gate.peek_next_day() is not None:
                    pending = intent

        # Close anything still open against the final session's quotes.
        if position is not None:
            chain = gate.chain(gate.now)
            exit_fill = self.fill_model.fill(
                chain, position.legs, Action.CLOSE, gate.now, position.contracts
            )
            if exit_fill.filled:
                result.trades.append(
                    self._close(position, exit_fill, gate.now, ExitReason.END_OF_BACKTEST)
                )
            else:
                result.unfillable.append(
                    UnfillableSignal(
                        day=gate.now,
                        symbol=gate.symbol,
                        signal_id=position.signal_id,
                        action="exit",
                        reason=(exit_fill.reject_reason or RejectReason.NO_CHAIN).value,
                        detail="position still open at the end of the backtest",
                    )
                )

        return result

    # -- entry -------------------------------------------------------------

    def _plan_entry(self, ctx: Context, chain: pd.DataFrame, signal_id: int) -> dict | None:
        """Choose the contract using the SIGNAL session's chain.

        Selection uses what was visible at signal time; only the *price* comes
        from the next session. That matches how a real order works: the contract
        is chosen on the evidence available, then filled at whatever the market
        offers next.
        """
        strategy = self.strategy
        spec = strategy.contract_spec(ctx)

        short_strike = None
        if strategy.needs_vertical:
            picked = select_vertical(chain, ctx.day, spec, strategy.spread_width(ctx))
            if picked is None:
                return None
            selected, short_strike = picked
        else:
            selected = select_contract(chain, ctx.day, spec)
            if selected is None:
                return None

        return {
            "legs": strategy.build_legs(selected, short_strike),
            "selected": selected,
            "signal_day": ctx.day,
            "signal_id": signal_id,
            "regime": self._regime(ctx.row),
            "contracts": strategy.sizing().contracts,
        }

    def _try_open(
        self, intent: dict, chain: pd.DataFrame, day: date, result: BacktestResult
    ) -> OpenPosition | None:
        strategy = self.strategy
        legs = intent["legs"]
        contracts = intent["contracts"]

        fill = self.fill_model.fill(chain, legs, Action.OPEN, day, contracts)

        if not fill.filled:
            result.unfillable.append(
                UnfillableSignal(
                    day=day,
                    symbol=result.symbols[0],
                    signal_id=intent["signal_id"],
                    action="entry",
                    reason=(fill.reject_reason or RejectReason.NO_CHAIN).value,
                    detail=fill.reject_detail,
                )
            )
            return None

        selected = intent["selected"]
        return OpenPosition(
            symbol=result.symbols[0],
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            structure_type=strategy.structure_type,
            direction=strategy.direction.value,
            legs=legs,
            contracts=contracts,
            entry_day=day,
            entry_price_per_unit=fill.price_per_unit,
            entry_commission=fill.commission,
            entry_fill=fill,
            signal_day=intent["signal_day"],
            signal_id=intent["signal_id"],
            expiration=selected.expiration,
            entry_delta=selected.delta,
            entry_iv=selected.implied_vol,
            entry_underlying=selected.underlying,
            regime_at_entry=intent["regime"],
        )
