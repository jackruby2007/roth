"""Reference strategy.

Exists to prove the pipeline runs end to end. It is deliberately simple and no
attempt has been made to make it profitable. Its results are a plumbing test,
not a finding, and the reporting layer says so.

The brief specifies entry at Monday's *open*. Phase 1 runs on end-of-day option
data, which carries one snapshot per contract per day and therefore has no
opening NBBO to fill against. Rather than invent an opening price, entry moves
to Monday's close: the signal fires on Monday and the engine fills at the next
session's close, which is the earliest honestly priced moment available. When
1-minute option quotes are ingested this becomes a genuine Monday-open entry
with no other change to the strategy.
"""

from __future__ import annotations

from roth.backtest.fills import Right
from roth.backtest.selection import ContractSpec
from roth.backtest.strategy import (
    Context,
    Direction,
    ExitRules,
    PositionSizing,
    Rule,
    RuleResult,
    Strategy,
    ThresholdRule,
)

MONDAY = 0
FRIDAY = 4


class ExitOnFridayRule(Rule):
    """Signals that the position should be closed once the week is over.

    Wired to `exit_on_signal_failure`, so the position is held while this rule
    passes and closed when it fails.
    """

    def __init__(self) -> None:
        super().__init__("before_friday_close", required=True)

    def evaluate(self, ctx: Context) -> RuleResult:
        dow = ctx.value("day_of_week")
        return RuleResult(
            rule_name=self.name,
            passed=dow is not None and dow < FRIDAY,
            actual_value=dow,
            threshold=FRIDAY,
            direction="<",
        )


class MondayCallReference(Strategy):
    """Buy a 30-delta call on Monday, exit at Friday's close.

    Nearest expiry beyond 7 days. One contract. No filters beyond the day of
    the week, on purpose: the point is to exercise selection, filling,
    journalling and reporting, not to find an edge.
    """

    name = "reference_monday_call"
    version = "1.0.0"
    structure_type = "long_call"
    direction = Direction.LONG

    def rules(self) -> list[Rule]:
        return [ThresholdRule("is_monday", "day_of_week", "==", MONDAY)]

    def contract_spec(self, ctx: Context) -> ContractSpec:
        return ContractSpec(
            right=Right.CALL,
            target_dte=10,
            target_delta=0.30,
            # "Nearest expiry beyond 7 days", bounded by the ingest horizon.
            min_dte=8,
            max_dte=45,
        )

    def exits(self) -> ExitRules:
        # Four sessions after a Monday entry lands on Friday. The expiry guard
        # is a safety net, not the intended exit.
        return ExitRules(
            time_stop_days=4,
            close_before_expiry_days=1,
            profit_target_pct=None,
            stop_loss_pct=None,
        )

    def sizing(self) -> PositionSizing:
        return PositionSizing(contracts=1)


class BuyAndHoldMarker(Strategy):
    """Placeholder used by the known-answer check.

    Buy-and-hold is an equity position, not an option structure, so the
    known-answer check walks the gate directly rather than going through the
    options engine. This class exists only to give that check a name and a
    version to report.
    """

    name = "buy_and_hold"
    version = "1.0.0"

    def rules(self) -> list[Rule]:
        return []

    def contract_spec(self, ctx: Context) -> ContractSpec:
        raise NotImplementedError("buy-and-hold does not select option contracts")


REFERENCE_STRATEGIES = {
    "reference_monday_call": MondayCallReference,
}
