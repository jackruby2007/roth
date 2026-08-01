"""Tests for the time gate, contract selection, and the engine loop.

The load-bearing test in this file is
`test_entry_fills_at_the_next_session_not_the_signal_session`. If that ever
breaks, every backtest result the harness produces is optimistic by an unknown
amount.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from roth.backtest.access import LookaheadError, TimeGate
from roth.backtest.engine import BacktestEngine
from roth.backtest.fills import Right
from roth.backtest.selection import ContractSpec, select_contract, select_vertical
from roth.backtest.strategy import (
    CategoryRule,
    Context,
    ExitRules,
    Strategy,
    ThresholdRule,
)
from roth.config import CostModel

DAYS = [date(2024, 6, 3) + timedelta(days=i) for i in range(10)]
NO_COST = CostModel(commission_per_contract=0.0, extra_slippage_per_contract=0.0)


def features(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": "SPY",
            "day": DAYS[:n],
            "close": [100.0 + i for i in range(n)],
            "rsi_14": [50.0 + i for i in range(n)],
            "trend_direction": ["up"] * n,
            "trend_strength": ["strong"] * n,
            "vol_bucket": ["mid"] * n,
            "day_type": ["range"] * n,
        }
    )


def chains(n: int = 10, ask_by_day: dict | None = None) -> pd.DataFrame:
    """One call contract per session, with a per-day ask so fills are traceable."""
    rows = []
    for i, day in enumerate(DAYS[:n]):
        ask = (ask_by_day or {}).get(day, 2.00 + i)
        rows.append(
            {
                "day": day,
                "symbol": "SPY",
                "expiration_date": date(2024, 7, 19),
                "strike_dollars": 100.0,
                "right": "C",
                "bid": ask - 0.20,
                "ask": ask,
                "bid_size": 10,
                "ask_size": 10,
                "delta": 0.50,
                "implied_vol": 0.20,
                "underlying_close": 100.0 + i,
            }
        )
    return pd.DataFrame(rows)


# -- TimeGate ---------------------------------------------------------------


def test_gate_starts_unstarted():
    gate = TimeGate(features(), chains(), "SPY")
    assert not gate.started
    with pytest.raises(LookaheadError, match="clock has not started"):
        _ = gate.now


def test_gate_advances_one_session_at_a_time():
    gate = TimeGate(features(), chains(), "SPY")
    gate.advance()
    assert gate.now == DAYS[0]
    gate.advance()
    assert gate.now == DAYS[1]


def test_gate_refuses_a_future_chain():
    gate = TimeGate(features(), chains(), "SPY")
    gate.advance()

    with pytest.raises(LookaheadError, match="lookahead"):
        gate.chain(DAYS[5])


def test_gate_refuses_a_future_feature_row():
    gate = TimeGate(features(), chains(), "SPY")
    gate.advance()
    with pytest.raises(LookaheadError):
        gate.row(DAYS[3])


def test_gate_allows_past_and_current():
    gate = TimeGate(features(), chains(), "SPY")
    for _ in range(4):
        gate.advance()

    assert gate.row(DAYS[0]) is not None
    assert not gate.chain(DAYS[3]).empty


def test_history_never_extends_past_the_cursor():
    gate = TimeGate(features(), chains(), "SPY")
    for _ in range(5):
        gate.advance()

    hist = gate.history()
    assert len(hist) == 5
    assert hist["day"].max() == gate.now


def test_history_lookback_limits_the_window():
    gate = TimeGate(features(), chains(), "SPY")
    for _ in range(6):
        gate.advance()
    assert len(gate.history(lookback=3)) == 3


def test_peek_next_day_reveals_the_date_but_not_the_data():
    gate = TimeGate(features(), chains(), "SPY")
    gate.advance()

    assert gate.peek_next_day() == DAYS[1]
    # Knowing a session exists must not let its prices be read.
    with pytest.raises(LookaheadError):
        gate.chain(DAYS[1])


def test_gate_exhausts_at_the_end():
    gate = TimeGate(features(3), chains(3), "SPY")
    assert gate.advance() and gate.advance() and gate.advance()
    assert not gate.advance()
    assert gate.exhausted


# -- selection --------------------------------------------------------------


def option_chain_for_selection() -> pd.DataFrame:
    rows = []
    for exp_offset, exp in ((7, date(2024, 6, 10)), (21, date(2024, 6, 24))):
        for strike, delta in ((95.0, 0.75), (100.0, 0.50), (105.0, 0.30), (110.0, 0.15)):
            rows.append(
                {
                    "day": date(2024, 6, 3),
                    "expiration_date": exp,
                    "strike_dollars": strike,
                    "right": "C",
                    "delta": delta,
                    "implied_vol": 0.20,
                    "underlying_close": 100.0,
                    "bid": 1.0,
                    "ask": 1.1,
                    "_dte": exp_offset,
                }
            )
    return pd.DataFrame(rows)


def test_selection_targets_delta():
    spec = ContractSpec(right=Right.CALL, target_dte=21, target_delta=0.30)
    picked = select_contract(option_chain_for_selection(), date(2024, 6, 3), spec)

    assert picked is not None
    assert picked.strike == 105.0
    assert picked.delta == pytest.approx(0.30)


def test_selection_targets_expiry():
    spec = ContractSpec(right=Right.CALL, target_dte=7, target_delta=0.50)
    picked = select_contract(option_chain_for_selection(), date(2024, 6, 3), spec)
    assert picked.expiration == date(2024, 6, 10)
    assert picked.dte == 7


def test_selection_respects_dte_bounds():
    spec = ContractSpec(
        right=Right.CALL, target_dte=7, target_delta=0.50, min_dte=14, max_dte=30
    )
    picked = select_contract(option_chain_for_selection(), date(2024, 6, 3), spec)
    assert picked.expiration == date(2024, 6, 24)


def test_selection_returns_none_when_no_expiry_qualifies():
    spec = ContractSpec(right=Right.CALL, target_dte=90, target_delta=0.50, min_dte=60)
    assert select_contract(option_chain_for_selection(), date(2024, 6, 3), spec) is None


def test_selection_targets_moneyness():
    spec = ContractSpec(right=Right.CALL, target_dte=21, target_moneyness=0.05)
    picked = select_contract(option_chain_for_selection(), date(2024, 6, 3), spec)
    assert picked.strike == 105.0


def test_vertical_requires_the_short_strike_to_be_listed():
    """Rounding to a nearby strike would silently change the position's risk, so
    an unlisted short strike means no trade."""
    spec = ContractSpec(right=Right.CALL, target_dte=21, target_delta=0.30)
    chain = option_chain_for_selection()

    assert select_vertical(chain, date(2024, 6, 3), spec, width=5.0) is not None
    assert select_vertical(chain, date(2024, 6, 3), spec, width=7.0) is None


# -- the engine's fill timing -----------------------------------------------


class AlwaysEnter(Strategy):
    name = "always"
    version = "1.0.0"
    structure_type = "long_call"

    def rules(self):
        return [ThresholdRule("always", "close", ">", 0)]

    def contract_spec(self, ctx):
        return ContractSpec(right=Right.CALL, target_dte=46, target_delta=0.50)

    def exits(self):
        return ExitRules(time_stop_days=2, close_before_expiry_days=None)


def test_entry_fills_at_the_next_session_not_the_signal_session():
    """The load-bearing test of the whole harness.

    Session 0 asks 2.00; session 1 asks 99.00. The signal fires on session 0.
    If the engine filled on the signal bar the entry would be 200; filling
    correctly on the next bar it must be 9900.
    """
    asks = {DAYS[0]: 2.00, DAYS[1]: 99.00}
    gate = TimeGate(features(), chains(ask_by_day=asks), "SPY")

    engine = BacktestEngine(AlwaysEnter(), costs=NO_COST)
    result = engine.run(gate)

    assert result.trades
    first = result.trades[0]
    assert first.entry_day == DAYS[1]
    assert first.entry_price == pytest.approx(9900.0)


def test_signal_day_and_entry_day_are_recorded_separately():
    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(AlwaysEnter(), costs=NO_COST).run(gate)

    first = result.trades[0]
    assert first.signal_day == DAYS[0]
    assert first.entry_day == DAYS[1]
    assert first.entry_day > first.signal_day


def test_every_signal_is_recorded_pass_or_fail():
    class Picky(AlwaysEnter):
        def rules(self):
            return [ThresholdRule("rsi_high", "rsi_14", ">", 55)]

    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(Picky(), costs=NO_COST).run(gate)

    assert len(result.signals) == 10
    assert any(not s.passed for s in result.signals)
    assert any(s.passed for s in result.signals)


def test_rule_evaluations_capture_failures_with_values():
    class Picky(AlwaysEnter):
        def rules(self):
            return [ThresholdRule("rsi_high", "rsi_14", ">", 55)]

    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(Picky(), costs=NO_COST).run(gate)

    evals = result.rule_evaluations_frame()
    assert len(evals) == 10
    failed = evals[~evals["passed"]]
    assert len(failed) > 0
    assert failed["actual_value"].notna().all()
    assert (failed["threshold"] == 55).all()


def test_rule_filter_summary_identifies_the_binding_rule():
    class TwoRules(AlwaysEnter):
        def rules(self):
            return [
                ThresholdRule("never_fails", "close", ">", 0),
                ThresholdRule("almost_always_fails", "rsi_14", ">", 1000),
            ]

    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(TwoRules(), costs=NO_COST).run(gate)

    summary = result.rule_filter_summary().set_index("rule_name")
    assert summary.loc["never_fails", "fail_rate"] == 0.0
    assert summary.loc["almost_always_fails", "fail_rate"] == 1.0


def test_unfillable_entries_are_counted_not_silently_dropped():
    """A chain of crossed quotes must produce rejections, never trades."""
    bad = chains()
    bad["bid"] = 99.0  # bid above ask everywhere

    gate = TimeGate(features(), bad, "SPY")
    result = BacktestEngine(AlwaysEnter(), costs=NO_COST).run(gate)

    assert result.trades == []
    assert result.unfillable_count > 0
    assert all(u.reason for u in result.unfillable)


def test_missing_feature_makes_a_rule_fail_not_pass():
    """NaN early in a long-window indicator must not be treated as a pass."""
    feats = features()
    feats["rsi_14"] = None

    class NeedsRsi(AlwaysEnter):
        def rules(self):
            return [ThresholdRule("rsi", "rsi_14", ">", 10)]

    gate = TimeGate(feats, chains(), "SPY")
    result = BacktestEngine(NeedsRsi(), costs=NO_COST).run(gate)

    assert result.trades == []
    assert all(not s.passed for s in result.signals)


def test_category_rule_matches_regime_labels():
    class TrendOnly(AlwaysEnter):
        def rules(self):
            return [CategoryRule("trend", "trend_direction", {"up"})]

    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(TrendOnly(), costs=NO_COST).run(gate)
    assert result.trades


def test_regime_labels_are_captured_at_entry():
    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(AlwaysEnter(), costs=NO_COST).run(gate)

    t = result.trades[0]
    assert t.trend_direction == "up"
    assert t.vol_bucket == "mid"


def test_time_stop_closes_the_position():
    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(AlwaysEnter(), costs=NO_COST).run(gate)

    assert result.trades
    assert result.trades[0].exit_reason in ("time_stop", "end_of_backtest")
    assert result.trades[0].holding_days >= 1


def test_r_multiple_uses_capital_at_risk():
    gate = TimeGate(features(), chains(), "SPY")
    result = BacktestEngine(AlwaysEnter(), costs=NO_COST).run(gate)

    t = result.trades[0]
    assert t.capital_at_risk > 0
    assert t.r_multiple == pytest.approx(t.net_pnl / t.capital_at_risk)


def test_context_cannot_reach_the_next_session():
    """Even a strategy actively trying to cheat is stopped by the gate.

    The cheat attempted here is the one that would actually work: reading the
    very next session, which is the bar the fill happens on.
    """
    seen: list = []

    class Cheater(AlwaysEnter):
        def rules(self):
            from roth.backtest.strategy import CallableRule, RuleResult

            def peek(ctx: Context):
                nxt = ctx.gate.peek_next_day()
                if nxt is None:
                    return RuleResult("peek", False, None, None, "==")
                try:
                    ctx.gate.chain(nxt)
                    seen.append("PEEKED")
                except LookaheadError:
                    seen.append("BLOCKED")
                return RuleResult("peek", False, None, None, "==")

            return [CallableRule("peek", peek)]

    gate = TimeGate(features(), chains(), "SPY")
    BacktestEngine(Cheater(), costs=NO_COST).run(gate)

    assert seen  # the cheat was actually attempted
    assert "PEEKED" not in seen
    assert set(seen) == {"BLOCKED"}


def test_reading_the_current_session_is_allowed():
    """The gate blocks the future, not the present. A strategy must be able to
    read the bar it is standing on."""
    gate = TimeGate(features(), chains(), "SPY")
    for _ in range(5):
        gate.advance()

    assert not gate.chain(gate.now).empty
    assert gate.row(gate.now) is not None
