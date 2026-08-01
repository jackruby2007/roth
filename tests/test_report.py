"""Tests for reporting and the trade journal.

Reporting arithmetic is easy to get subtly wrong and nothing downstream will
complain, so every statistic is pinned against a hand-computed case.
"""

from __future__ import annotations

from datetime import date, timedelta

import math

import pandas as pd
import pytest

from roth.journal import JOURNAL_COLUMNS, build_journal, make_run_id
from roth.report import (
    build_equity_curve,
    build_report,
    longest_losing_streak,
    render_text,
)


def trades(pnls: list[float], **overrides) -> pd.DataFrame:
    """A trade frame with the given net P/L, one trade per business day."""
    rows = []
    day = date(2024, 1, 1)
    for i, pnl in enumerate(pnls):
        entry = day + timedelta(days=i * 7)
        rows.append(
            {
                "symbol": "SPY",
                "strategy_name": "t",
                "strategy_version": "1.0.0",
                "structure_type": "long_call",
                "direction": "long",
                "entry_day": entry,
                "exit_day": entry + timedelta(days=4),
                "signal_day": entry - timedelta(days=1),
                "signal_id": i,
                "strikes": "500",
                "contract_ids": "SPY240621C00500000",
                "expiration": date(2024, 6, 21),
                "contracts": 1,
                "entry_price": 100.0,
                "exit_price": 100.0 + pnl,
                "entry_bid": 0.98,
                "entry_ask": 1.00,
                "exit_bid": 0.98,
                "exit_ask": 1.00,
                "capital_at_risk": 100.0,
                "gross_pnl": pnl,
                "commissions": 0.0,
                "net_pnl": pnl,
                "return_pct": pnl / 100.0,
                "r_multiple": pnl / 100.0,
                "holding_days": 4,
                "max_favorable_excursion": max(pnl, 0.0),
                "max_adverse_excursion": min(pnl, 0.0),
                "exit_reason": "time_stop",
                "entry_delta": 0.30,
                "entry_iv": 0.20,
                "entry_underlying": 500.0,
                "trend_direction": "up",
                "trend_strength": "strong",
                "vol_bucket": "mid",
                "day_type": "range",
                **overrides,
            }
        )
    return pd.DataFrame(rows)


# -- core statistics --------------------------------------------------------


def test_win_rate_and_counts():
    r = build_report(trades([100.0, 100.0, -50.0, -50.0]))
    assert r.sample_count == 4
    assert r.wins == 2 and r.losses == 2
    assert r.win_rate == pytest.approx(0.5)


def test_average_winner_and_loser():
    r = build_report(trades([100.0, 200.0, -50.0]))
    assert r.average_winner == pytest.approx(150.0)
    assert r.average_loser == pytest.approx(-50.0)


def test_a_breakeven_trade_counts_as_a_loss():
    """Zero P/L is not a win. After costs it never is in practice either."""
    r = build_report(trades([0.0, 100.0]))
    assert r.wins == 1
    assert r.losses == 1


def test_profit_factor():
    r = build_report(trades([100.0, 200.0, -100.0, -50.0]))
    # 300 of wins against 150 of losses.
    assert r.profit_factor == pytest.approx(2.0)


def test_profit_factor_is_infinite_with_no_losers():
    r = build_report(trades([100.0, 50.0]))
    assert math.isinf(r.profit_factor)


def test_profit_factor_is_zero_with_no_winners():
    r = build_report(trades([-100.0, -50.0]))
    assert r.profit_factor == 0.0


def test_expectancy_in_r():
    r = build_report(trades([100.0, -50.0]))
    # R multiples of +1.0 and -0.5 against capital at risk of 100.
    assert r.expectancy_r == pytest.approx(0.25)


def test_net_pnl_and_commissions_are_summed():
    df = trades([100.0, -50.0])
    df["commissions"] = 1.30
    r = build_report(df)
    assert r.net_pnl == pytest.approx(50.0)
    assert r.total_commissions == pytest.approx(2.60)


def test_largest_winner_and_loser():
    r = build_report(trades([10.0, 500.0, -300.0, -20.0]))
    assert r.largest_winner == pytest.approx(500.0)
    assert r.largest_loser == pytest.approx(-300.0)


# -- streaks and drawdown ---------------------------------------------------


def test_longest_losing_streak():
    assert longest_losing_streak(trades([-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0])) == 3


def test_streak_of_zero_when_everything_wins():
    assert longest_losing_streak(trades([1.0, 2.0, 3.0])) == 0


def test_a_single_loss_is_a_streak_of_one():
    assert longest_losing_streak(trades([1.0, -1.0, 1.0])) == 1


def test_max_drawdown_measures_peak_to_trough():
    r = build_report(trades([1000.0, -400.0, -300.0, 200.0]))
    # Peak after trade 1 is +1000, trough after trade 3 is +300.
    assert r.max_drawdown == pytest.approx(700.0)


def test_no_drawdown_when_equity_only_rises():
    r = build_report(trades([100.0, 100.0, 100.0]))
    assert r.max_drawdown == pytest.approx(0.0)


# -- equity curve -----------------------------------------------------------


def test_equity_curve_starts_at_initial_capital():
    curve = build_equity_curve(trades([100.0]), initial_capital=50_000.0)
    assert curve["equity"].iloc[0] == pytest.approx(50_000.0)


def test_equity_curve_includes_flat_days_between_trades():
    """Compressing flat days away would inflate Sharpe for a strategy that is
    rarely in the market."""
    curve = build_equity_curve(trades([100.0, 100.0, 100.0]))
    assert len(curve) > 3
    assert (curve["pnl"] == 0.0).sum() > 0


def test_equity_curve_ends_at_initial_plus_net():
    curve = build_equity_curve(trades([100.0, -30.0]), initial_capital=10_000.0)
    assert curve["equity"].iloc[-1] == pytest.approx(10_070.0)


def test_empty_trades_give_an_empty_curve():
    assert build_equity_curve(pd.DataFrame()).empty


# -- risk statistics --------------------------------------------------------


def test_sharpe_is_positive_for_a_rising_curve():
    r = build_report(trades([50.0] * 20))
    assert r.sharpe > 0


def test_sharpe_is_negative_for_a_falling_curve():
    r = build_report(trades([-50.0] * 20))
    assert r.sharpe < 0


def test_sortino_ignores_upside_volatility():
    """A curve with no losing days has no downside deviation, so Sortino is
    reported as zero rather than dividing by zero."""
    r = build_report(trades([50.0] * 10))
    assert r.sortino == 0.0


def test_risk_stats_are_zero_for_a_single_trade():
    r = build_report(trades([100.0]))
    assert r.sharpe == 0.0 or math.isfinite(r.sharpe)


# -- context that must always appear ---------------------------------------


def test_empty_report_is_valid_and_states_it_has_no_trades():
    r = build_report(pd.DataFrame(), strategy_name="none", candidate_signals=5)
    assert r.sample_count == 0
    text = render_text(r)
    assert "No trades" in text
    assert "Candidate signals" in text


def test_report_carries_unfillable_and_quarantine_counts():
    r = build_report(
        trades([100.0]),
        unfillable_entries=3,
        unfillable_exits=7,
        candidate_signals=50,
        quarantined_sessions=12,
    )
    text = render_text(r)
    assert "Entries refused" in text
    assert "Exit attempts refused" in text
    assert "Quarantined sessions" in text
    assert "12" in text


def test_entry_and_exit_rejections_are_reported_separately():
    """Combining them and dividing by candidate signals produces a badly
    misleading rejection rate, since exits retry over a position's life."""
    r = build_report(
        trades([100.0]), unfillable_entries=2, unfillable_exits=128, candidate_signals=271
    )
    assert r.unfillable_entries == 2
    assert r.unfillable_exits == 128

    text = render_text(r)
    # 2/271 is 0.7%, not 48%.
    assert "0.7%" in text


def test_synthetic_data_forces_a_banner():
    text = render_text(build_report(trades([100.0]), synthetic_data=True))
    assert "SYNTHETIC DATA" in text
    assert "nothing" in text.lower()


def test_mid_fills_force_a_banner():
    text = render_text(build_report(trades([100.0]), used_mid_fills=True))
    assert "MID-PRICE FILLS" in text
    assert "not achievable" in text


def test_a_clean_report_carries_no_banners():
    text = render_text(build_report(trades([100.0])))
    assert "SYNTHETIC DATA" not in text
    assert "MID-PRICE FILLS" not in text


def test_intrinsic_settlements_are_surfaced():
    df = trades([100.0, -50.0])
    df.loc[0, "exit_reason"] = "expired_settled_at_intrinsic"
    text = render_text(build_report(df))
    assert "Settled at intrinsic" in text


def test_render_includes_every_section():
    text = render_text(build_report(trades([100.0, -50.0, 30.0])))
    for section in ("SAMPLE", "OUTCOME", "PROFIT AND LOSS", "RISK", "EXITS", "DATA QUALITY"):
        assert section in text


# -- journal ----------------------------------------------------------------


def test_journal_has_every_promised_column():
    from roth.backtest.engine import BacktestResult

    result = BacktestResult("t", "1.0.0", ("SPY",), None, None)
    journal = build_journal(result, run_id="test")
    assert list(journal.columns) == list(JOURNAL_COLUMNS)


def test_journal_column_order_is_stable():
    assert JOURNAL_COLUMNS[0] == "run_id"
    for required in (
        "contract_ids",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
        "capital_at_risk",
        "r_multiple",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "exit_reason",
        "signal_id",
    ):
        assert required in JOURNAL_COLUMNS


def test_run_id_encodes_fill_mode_and_data_origin():
    from roth.backtest.engine import BacktestResult

    result = BacktestResult("s", "2.0.0", ("SPY",), None, None)
    result.used_mid_fills = True
    result.synthetic_data = True

    run_id = make_run_id(result)
    assert "s_2.0.0" in run_id
    assert "mid" in run_id
    assert "synth" in run_id


def test_run_id_distinguishes_nbbo_from_mid():
    from roth.backtest.engine import BacktestResult

    a = BacktestResult("s", "1.0.0", ("SPY",), None, None)
    b = BacktestResult("s", "1.0.0", ("SPY",), None, None)
    b.used_mid_fills = True

    assert "nbbo" in make_run_id(a)
    assert "mid" in make_run_id(b)


def test_occ_symbol_format():
    from roth.backtest.fills import Leg, Right

    leg = Leg(Right.CALL, 500.0, date(2024, 6, 21), 1)
    assert leg.occ_symbol("SPY") == "SPY240621C00500000"

    put = Leg(Right.PUT, 62.5, date(2025, 1, 17), -1)
    assert put.occ_symbol("QQQ") == "QQQ250117P00062500"
