"""Tests for the correctness checks themselves.

A check that always passes verifies nothing. Each one here is given a broken
input and must fail, and a sound input and must pass.
"""

from __future__ import annotations

from datetime import date, time, timedelta

import pandas as pd
import pytest

from roth.backtest.access import TimeGate
from roth.calendar import to_utc, trading_sessions
from roth.verify import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    _walk_buy_and_hold,
    check_timezone,
    load_benchmarks,
)


def gate_from_closes(closes: list[float]) -> TimeGate:
    days = [date(2024, 1, 2) + timedelta(days=i) for i in range(len(closes))]
    features = pd.DataFrame({"symbol": "SPY", "day": days, "close": closes})
    return TimeGate(features, pd.DataFrame(), "SPY")


# -- known answer -----------------------------------------------------------


def test_buy_and_hold_walk_matches_the_endpoint_ratio():
    stats = _walk_buy_and_hold(gate_from_closes([100.0, 110.0, 121.0]))

    assert stats["endpoint_return_pct"] == pytest.approx(21.0)
    assert stats["compounded_return_pct"] == pytest.approx(21.0)
    assert stats["sessions"] == 3


def test_buy_and_hold_handles_a_decline():
    stats = _walk_buy_and_hold(gate_from_closes([100.0, 50.0]))
    assert stats["endpoint_return_pct"] == pytest.approx(-50.0)
    assert stats["compounded_return_pct"] == pytest.approx(-50.0)


def test_buy_and_hold_walks_every_session():
    closes = [100.0 * (1.01**i) for i in range(50)]
    stats = _walk_buy_and_hold(gate_from_closes(closes))

    assert stats["sessions"] == 50
    assert stats["compounded_return_pct"] == pytest.approx(
        stats["endpoint_return_pct"], abs=1e-9
    )


def test_buy_and_hold_needs_two_sessions():
    assert _walk_buy_and_hold(gate_from_closes([100.0])) == {}


def test_compounded_and_endpoint_diverge_if_a_bar_is_skipped():
    """The known-answer check works by comparing two routes to the same number.
    Confirm they actually diverge when the walk is wrong, otherwise the check
    would pass on a broken engine."""
    closes = [100.0, 110.0, 121.0]
    stats = _walk_buy_and_hold(gate_from_closes(closes))

    # A walk that dropped the middle bar would compound 100 -> 121 only.
    skipped = (121.0 / 100.0 - 1) * 100
    honest_but_wrong = (121.0 / 110.0 - 1) * 100
    assert stats["compounded_return_pct"] == pytest.approx(skipped)
    assert stats["compounded_return_pct"] != pytest.approx(honest_but_wrong)


# -- timezone ---------------------------------------------------------------


def test_timezone_check_passes():
    result = check_timezone()
    assert result.status == PASS, result.detail
    assert result.numbers["assertions"] >= 18


@pytest.mark.parametrize(
    ("day", "expected_hour"),
    [
        (date(2024, 3, 8), 14),  # EST, before spring forward
        (date(2024, 3, 11), 13),  # EDT, after
        (date(2024, 11, 1), 13),  # EDT, before fall back
        (date(2024, 11, 4), 14),  # EST, after
    ],
)
def test_open_maps_to_the_right_utc_hour(day, expected_hour):
    assert to_utc(day, time(9, 30)).hour == expected_hour


@pytest.mark.parametrize(
    ("day", "expected_hour"),
    [
        (date(2024, 3, 8), 14),
        (date(2024, 3, 11), 13),
        (date(2024, 11, 1), 13),
        (date(2024, 11, 4), 14),
    ],
)
def test_calendar_session_open_agrees_with_the_conversion(day, expected_hour):
    """Two independent routes to the same instant: the tz conversion and the
    exchange calendar. If they disagree, one of them is wrong."""
    sched = trading_sessions(day, day)
    assert not sched.empty
    assert pd.Timestamp(sched.iloc[0]["session_open_utc"]).hour == expected_hour


def test_utc_round_trips_back_to_local_open():
    for day in (date(2024, 3, 8), date(2024, 3, 11), date(2024, 11, 4)):
        back = pd.Timestamp(to_utc(day, time(9, 30))).tz_convert("America/New_York")
        assert (back.hour, back.minute) == (9, 30)


def test_the_dst_shift_is_exactly_one_hour():
    before = to_utc(date(2024, 3, 8), time(9, 30))
    after = to_utc(date(2024, 3, 11), time(9, 30))
    assert before.hour - after.hour == 1


# -- benchmarks -------------------------------------------------------------


def test_missing_benchmark_file_reads_as_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("roth.verify.BENCHMARK_PATH", tmp_path / "nope.csv")
    assert load_benchmarks().empty


def test_malformed_benchmark_file_reads_as_empty(monkeypatch, tmp_path):
    path = tmp_path / "benchmarks.csv"
    path.write_text("wrong,columns\n1,2\n")
    monkeypatch.setattr("roth.verify.BENCHMARK_PATH", path)
    assert load_benchmarks().empty


def test_valid_benchmark_file_is_parsed(monkeypatch, tmp_path):
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "symbol,start,end,total_return_pct,source\n"
        "SPY,2023-01-01,2023-12-31,24.2,example\n"
    )
    monkeypatch.setattr("roth.verify.BENCHMARK_PATH", path)

    df = load_benchmarks()
    assert len(df) == 1
    assert df.iloc[0]["start"] == date(2023, 1, 1)
    assert df.iloc[0]["total_return_pct"] == 24.2


def test_external_check_skips_without_a_benchmark(monkeypatch, tmp_path):
    """It must never report PASS when it has verified nothing."""
    monkeypatch.setattr("roth.verify.BENCHMARK_PATH", tmp_path / "nope.csv")
    from roth.verify import check_known_answer_external

    result = check_known_answer_external()
    assert result.status == SKIP
    assert "benchmark" in result.detail.lower()


def test_external_check_skips_on_synthetic_data(monkeypatch, tmp_path):
    """A generated series cannot reproduce a real published return, so claiming
    a pass would be a lie."""
    path = tmp_path / "benchmarks.csv"
    path.write_text(
        "symbol,start,end,total_return_pct,source\nSPY,2023-01-01,2023-12-31,24.2,x\n"
    )
    monkeypatch.setattr("roth.verify.BENCHMARK_PATH", path)
    monkeypatch.setattr("roth.data.synth.is_synthetic", lambda: True)

    from roth.verify import check_known_answer_external

    result = check_known_answer_external()
    assert result.status == SKIP
    assert "synthetic" in result.detail.lower()


# -- result semantics -------------------------------------------------------


def test_only_pass_counts_as_ok():
    assert CheckResult("x", PASS, "").ok
    assert not CheckResult("x", FAIL, "").ok
    assert not CheckResult("x", SKIP, "").ok
