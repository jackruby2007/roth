"""Tests for the data quality layer.

Each check gets a frame containing the exact defect it is supposed to catch,
plus at least one case it must *not* flag. A check that fires on everything is
as useless as one that never fires.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from roth.calendar import session_days, trading_sessions
from roth.quality import (
    SEVERITY_FLAG,
    SEVERITY_QUARANTINE,
    QualityReport,
    check_daily_bars,
    check_minute_continuity,
    check_option_chain_shape,
    check_option_quotes,
    check_price_gaps,
    check_missing_sessions,
    expected_minutes_by_session,
    exclude_quarantined,
)


def daily_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000}
    return pd.DataFrame([{**base, **r} for r in rows])


def option_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "day": date(2024, 6, 3),
        "expiration_date": date(2024, 6, 21),
        "strike_dollars": 500.0,
        "right": "C",
        "bid": 1.00,
        "ask": 1.05,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# -- missing sessions -------------------------------------------------------


def test_missing_sessions_are_quarantined():
    days = session_days(date(2024, 6, 3), date(2024, 6, 7))
    present = daily_frame([{"day": d} for d in days[:3]])

    findings = check_missing_sessions("SPY", present, date(2024, 6, 3), date(2024, 6, 7))

    assert len(findings) == 2
    assert {f.day for f in findings} == set(days[3:])
    assert all(f.severity == SEVERITY_QUARANTINE for f in findings)


def test_complete_data_produces_no_missing_session_findings():
    days = session_days(date(2024, 6, 3), date(2024, 6, 7))
    present = daily_frame([{"day": d} for d in days])
    assert check_missing_sessions("SPY", present, date(2024, 6, 3), date(2024, 6, 7)) == []


def test_weekends_are_never_reported_missing():
    days = session_days(date(2024, 6, 3), date(2024, 6, 9))
    present = daily_frame([{"day": d} for d in days])
    findings = check_missing_sessions("SPY", present, date(2024, 6, 3), date(2024, 6, 9))
    assert findings == []


# -- daily bars -------------------------------------------------------------


def test_null_price_is_quarantined():
    df = daily_frame([{"day": date(2024, 6, 3), "close": None}])
    findings = check_daily_bars("SPY", df)
    assert [f.check for f in findings] == ["null_price"]


def test_zero_range_is_quarantined():
    df = daily_frame([{"day": date(2024, 6, 3), "high": 100.0, "low": 100.0}])
    assert "zero_range" in [f.check for f in check_daily_bars("SPY", df)]


def test_zero_volume_is_quarantined():
    df = daily_frame([{"day": date(2024, 6, 3), "volume": 0}])
    assert "zero_volume" in [f.check for f in check_daily_bars("SPY", df)]


def test_inverted_bar_is_quarantined():
    df = daily_frame([{"day": date(2024, 6, 3), "high": 98.0, "low": 102.0}])
    assert "inverted_bar" in [f.check for f in check_daily_bars("SPY", df)]


def test_a_healthy_bar_produces_nothing():
    df = daily_frame([{"day": date(2024, 6, 3)}])
    assert check_daily_bars("SPY", df) == []


# -- price gaps -------------------------------------------------------------


def test_large_unexplained_gap_is_quarantined():
    df = daily_frame(
        [
            {"day": date(2024, 6, 3), "close": 100.0},
            {"day": date(2024, 6, 4), "close": 130.0},
        ]
    )
    findings = check_price_gaps("SPY", df)
    assert [f.check for f in findings] == ["price_gap"]
    assert findings[0].severity == SEVERITY_QUARANTINE


def test_a_normal_move_is_not_a_gap():
    df = daily_frame(
        [
            {"day": date(2024, 6, 3), "close": 100.0},
            {"day": date(2024, 6, 4), "close": 102.0},
        ]
    )
    assert check_price_gaps("SPY", df) == []


def test_gap_on_a_known_split_date_is_flagged_not_quarantined():
    """SPY split 2-for-1 on 2005-06-09. That halving is a corporate action, not
    a corrupt bar, and quarantining it would discard a perfectly good session."""
    df = daily_frame(
        [
            {"day": date(2005, 6, 8), "close": 240.0},
            {"day": date(2005, 6, 9), "close": 120.0},
        ]
    )
    findings = check_price_gaps("SPY", df)

    assert len(findings) == 1
    assert findings[0].check == "price_gap_explained_by_split"
    assert findings[0].severity == SEVERITY_FLAG


# -- minute continuity ------------------------------------------------------


def minute_frame(day: date, n: int, tz: str = "UTC") -> pd.DataFrame:
    ts = pd.date_range(f"{day} 13:30", periods=n, freq="1min", tz=tz)
    return pd.DataFrame({"day": day, "ts_utc": ts, "close": 100.0})


def test_complete_session_passes():
    assert check_minute_continuity("SPY", minute_frame(date(2024, 6, 3), 390)) == []


def test_short_session_is_quarantined():
    findings = check_minute_continuity("SPY", minute_frame(date(2024, 6, 3), 300))
    assert [f.check for f in findings] == ["missing_minute_bars"]


def test_early_close_with_its_correct_bar_count_is_not_quarantined():
    """2024-07-03 closed at 13:00 ET: 210 bars is complete, not missing 180."""
    expected = expected_minutes_by_session(date(2024, 7, 3), date(2024, 7, 3))
    assert expected[date(2024, 7, 3)] == 210

    assert check_minute_continuity("SPY", minute_frame(date(2024, 7, 3), 210)) == []


def test_early_close_measured_against_390_would_have_been_wrong():
    """Guard the regression directly: a full 390 bars on a half day is itself
    suspicious, but 210 must pass."""
    findings = check_minute_continuity("SPY", minute_frame(date(2024, 7, 3), 150))
    assert [f.check for f in findings] == ["missing_minute_bars"]


def test_naive_timestamps_are_rejected():
    df = minute_frame(date(2024, 6, 3), 390)
    df["ts_utc"] = df["ts_utc"].dt.tz_localize(None)
    findings = check_minute_continuity("SPY", df)
    assert [f.check for f in findings] == ["naive_timestamp"]


def test_non_utc_timestamps_are_rejected():
    df = minute_frame(date(2024, 6, 3), 390, tz="America/New_York")
    findings = check_minute_continuity("SPY", df)
    assert "non_utc_timestamp" in [f.check for f in findings]


def test_duplicate_timestamps_are_caught():
    df = minute_frame(date(2024, 6, 3), 390)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert "duplicate_timestamps" in [f.check for f in check_minute_continuity("SPY", df)]


# -- option quotes ----------------------------------------------------------


def test_isolated_crossed_quotes_are_flagged_not_quarantined():
    rows = [{"bid": 1.0, "ask": 1.05} for _ in range(199)]
    rows.append({"bid": 1.10, "ask": 1.00})  # crossed
    findings = check_option_quotes("SPY", option_frame(rows))

    crossed = [f for f in findings if f.check == "crossed_quotes"]
    assert len(crossed) == 1
    assert crossed[0].severity == SEVERITY_FLAG


def test_widespread_crossed_quotes_quarantine_the_session():
    """One crossed quote is feed noise. A tenth of the chain is a broken feed."""
    rows = [{"bid": 1.0, "ask": 1.05} for _ in range(90)]
    rows += [{"bid": 1.10, "ask": 1.00} for _ in range(10)]
    findings = check_option_quotes("SPY", option_frame(rows))

    bad = [f for f in findings if f.check == "crossed_quotes_widespread"]
    assert len(bad) == 1
    assert bad[0].severity == SEVERITY_QUARANTINE


def test_wide_spreads_are_flagged_never_quarantined():
    """A far-OTM contract quoted 2.00 wide on a 0.10 mid is a real market, not
    corrupt data. Quarantining it would throw away a valid session."""
    rows = [{"bid": 0.05, "ask": 2.05} for _ in range(10)]
    findings = check_option_quotes("SPY", option_frame(rows))

    wide = [f for f in findings if f.check == "wide_spread"]
    assert len(wide) == 1
    assert wide[0].severity == SEVERITY_FLAG
    assert all(f.severity != SEVERITY_QUARANTINE for f in findings)


def test_tight_market_produces_no_findings():
    rows = [{"bid": 1.00, "ask": 1.02} for _ in range(50)]
    assert check_option_quotes("SPY", option_frame(rows)) == []


def test_thin_chain_is_quarantined():
    rows = []
    for d in (date(2024, 6, 3), date(2024, 6, 4)):
        rows += [{"day": d, "strike_dollars": float(s)} for s in range(400, 600)]
    rows += [{"day": date(2024, 6, 5), "strike_dollars": 500.0}]

    findings = check_option_chain_shape("SPY", option_frame(rows))
    assert [f.day for f in findings] == [date(2024, 6, 5)]
    assert findings[0].severity == SEVERITY_QUARANTINE


# -- report and exclusion ---------------------------------------------------


def test_report_separates_quarantine_from_flags():
    report = QualityReport(date(2024, 1, 1), date(2024, 12, 31), ("SPY",))
    report.findings = [
        *check_daily_bars("SPY", daily_frame([{"day": date(2024, 6, 3), "volume": 0}])),
        *check_option_quotes("SPY", option_frame([{"bid": 0.05, "ask": 2.05}])),
    ]

    assert len(report.quarantined_days) == 1
    assert len(report.flags) == 1


def test_exclude_quarantined_drops_only_matching_symbol_days(monkeypatch):
    monkeypatch.setattr(
        "roth.quality.load_quarantine",
        lambda: {("SPY", date(2024, 6, 4))},
    )
    df = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "QQQ"],
            "day": [date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 4)],
        }
    )
    kept, dropped = exclude_quarantined(df)

    assert dropped == 1
    assert len(kept) == 2
    # QQQ on the same day is untouched: quarantine is per symbol.
    assert ("QQQ", date(2024, 6, 4)) in list(zip(kept["symbol"], kept["day"], strict=False))


def test_exclude_is_a_noop_when_nothing_is_quarantined(monkeypatch):
    monkeypatch.setattr("roth.quality.load_quarantine", lambda: set())
    df = pd.DataFrame({"symbol": ["SPY"], "day": [date(2024, 6, 3)]})
    kept, dropped = exclude_quarantined(df)
    assert dropped == 0
    assert len(kept) == 1


@pytest.mark.parametrize("day,expected", [(date(2024, 7, 3), 210), (date(2024, 6, 3), 390)])
def test_expected_minutes_matches_the_exchange_schedule(day, expected):
    assert expected_minutes_by_session(day, day)[day] == expected


def test_expected_minutes_is_consistent_with_the_calendar():
    sched = trading_sessions(date(2024, 11, 25), date(2024, 11, 30))
    expected = expected_minutes_by_session(date(2024, 11, 25), date(2024, 11, 30))
    early = set(sched[sched["is_early_close"]]["day"])

    assert early  # Thanksgiving Friday
    for day in early:
        assert expected[day] < 390
