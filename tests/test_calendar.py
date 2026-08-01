"""Tests for the trading calendar.

Timezone correctness and OPEX edge cases are the two things most likely to be
quietly wrong, so both are pinned here against known-good dates.
"""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from roth.calendar import (
    build_calendar,
    first_friday,
    load_event_dates,
    monthly_opex,
    nfp_dates_by_rule,
    third_friday,
    to_utc,
    trading_sessions,
)


# -- sessions ---------------------------------------------------------------


def test_sessions_exclude_weekends_and_holidays():
    df = trading_sessions(date(2025, 12, 24), date(2025, 12, 26))
    days = set(df["day"])
    assert date(2025, 12, 25) not in days  # Christmas
    assert date(2025, 12, 24) in days  # early close, still a session


def test_christmas_eve_is_flagged_as_an_early_close():
    df = trading_sessions(date(2025, 12, 24), date(2025, 12, 24))
    assert bool(df.iloc[0]["is_early_close"]) is True


def test_a_normal_session_is_not_an_early_close():
    df = trading_sessions(date(2026, 6, 10), date(2026, 6, 10))
    assert bool(df.iloc[0]["is_early_close"]) is False


# -- timezone ---------------------------------------------------------------


def test_open_maps_to_1330_utc_in_summer():
    """09:30 America/New_York is 13:30 UTC while daylight saving is in effect."""
    assert to_utc(date(2026, 7, 1), time(9, 30)).strftime("%H:%M") == "13:30"


def test_open_maps_to_1430_utc_in_winter():
    """09:30 America/New_York is 14:30 UTC on standard time."""
    assert to_utc(date(2026, 1, 7), time(9, 30)).strftime("%H:%M") == "14:30"


def test_session_open_shifts_across_the_spring_transition():
    """DST began 2026-03-08. Sessions either side must differ by one hour in UTC."""
    before = trading_sessions(date(2026, 3, 6), date(2026, 3, 6)).iloc[0]
    after = trading_sessions(date(2026, 3, 9), date(2026, 3, 9)).iloc[0]

    assert pd.Timestamp(before["session_open_utc"]).strftime("%H:%M") == "14:30"
    assert pd.Timestamp(after["session_open_utc"]).strftime("%H:%M") == "13:30"


def test_session_open_shifts_across_the_autumn_transition():
    """DST ended 2026-11-01. The shift must run the other way too."""
    before = trading_sessions(date(2026, 10, 30), date(2026, 10, 30)).iloc[0]
    after = trading_sessions(date(2026, 11, 2), date(2026, 11, 2)).iloc[0]

    assert pd.Timestamp(before["session_open_utc"]).strftime("%H:%M") == "13:30"
    assert pd.Timestamp(after["session_open_utc"]).strftime("%H:%M") == "14:30"


def test_everything_is_stored_as_utc():
    df = trading_sessions(date(2026, 6, 1), date(2026, 6, 5))
    assert str(pd.Series(df["session_open_utc"]).dt.tz) == "UTC"


# -- OPEX -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2026, 1, date(2026, 1, 16)),
        (2026, 3, date(2026, 3, 20)),
        (2025, 6, date(2025, 6, 20)),
        (2024, 12, date(2024, 12, 20)),
    ],
)
def test_third_friday(year, month, expected):
    assert third_friday(year, month) == expected


def test_every_month_has_exactly_one_monthly_opex():
    sessions = set(trading_sessions(date(2018, 1, 1), date(2026, 12, 31))["day"])
    opex = monthly_opex(date(2018, 1, 1), date(2026, 12, 31), sessions)
    assert len(opex) == 9 * 12


def test_good_friday_opex_moves_to_thursday():
    """April 2025 expiration: the third Friday was Good Friday, so it shifts back."""
    sessions = set(trading_sessions(date(2025, 4, 1), date(2025, 4, 30))["day"])
    assert third_friday(2025, 4) == date(2025, 4, 18)
    assert date(2025, 4, 18) not in sessions  # Good Friday, market closed

    opex = monthly_opex(date(2025, 4, 1), date(2025, 4, 30), sessions)
    assert opex == {date(2025, 4, 17)}


def test_every_opex_lands_on_a_trading_session():
    sessions = set(trading_sessions(date(2018, 1, 1), date(2026, 12, 31))["day"])
    opex = monthly_opex(date(2018, 1, 1), date(2026, 12, 31), sessions)
    assert opex <= sessions


def test_quarterly_opex_is_four_per_year():
    df, _ = build_calendar(date(2024, 1, 1), date(2024, 12, 31))
    assert int(df["is_quarterly_opex"].sum()) == 4


def test_opex_week_covers_the_whole_week():
    df, _ = build_calendar(date(2026, 3, 1), date(2026, 3, 31))
    week = df[df["is_opex_week"]]["day"].tolist()
    assert date(2026, 3, 16) in week  # Monday
    assert date(2026, 3, 20) in week  # Friday, expiration itself


# -- events -----------------------------------------------------------------


def test_nfp_rule_gives_one_date_per_month():
    dates = nfp_dates_by_rule(date(2026, 1, 1), date(2026, 12, 31))
    assert len(dates) == 12
    assert all(d.weekday() == 4 for d in dates)


def test_first_friday():
    assert first_friday(2026, 1) == date(2026, 1, 2)
    assert first_friday(2026, 5) == date(2026, 5, 1)


def test_missing_event_file_reports_unavailable_not_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("roth.calendar.RAW_CALENDAR", tmp_path)
    dates, avail = load_event_dates("fomc")
    assert dates == set()
    assert avail.available is False
    assert avail.source == "missing"


def test_supplied_event_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr("roth.calendar.RAW_CALENDAR", tmp_path)
    (tmp_path / "fomc_dates.csv").write_text("date\n2026-01-28\n2026-03-18\n")

    dates, avail = load_event_dates("fomc")
    assert avail.available is True
    assert dates == {date(2026, 1, 28), date(2026, 3, 18)}


def test_unavailable_event_flag_is_null_not_false(monkeypatch, tmp_path):
    """The distinction that matters: a missing FOMC list must never make every
    day look like a confirmed non-FOMC day."""
    monkeypatch.setattr("roth.calendar.RAW_CALENDAR", tmp_path)
    df, availability = build_calendar(date(2026, 1, 1), date(2026, 1, 31))

    assert df["is_fomc_day"].isna().all()
    assert not (df["is_fomc_day"] == False).any()  # noqa: E712

    fomc = next(a for a in availability if a.name == "fomc")
    assert fomc.available is False


def test_supplied_fomc_file_populates_the_flag(monkeypatch, tmp_path):
    monkeypatch.setattr("roth.calendar.RAW_CALENDAR", tmp_path)
    (tmp_path / "fomc_dates.csv").write_text("date\n2026-01-28\n")

    df, _ = build_calendar(date(2026, 1, 1), date(2026, 1, 31))
    assert int(df["is_fomc_day"].sum()) == 1
    assert bool(df.loc[df["day"] == date(2026, 1, 28), "is_fomc_day"].iloc[0]) is True
