"""Tests for the market session state machine.

The bot's whole cadence hangs off `phase`, so the interesting cases are the
boundaries: the two DST transitions, half-days, holidays, and the weekend.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from roth.news.session import SessionClock, phase_at


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def clock() -> SessionClock:
    return SessionClock()


# -- the ordinary day -------------------------------------------------------


@pytest.mark.parametrize(
    "when,expected",
    [
        # Tuesday 2026-08-11, EDT so ET is UTC-4.
        ("2026-08-11T07:59", "overnight"),   # 03:59 ET
        ("2026-08-11T08:00", "premarket"),   # 04:00 ET, pre-market opens
        ("2026-08-11T13:29", "premarket"),   # 09:29 ET
        ("2026-08-11T13:30", "regular"),     # 09:30 ET, opening bell
        ("2026-08-11T19:59", "regular"),     # 15:59 ET
        ("2026-08-11T20:00", "afterhours"),  # 16:00 ET, closing bell
        ("2026-08-11T23:59", "afterhours"),  # 19:59 ET
        ("2026-08-12T00:00", "overnight"),   # 20:00 ET, after-hours ends
    ],
)
def test_phase_boundaries_through_a_normal_session(clock, when, expected):
    assert clock.state(utc(when)).phase == expected


# -- daylight saving --------------------------------------------------------


def test_open_is_1330_utc_in_summer(clock):
    assert clock.state(utc("2026-08-11T13:29")).phase == "premarket"
    assert clock.state(utc("2026-08-11T13:30")).phase == "regular"


def test_open_is_1430_utc_in_winter(clock):
    """The same wall-clock open is an hour later in UTC on standard time."""
    assert clock.state(utc("2026-01-07T14:29")).phase == "premarket"
    assert clock.state(utc("2026-01-07T14:30")).phase == "regular"
    # 13:30 UTC is 08:30 ET in January -- still pre-market, not the open.
    assert clock.state(utc("2026-01-07T13:30")).phase == "premarket"


def test_premarket_start_tracks_the_offset(clock):
    """04:00 ET is 08:00 UTC in summer and 09:00 UTC in winter."""
    assert clock.state(utc("2026-08-11T08:00")).phase == "premarket"
    assert clock.state(utc("2026-01-07T08:59")).phase == "overnight"
    assert clock.state(utc("2026-01-07T09:00")).phase == "premarket"


# -- non-sessions -----------------------------------------------------------


def test_saturday_is_closed(clock):
    assert clock.state(utc("2026-08-15T18:00")).phase == "closed"


def test_sunday_evening_is_overnight_because_monday_opens(clock):
    """Within a day of the next pre-market, so the bot wakes up rather than
    sleeping through the open."""
    assert clock.state(utc("2026-08-16T23:00")).phase == "overnight"


def test_independence_day_observed_is_not_a_session(clock):
    """2026-07-04 is a Saturday, so the holiday is observed Friday the 3rd."""
    assert clock.state(utc("2026-07-03T17:00")).phase != "regular"


def test_thanksgiving_is_not_a_session(clock):
    assert clock.state(utc("2026-11-26T16:00")).phase != "regular"


def test_early_close_is_flagged_and_ends_the_session_at_1300_et(clock):
    """The day after Thanksgiving closes at 13:00 ET, i.e. 18:00 UTC."""
    state = clock.state(utc("2026-11-27T17:59"))
    assert state.phase == "regular"
    assert state.session is not None and state.session.is_early_close
    assert clock.state(utc("2026-11-27T18:00")).phase == "afterhours"


# -- derived quantities -----------------------------------------------------


def test_seconds_to_open_is_zero_during_the_session(clock):
    assert clock.state(utc("2026-08-11T15:00")).seconds_to_open == 0.0


def test_seconds_to_open_counts_down_in_premarket(clock):
    state = clock.state(utc("2026-08-11T13:00"))
    assert state.seconds_to_open == pytest.approx(30 * 60)


def test_next_change_stops_at_the_opening_bell(clock):
    state = clock.state(utc("2026-08-11T13:00"))
    assert state.next_change_utc == utc("2026-08-11T13:30")


def test_state_requires_an_aware_datetime(clock):
    with pytest.raises(ValueError):
        clock.state(datetime(2026, 8, 11, 13, 0))


def test_describe_names_an_early_close(clock):
    assert "early close" in clock.state(utc("2026-11-27T17:00")).describe()


def test_module_level_helper_agrees_with_the_clock():
    assert phase_at(utc("2026-08-11T13:30")) == "regular"
