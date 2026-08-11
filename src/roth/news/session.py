"""What phase of the trading day it is, right now.

The exchange calendar is authoritative for the regular session, so holidays,
half-days, and the two DST transitions are handled by not hardcoding them. Only
the extended-hours window is a constant here, because pre- and post-market are
broker conventions rather than exchange sessions and the calendar has no
opinion about them.

Phases:

    overnight   between the after-hours close and the next pre-market open
    premarket   04:00 ET until the opening bell
    regular     the exchange session, whenever the calendar says it runs
    afterhours  the closing bell until 20:00 ET
    closed      weekends and holidays with no session tomorrow
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from roth.calendar import trading_sessions
from roth.news.config import AFTERHOURS_CLOSE, MARKET_TZ, PREMARKET_OPEN

PHASES = ("overnight", "premarket", "regular", "afterhours", "closed")


def _as_utc(value) -> datetime:
    """Coerce whatever pandas handed back into a tz-aware UTC datetime."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _et_wall(day: date, wall) -> datetime:
    """A wall-clock America/New_York time on `day`, expressed in UTC."""
    ts = pd.Timestamp(datetime.combine(day, wall)).tz_localize(MARKET_TZ)
    return ts.tz_convert("UTC").to_pydatetime()


@dataclass(frozen=True)
class Session:
    day: date
    open_utc: datetime
    close_utc: datetime
    is_early_close: bool

    @property
    def premarket_start_utc(self) -> datetime:
        return _et_wall(self.day, PREMARKET_OPEN)

    @property
    def afterhours_end_utc(self) -> datetime:
        return _et_wall(self.day, AFTERHOURS_CLOSE)


@dataclass(frozen=True)
class PhaseState:
    """The answer to "what is happening now, and what happens next"."""

    phase: str
    now_utc: datetime
    session: Session | None
    next_session: Session | None
    # When the current phase gives way to the next one. None when nothing is
    # scheduled within the lookahead window.
    next_change_utc: datetime | None

    @property
    def is_open(self) -> bool:
        return self.phase == "regular"

    @property
    def seconds_to_next_change(self) -> float | None:
        if self.next_change_utc is None:
            return None
        return max(0.0, (self.next_change_utc - self.now_utc).total_seconds())

    @property
    def seconds_to_open(self) -> float | None:
        """Until the next opening bell. Zero while the session is running."""
        if self.phase == "regular":
            return 0.0
        target = self.next_session or self.session
        if target is None:
            return None
        if self.session is not None and self.now_utc < self.session.open_utc:
            target = self.session
        return max(0.0, (target.open_utc - self.now_utc).total_seconds())

    def describe(self) -> str:
        label = {
            "premarket": "pre-market",
            "regular": "regular session",
            "afterhours": "after hours",
            "overnight": "overnight",
            "closed": "closed",
        }[self.phase]
        if self.phase == "regular" and self.session is not None:
            close_et = self.session.close_utc.astimezone(_tz())
            suffix = " (early close)" if self.session.is_early_close else ""
            return f"{label}, closes {close_et:%H:%M} ET{suffix}"
        if self.phase in ("premarket", "overnight", "closed"):
            secs = self.seconds_to_open
            if secs is not None:
                return f"{label}, {_humanize(secs)} to the open"
        return label


def _tz():
    import zoneinfo

    return zoneinfo.ZoneInfo(MARKET_TZ)


def _humanize(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


class SessionClock:
    """Phase lookups backed by a cached slice of the exchange calendar.

    The calendar is rebuilt at most once a day. Reaching into
    pandas_market_calendars on every sixty-second poll would work, but it costs
    tens of milliseconds and allocates a DataFrame each time for an answer that
    changes once per midnight.
    """

    def __init__(self, lookahead_days: int = 10) -> None:
        self.lookahead_days = lookahead_days
        self._sessions: list[Session] = []
        self._loaded_for: date | None = None

    def _ensure_loaded(self, today: date) -> None:
        if self._loaded_for == today and self._sessions:
            return
        # Reach back a few days so a Saturday still knows about Friday's
        # session, which the after-hours window can still be inside.
        start = today - timedelta(days=5)
        end = today + timedelta(days=self.lookahead_days)
        df = trading_sessions(start, end)
        self._sessions = [
            Session(
                day=row.day,
                open_utc=_as_utc(row.session_open_utc),
                close_utc=_as_utc(row.session_close_utc),
                is_early_close=bool(row.is_early_close),
            )
            for row in df.itertuples()
        ]
        self._loaded_for = today

    def sessions(self, now_utc: datetime | None = None) -> list[Session]:
        now_utc = now_utc or datetime.now(timezone.utc)
        self._ensure_loaded(now_utc.astimezone(_tz()).date())
        return list(self._sessions)

    def session_on(self, day: date, now_utc: datetime | None = None) -> Session | None:
        for s in self.sessions(now_utc):
            if s.day == day:
                return s
        return None

    def state(self, now_utc: datetime | None = None) -> PhaseState:
        now_utc = now_utc or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware")
        now_utc = now_utc.astimezone(timezone.utc)

        sessions = self.sessions(now_utc)
        today = now_utc.astimezone(_tz()).date()

        today_session = next((s for s in sessions if s.day == today), None)
        future = [s for s in sessions if s.open_utc > now_utc]
        next_session = future[0] if future else None

        # A session that has already opened but whose after-hours window is
        # still running. On a Friday night this is still Friday's session.
        current = today_session
        if current is None:
            prior = [s for s in sessions if s.day < today]
            if prior:
                candidate = prior[-1]
                if now_utc < candidate.afterhours_end_utc:
                    current = candidate

        if current is not None:
            if now_utc < current.premarket_start_utc:
                return PhaseState(
                    "overnight", now_utc, current, next_session, current.premarket_start_utc
                )
            if now_utc < current.open_utc:
                return PhaseState(
                    "premarket", now_utc, current, next_session, current.open_utc
                )
            if now_utc < current.close_utc:
                return PhaseState(
                    "regular", now_utc, current, next_session, current.close_utc
                )
            if now_utc < current.afterhours_end_utc:
                return PhaseState(
                    "afterhours", now_utc, current, next_session, current.afterhours_end_utc
                )

        # Nothing running. Overnight when the next pre-market opens within a
        # day; genuinely closed across a weekend or holiday stretch.
        if next_session is not None:
            start = next_session.premarket_start_utc
            phase = "overnight" if (start - now_utc) <= timedelta(hours=24) else "closed"
            return PhaseState(phase, now_utc, None, next_session, start)

        return PhaseState("closed", now_utc, None, None, None)


def phase_at(now_utc: datetime) -> str:
    """Convenience for tests and one-off checks."""
    return SessionClock().state(now_utc).phase
