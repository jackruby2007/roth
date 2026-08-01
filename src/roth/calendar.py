"""Trading calendar and event dates.

Everything here is small, static, and independent of the market data feed.

Three categories, and the difference between them matters:

* **Computed exactly** -- NYSE sessions and early closes come from the exchange
  calendar; monthly and quarterly OPEX are the third Friday by definition.
  These are authoritative.
* **Rule-derived** -- NFP is released on the first Friday of the month as a
  rule, but the Bureau of Labor Statistics deviates from it. Flagged as
  rule-derived so it is never mistaken for an authoritative list.
* **Externally supplied** -- FOMC decision days and CPI release days follow no
  derivable rule. They are read from CSV files under `data/raw/calendar/`.
  If a file is absent the corresponding flag is reported as UNAVAILABLE rather
  than silently set to False, because a silent False would quietly mislabel
  every event day in the dataset.

That last distinction is the whole reason this module reports availability
instead of just returning a column of booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
import calendar as calendar_module
from datetime import date, datetime, time, timedelta

import pandas as pd
import pandas_market_calendars as mcal

from roth.config import MARKET_TZ
from roth.paths import RAW_CALENDAR

NYSE_REGULAR_CLOSE = time(16, 0)

# Event files the user supplies. Format: a CSV with a single `date` column of
# YYYY-MM-DD values, header included.
EVENT_FILES: dict[str, str] = {
    "fomc": "fomc_dates.csv",
    "cpi": "cpi_dates.csv",
    "nfp": "nfp_dates.csv",
}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def trading_sessions(start: date, end: date) -> pd.DataFrame:
    """Every NYSE session in range, with open and close stamped in UTC.

    Columns: day, session_open_utc, session_close_utc, is_early_close.

    Timezone handling lives here and nowhere else. The exchange calendar is
    authoritative for when a session actually opened, including the DST
    transitions that move 09:30 ET between 13:30 and 14:30 UTC.
    """
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    if sched.empty:
        return pd.DataFrame(
            columns=["day", "session_open_utc", "session_close_utc", "is_early_close"]
        )

    opens = sched["market_open"].dt.tz_convert("UTC")
    closes = sched["market_close"].dt.tz_convert("UTC")
    local_close = sched["market_close"].dt.tz_convert(MARKET_TZ)

    return pd.DataFrame(
        {
            "day": [ts.date() for ts in sched.index],
            "session_open_utc": opens.to_numpy(),
            "session_close_utc": closes.to_numpy(),
            "is_early_close": (local_close.dt.time != NYSE_REGULAR_CLOSE).to_numpy(),
        }
    )


def session_days(start: date, end: date) -> list[date]:
    df = trading_sessions(start, end)
    return list(df["day"]) if not df.empty else []


# ---------------------------------------------------------------------------
# OPEX -- computed exactly
# ---------------------------------------------------------------------------


def third_friday(year: int, month: int) -> date:
    """Standard monthly option expiration: the third Friday of the month."""
    d = date(year, month, 1)
    # weekday(): Monday is 0, Friday is 4.
    first_friday_day = 1 + (4 - d.weekday()) % 7
    return date(year, month, first_friday_day + 14)


def monthly_opex(start: date, end: date, sessions: set[date] | None = None) -> set[date]:
    """Monthly expiration dates.

    Normally the third Friday. When that Friday is an exchange holiday -- in
    practice Good Friday, which collides with April expiration every few years
    -- expiration moves to the preceding trading day. Without this adjustment
    those months have no OPEX flag at all, because the third Friday is not a
    session.
    """
    out: set[date] = set()
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            d = third_friday(year, month)
            if sessions is not None and d not in sessions:
                # Walk back to the previous trading day.
                probe = d - timedelta(days=1)
                for _ in range(7):
                    if probe in sessions:
                        d = probe
                        break
                    probe -= timedelta(days=1)
                else:
                    continue
            if start <= d <= end:
                out.add(d)
    return out


def quarterly_opex(start: date, end: date, sessions: set[date] | None = None) -> set[date]:
    """Quarterly expiration: third Friday of March, June, September, December."""
    return {d for d in monthly_opex(start, end, sessions) if d.month in (3, 6, 9, 12)}


def opex_weeks(opex_days: set[date]) -> set[date]:
    """Every Monday-to-Friday day belonging to a week containing an OPEX date."""
    out: set[date] = set()
    for d in opex_days:
        monday = d - timedelta(days=d.weekday())
        for offset in range(5):
            out.add(monday + timedelta(days=offset))
    return out


# ---------------------------------------------------------------------------
# NFP -- rule-derived
# ---------------------------------------------------------------------------


def first_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    return date(year, month, 1 + (4 - d.weekday()) % 7)


def nfp_dates_by_rule(start: date, end: date) -> set[date]:
    """First Friday of each month.

    A rule, not a source. The BLS shifts the release when the first Friday
    falls too close to the reference period. Treat as approximate.
    """
    out: set[date] = set()
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            d = first_friday(year, month)
            if start <= d <= end:
                out.add(d)
    return out


# ---------------------------------------------------------------------------
# Externally supplied event dates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventAvailability:
    name: str
    available: bool
    source: str
    count: int
    path: str


def load_event_dates(name: str) -> tuple[set[date], EventAvailability]:
    """Load an event date list from `data/raw/calendar/<name>_dates.csv`.

    Returns the dates and an availability record. When the file is missing the
    dates are empty and `available` is False -- callers must surface that
    rather than treating the absence as "no events occurred".
    """
    filename = EVENT_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown event calendar: {name!r}")

    path = RAW_CALENDAR / filename
    if not path.exists():
        return set(), EventAvailability(name, False, "missing", 0, str(path))

    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.strip().lower() in ("date", "day")), None)
    if col is None:
        return set(), EventAvailability(name, False, "malformed", 0, str(path))

    dates = {
        d.date()
        for d in pd.to_datetime(df[col], errors="coerce").dropna()
    }
    return dates, EventAvailability(name, True, "file", len(dates), str(path))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _pad_range(start: date, end: date) -> tuple[date, date]:
    """Widen a range to whole quarters, plus a week either side.

    Flags like `is_quarter_end` and `is_short_week` are properties of the
    calendar, not of whatever slice of data happens to be loaded. Computing them
    on a truncated range makes the last loaded session look like a quarter end
    and the final partial week look short -- so the same day would carry
    different labels depending on when the table was built. Padding, then
    trimming back, keeps every flag a function of the calendar alone.
    """
    q_start = date(start.year, 3 * ((start.month - 1) // 3) + 1, 1)
    q_end_month = 3 * ((end.month - 1) // 3) + 3
    last_day = calendar_module.monthrange(end.year, q_end_month)[1]
    q_end = date(end.year, q_end_month, last_day)
    return q_start - timedelta(days=7), q_end + timedelta(days=7)


def build_calendar(start: date, end: date) -> tuple[pd.DataFrame, list[EventAvailability]]:
    """One row per trading session with every calendar flag attached.

    Returns the frame and the availability record for each externally supplied
    event calendar, so a report can state plainly which flags are real.
    """
    requested_start, requested_end = start, end
    pad_start, pad_end = _pad_range(start, end)
    df = trading_sessions(pad_start, pad_end)
    if df.empty:
        return df, []

    # Flags are computed over the padded span, then trimmed below.
    start, end = pad_start, pad_end

    days = pd.Series(df["day"])

    session_set = set(df["day"])
    m_opex = monthly_opex(start, end, session_set)
    q_opex = quarterly_opex(start, end, session_set)
    opex_week_days = opex_weeks(m_opex)

    df["day_of_week"] = days.map(lambda d: d.weekday())
    df["day_of_month"] = days.map(lambda d: d.day)
    df["month"] = days.map(lambda d: d.month)
    df["is_monthly_opex"] = days.isin(m_opex)
    df["is_quarterly_opex"] = days.isin(q_opex)
    df["is_opex_week"] = days.isin(opex_week_days)
    df["is_quarter_end"] = days.map(_is_last_session_of_quarter_factory(df))

    # A holiday-shortened week is any week with fewer than five sessions, or one
    # containing an early close.
    week_key = days.map(lambda d: (d.isocalendar().year, d.isocalendar().week))
    sessions_in_week = week_key.map(week_key.value_counts())
    df["is_short_week"] = (sessions_in_week < 5) | (
        week_key.isin(week_key[df["is_early_close"]].unique())
    )

    availability: list[EventAvailability] = []

    for name in ("fomc", "cpi"):
        dates, avail = load_event_dates(name)
        availability.append(avail)
        df[f"is_{name}_day"] = days.isin(dates) if avail.available else pd.NA

    nfp_dates, nfp_avail = load_event_dates("nfp")
    if nfp_avail.available:
        df["is_nfp_day"] = days.isin(nfp_dates)
        availability.append(nfp_avail)
    else:
        df["is_nfp_day"] = days.isin(nfp_dates_by_rule(start, end))
        availability.append(
            EventAvailability("nfp", True, "rule-derived (first Friday)", 0, str(RAW_CALENDAR))
        )

    # Trim back to what was asked for, now that every flag has been computed
    # over whole quarters and whole weeks.
    mask = (df["day"] >= requested_start) & (df["day"] <= requested_end)
    return df[mask].reset_index(drop=True), availability


def _is_last_session_of_quarter_factory(df: pd.DataFrame):
    by_quarter: dict[tuple[int, int], date] = {}
    for d in df["day"]:
        key = (d.year, (d.month - 1) // 3)
        if key not in by_quarter or d > by_quarter[key]:
            by_quarter[key] = d
    last_days = set(by_quarter.values())
    return lambda d: d in last_days


def to_utc(day: date, local_time: time) -> datetime:
    """Convert a wall-clock market time on a given day to UTC.

    Used by the timezone correctness test: 09:30 America/New_York is 13:30 UTC
    in summer and 14:30 UTC in winter, and the harness must get both right.
    """
    naive = datetime.combine(day, local_time)
    localized = pd.Timestamp(naive).tz_localize(MARKET_TZ)
    return localized.tz_convert("UTC").to_pydatetime()
