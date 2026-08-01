"""Ingestion.

Pulls data from ThetaData and lands it in `data/raw/`, immutably, one partition
per symbol per day. Every downloader is resumable: the manifest is consulted
first and days already on disk are skipped, so an interrupted backfill picks up
where it stopped instead of starting over.

Nothing in this module transforms data. Whatever the provider returned is what
lands on disk, with two mechanical exceptions that are lossless and reversible:
ThetaData's integer-scaled strikes are widened to dollars, and its
`ms_of_day` + `date` pair is resolved into a real UTC timestamp. Both are
recorded as *additional* columns; the originals are kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from roth.calendar import build_calendar, session_days
from roth.config import BOUNDS, SYMBOLS
from roth.data.pilot import _parse_csv_pages
from roth.data.thetadata import ThetaClient, ThetaError, ThetaNoData
from roth.paths import (
    RAW_CALENDAR,
    RAW_OPTION_EOD,
    RAW_UNDERLYING_DAILY,
    RAW_UNDERLYING_MINUTE,
    RAW_VIX,
    ensure_dirs,
)
from roth.storage import ImmutableRawError, already_downloaded, write_derived, write_raw

# Dataset names as recorded in the manifest.
DS_OPTION_EOD = "option_eod"
DS_UNDERLYING_DAILY = "underlying_daily"
DS_UNDERLYING_MINUTE = "underlying_minute"
DS_VIX = "vix_daily"


@dataclass
class IngestResult:
    dataset: str
    symbol: str
    days_requested: int
    days_written: int
    days_skipped: int
    rows_written: int
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.dataset}/{self.symbol}: {self.days_written} days written, "
            f"{self.days_skipped} already present, {self.rows_written:,} rows, "
            f"{len(self.failures)} failures"
        )


# ---------------------------------------------------------------------------
# Shared normalisation
# ---------------------------------------------------------------------------


def _utc_timestamp(day_col: pd.Series, ms_col: pd.Series) -> pd.Series:
    """Resolve ThetaData's (date, ms_of_day) pair into a UTC timestamp.

    ThetaData stamps `ms_of_day` in Eastern wall-clock milliseconds since
    midnight. Localising to America/New_York before converting is what makes
    this correct across daylight saving transitions; adding a fixed offset is
    not.
    """
    days = pd.to_datetime(day_col.astype(int).astype(str), format="%Y%m%d")
    naive = days + pd.to_timedelta(ms_col.astype("int64"), unit="ms")
    return (
        naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )


def _normalise_option_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns to a raw option frame without dropping originals."""
    if "strike" in df.columns:
        df["strike_dollars"] = df["strike"] / 1000.0
    if "expiration" in df.columns:
        df["expiration_date"] = pd.to_datetime(
            df["expiration"].astype(int).astype(str), format="%Y%m%d"
        ).dt.date
    if "date" in df.columns:
        df["day"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d").dt.date
        if "ms_of_day" in df.columns:
            df["ts_utc"] = _utc_timestamp(df["date"], df["ms_of_day"])
    return df


# ---------------------------------------------------------------------------
# Underlying
# ---------------------------------------------------------------------------


def ingest_underlying_daily(
    symbol: str, start: date, end: date, replace: bool = False
) -> IngestResult:
    """Daily OHLCV for one underlying.

    Written as a single partition per year rather than per day: daily bars are
    tiny and one file per session would be thousands of near-empty files.
    """
    ensure_dirs()
    result = IngestResult(DS_UNDERLYING_DAILY, symbol, 0, 0, 0, 0)
    done = already_downloaded(DS_UNDERLYING_DAILY, symbol)

    with ThetaClient() as client:
        client.check_connection()
        for year in range(start.year, end.year + 1):
            y_start = max(start, date(year, 1, 1))
            y_end = min(end, date(year, 12, 31))
            marker = date(year, 1, 1)
            result.days_requested += 1

            if marker in done and not replace:
                result.days_skipped += 1
                continue

            try:
                pages = [p.text for p in client.stock_ohlc(symbol, y_start, y_end, interval_ms=0)]
            except ThetaNoData:
                continue
            except ThetaError as exc:
                result.failures.append(f"{symbol} {year}: {exc}")
                continue

            df = _parse_csv_pages(pages)
            if df.empty:
                continue

            df["day"] = pd.to_datetime(
                df["date"].astype(int).astype(str), format="%Y%m%d"
            ).dt.date
            df["symbol"] = symbol

            write_raw(df, RAW_UNDERLYING_DAILY, DS_UNDERLYING_DAILY, symbol, marker, replace)
            result.days_written += 1
            result.rows_written += len(df)

    return result


def ingest_underlying_minute(
    symbol: str, start: date, end: date, replace: bool = False
) -> IngestResult:
    """1-minute OHLCV for one underlying, one partition per session."""
    ensure_dirs()
    result = IngestResult(DS_UNDERLYING_MINUTE, symbol, 0, 0, 0, 0)
    done = already_downloaded(DS_UNDERLYING_MINUTE, symbol)
    sessions = session_days(start, end)
    result.days_requested = len(sessions)

    with ThetaClient() as client:
        client.check_connection()
        for session in sessions:
            if session in done and not replace:
                result.days_skipped += 1
                continue

            try:
                pages = [
                    p.text
                    for p in client.stock_ohlc(
                        symbol, session, session, interval_ms=60_000, rth_only=BOUNDS.rth_only
                    )
                ]
            except ThetaNoData:
                continue
            except ThetaError as exc:
                result.failures.append(f"{symbol} {session}: {exc}")
                continue

            df = _parse_csv_pages(pages)
            if df.empty:
                continue

            df["day"] = session
            df["symbol"] = symbol
            if "ms_of_day" in df.columns and "date" in df.columns:
                df["ts_utc"] = _utc_timestamp(df["date"], df["ms_of_day"])

            write_raw(df, RAW_UNDERLYING_MINUTE, DS_UNDERLYING_MINUTE, symbol, session, replace)
            result.days_written += 1
            result.rows_written += len(df)

    return result


def ingest_vix_daily(start: date, end: date, replace: bool = False) -> IngestResult:
    """VIX daily close. Stored under the pseudo-symbol VIX."""
    ensure_dirs()
    result = IngestResult(DS_VIX, "VIX", 0, 0, 0, 0)
    done = already_downloaded(DS_VIX, "VIX")

    with ThetaClient() as client:
        client.check_connection()
        for year in range(start.year, end.year + 1):
            marker = date(year, 1, 1)
            result.days_requested += 1
            if marker in done and not replace:
                result.days_skipped += 1
                continue

            y_start = max(start, date(year, 1, 1))
            y_end = min(end, date(year, 12, 31))
            try:
                pages = [p.text for p in client.stock_ohlc("VIX", y_start, y_end, interval_ms=0)]
            except ThetaNoData:
                continue
            except ThetaError as exc:
                result.failures.append(f"VIX {year}: {exc}")
                continue

            df = _parse_csv_pages(pages)
            if df.empty:
                continue
            df["day"] = pd.to_datetime(
                df["date"].astype(int).astype(str), format="%Y%m%d"
            ).dt.date
            df["symbol"] = "VIX"

            write_raw(df, RAW_VIX, DS_VIX, "VIX", marker, replace)
            result.days_written += 1
            result.rows_written += len(df)

    return result


# ---------------------------------------------------------------------------
# Option chains, EOD
# ---------------------------------------------------------------------------


def ingest_option_eod(
    symbol: str,
    start: date,
    end: date,
    spot_by_day: dict[date, float] | None = None,
    replace: bool = False,
) -> IngestResult:
    """End-of-day option chains, bounded to +/-10% of spot and 60 days to expiry.

    The bounds are applied here, at download time. Pulling the whole chain and
    filtering later would defeat the point: the unbounded dataset is the size
    bomb this harness exists to avoid.
    """
    ensure_dirs()
    result = IngestResult(DS_OPTION_EOD, symbol, 0, 0, 0, 0)
    done = already_downloaded(DS_OPTION_EOD, symbol)
    sessions = session_days(start, end)
    result.days_requested = len(sessions)

    if spot_by_day is None:
        spot_by_day = load_spot_by_day(symbol)

    with ThetaClient() as client:
        client.check_connection()
        all_expirations = client.list_expirations(symbol)

        for session in sessions:
            if session in done and not replace:
                result.days_skipped += 1
                continue

            spot = spot_by_day.get(session)
            if spot is None:
                result.failures.append(
                    f"{symbol} {session}: no underlying close on file, cannot bound strikes. "
                    "Ingest underlying daily data first."
                )
                continue

            lo = spot * (1 - BOUNDS.strike_pct)
            hi = spot * (1 + BOUNDS.strike_pct)
            in_window = [
                e
                for e in all_expirations
                if session <= e <= session + timedelta(days=BOUNDS.max_dte)
            ]

            frames: list[pd.DataFrame] = []
            for exp in in_window:
                try:
                    pages = [
                        p.text for p in client.bulk_option_eod(symbol, exp, session, session)
                    ]
                except ThetaNoData:
                    continue
                except ThetaError as exc:
                    result.failures.append(f"{symbol} {session} exp={exp}: {exc}")
                    continue

                df = _parse_csv_pages(pages)
                if df.empty or "strike" not in df.columns:
                    continue

                df = _normalise_option_frame(df)
                df = df[(df["strike_dollars"] >= lo) & (df["strike_dollars"] <= hi)]
                if not df.empty:
                    frames.append(df)

            if not frames:
                continue

            day_df = pd.concat(frames, ignore_index=True)
            day_df["symbol"] = symbol
            day_df["day"] = session
            day_df["underlying_close"] = spot

            try:
                write_raw(day_df, RAW_OPTION_EOD, DS_OPTION_EOD, symbol, session, replace)
            except ImmutableRawError as exc:
                result.failures.append(str(exc))
                continue

            result.days_written += 1
            result.rows_written += len(day_df)

    return result


def load_spot_by_day(symbol: str) -> dict[date, float]:
    """Closing price per session from already-ingested daily bars."""
    from roth.storage import read_dataset

    df = read_dataset(RAW_UNDERLYING_DAILY, symbols=[symbol])
    if df.empty or "close" not in df.columns:
        return {}
    return dict(zip(df["day"], df["close"].astype(float), strict=False))


# ---------------------------------------------------------------------------
# Calendar -- no network required
# ---------------------------------------------------------------------------


def ingest_calendar(start: date, end: date) -> tuple[pd.DataFrame, list]:
    """Build the trading calendar and write it to derived storage.

    This one needs no data feed at all, so it runs and is verifiable before any
    subscription exists.
    """
    ensure_dirs()
    df, availability = build_calendar(start, end)
    if not df.empty:
        write_derived(df, RAW_CALENDAR, "trading_calendar")
    return df, availability


def default_backfill_range() -> tuple[date, date]:
    """A sensible default window: 2018 through yesterday."""
    return date(2018, 1, 1), datetime.now(timezone.utc).date() - timedelta(days=1)


__all__ = [
    "DS_OPTION_EOD",
    "DS_UNDERLYING_DAILY",
    "DS_UNDERLYING_MINUTE",
    "DS_VIX",
    "IngestResult",
    "SYMBOLS",
    "default_backfill_range",
    "ingest_calendar",
    "ingest_option_eod",
    "ingest_underlying_daily",
    "ingest_underlying_minute",
    "ingest_vix_daily",
    "load_spot_by_day",
]
