"""Pilot download.

Downloads exactly one month of SPY option quote data at the intended resolution
and strike/expiry bounds, then reports two measured numbers:

1. Actual disk footprint per month
2. Actual wall-clock download time per month

Those two numbers get extrapolated to the full history so the real cost of a
backfill is known before a subscription tier is chosen. Nothing else in the
harness depends on this module; it exists purely to inform that decision.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as mcal

from roth.config import BOUNDS
from roth.data.sizing import SizingAssumptions, save_assumptions
from roth.data.thetadata import ThetaClient, ThetaNoData
from roth.paths import PILOT, dir_size_bytes, ensure_dirs


@dataclass
class PilotResult:
    symbol: str
    start: date
    end: date
    trading_days: int
    expirations_touched: int
    rows_written: int
    bytes_on_disk: int
    bytes_over_wire: int
    wall_clock_seconds: float
    requests_made: int
    failures: list[str] = field(default_factory=list)

    @property
    def bytes_per_row(self) -> float:
        return self.bytes_on_disk / self.rows_written if self.rows_written else 0.0

    @property
    def seconds_per_request(self) -> float:
        return self.wall_clock_seconds / self.requests_made if self.requests_made else 0.0

    def to_assumptions(self) -> SizingAssumptions:
        return SizingAssumptions(
            bytes_per_quote_row=round(self.bytes_per_row, 3),
            seconds_per_bulk_request=round(self.seconds_per_request, 4),
            measured=True,
            measured_from=(
                f"{self.symbol} pilot {self.start.isoformat()}..{self.end.isoformat()} "
                f"({self.rows_written:,} rows, {self.requests_made:,} requests)"
            ),
        )


def nyse_sessions(start: date, end: date) -> list[date]:
    """Trading days between start and end inclusive, per the NYSE calendar."""
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.date() for ts in sched.index]


def _parse_csv_pages(pages: list[str]) -> pd.DataFrame:
    """Concatenate ThetaData CSV pages into one frame.

    Every page carries its own header row, so pages after the first have theirs
    stripped before concatenation.
    """
    frames: list[pd.DataFrame] = []
    header: str | None = None
    for text in pages:
        text = text.strip()
        if not text:
            continue
        lines = text.splitlines()
        if header is None:
            header = lines[0]
            body = lines[1:]
        else:
            body = lines[1:] if lines[0] == header else lines
        if not body:
            continue
        frames.append(pd.read_csv(io.StringIO(header + "\n" + "\n".join(body))))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _daily_spot(client: ThetaClient, symbol: str, start: date, end: date) -> dict[date, float]:
    """Closing price per session, used to centre the +/-10% strike band."""
    pages = [p.text for p in client.stock_ohlc(symbol, start, end, interval_ms=0)]
    df = _parse_csv_pages(pages)
    if df.empty:
        return {}

    date_col = "date" if "date" in df.columns else df.columns[-1]
    close_col = next((c for c in ("close", "Close") if c in df.columns), None)
    if close_col is None:
        return {}

    out: dict[date, float] = {}
    for raw_date, close in zip(df[date_col], df[close_col], strict=False):
        s = str(int(raw_date))
        out[date(int(s[:4]), int(s[4:6]), int(s[6:8]))] = float(close)
    return out


def run_pilot(
    symbol: str = "SPY",
    start: date | None = None,
    end: date | None = None,
    write_parquet: bool = True,
) -> PilotResult:
    """Download one month of option quotes and measure what it cost.

    Defaults to the most recent complete calendar month.
    """
    ensure_dirs()

    if start is None or end is None:
        today = date.today()
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)

    out_dir = PILOT / f"{symbol}_{start:%Y%m}"
    if out_dir.exists():
        for f in out_dir.rglob("*"):
            if f.is_file():
                f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    sessions = nyse_sessions(start, end)
    failures: list[str] = []
    rows_written = 0
    expirations_touched: set[date] = set()

    t0 = time.monotonic()
    with ThetaClient() as client:
        client.check_connection()

        spot_by_day = _daily_spot(client, symbol, start, end)
        all_expirations = client.list_expirations(symbol)

        for session in sessions:
            spot = spot_by_day.get(session)
            if spot is None:
                failures.append(f"{session}: no underlying close, skipped")
                continue

            lo_strike = spot * (1 - BOUNDS.strike_pct)
            hi_strike = spot * (1 + BOUNDS.strike_pct)

            in_window = [
                e for e in all_expirations if session <= e <= session + timedelta(days=BOUNDS.max_dte)
            ]

            day_frames: list[pd.DataFrame] = []
            for exp in in_window:
                try:
                    pages = [
                        p.text
                        for p in client.bulk_option_quotes(
                            symbol,
                            exp,
                            session,
                            session,
                            interval_ms=BOUNDS.interval_ms,
                            rth_only=BOUNDS.rth_only,
                        )
                    ]
                except ThetaNoData:
                    continue
                except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
                    failures.append(f"{session} exp={exp}: {exc}")
                    continue

                df = _parse_csv_pages(pages)
                if df.empty or "strike" not in df.columns:
                    continue

                # ThetaData reports strikes multiplied by 1000.
                df["strike_dollars"] = df["strike"] / 1000.0
                df = df[
                    (df["strike_dollars"] >= lo_strike) & (df["strike_dollars"] <= hi_strike)
                ]
                if df.empty:
                    continue

                expirations_touched.add(exp)
                day_frames.append(df)

            if not day_frames:
                continue

            day_df = pd.concat(day_frames, ignore_index=True)
            rows_written += len(day_df)

            if write_parquet:
                day_df.to_parquet(
                    out_dir / f"date={session:%Y%m%d}.parquet",
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )

        wall = time.monotonic() - t0
        requests_made = client.requests_made
        bytes_over_wire = client.bytes_received

    return PilotResult(
        symbol=symbol,
        start=start,
        end=end,
        trading_days=len(sessions),
        expirations_touched=len(expirations_touched),
        rows_written=rows_written,
        bytes_on_disk=dir_size_bytes(out_dir),
        bytes_over_wire=bytes_over_wire,
        wall_clock_seconds=wall,
        requests_made=requests_made,
        failures=failures,
    )


def persist_pilot(result: PilotResult) -> None:
    """Record the pilot outcome and promote its measurements into the model."""
    import json

    PILOT.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["start"] = result.start.isoformat()
    payload["end"] = result.end.isoformat()
    payload["bytes_per_row"] = result.bytes_per_row
    payload["seconds_per_request"] = result.seconds_per_request
    (PILOT / f"pilot_{result.symbol}_{result.start:%Y%m}.json").write_text(
        json.dumps(payload, indent=2)
    )
    save_assumptions(result.to_assumptions())
