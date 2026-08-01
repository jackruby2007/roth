"""Data quality layer.

Runs before any research and fails loudly. Bad days go into a quarantine table
with a reason, are excluded from research by default, and the count of excluded
days appears in every report.

**The separation that matters here.** This layer quarantines *sessions*. It does
not reject individual quotes -- that is the fill model's job, at trade time,
where a single unusable quote means one rejected signal rather than a discarded
day. Mixing the two would either throw away good days because of one bad
contract, or silently fill against a crossed market.

So:

* A handful of crossed quotes in a chain is normal feed noise. It is *flagged*,
  and the fill model refuses those specific contracts.
* A chain where 5% of quotes are crossed is a broken feed. The session is
  *quarantined*.
* A wide spread is not a data error at all. Far-OTM contracts genuinely trade
  200% of mid wide. It is recorded, never quarantined, and the fill model
  declines to trade it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from roth.calendar import session_days, trading_sessions
from roth.config import KNOWN_SPLITS, QUALITY
from roth.paths import (
    QUARANTINE,
    RAW_OPTION_EOD,
    RAW_UNDERLYING_DAILY,
    RAW_UNDERLYING_MINUTE,
)
from roth.storage import read_dataset, write_derived

EXPECTED_MINUTES = 390

# A session is quarantined when this fraction of its option quotes are crossed.
# Below it, individual quotes are flagged and the fill model handles them.
CROSSED_QUOTE_QUARANTINE_RATE = 0.05

SEVERITY_QUARANTINE = "quarantine"
SEVERITY_FLAG = "flag"


@dataclass
class Finding:
    symbol: str
    day: date | None
    check: str
    severity: str
    detail: str
    count: int = 1


@dataclass
class QualityReport:
    start: date
    end: date
    symbols: tuple[str, ...]
    findings: list[Finding] = field(default_factory=list)
    sessions_expected: int = 0
    sessions_present: int = 0

    # -- derived views -----------------------------------------------------

    @property
    def quarantined_days(self) -> set[tuple[str, date]]:
        return {
            (f.symbol, f.day)
            for f in self.findings
            if f.severity == SEVERITY_QUARANTINE and f.day is not None
        }

    @property
    def flags(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_FLAG]

    def by_check(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["check", "severity", "sessions", "occurrences"])
        df = pd.DataFrame([asdict(f) for f in self.findings])
        return (
            df.groupby(["check", "severity"], as_index=False)
            .agg(sessions=("day", "nunique"), occurrences=("count", "sum"))
            .sort_values(["severity", "occurrences"], ascending=[True, False])
        )

    def quarantine_frame(self) -> pd.DataFrame:
        rows = [
            asdict(f) for f in self.findings if f.severity == SEVERITY_QUARANTINE and f.day
        ]
        if not rows:
            return pd.DataFrame(columns=["symbol", "day", "check", "severity", "detail", "count"])
        return pd.DataFrame(rows).sort_values(["symbol", "day"])


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_missing_sessions(
    symbol: str, daily: pd.DataFrame, start: date, end: date
) -> list[Finding]:
    """Trading days the NYSE calendar knows about that the data does not have."""
    expected = set(session_days(start, end))
    present = set(daily["day"]) if not daily.empty else set()
    missing = sorted(expected - present)

    return [
        Finding(symbol, day, "missing_session", SEVERITY_QUARANTINE, "no bar for this session")
        for day in missing
    ]


def check_daily_bars(symbol: str, daily: pd.DataFrame) -> list[Finding]:
    """Null prices, zero range, and zero volume in the daily series."""
    out: list[Finding] = []
    if daily.empty:
        return out

    price_cols = [c for c in ("open", "high", "low", "close") if c in daily.columns]

    null_mask = daily[price_cols].isna().any(axis=1)
    for day in daily.loc[null_mask, "day"]:
        out.append(Finding(symbol, day, "null_price", SEVERITY_QUARANTINE, "null OHLC value"))

    zero_range = daily["high"] == daily["low"]
    for day in daily.loc[zero_range, "day"]:
        out.append(
            Finding(symbol, day, "zero_range", SEVERITY_QUARANTINE, "high equals low")
        )

    if "volume" in daily.columns:
        zero_vol = daily["volume"].fillna(0) <= 0
        for day in daily.loc[zero_vol, "day"]:
            out.append(
                Finding(symbol, day, "zero_volume", SEVERITY_QUARANTINE, "no volume recorded")
            )

    inverted = daily["high"] < daily["low"]
    for day in daily.loc[inverted, "day"]:
        out.append(
            Finding(symbol, day, "inverted_bar", SEVERITY_QUARANTINE, "high below low")
        )

    return out


def check_price_gaps(symbol: str, daily: pd.DataFrame) -> list[Finding]:
    """Close-to-close moves beyond the threshold, cross-checked against splits.

    A 50% overnight move is either a corrupt bar or a stock split. Checking the
    known split dates is what tells them apart; without it every split date
    would be quarantined and a genuinely broken bar would look the same as one.
    """
    out: list[Finding] = []
    if daily.empty or len(daily) < 2:
        return out

    df = daily.sort_values("day").reset_index(drop=True)
    prev_close = df["close"].shift(1)
    move = (df["close"] - prev_close) / prev_close

    split_dates = {
        pd.to_datetime(d).date() for d, _ in KNOWN_SPLITS.get(symbol, ())
    }

    big = move.abs() > QUALITY.max_daily_gap_pct
    for idx in df.index[big.fillna(False)]:
        day = df.loc[idx, "day"]
        pct = float(move.loc[idx])

        if day in split_dates:
            out.append(
                Finding(
                    symbol,
                    day,
                    "price_gap_explained_by_split",
                    SEVERITY_FLAG,
                    f"{pct:+.1%} move on a known split date",
                )
            )
        else:
            out.append(
                Finding(
                    symbol,
                    day,
                    "price_gap",
                    SEVERITY_QUARANTINE,
                    f"{pct:+.1%} close-to-close move, no known corporate action",
                )
            )
    return out


def expected_minutes_by_session(start: date, end: date) -> dict[date, int]:
    """Minutes each session should contain, from its actual open and close.

    Not a constant 390. Half-days around Thanksgiving and Christmas close at
    13:00 ET and contain 210 bars; comparing those against 390 would quarantine
    every early close in the dataset as if it were a partial download.
    """
    sched = trading_sessions(start, end)
    if sched.empty:
        return {}
    opens = pd.to_datetime(pd.Series(sched["session_open_utc"]), utc=True)
    closes = pd.to_datetime(pd.Series(sched["session_close_utc"]), utc=True)
    minutes = ((closes - opens).dt.total_seconds() // 60).astype(int)
    return dict(zip(sched["day"], minutes, strict=False))


def check_minute_continuity(symbol: str, minute: pd.DataFrame) -> list[Finding]:
    """Sessions missing too many minute bars, and any timezone irregularity."""
    out: list[Finding] = []
    if minute.empty:
        return out

    ts = pd.Series(minute["ts_utc"])
    if ts.dt.tz is None:
        out.append(
            Finding(
                symbol,
                None,
                "naive_timestamp",
                SEVERITY_QUARANTINE,
                "minute timestamps carry no timezone; everything must be UTC",
            )
        )
        return out

    if str(ts.dt.tz) != "UTC":
        out.append(
            Finding(
                symbol,
                None,
                "non_utc_timestamp",
                SEVERITY_QUARANTINE,
                f"minute timestamps are {ts.dt.tz}, expected UTC",
            )
        )

    counts = minute.groupby("day").size()
    expected = expected_minutes_by_session(min(counts.index), max(counts.index))

    for day, n in counts.items():
        want = expected.get(day, EXPECTED_MINUTES)
        if n < want * (1 - QUALITY.max_missing_minute_fraction):
            out.append(
                Finding(
                    symbol,
                    day,
                    "missing_minute_bars",
                    SEVERITY_QUARANTINE,
                    f"{n} of {want} bars present",
                )
            )

    dupes = minute.duplicated(subset=["day", "ts_utc"]).sum()
    if dupes:
        out.append(
            Finding(
                symbol,
                None,
                "duplicate_timestamps",
                SEVERITY_QUARANTINE,
                f"{int(dupes)} duplicate (day, timestamp) rows",
                count=int(dupes),
            )
        )

    return out


def check_option_quotes(symbol: str, options: pd.DataFrame) -> list[Finding]:
    """Crossed markets and absurd spreads.

    Crossed quotes are counted per session. A few are feed noise and get
    flagged; a session where a meaningful fraction are crossed is a broken feed
    and gets quarantined.
    """
    out: list[Finding] = []
    if options.empty:
        return out

    df = options
    crossed = df["bid"] > df["ask"]
    mid = (df["bid"] + df["ask"]) / 2
    wide = (df["ask"] - df["bid"]) > QUALITY.wide_spread_pct_of_mid * mid.clip(lower=0.01)
    nonpositive = (df["bid"] < 0) | (df["ask"] <= 0)

    per_day = df.assign(_crossed=crossed, _wide=wide, _bad=nonpositive).groupby("day")

    for day, grp in per_day:
        n = len(grp)
        n_crossed = int(grp["_crossed"].sum())
        n_wide = int(grp["_wide"].sum())
        n_bad = int(grp["_bad"].sum())

        if n_crossed:
            rate = n_crossed / n
            if rate >= CROSSED_QUOTE_QUARANTINE_RATE:
                out.append(
                    Finding(
                        symbol,
                        day,
                        "crossed_quotes_widespread",
                        SEVERITY_QUARANTINE,
                        f"{n_crossed} of {n} quotes crossed ({rate:.1%})",
                        count=n_crossed,
                    )
                )
            else:
                out.append(
                    Finding(
                        symbol,
                        day,
                        "crossed_quotes",
                        SEVERITY_FLAG,
                        f"{n_crossed} of {n} quotes crossed; the fill model will reject them",
                        count=n_crossed,
                    )
                )

        if n_wide:
            out.append(
                Finding(
                    symbol,
                    day,
                    "wide_spread",
                    SEVERITY_FLAG,
                    f"{n_wide} of {n} quotes wider than "
                    f"{QUALITY.wide_spread_pct_of_mid:.0%} of mid; untradeable, not corrupt",
                    count=n_wide,
                )
            )

        if n_bad:
            out.append(
                Finding(
                    symbol,
                    day,
                    "nonpositive_quote",
                    SEVERITY_FLAG,
                    f"{n_bad} of {n} quotes with a negative bid or non-positive ask",
                    count=n_bad,
                )
            )

    return out


def check_option_chain_shape(symbol: str, options: pd.DataFrame) -> list[Finding]:
    """Sessions whose chain is implausibly thin, which usually means a partial
    download rather than a real market condition."""
    out: list[Finding] = []
    if options.empty:
        return out

    per_day = options.groupby("day").agg(
        contracts=("strike_dollars", "size"),
        expirations=("expiration_date", "nunique"),
    )
    if per_day.empty:
        return out

    median_contracts = float(per_day["contracts"].median())
    floor = max(20.0, median_contracts * 0.25)

    for day, row in per_day.iterrows():
        if row["contracts"] < floor:
            out.append(
                Finding(
                    symbol,
                    day,
                    "thin_chain",
                    SEVERITY_QUARANTINE,
                    f"{int(row['contracts'])} contracts against a median of "
                    f"{median_contracts:.0f}; likely a partial download",
                )
            )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_quality(
    symbols: tuple[str, ...],
    start: date,
    end: date,
    include_options: bool = True,
    include_minute: bool = True,
) -> QualityReport:
    report = QualityReport(start=start, end=end, symbols=symbols)
    report.sessions_expected = len(session_days(start, end))

    for symbol in symbols:
        daily = read_dataset(RAW_UNDERLYING_DAILY, symbols=[symbol], start=start, end=end)
        if not daily.empty:
            daily = daily.sort_values("day").reset_index(drop=True)
            report.sessions_present = max(report.sessions_present, daily["day"].nunique())

        report.findings += check_missing_sessions(symbol, daily, start, end)
        report.findings += check_daily_bars(symbol, daily)
        report.findings += check_price_gaps(symbol, daily)

        if include_minute:
            minute = read_dataset(
                RAW_UNDERLYING_MINUTE, symbols=[symbol], start=start, end=end
            )
            report.findings += check_minute_continuity(symbol, minute)

        if include_options:
            options = read_dataset(RAW_OPTION_EOD, symbols=[symbol], start=start, end=end)
            report.findings += check_option_quotes(symbol, options)
            report.findings += check_option_chain_shape(symbol, options)

    return report


def persist_quarantine(report: QualityReport) -> pd.DataFrame:
    """Write the quarantine table. Research reads this to exclude bad days."""
    df = report.quarantine_frame()
    write_derived(df, QUARANTINE, "quarantine")
    return df


def load_quarantine() -> set[tuple[str, date]]:
    """Quarantined (symbol, day) pairs. Empty if quality has never been run."""
    path = QUARANTINE / "quarantine.parquet"
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    if df.empty or "day" not in df.columns:
        return set()
    days = pd.to_datetime(df["day"]).dt.date
    return set(zip(df["symbol"], days, strict=False))


def exclude_quarantined(df: pd.DataFrame, symbol_col: str = "symbol") -> tuple[pd.DataFrame, int]:
    """Drop quarantined sessions from a frame, returning it and the drop count."""
    bad = load_quarantine()
    if not bad or df.empty or "day" not in df.columns:
        return df, 0

    days = pd.to_datetime(df["day"]).dt.date
    keys = list(zip(df[symbol_col], days, strict=False))
    mask = np.array([k not in bad for k in keys])
    return df[mask], int((~mask).sum())
