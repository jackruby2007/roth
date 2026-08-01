"""Dataset size and download-time model.

This exists to answer one question before any money is spent: how big is the
full backfill, and how long does it take to pull?

There are two ways to get those numbers:

* MEASURED -- run `roth pilot`, which downloads one real month and records the
  actual bytes-per-row and seconds-per-request. This is the number to trust.
* MODELED  -- the fallback below, which derives the same figures from option
  chain structure. It is arithmetic on top of stated assumptions, not a
  measurement, and every assumption it uses is printed alongside the result.

The model exists so the shape of the problem is visible without a subscription.
It is not a substitute for the pilot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from roth.config import BOUNDS
from roth.paths import PILOT

MINUTES_PER_RTH_SESSION = 390
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class EraAssumption:
    """Chain structure for a span of years.

    `expirations_in_window` is the count of listed expirations falling within
    BOUNDS.max_dte days of a given trade date. It grew enormously over the last
    decade: monthly-only, then weeklies, then Mon/Wed/Fri, then daily.
    """

    first_year: int
    last_year: int
    expirations_in_window: int
    avg_spot: float
    note: str


# SPY chain structure by era. `expirations_in_window` for the current era is
# anchored on a live observation: on 2026-08-01 SPY listed 18 expirations
# within 60 days. Earlier eras follow the listing history of SPY weeklies.
SPY_ERAS: tuple[EraAssumption, ...] = (
    EraAssumption(2018, 2019, 13, 280.0, "Mon/Wed/Fri weeklies"),
    EraAssumption(2020, 2021, 16, 380.0, "Mon/Wed/Fri weeklies, elevated listings"),
    EraAssumption(2022, 2023, 18, 420.0, "daily expirations phased in"),
    EraAssumption(2024, 2026, 18, 620.0, "daily expirations, observed 18 on 2026-08-01"),
)

QQQ_ERAS: tuple[EraAssumption, ...] = (
    EraAssumption(2018, 2019, 12, 175.0, "Mon/Wed/Fri weeklies"),
    EraAssumption(2020, 2021, 15, 300.0, "Mon/Wed/Fri weeklies"),
    EraAssumption(2022, 2023, 17, 330.0, "daily expirations phased in"),
    EraAssumption(2024, 2026, 17, 520.0, "daily expirations"),
)

ERAS: dict[str, tuple[EraAssumption, ...]] = {"SPY": SPY_ERAS, "QQQ": QQQ_ERAS}


@dataclass
class SizingAssumptions:
    """Every number the model depends on, in one place.

    When `roth pilot` runs, it overwrites `bytes_per_quote_row` and
    `seconds_per_bulk_request` with measured values and flips `measured` to
    True.
    """

    # Strike spacing near the money, in dollars. SPY and QQQ both list $1
    # strikes around spot and widen further out; $1 is the binding figure since
    # the +/-10% band sits in the dense region.
    strike_spacing: float = 1.0

    # Fraction of the +/-10% band that is actually listed and quoted. Not every
    # nominal strike in the band carries a two-sided quote every minute.
    listed_fraction: float = 0.85

    # Compressed parquet bytes per intraday quote row. Quote rows are highly
    # compressible: sorted timestamps delta-encode, prices and sizes fit in
    # int32. Range checked below; this is the midpoint.
    bytes_per_quote_row: float = 18.0

    # Compressed parquet bytes per EOD chain row. Wider (greeks, IV, OI,
    # volume) so materially larger per row.
    bytes_per_eod_row: float = 70.0

    # Wall-clock seconds per bulk request. One bulk request covers all strikes
    # of one expiration for one day.
    seconds_per_bulk_request: float = 2.0

    measured: bool = False
    measured_from: str | None = None

    def bytes_per_quote_row_range(self) -> tuple[float, float]:
        if self.measured:
            return (self.bytes_per_quote_row, self.bytes_per_quote_row)
        return (12.0, 25.0)


@dataclass
class YearEstimate:
    year: int
    symbol: str
    expirations_in_window: int
    strikes_in_band: int
    contracts_in_scope: int
    quote_rows: int
    quote_bytes_low: float
    quote_bytes_high: float
    eod_rows: int
    eod_bytes: float
    bulk_requests: int
    download_seconds: float


@dataclass
class SizingReport:
    assumptions: SizingAssumptions
    years: list[YearEstimate] = field(default_factory=list)

    # -- aggregates --------------------------------------------------------

    def _sum(self, attr: str, since: int | None = None) -> float:
        return sum(getattr(y, attr) for y in self.years if since is None or y.year >= since)

    def totals(self, since: int | None = None) -> dict[str, float]:
        return {
            "quote_rows": self._sum("quote_rows", since),
            "quote_bytes_low": self._sum("quote_bytes_low", since),
            "quote_bytes_high": self._sum("quote_bytes_high", since),
            "eod_rows": self._sum("eod_rows", since),
            "eod_bytes": self._sum("eod_bytes", since),
            "bulk_requests": self._sum("bulk_requests", since),
            "download_seconds": self._sum("download_seconds", since),
        }


def _era_for(symbol: str, year: int) -> EraAssumption:
    eras = ERAS[symbol]
    for era in eras:
        if era.first_year <= year <= era.last_year:
            return era
    # Outside the tabulated range: fall back to the nearest era.
    return eras[0] if year < eras[0].first_year else eras[-1]


def strikes_in_band(spot: float, assumptions: SizingAssumptions) -> int:
    """Count of listed strikes within +/- BOUNDS.strike_pct of spot.

    A +/-10% band around spot spans 0.2 * spot dollars. At $1 spacing that is
    0.2 * spot strikes, scaled by the fraction actually listed and quoted.
    """
    band_width = 2.0 * BOUNDS.strike_pct * spot
    nominal = band_width / assumptions.strike_spacing
    return max(1, int(round(nominal * assumptions.listed_fraction)))


def estimate_year(symbol: str, year: int, assumptions: SizingAssumptions) -> YearEstimate:
    era = _era_for(symbol, year)
    n_strikes = strikes_in_band(era.avg_spot, assumptions)

    # Both calls and puts.
    contracts = era.expirations_in_window * n_strikes * 2

    quote_rows = contracts * MINUTES_PER_RTH_SESSION * TRADING_DAYS_PER_YEAR
    lo, hi = assumptions.bytes_per_quote_row_range()

    eod_rows = contracts * TRADING_DAYS_PER_YEAR

    # One bulk request per (expiration, trading day). Quotes and EOD are
    # separate endpoints, hence the doubling.
    bulk_requests = era.expirations_in_window * TRADING_DAYS_PER_YEAR * 2

    return YearEstimate(
        year=year,
        symbol=symbol,
        expirations_in_window=era.expirations_in_window,
        strikes_in_band=n_strikes,
        contracts_in_scope=contracts,
        quote_rows=quote_rows,
        quote_bytes_low=quote_rows * lo,
        quote_bytes_high=quote_rows * hi,
        eod_rows=eod_rows,
        eod_bytes=eod_rows * assumptions.bytes_per_eod_row,
        bulk_requests=bulk_requests,
        download_seconds=bulk_requests * assumptions.seconds_per_bulk_request,
    )


def build_report(
    symbols: tuple[str, ...],
    first_year: int,
    last_year: int,
    assumptions: SizingAssumptions | None = None,
) -> SizingReport:
    assumptions = assumptions or load_assumptions()
    report = SizingReport(assumptions=assumptions)
    for year in range(first_year, last_year + 1):
        for symbol in symbols:
            report.years.append(estimate_year(symbol, year, assumptions))
    return report


# ---------------------------------------------------------------------------
# Persistence: the pilot writes measured assumptions here, the model reads them
# ---------------------------------------------------------------------------

ASSUMPTIONS_PATH = PILOT / "measured_assumptions.json"


def save_assumptions(assumptions: SizingAssumptions, path: Path = ASSUMPTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(assumptions), indent=2))


def load_assumptions(path: Path = ASSUMPTIONS_PATH) -> SizingAssumptions:
    """Measured assumptions if a pilot has been run, otherwise the modeled defaults."""
    if path.exists():
        try:
            return SizingAssumptions(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            pass
    return SizingAssumptions()
