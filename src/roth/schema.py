"""Canonical column schemas.

These define the contract between whatever produced a table and everything that
reads it. Both the ThetaData ingest path and the synthetic fixture generator
emit exactly these columns, so no downstream code can tell the difference --
which is the point. When real data replaces fixtures, nothing below the storage
layer changes.

Timestamps are UTC everywhere. Prices are dollars. Strikes are dollars, not
ThetaData's integer-scaled form.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Underlying bars
# ---------------------------------------------------------------------------

UNDERLYING_DAILY: tuple[str, ...] = (
    "symbol",
    "day",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

UNDERLYING_MINUTE: tuple[str, ...] = (
    "symbol",
    "day",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

VIX_DAILY: tuple[str, ...] = (
    "symbol",
    "day",
    "open",
    "high",
    "low",
    "close",
)

# ---------------------------------------------------------------------------
# Option chains
# ---------------------------------------------------------------------------

OPTION_EOD: tuple[str, ...] = (
    "symbol",
    "day",
    "expiration_date",
    "strike_dollars",
    "right",  # "C" or "P"
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "bid",
    "ask",
    "bid_size",
    "ask_size",
    "implied_vol",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "underlying_close",
)


@dataclass(frozen=True)
class DataOrigin:
    """Where a dataset came from.

    This is not decoration. A backtest run against synthetic data must say so in
    every report it produces, or someone will eventually mistake a plumbing test
    for a finding.
    """

    REAL: str = "real"
    SYNTHETIC: str = "synthetic"


ORIGIN = DataOrigin()

# A marker file dropped alongside synthetic data. Reporting checks for it and
# refuses to present results without a warning banner.
SYNTHETIC_MARKER = "_SYNTHETIC_DATA"


def missing_columns(actual: object, expected: tuple[str, ...]) -> list[str]:
    """Columns required by a schema that a frame does not have."""
    have = set(getattr(actual, "columns", actual))
    return [c for c in expected if c not in have]
