"""Configuration for ingestion bounds and execution assumptions.

These are the knobs that decide how big the dataset gets. They are deliberately
conservative: full option chains at 1-minute resolution across all strikes and
expirations is a size bomb, so both the strike range and the expiration horizon
are bounded here and enforced at download time, not at query time.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

SYMBOLS: tuple[str, ...] = ("SPY", "QQQ")

# SPX is deliberately excluded from Phase 1: its historical feed has known
# quirks (settlement conventions, AM/PM expiration collisions) that are not
# worth debugging before the harness itself is trusted.

MARKET_TZ = "America/New_York"
STORAGE_TZ = "UTC"


# ---------------------------------------------------------------------------
# Ingestion bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestBounds:
    """Bounds applied to every option chain download."""

    # Only expirations within this many calendar days of the trade date.
    max_dte: int = 60

    # Only strikes within this fraction of spot. 0.10 == +/- 10%.
    strike_pct: float = 0.10

    # Intraday quote resolution in milliseconds. 60_000 == 1 minute.
    interval_ms: int = 60_000

    # Regular trading hours only. Extended-hours option quotes are thin and
    # mostly noise for the hypotheses this harness is meant to test.
    rth_only: bool = True


BOUNDS = IngestBounds()


# ---------------------------------------------------------------------------
# Execution assumptions (used by the fill model, not by ingestion)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Execution cost assumptions.

    The defaults here are intentionally realistic-pessimistic. `extra_slippage`
    exists so sensitivity to worse execution can be tested without editing code.
    """

    # Per-contract commission, each way.
    commission_per_contract: float = 0.65

    # Additional slippage in dollars per contract, applied on top of crossing
    # the spread. Default zero: crossing the spread is already the base cost.
    extra_slippage_per_contract: float = 0.0

    # A quote whose spread exceeds this fraction of mid is treated as
    # untradeable. Signals needing such a quote are rejected, never filled at a
    # synthetic price.
    max_spread_pct_of_mid: float = 0.50


COSTS = CostModel()


# ---------------------------------------------------------------------------
# ThetaData connection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThetaConfig:
    """Connection settings for a locally running Theta Terminal.

    Theta Terminal is a Java process that exposes a local REST API. It only
    needs to be running during a download, never afterwards. Nothing in this
    harness connects to it at research time.
    """

    host: str = "127.0.0.1"
    port: int = 25510
    timeout_seconds: float = 120.0
    max_retries: int = 4

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


THETA = ThetaConfig()


# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityThresholds:
    # Underlying bar-to-bar move beyond this fraction is flagged as a gap and
    # cross-checked against known split/dividend dates before quarantine.
    max_daily_gap_pct: float = 0.10
    max_minute_gap_pct: float = 0.05

    # A session missing more than this fraction of its expected minute bars is
    # quarantined.
    max_missing_minute_fraction: float = 0.02

    # Option spread beyond this fraction of mid is flagged (not necessarily
    # quarantined -- wide spreads are real, they just are not tradeable).
    wide_spread_pct_of_mid: float = 0.50


QUALITY = QualityThresholds()


# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

# Known SPY / QQQ splits. Used to distinguish a real corporate action from a
# corrupt bar when a large price gap is detected.
KNOWN_SPLITS: dict[str, tuple[tuple[str, float], ...]] = {
    # (effective_date, ratio) -- ratio > 1 means the price divided by ratio.
    "SPY": (("2005-06-09", 2.0),),
    "QQQ": (("1999-03-20", 2.0), ("2000-03-20", 2.0)),
}
