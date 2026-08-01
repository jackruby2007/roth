"""Synthetic fixture generator.

Produces a complete, self-consistent dataset in the canonical schema so the
whole pipeline -- quality checks, features, backtest, fill model, reporting --
can be exercised and verified without a data subscription.

**What this data is for and what it is not for.**

It proves the plumbing works: that features are causal, that fills happen at the
next bar against real quoted bid/ask, that the reporting arithmetic is right.

It proves nothing whatsoever about whether a strategy has edge. The price
process is a random walk with no predictable structure in it by construction, so
any positive expectancy found here is noise. Every dataset written by this
module drops a `_SYNTHETIC_DATA` marker file, and the reporting layer refuses to
present results without a warning banner when it sees one.

Deliberate imperfections are included -- crossed markets, absurd spreads,
missing bars, a price gap -- because a quality layer that has never caught
anything is untested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from roth.calendar import session_days, trading_sessions
from roth.config import BOUNDS
from roth.paths import (
    RAW,
    RAW_OPTION_EOD,
    RAW_UNDERLYING_DAILY,
    RAW_UNDERLYING_MINUTE,
    RAW_VIX,
    ensure_dirs,
)
from roth.pricing import greeks as bs_greeks
from roth.pricing import price as bs_price
from roth.schema import SYNTHETIC_MARKER
from roth.storage import write_raw

MINUTES_PER_SESSION = 390
RISK_FREE = 0.04
DIV_YIELD = 0.013


@dataclass(frozen=True)
class SynthSpec:
    """Parameters of the generated world."""

    symbol: str
    start_price: float
    annual_drift: float = 0.08
    base_vol: float = 0.16
    # Volatility regime switching, so regime tagging has real structure to find.
    vol_regimes: tuple[float, ...] = (0.10, 0.16, 0.30)
    regime_persistence: float = 0.985
    seed: int = 20260801

    # Deliberate defects, as a fraction of sessions.
    crossed_quote_rate: float = 0.002
    absurd_spread_rate: float = 0.004
    # Fraction of *sessions* that lose a block of minute bars.
    missing_minute_rate: float = 0.02


SPECS: dict[str, SynthSpec] = {
    "SPY": SynthSpec("SPY", start_price=270.0, seed=20260801),
    "QQQ": SynthSpec("QQQ", start_price=160.0, base_vol=0.21, seed=20260802),
}


# ---------------------------------------------------------------------------
# Underlying
# ---------------------------------------------------------------------------


def _simulate_daily(spec: SynthSpec, days: list[date]) -> pd.DataFrame:
    """Geometric Brownian motion with a persistent volatility regime.

    The regime process is what makes VIX percentile terciles and ADX buckets
    mean something in the fixture. The return process itself is memoryless.
    """
    rng = np.random.default_rng(spec.seed)
    n = len(days)

    regime_idx = np.empty(n, dtype=int)
    current = 1
    for i in range(n):
        if rng.random() > spec.regime_persistence:
            current = int(rng.integers(0, len(spec.vol_regimes)))
        regime_idx[i] = current
    vols = np.array([spec.vol_regimes[i] for i in regime_idx])

    dt = 1.0 / 252.0
    shocks = rng.standard_normal(n)
    log_returns = (spec.annual_drift - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * shocks

    closes = spec.start_price * np.exp(np.cumsum(log_returns))

    # Intraday range scaled to that day's volatility.
    daily_vol = vols * np.sqrt(dt)
    opens = closes * np.exp(rng.normal(0, daily_vol * 0.3))
    spans = np.abs(rng.normal(0, daily_vol * 0.8)) * closes
    highs = np.maximum(opens, closes) + spans * 0.5
    lows = np.minimum(opens, closes) - spans * 0.5

    volumes = rng.lognormal(mean=18.0, sigma=0.35, size=n) * (1 + 2 * (vols - spec.base_vol))

    return pd.DataFrame(
        {
            "symbol": spec.symbol,
            "day": days,
            "open": np.round(opens, 2),
            "high": np.round(highs, 2),
            "low": np.round(lows, 2),
            "close": np.round(closes, 2),
            "volume": volumes.astype("int64"),
            "_realized_vol": vols,
        }
    )


def _simulate_minutes(
    spec: SynthSpec,
    day: date,
    day_row: pd.Series,
    session_open: pd.Timestamp,
    rng,
    n_minutes: int = MINUTES_PER_SESSION,
) -> pd.DataFrame:
    """A Brownian bridge from the open to the close, with a U-shaped volume day.

    A bridge rather than a fresh random walk, so the minute bars are consistent
    with the daily bar they belong to. Inconsistent daily and minute data would
    make the quality layer's cross-checks meaningless.

    `n_minutes` comes from the session's actual length, so early closes produce
    short sessions the way real data does.
    """
    n = n_minutes
    o, c = float(day_row["open"]), float(day_row["close"])
    hi, lo = float(day_row["high"]), float(day_row["low"])

    steps = rng.standard_normal(n)
    walk = np.cumsum(steps)
    # Pin the endpoint: bridge = walk - t * walk[-1]
    t = np.arange(1, n + 1) / n
    bridge = walk - t * walk[-1]

    scale = (hi - lo) / 4.0 if hi > lo else abs(c) * 0.001
    denom = np.abs(bridge).max() or 1.0
    path = o + (c - o) * t + bridge * scale / denom

    # Nudge the extremes so the session's high and low actually occur.
    path[int(n * 0.3)] = hi
    path[int(n * 0.7)] = lo
    path[-1] = c

    spread = np.abs(rng.normal(0, scale * 0.05, n))
    highs = path + spread
    lows = path - spread
    opens = np.concatenate([[o], path[:-1]])

    # U-shaped intraday volume: heavy at the open and the close.
    shape = 1.0 + 2.0 * (np.linspace(-1, 1, n) ** 2)
    volume = (float(day_row["volume"]) / max(n, 1)) * shape * rng.lognormal(0, 0.2, n)

    ts = pd.date_range(session_open, periods=n, freq="1min", tz="UTC")

    df = pd.DataFrame(
        {
            "symbol": spec.symbol,
            "day": day,
            "ts_utc": ts,
            "open": np.round(opens, 2),
            "high": np.round(np.maximum(highs, np.maximum(opens, path)), 2),
            "low": np.round(np.minimum(lows, np.minimum(opens, path)), 2),
            "close": np.round(path, 2),
            "volume": volume.astype("int64"),
        }
    )

    # Deliberate gaps, so the quality layer has missing bars to find.
    if rng.random() < spec.missing_minute_rate:
        drop_start = int(rng.integers(30, n - 30))
        df = df.drop(index=range(drop_start, min(drop_start + 12, n))).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Option chains
# ---------------------------------------------------------------------------


def _expirations_for(day: date, all_expirations: list[date]) -> list[date]:
    return [e for e in all_expirations if day <= e <= day + timedelta(days=BOUNDS.max_dte)]


def _build_expiration_grid(days: list[date]) -> list[date]:
    """Every Friday plus month-end, which approximates a modern weekly chain."""
    out: set[date] = set()
    if not days:
        return []
    cur, last = days[0], days[-1] + timedelta(days=BOUNDS.max_dte + 7)
    while cur <= last:
        if cur.weekday() == 4:
            out.add(cur)
        cur += timedelta(days=1)
    return sorted(out)


def _iv_surface(moneyness: float, dte_years: float, base_iv: float, rng) -> float:
    """A plausible smile and term structure.

    Downside skew is the dominant real feature of index option surfaces: puts
    below spot trade at materially higher implied vol than calls above it.
    """
    skew = -0.55 * moneyness
    smile = 1.8 * moneyness**2
    term = 0.02 * np.log(max(dte_years, 1 / 365) / (30 / 365))
    iv = base_iv * (1.0 + skew + smile) + term + rng.normal(0, 0.004)
    return float(np.clip(iv, 0.04, 3.0))


def _spread_for(mid: float, dte_years: float, abs_delta: float, rng) -> float:
    """Bid-ask width as a function of price, tenor and moneyness.

    Cheap far-OTM contracts have proportionally enormous spreads; that is the
    single most important execution fact this harness has to respect, so the
    fixture reproduces it rather than assuming a uniform penny-wide market.
    """
    # Deep-OTM theoreticals can come back as a tiny negative from floating point
    # cancellation; clamp before the square root.
    base = 0.01 + 0.02 * max(mid, 0.0) ** 0.5
    otm_penalty = 1.0 + 2.5 * (1.0 - abs_delta) ** 2
    tenor_penalty = 1.0 + 1.5 * dte_years
    return float(max(0.01, base * otm_penalty * tenor_penalty * rng.lognormal(0, 0.25)))


def _chain_for_day(
    spec: SynthSpec,
    day: date,
    spot: float,
    base_iv: float,
    expirations: list[date],
    rng,
) -> pd.DataFrame:
    lo_strike = spot * (1 - BOUNDS.strike_pct)
    hi_strike = spot * (1 + BOUNDS.strike_pct)
    strikes = np.arange(np.ceil(lo_strike), np.floor(hi_strike) + 1, 1.0)

    rows: list[dict] = []
    for exp in expirations:
        dte_days = (exp - day).days
        t = max(dte_days, 0) / 365.0
        for k in strikes:
            moneyness = float(np.log(k / spot))
            iv = _iv_surface(moneyness, t, base_iv, rng)
            for right in ("C", "P"):
                theo = bs_price(spot, float(k), t, RISK_FREE, iv, right, DIV_YIELD)
                g = bs_greeks(spot, float(k), t, RISK_FREE, iv, right, DIV_YIELD)

                spread = _spread_for(theo, t, abs(g["delta"]), rng)
                bid = max(0.0, theo - spread / 2)
                ask = theo + spread / 2

                # Deliberate defects for the quality layer to catch.
                if rng.random() < spec.crossed_quote_rate:
                    bid, ask = ask, bid  # crossed market
                elif rng.random() < spec.absurd_spread_rate:
                    ask = bid + max(theo, 0.05) * 4.0  # spread far beyond 50% of mid

                traded = rng.random() < 0.35
                rows.append(
                    {
                        "symbol": spec.symbol,
                        "day": day,
                        "expiration_date": exp,
                        "strike_dollars": float(k),
                        "right": right,
                        "open": round(theo * float(rng.normal(1.0, 0.02)), 2),
                        "high": round(theo * float(rng.uniform(1.0, 1.06)), 2),
                        "low": round(theo * float(rng.uniform(0.94, 1.0)), 2),
                        "close": round(theo, 2),
                        "volume": int(rng.lognormal(4.5, 1.5)) if traded else 0,
                        "open_interest": int(rng.lognormal(6.0, 1.2)),
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "bid_size": int(rng.integers(1, 200)),
                        "ask_size": int(rng.integers(1, 200)),
                        "implied_vol": round(iv, 4),
                        "delta": round(g["delta"], 4),
                        "gamma": round(g["gamma"], 6),
                        "theta": round(g["theta"], 4),
                        "vega": round(g["vega"], 4),
                        "rho": round(g["rho"], 4),
                        "underlying_close": spot,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def mark_synthetic(reason: str, spec_info: dict) -> None:
    """Drop the marker that forces every report to declare the data synthetic."""
    ensure_dirs()
    (RAW / SYNTHETIC_MARKER).write_text(
        json.dumps(
            {
                "origin": "synthetic",
                "reason": reason,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "warning": (
                    "This dataset was generated, not observed. Strategy results "
                    "computed from it demonstrate that the pipeline runs. They say "
                    "nothing about whether any strategy has edge."
                ),
                **spec_info,
            },
            indent=2,
        )
    )


def is_synthetic() -> bool:
    return (RAW / SYNTHETIC_MARKER).exists()


def synthetic_info() -> dict | None:
    path = RAW / SYNTHETIC_MARKER
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"origin": "synthetic"}


def generate(
    start: date,
    end: date,
    symbols: tuple[str, ...] = ("SPY", "QQQ"),
    with_minute: bool = True,
    with_options: bool = True,
) -> dict[str, int]:
    """Generate a full synthetic dataset and write it to the raw store."""
    ensure_dirs()
    counts: dict[str, int] = {}

    sessions_df = trading_sessions(start, end)
    days = list(sessions_df["day"])
    opens = list(pd.to_datetime(sessions_df["session_open_utc"], utc=True))
    closes = list(pd.to_datetime(sessions_df["session_close_utc"], utc=True))
    if not days:
        raise ValueError(f"No trading sessions between {start} and {end}")

    # Real session length, so early closes yield short sessions rather than a
    # uniform 390 bars the quality layer would never be tested against.
    session_minutes = [int((c - o).total_seconds() // 60) for o, c in zip(opens, closes, strict=True)]

    expirations = _build_expiration_grid(days)
    vix_source: pd.DataFrame | None = None

    for symbol in symbols:
        spec = SPECS.get(symbol) or SynthSpec(symbol, start_price=300.0)
        rng = np.random.default_rng(spec.seed)

        daily = _simulate_daily(spec, days)
        if vix_source is None:
            vix_source = daily.copy()

        realized = daily.pop("_realized_vol")
        write_raw(
            daily, RAW_UNDERLYING_DAILY, "underlying_daily", symbol, date(days[0].year, 1, 1)
        )
        counts[f"{symbol}_daily"] = len(daily)

        if with_minute:
            minute_rows = 0
            for i, day in enumerate(days):
                mdf = _simulate_minutes(
                    spec, day, daily.iloc[i], opens[i], rng, n_minutes=session_minutes[i]
                )
                write_raw(mdf, RAW_UNDERLYING_MINUTE, "underlying_minute", symbol, day)
                minute_rows += len(mdf)
            counts[f"{symbol}_minute"] = minute_rows

        if with_options:
            option_rows = 0
            for i, day in enumerate(days):
                spot = float(daily.iloc[i]["close"])
                base_iv = float(realized.iloc[i]) * 1.15  # IV trades above realized
                exps = _expirations_for(day, expirations)
                if not exps:
                    continue
                chain = _chain_for_day(spec, day, spot, base_iv, exps, rng)
                write_raw(chain, RAW_OPTION_EOD, "option_eod", symbol, day)
                option_rows += len(chain)
            counts[f"{symbol}_option_eod"] = option_rows

    # VIX, derived from the first symbol's volatility regime so it is coherent
    # with the option surface rather than an independent random series.
    if vix_source is not None:
        vix = pd.DataFrame(
            {
                "symbol": "VIX",
                "day": days,
                "close": np.round(vix_source["_realized_vol"].to_numpy() * 100 * 1.15, 2),
            }
        )
        vix["open"] = vix["close"]
        vix["high"] = vix["close"] * 1.03
        vix["low"] = vix["close"] * 0.97
        write_raw(vix, RAW_VIX, "vix_daily", "VIX", date(days[0].year, 1, 1))
        counts["vix_daily"] = len(vix)

    mark_synthetic(
        reason="generated by roth.data.synth for pipeline verification",
        spec_info={
            "symbols": list(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sessions": len(days),
        },
    )
    return counts


__all__ = [
    "SPECS",
    "SynthSpec",
    "generate",
    "is_synthetic",
    "mark_synthetic",
    "synthetic_info",
    "session_days",
]
