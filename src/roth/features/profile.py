"""Volume profile, computed per session from minute bars.

Produces value area high, value area low, point of control, and the high and
low volume nodes.

Causality: every value here describes a session that has *finished*. The
feature builder stamps them as available at the following session's open, never
the session they describe. That shift is the whole reason this module returns a
per-session table rather than writing into the session's own row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of session volume contained in the value area. 70% is the
# conventional definition, being roughly one standard deviation.
VALUE_AREA_FRACTION = 0.70


def _price_bins(prices: pd.Series, tick: float) -> np.ndarray:
    lo = np.floor(prices.min() / tick) * tick
    hi = np.ceil(prices.max() / tick) * tick
    if hi <= lo:
        hi = lo + tick
    return np.arange(lo, hi + tick, tick)


def session_profile(minute: pd.DataFrame, tick: float = 0.25) -> dict[str, float]:
    """Volume profile statistics for one session's minute bars.

    Volume is attributed to each bar's typical price (H+L+C)/3, which is the
    standard approximation when true price-level volume is unavailable.
    """
    if minute.empty:
        return {}

    typical = (minute["high"] + minute["low"] + minute["close"]) / 3.0
    volume = minute["volume"].astype(float)

    bins = _price_bins(typical, tick)
    if len(bins) < 2:
        return {}

    idx = np.clip(np.digitize(typical, bins) - 1, 0, len(bins) - 2)
    hist = np.zeros(len(bins) - 1)
    np.add.at(hist, idx, volume.to_numpy())

    total = hist.sum()
    if total <= 0:
        return {}

    centres = (bins[:-1] + bins[1:]) / 2.0
    poc_idx = int(hist.argmax())

    # Grow the value area outward from the point of control, always taking the
    # heavier adjacent level, until 70% of volume is enclosed.
    target = total * VALUE_AREA_FRACTION
    lo_idx = hi_idx = poc_idx
    covered = hist[poc_idx]

    while covered < target and (lo_idx > 0 or hi_idx < len(hist) - 1):
        below = hist[lo_idx - 1] if lo_idx > 0 else -1.0
        above = hist[hi_idx + 1] if hi_idx < len(hist) - 1 else -1.0
        if above >= below:
            hi_idx += 1
            covered += hist[hi_idx]
        else:
            lo_idx -= 1
            covered += hist[lo_idx]

    nonzero = hist[hist > 0]
    hvn_threshold = np.percentile(nonzero, 80) if len(nonzero) else 0.0
    lvn_threshold = np.percentile(nonzero, 20) if len(nonzero) else 0.0

    hvn_prices = centres[hist >= hvn_threshold]
    lvn_prices = centres[(hist > 0) & (hist <= lvn_threshold)]

    return {
        "poc": float(centres[poc_idx]),
        "value_area_high": float(centres[hi_idx]),
        "value_area_low": float(centres[lo_idx]),
        "value_area_volume_pct": float(covered / total * 100.0),
        "hvn_high": float(hvn_prices.max()) if len(hvn_prices) else np.nan,
        "hvn_low": float(hvn_prices.min()) if len(hvn_prices) else np.nan,
        "lvn_high": float(lvn_prices.max()) if len(lvn_prices) else np.nan,
        "lvn_low": float(lvn_prices.min()) if len(lvn_prices) else np.nan,
    }


def profiles_by_session(minute: pd.DataFrame, tick: float = 0.25) -> pd.DataFrame:
    """One volume profile per session.

    The returned rows are stamped with the session they *describe*. Shifting
    them to when they become usable is the builder's job, so that the shift
    happens in exactly one place.
    """
    if minute.empty:
        return pd.DataFrame()

    rows = []
    for day, grp in minute.groupby("day"):
        stats = session_profile(grp, tick=tick)
        if stats:
            rows.append({"day": day, **stats})

    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True) if rows else pd.DataFrame()


def session_vwap(minute: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP, expanding from each session's first bar.

    Anchored and expanding, not rolling: at 10:15 it reflects 09:30 to 10:15 and
    nothing else. It resets at every session boundary.
    """
    if minute.empty:
        return pd.Series(dtype=float)

    typical = (minute["high"] + minute["low"] + minute["close"]) / 3.0
    pv = typical * minute["volume"]

    grouped = minute.assign(_pv=pv).groupby("day")
    cum_pv = grouped["_pv"].cumsum()
    cum_vol = grouped["volume"].cumsum()

    return cum_pv / cum_vol.replace(0, np.nan)


def opening_range(minute: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Opening range high and low for each session.

    Stamped with the timestamp at which the range *completes*, because that is
    the first moment it could be acted on. A 30-minute opening range is not a
    signal at 09:31.
    """
    if minute.empty:
        return pd.DataFrame()

    rows = []
    for day, grp in minute.groupby("day"):
        grp = grp.sort_values("ts_utc")
        window = grp.head(minutes)
        if len(window) < minutes:
            continue
        rows.append(
            {
                "day": day,
                f"or{minutes}_high": float(window["high"].max()),
                f"or{minutes}_low": float(window["low"].min()),
                f"or{minutes}_available_at": window["ts_utc"].iloc[-1],
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()
