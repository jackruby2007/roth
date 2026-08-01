"""Feature store construction.

**The stamping convention, stated once.**

Every row in the daily feature table is keyed by `(symbol, day)` and contains
only what was knowable **by the close of that session**. Nothing in row D
depends on any observation after D's close.

That convention pairs with the backtest engine, which evaluates signals on bar
close and fills at the *next* bar. Together they make lookahead structurally
impossible rather than merely discouraged: a strategy reading row D and filling
at D+1 cannot see the future even if it tries.

Two consequences worth being explicit about:

* Columns named `prev_*` refer to session D-1. `prev_high` in row D is the high
  of the previous session, which is exactly what a strategy trading session D
  can know.
* Volume profile columns in row D describe session **D-1**, because a profile of
  session D is not complete until D's close. This satisfies the rule that daily
  features are stamped as available at the following session, and the shift
  happens here, in one place, rather than being re-derived by every caller.

Calendar flags are the one exception to shifting, and deliberately so: the fact
that next Friday is an OPEX day is knowable years in advance. Shifting a known
calendar would throw away real information.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from roth.calendar import build_calendar
from roth.config import SYMBOLS
from roth.features import technicals as ta
from roth.features.profile import opening_range, profiles_by_session, session_vwap
from roth.paths import (
    FEATURES,
    RAW_OPTION_EOD,
    RAW_UNDERLYING_DAILY,
    RAW_UNDERLYING_MINUTE,
    RAW_VIX,
)
from roth.storage import read_dataset, write_derived

# Lookback used for percentile and rank features. One trading year.
PERCENTILE_WINDOW = 252

# Columns that describe the *previous* session and are therefore shifted.
PROFILE_COLUMNS = (
    "poc",
    "value_area_high",
    "value_area_low",
    "value_area_volume_pct",
    "hvn_high",
    "hvn_low",
    "lvn_high",
    "lvn_low",
)


# ---------------------------------------------------------------------------
# Daily features
# ---------------------------------------------------------------------------


def _price_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Previous-session levels and the current session's own realised range."""
    out = pd.DataFrame(index=df.index)

    # Explicitly the prior session. Row D holds session D-1's levels.
    out["prev_high"] = df["high"].shift(1)
    out["prev_low"] = df["low"].shift(1)
    out["prev_close"] = df["close"].shift(1)
    out["prev_range"] = out["prev_high"] - out["prev_low"]

    # Knowable at D's close: D's own bar.
    out["daily_range"] = df["high"] - df["low"]
    out["daily_range_pct"] = out["daily_range"] / df["close"]

    # The overnight gap is knowable once D has opened.
    out["overnight_gap"] = df["open"] - out["prev_close"]
    out["overnight_gap_pct"] = out["overnight_gap"] / out["prev_close"]

    out["close_vs_prev_high"] = df["close"] - out["prev_high"]
    out["close_vs_prev_low"] = df["close"] - out["prev_low"]
    return out


def _technicals(df: pd.DataFrame) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    for window in (10, 20, 50, 100, 200):
        out[f"sma_{window}"] = ta.sma(close, window)
    for span in (8, 21, 50):
        out[f"ema_{span}"] = ta.ema(close, span)

    out["atr_14"] = ta.atr(high, low, close, 14)
    out["atr_pct"] = out["atr_14"] / close

    out["rsi_14"] = ta.rsi(close, 14)

    macd_line, macd_signal, macd_hist = ta.macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    plus_di, minus_di, adx = ta.directional_movement(high, low, close, 14)
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx_14"] = adx

    bb_u, bb_m, bb_l = ta.bollinger(close, 20, 2.0)
    out["bb_upper"] = bb_u
    out["bb_mid"] = bb_m
    out["bb_lower"] = bb_l
    out["bb_width"] = (bb_u - bb_l) / bb_m

    kc_u, kc_m, kc_l = ta.keltner(high, low, close, 20, 10, 2.0)
    out["kc_upper"] = kc_u
    out["kc_mid"] = kc_m
    out["kc_lower"] = kc_l

    stoch_k, stoch_d = ta.stochastic(high, low, close, 14, 3, 3)
    out["stoch_k"] = stoch_k
    out["stoch_d"] = stoch_d

    for window in (5, 10, 21, 63):
        out[f"realized_vol_{window}"] = ta.realized_volatility(close, window)

    return out


def _option_derived(symbol: str, days: pd.Index, start: date, end: date) -> pd.DataFrame:
    """ATM implied vol, IV rank, IV percentile and expected move.

    Derived from the option chain of the session itself, so it is knowable at
    that session's close -- the same stamp as everything else in the table.
    """
    chains = read_dataset(RAW_OPTION_EOD, symbols=[symbol], start=start, end=end)
    out = pd.DataFrame(index=days)
    if chains.empty:
        return out

    chains = chains.copy()
    chains["dte"] = (
        pd.to_datetime(chains["expiration_date"]) - pd.to_datetime(chains["day"])
    ).dt.days
    chains["moneyness"] = (chains["strike_dollars"] / chains["underlying_close"] - 1).abs()

    # The ATM contract nearest to 30 days, which is the conventional anchor for
    # an implied-vol series.
    near = chains[(chains["dte"] >= 20) & (chains["dte"] <= 45)]
    if near.empty:
        near = chains

    near = near.sort_values(["day", "moneyness", "dte"])
    atm = near.groupby("day").first()

    series = pd.DataFrame(index=days)
    series["atm_iv"] = atm["implied_vol"].reindex(days)
    series["atm_iv_dte"] = atm["dte"].reindex(days)

    series["iv_rank"] = ta.rolling_rank(series["atm_iv"], PERCENTILE_WINDOW)
    series["iv_percentile"] = ta.rolling_percentile(series["atm_iv"], PERCENTILE_WINDOW)

    # Expected move to the anchor expiry: S * IV * sqrt(T).
    underlying = atm["underlying_close"].reindex(days)
    series["expected_move"] = (
        underlying * series["atm_iv"] * np.sqrt(series["atm_iv_dte"] / 365.0)
    )
    series["expected_move_pct"] = series["expected_move"] / underlying

    return series


def _vix_features(days: pd.Index, start: date, end: date) -> pd.DataFrame:
    vix = read_dataset(RAW_VIX, symbols=["VIX"], start=start, end=end)
    out = pd.DataFrame(index=days)
    if vix.empty:
        return out

    vix = vix.sort_values("day").set_index("day")
    out["vix"] = vix["close"].reindex(days)
    out["vix_percentile"] = ta.rolling_percentile(out["vix"], PERCENTILE_WINDOW)
    out["vix_rank"] = ta.rolling_rank(out["vix"], PERCENTILE_WINDOW)
    return out


def _profile_features(symbol: str, days: pd.Index, start: date, end: date) -> pd.DataFrame:
    """Volume profile of the PREVIOUS session, stamped on the current one.

    The shift lives here and nowhere else. A profile of session D is not
    complete until D closes, so row D carries session D-1's profile.
    """
    minute = read_dataset(RAW_UNDERLYING_MINUTE, symbols=[symbol], start=start, end=end)
    out = pd.DataFrame(index=days)
    if minute.empty:
        return out

    profiles = profiles_by_session(minute)
    if profiles.empty:
        return out

    profiles = profiles.set_index("day").reindex(days)
    shifted = profiles[list(PROFILE_COLUMNS)].shift(1)
    shifted.columns = [f"prev_{c}" for c in shifted.columns]

    for minutes in (5, 15, 30):
        rng = opening_range(minute, minutes)
        if rng.empty:
            continue
        rng = rng.set_index("day").reindex(days)
        # The opening range of session D completes during session D, so it is
        # knowable by D's close and is not shifted.
        out[f"or{minutes}_high"] = rng[f"or{minutes}_high"]
        out[f"or{minutes}_low"] = rng[f"or{minutes}_low"]

    vwap = session_vwap(minute)
    if not vwap.empty:
        closing_vwap = minute.assign(_v=vwap).sort_values("ts_utc").groupby("day")["_v"].last()
        out["session_vwap_close"] = closing_vwap.reindex(days)

    return pd.concat([out, shifted], axis=1)


# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------


def _regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Trend, volatility and day-type labels.

    Not an engine. A labelling function whose output is stored as ordinary
    feature columns, so any hypothesis can filter on it.
    """
    out = pd.DataFrame(index=df.index)

    # Trend direction from the moving average relationship.
    above_fast = df["close"] > df["ema_21"]
    fast_above_slow = df["ema_21"] > df["ema_50"]
    out["trend_direction"] = np.select(
        [above_fast & fast_above_slow, ~above_fast & ~fast_above_slow],
        ["up", "down"],
        default="sideways",
    )

    # Trend strength from ADX, using the conventional thresholds.
    adx = df["adx_14"]
    out["trend_strength"] = pd.Series(
        np.select(
            [adx < 20, adx < 25, adx >= 25],
            ["weak", "moderate", "strong"],
            default="unknown",
        ),
        index=df.index,
    ).where(adx.notna())

    # Volatility bucket from VIX percentile terciles.
    if "vix_percentile" in df.columns:
        pct = df["vix_percentile"]
        out["vol_bucket"] = pd.Series(
            np.select(
                [pct < 33.33, pct < 66.67, pct >= 66.67],
                ["low", "mid", "high"],
                default="unknown",
            ),
            index=df.index,
        ).where(pct.notna())
    else:
        out["vol_bucket"] = pd.NA

    # Range day versus trend day: how much of the session's range the close
    # retained. A close near an extreme is a trend day.
    span = (df["high"] - df["low"]).replace(0, np.nan)
    close_position = (df["close"] - df["low"]) / span
    out["close_position_in_range"] = close_position
    out["day_type"] = pd.Series(
        np.select(
            [close_position >= 0.75, close_position <= 0.25],
            ["trend_up", "trend_down"],
            default="range",
        ),
        index=df.index,
    ).where(close_position.notna())

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_daily_features(symbol: str, start: date | None = None, end: date | None = None):
    """Build the daily feature table for one symbol.

    Every column is knowable by the close of the session in its row.
    """
    daily = read_dataset(RAW_UNDERLYING_DAILY, symbols=[symbol], start=start, end=end)
    if daily.empty:
        return pd.DataFrame()

    daily = daily.sort_values("day").drop_duplicates("day").reset_index(drop=True)
    daily = daily.set_index("day")

    span_start = start or daily.index.min()
    span_end = end or daily.index.max()

    parts = [
        daily[["open", "high", "low", "close", "volume"]],
        _price_structure(daily),
        _technicals(daily),
        _vix_features(daily.index, span_start, span_end),
        _option_derived(symbol, daily.index, span_start, span_end),
        _profile_features(symbol, daily.index, span_start, span_end),
    ]
    features = pd.concat(parts, axis=1)

    features = pd.concat([features, _regime_labels(features)], axis=1)

    # Calendar flags are knowable in advance and are deliberately not shifted.
    cal, _ = build_calendar(span_start, span_end)
    if not cal.empty:
        cal_cols = [
            c
            for c in cal.columns
            if c.startswith("is_") or c in ("day_of_week", "day_of_month", "month")
        ]
        features = features.join(cal.set_index("day")[cal_cols])

    features = features.reset_index().rename(columns={"index": "day"})
    features.insert(0, "symbol", symbol)
    return features


def build_all(symbols: tuple[str, ...] = SYMBOLS, start=None, end=None) -> dict[str, int]:
    """Build and persist feature tables for every symbol."""
    counts: dict[str, int] = {}
    for symbol in symbols:
        df = build_daily_features(symbol, start, end)
        if df.empty:
            counts[symbol] = 0
            continue
        write_derived(df, FEATURES, f"daily_{symbol}")
        counts[symbol] = len(df)
    return counts


def load_features(symbol: str) -> pd.DataFrame:
    path = FEATURES / f"daily_{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "day" in df.columns and len(df):
        df["day"] = pd.to_datetime(df["day"]).dt.date
    return df
