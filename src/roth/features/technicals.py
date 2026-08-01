"""Technical indicators.

Every function here is causal by construction: the value at index i depends
only on rows 0..i. Two rules make that true and both are easy to break by
accident, so they are stated once here and followed everywhere below.

1. **No centred windows.** `rolling(...)` defaults to a trailing window, which
   is what we want. `center=True` would look forward and must never appear.
2. **`min_periods` equals the window.** A 200-day moving average computed from
   20 observations is not a 200-day moving average; it is a different, shorter
   indicator wearing the same column name. Partial windows return NaN instead,
   and the backtest simply has no signal until enough history exists.

Wilder's smoothing (RSI, ATR, ADX) uses `ewm(alpha=1/n, adjust=False)`, which
is the recursive form and depends only on prior values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range. Uses the *previous* close, never the current one."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range, Wilder smoothed."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index, Wilder smoothed."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # All-gain windows have zero average loss, which is RSI 100 by definition.
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def directional_movement(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """+DI, -DI and ADX, Wilder smoothed.

    ADX is the trend-strength half of the regime label, so it has to be right.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    alpha = 1 / window
    atr_ = true_range(high, low, close).ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=window).mean() / atr_

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    return plus_di, minus_di, adx


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger upper, middle, lower."""
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    return mid + num_std * sd, mid, mid - num_std * sd


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner upper, middle, lower."""
    mid = ema(close, window)
    band = multiplier * atr(high, low, close, atr_window)
    return mid + band, mid, mid - band


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K and %D."""
    lowest = low.rolling(window, min_periods=window).min()
    highest = high.rolling(window, min_periods=window).max()

    raw_k = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(smooth_d, min_periods=smooth_d).mean()
    return k, d


def realized_volatility(close: pd.Series, window: int, periods_per_year: int = 252) -> pd.Series:
    """Annualised realised volatility from close-to-close log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(periods_per_year)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the latest value within its own trailing window.

    Used for IV rank and VIX percentile. The window includes the current value
    and nothing after it, which is what makes it usable as a signal.
    """

    def _rank(values: np.ndarray) -> float:
        return float((values <= values[-1]).sum() - 1) / (len(values) - 1) * 100.0

    return series.rolling(window, min_periods=window).apply(_rank, raw=True)


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    """Position of the latest value between its trailing min and max, 0-100.

    This is the "IV rank" convention, which differs from IV percentile: rank is
    where the value sits in the observed range, percentile is what fraction of
    observations it exceeds.
    """
    lo = series.rolling(window, min_periods=window).min()
    hi = series.rolling(window, min_periods=window).max()
    return 100 * (series - lo) / (hi - lo).replace(0, np.nan)
