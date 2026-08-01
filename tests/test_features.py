"""Tests for the feature store.

Two concerns. The indicators have to be arithmetically right, and none of them
may look forward. The second concern gets the most attention here, because a
lookahead bug produces beautiful backtest results rather than an error.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from roth.features import technicals as ta
from roth.features.causality import verify_causality
from roth.features.profile import opening_range, session_profile, session_vwap


@pytest.fixture
def close() -> pd.Series:
    rng = np.random.default_rng(7)
    return pd.Series(100 + np.cumsum(rng.standard_normal(300)))


@pytest.fixture
def ohlc(close: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    span = np.abs(rng.standard_normal(len(close))) + 0.5
    return pd.DataFrame(
        {"high": close + span, "low": close - span, "close": close, "open": close.shift(1)}
    )


# -- indicator arithmetic ---------------------------------------------------


def test_sma_matches_a_hand_computed_mean(close):
    result = ta.sma(close, 10)
    assert result.iloc[9] == pytest.approx(close.iloc[0:10].mean())
    assert result.iloc[50] == pytest.approx(close.iloc[41:51].mean())


def test_partial_windows_are_nan_not_a_shorter_average(close):
    """A 200-day average computed from 20 points is a different indicator
    wearing the same name. It must be NaN instead."""
    result = ta.sma(close, 200)
    assert result.iloc[:199].isna().all()
    assert result.iloc[199:].notna().all()


def test_ema_respects_min_periods(close):
    assert ta.ema(close, 50).iloc[:49].isna().all()


def test_true_range_uses_the_previous_close(ohlc):
    tr = ta.true_range(ohlc["high"], ohlc["low"], ohlc["close"])
    i = 100
    expected = max(
        ohlc["high"].iloc[i] - ohlc["low"].iloc[i],
        abs(ohlc["high"].iloc[i] - ohlc["close"].iloc[i - 1]),
        abs(ohlc["low"].iloc[i] - ohlc["close"].iloc[i - 1]),
    )
    assert tr.iloc[i] == pytest.approx(expected)


def test_rsi_is_bounded(close):
    r = ta.rsi(close, 14).dropna()
    assert r.between(0, 100).all()


def test_rsi_is_100_when_every_change_is_a_gain():
    rising = pd.Series(np.arange(100, 140, dtype=float))
    assert ta.rsi(rising, 14).dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_when_every_change_is_a_loss():
    falling = pd.Series(np.arange(140, 100, -1, dtype=float))
    assert ta.rsi(falling, 14).dropna().iloc[-1] == pytest.approx(0.0)


def test_macd_histogram_is_line_minus_signal(close):
    line, signal, hist = ta.macd(close)
    valid = hist.notna()
    assert np.allclose((line - signal)[valid], hist[valid])


def test_adx_is_bounded_and_high_in_a_strong_trend(ohlc):
    _, _, adx = ta.directional_movement(ohlc["high"], ohlc["low"], ohlc["close"], 14)
    assert adx.dropna().between(0, 100).all()

    n = 100
    trend_close = pd.Series(np.linspace(100, 200, n))
    trend = pd.DataFrame(
        {"high": trend_close + 1, "low": trend_close - 1, "close": trend_close}
    )
    _, _, trend_adx = ta.directional_movement(trend["high"], trend["low"], trend["close"], 14)
    assert trend_adx.dropna().iloc[-1] > 50


def test_bollinger_bands_bracket_the_middle(close):
    upper, mid, lower = ta.bollinger(close, 20, 2.0)
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (lower[valid] <= mid[valid]).all()


def test_stochastic_is_bounded(ohlc):
    k, d = ta.stochastic(ohlc["high"], ohlc["low"], ohlc["close"])
    assert k.dropna().between(0, 100).all()
    assert d.dropna().between(0, 100).all()


def test_realized_volatility_rises_with_dispersion():
    calm = pd.Series(100 + np.sin(np.arange(300) / 10) * 0.1)
    wild = pd.Series(100 + np.sin(np.arange(300) / 10) * 10)
    assert (
        ta.realized_volatility(wild, 21).dropna().mean()
        > ta.realized_volatility(calm, 21).dropna().mean()
    )


def test_rolling_percentile_is_bounded_and_ranks_extremes_correctly():
    rising = pd.Series(np.arange(100, dtype=float))
    pct = ta.rolling_percentile(rising, 20).dropna()
    assert pct.between(0, 100).all()
    # A monotonically rising series is always at the top of its own window.
    assert pct.iloc[-1] == pytest.approx(100.0)


def test_rolling_rank_places_value_between_window_extremes():
    series = pd.Series([10.0] * 19 + [20.0])
    assert ta.rolling_rank(series, 20).iloc[-1] == pytest.approx(100.0)


# -- no indicator may look forward -----------------------------------------

INDICATORS = {
    "sma": lambda o: ta.sma(o["close"], 20),
    "ema": lambda o: ta.ema(o["close"], 20),
    "atr": lambda o: ta.atr(o["high"], o["low"], o["close"], 14),
    "rsi": lambda o: ta.rsi(o["close"], 14),
    "macd": lambda o: ta.macd(o["close"])[0],
    "adx": lambda o: ta.directional_movement(o["high"], o["low"], o["close"], 14)[2],
    "bollinger_upper": lambda o: ta.bollinger(o["close"], 20)[0],
    "keltner_upper": lambda o: ta.keltner(o["high"], o["low"], o["close"])[0],
    "stoch_k": lambda o: ta.stochastic(o["high"], o["low"], o["close"])[0],
    "realized_vol": lambda o: ta.realized_volatility(o["close"], 21),
    "rolling_percentile": lambda o: ta.rolling_percentile(o["close"], 20),
    "rolling_rank": lambda o: ta.rolling_rank(o["close"], 20),
}


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_value_does_not_change_when_future_rows_are_removed(name, ohlc):
    """Truncating the input must not alter any value that came before the cut.

    This is the same invariance the feature-store check applies, at the level of
    a single indicator, so a broken one is caught at its source.
    """
    fn = INDICATORS[name]
    cut = 200

    full = fn(ohlc).iloc[:cut]
    truncated = fn(ohlc.iloc[:cut])

    both_na = full.isna() & truncated.isna()
    pd.testing.assert_series_equal(
        full[~both_na], truncated[~both_na], check_names=False, rtol=1e-9
    )


# -- volume profile ---------------------------------------------------------


def minute_session(day: date, prices: list[float], volumes: list[float]) -> pd.DataFrame:
    ts = pd.date_range(f"{day} 14:30", periods=len(prices), freq="1min", tz="UTC")
    p = pd.Series(prices)
    return pd.DataFrame(
        {
            "day": day,
            "ts_utc": ts,
            "high": p + 0.1,
            "low": p - 0.1,
            "close": p,
            "volume": volumes,
        }
    )


def test_point_of_control_is_the_heaviest_price():
    prices = [100.0] * 10 + [105.0] * 50 + [110.0] * 10
    volumes = [100.0] * 70
    stats = session_profile(minute_session(date(2024, 6, 3), prices, volumes), tick=1.0)
    assert stats["poc"] == pytest.approx(105.0, abs=1.0)


def test_value_area_brackets_the_point_of_control():
    rng = np.random.default_rng(3)
    prices = list(100 + rng.standard_normal(200) * 2)
    stats = session_profile(minute_session(date(2024, 6, 3), prices, [100.0] * 200), tick=0.25)
    assert stats["value_area_low"] <= stats["poc"] <= stats["value_area_high"]


def test_value_area_holds_about_seventy_percent_of_volume():
    rng = np.random.default_rng(4)
    prices = list(100 + rng.standard_normal(400) * 2)
    stats = session_profile(minute_session(date(2024, 6, 3), prices, [100.0] * 400), tick=0.25)
    assert 68.0 <= stats["value_area_volume_pct"] <= 85.0


def test_session_vwap_resets_each_session():
    a = minute_session(date(2024, 6, 3), [100.0] * 10, [100.0] * 10)
    b = minute_session(date(2024, 6, 4), [200.0] * 10, [100.0] * 10)
    combined = pd.concat([a, b], ignore_index=True)

    vwap = session_vwap(combined)
    # Second session must not be dragged toward the first.
    assert vwap.iloc[10] == pytest.approx(200.0, abs=0.2)


def test_session_vwap_is_expanding_not_rolling():
    prices = [100.0] * 5 + [200.0] * 5
    df = minute_session(date(2024, 6, 3), prices, [100.0] * 10)
    vwap = session_vwap(df)

    assert vwap.iloc[0] == pytest.approx(100.0, abs=0.2)
    # By the final bar it reflects the whole session, not just recent bars.
    assert vwap.iloc[-1] == pytest.approx(150.0, abs=0.2)


def test_opening_range_uses_only_the_first_n_minutes():
    prices = [100.0] * 5 + [500.0] * 25
    df = minute_session(date(2024, 6, 3), prices, [100.0] * 30)

    rng5 = opening_range(df, 5)
    assert rng5.iloc[0]["or5_high"] == pytest.approx(100.1)
    # The 500 spike happens after the range closes and must not appear in it.
    assert rng5.iloc[0]["or5_low"] == pytest.approx(99.9)


def test_opening_range_is_stamped_when_it_completes():
    df = minute_session(date(2024, 6, 3), [100.0] * 30, [100.0] * 30)
    rng30 = opening_range(df, 30)
    stamped = rng30.iloc[0]["or30_available_at"]
    assert stamped == df["ts_utc"].iloc[29]


def test_incomplete_opening_range_is_omitted():
    df = minute_session(date(2024, 6, 3), [100.0] * 10, [100.0] * 10)
    assert opening_range(df, 30).empty


# -- the causality checker itself must work ---------------------------------


def _fake_builder(leak: bool):
    """A miniature feature builder, optionally with a deliberate lookahead."""
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(120)]
    rng = np.random.default_rng(11)
    closes = 100 + np.cumsum(rng.standard_normal(120))

    def build(symbol, start=None, end=None):
        df = pd.DataFrame({"symbol": symbol, "day": days, "close": closes})
        if end is not None:
            df = df[df["day"] <= end]
        df = df.reset_index(drop=True)
        df["causal_sma"] = df["close"].rolling(10, min_periods=10).mean()
        if leak:
            # Centred window: reads five rows into the future.
            df["leaky_sma"] = df["close"].rolling(10, min_periods=10, center=True).mean()
        return df

    return build


def test_causality_check_passes_a_clean_builder():
    result = verify_causality("SPY", n_cutoffs=3, builder=_fake_builder(leak=False))
    assert result.passed, result.summary()
    assert result.columns_checked >= 2


def test_causality_check_catches_a_centred_window():
    """The negative control. If this ever passes, the checker is not working and
    every other causality guarantee in the harness is worthless."""
    result = verify_causality("SPY", n_cutoffs=3, builder=_fake_builder(leak=True))

    assert not result.passed
    assert {v.column for v in result.violations} == {"leaky_sma"}


def test_causality_check_reports_a_usable_example():
    result = verify_causality("SPY", n_cutoffs=3, builder=_fake_builder(leak=True))
    v = result.violations[0]
    assert v.example_day is not None
    assert v.rows_differing > 0
