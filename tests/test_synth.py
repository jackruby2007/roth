"""Tests for the synthetic fixture generator.

Two jobs here. First, confirm the fixture is realistic enough that code
validated against it will also work against real data. Second, confirm it is
impossible to mistake for real data.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from roth.data.synth import _build_expiration_grid, generate, is_synthetic, synthetic_info
from roth.schema import OPTION_EOD, UNDERLYING_DAILY, missing_columns
from roth.storage import read_dataset

START, END = date(2024, 1, 1), date(2024, 2, 29)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One generated dataset shared by every test in this module."""
    root = tmp_path_factory.mktemp("synthdata")
    import roth.data.synth as synth
    import roth.paths as paths
    import roth.storage as storage

    raw = root / "raw"
    patches = {
        "RAW": raw,
        "RAW_OPTION_EOD": raw / "option_eod",
        "RAW_UNDERLYING_DAILY": raw / "underlying_daily",
        "RAW_UNDERLYING_MINUTE": raw / "underlying_minute",
        "RAW_VIX": raw / "vix",
    }
    originals_synth = {k: getattr(synth, k) for k in patches}
    original_manifest = storage.MANIFEST_PATH
    original_dirs = paths.ALL_DIRS

    for k, v in patches.items():
        setattr(synth, k, v)
    storage.MANIFEST_PATH = raw / "_manifest.jsonl"
    paths.ALL_DIRS = list(patches.values())

    try:
        counts = generate(START, END, symbols=("SPY",))
        yield {"counts": counts, "paths": patches}
    finally:
        for k, v in originals_synth.items():
            setattr(synth, k, v)
        storage.MANIFEST_PATH = original_manifest
        paths.ALL_DIRS = original_dirs


# -- schema conformance -----------------------------------------------------


def test_daily_matches_the_canonical_schema(dataset):
    df = read_dataset(dataset["paths"]["RAW_UNDERLYING_DAILY"], symbols=["SPY"])
    assert missing_columns(df, UNDERLYING_DAILY) == []


def test_option_eod_matches_the_canonical_schema(dataset):
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    assert missing_columns(df, OPTION_EOD) == []


# -- the data has to behave like markets ------------------------------------


def test_daily_bars_are_internally_consistent(dataset):
    df = read_dataset(dataset["paths"]["RAW_UNDERLYING_DAILY"], symbols=["SPY"])
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()
    assert (df["volume"] > 0).all()


def test_minute_bars_are_internally_consistent(dataset):
    df = read_dataset(dataset["paths"]["RAW_UNDERLYING_MINUTE"], symbols=["SPY"])
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["close"]).all()


def test_minute_bars_are_stamped_in_utc(dataset):
    df = read_dataset(dataset["paths"]["RAW_UNDERLYING_MINUTE"], symbols=["SPY"])
    assert str(pd.Series(df["ts_utc"]).dt.tz) == "UTC"


def test_minute_closes_reconcile_to_the_daily_bar(dataset):
    """The last minute of a session must equal that session's daily close, or
    the two datasets disagree and every cross-check built on them is void."""
    daily = read_dataset(dataset["paths"]["RAW_UNDERLYING_DAILY"], symbols=["SPY"])
    minute = read_dataset(dataset["paths"]["RAW_UNDERLYING_MINUTE"], symbols=["SPY"])

    last = minute.sort_values("ts_utc").groupby("day")["close"].last()
    merged = daily.set_index("day")["close"].to_frame("daily").join(last.to_frame("minute"))
    merged = merged.dropna()

    assert len(merged) > 20
    assert (merged["daily"] - merged["minute"]).abs().max() < 0.01


def test_call_prices_fall_as_strike_rises(dataset):
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    day = df["day"].max()
    exp = sorted(df.loc[df["day"] == day, "expiration_date"].unique())[1]

    calls = df[
        (df["day"] == day) & (df["expiration_date"] == exp) & (df["right"] == "C")
    ].sort_values("strike_dollars")

    assert len(calls) > 10
    # Allow a cent of rounding noise.
    assert (calls["close"].diff().dropna() <= 0.011).all()


def test_deltas_are_in_range_and_signed_correctly(dataset):
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    calls, puts = df[df["right"] == "C"], df[df["right"] == "P"]

    assert calls["delta"].between(0.0, 1.0).all()
    assert puts["delta"].between(-1.0, 0.0).all()


def test_the_surface_has_downside_skew(dataset):
    """Index options skew: OTM puts carry higher implied vol than OTM calls.
    Without this, any delta-targeting logic is being tested against a world
    that does not exist."""
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    day = df["day"].max()
    d = df[df["day"] == day]
    spot = d["underlying_close"].iloc[0]

    otm_puts = d[(d["right"] == "P") & (d["strike_dollars"] < spot * 0.96)]
    otm_calls = d[(d["right"] == "C") & (d["strike_dollars"] > spot * 1.04)]

    assert otm_puts["implied_vol"].mean() > otm_calls["implied_vol"].mean()


def test_expirations_stay_inside_the_configured_horizon(dataset):
    from roth.config import BOUNDS

    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    dte = (pd.to_datetime(df["expiration_date"]) - pd.to_datetime(df["day"])).dt.days
    assert dte.min() >= 0
    assert dte.max() <= BOUNDS.max_dte


def test_strikes_stay_inside_the_configured_band(dataset):
    from roth.config import BOUNDS

    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    ratio = df["strike_dollars"] / df["underlying_close"]
    assert ratio.min() >= 1 - BOUNDS.strike_pct - 0.01
    assert ratio.max() <= 1 + BOUNDS.strike_pct + 0.01


def test_expiration_grid_is_all_fridays():
    grid = _build_expiration_grid([date(2024, 1, 2), date(2024, 3, 29)])
    assert grid
    assert all(d.weekday() == 4 for d in grid)


# -- deliberate defects for the quality layer -------------------------------


def test_some_quotes_are_crossed(dataset):
    """A quality layer that has never seen a crossed market is untested."""
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    assert (df["bid"] > df["ask"]).sum() > 0


def test_some_spreads_are_absurd(dataset):
    df = read_dataset(dataset["paths"]["RAW_OPTION_EOD"], symbols=["SPY"])
    mid = ((df["bid"] + df["ask"]) / 2).clip(lower=0.01)
    assert ((df["ask"] - df["bid"]) > 0.5 * mid).sum() > 0


# -- impossible to mistake for real -----------------------------------------


def test_generated_data_is_marked_synthetic(dataset):
    marker = dataset["paths"]["RAW"] / "_SYNTHETIC_DATA"
    assert marker.exists()


def test_the_marker_explains_itself(dataset, monkeypatch):
    monkeypatch.setattr("roth.data.synth.RAW", dataset["paths"]["RAW"])
    assert is_synthetic() is True

    info = synthetic_info()
    assert info["origin"] == "synthetic"
    assert "nothing about whether any strategy has edge" in info["warning"]


def test_real_data_is_not_flagged_synthetic(monkeypatch, tmp_path):
    monkeypatch.setattr("roth.data.synth.RAW", tmp_path)
    assert is_synthetic() is False
    assert synthetic_info() is None


# -- determinism ------------------------------------------------------------


def test_generation_is_deterministic(tmp_path):
    """Same seed, same data. Otherwise no result is reproducible."""
    import roth.data.synth as synth
    import roth.storage as storage

    def build(target):
        raw = target / "raw"
        patches = {
            "RAW": raw,
            "RAW_OPTION_EOD": raw / "option_eod",
            "RAW_UNDERLYING_DAILY": raw / "underlying_daily",
            "RAW_UNDERLYING_MINUTE": raw / "underlying_minute",
            "RAW_VIX": raw / "vix",
        }
        originals = {k: getattr(synth, k) for k in patches}
        original_manifest = storage.MANIFEST_PATH
        for k, v in patches.items():
            setattr(synth, k, v)
        storage.MANIFEST_PATH = raw / "_manifest.jsonl"
        try:
            generate(START, date(2024, 1, 31), symbols=("SPY",), with_minute=False)
            return read_dataset(patches["RAW_UNDERLYING_DAILY"], symbols=["SPY"])
        finally:
            for k, v in originals.items():
                setattr(synth, k, v)
            storage.MANIFEST_PATH = original_manifest

    first = build(tmp_path / "a")
    second = build(tmp_path / "b")
    pd.testing.assert_frame_equal(first, second)
