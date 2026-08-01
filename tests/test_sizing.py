"""Tests for the dataset size model.

These check the arithmetic, not the assumptions. The assumptions are only
validated by an actual pilot download.
"""

from __future__ import annotations

import json

from roth.config import BOUNDS
from roth.data.sizing import (
    MINUTES_PER_RTH_SESSION,
    TRADING_DAYS_PER_YEAR,
    SizingAssumptions,
    build_report,
    estimate_year,
    load_assumptions,
    save_assumptions,
    strikes_in_band,
)


def test_strike_band_scales_with_spot():
    """A +/-10% band around a higher spot contains proportionally more $1 strikes."""
    a = SizingAssumptions(listed_fraction=1.0, strike_spacing=1.0)
    assert strikes_in_band(100.0, a) == 20  # 0.2 * 100
    assert strikes_in_band(750.0, a) == 150  # 0.2 * 750


def test_strike_band_respects_configured_pct():
    a = SizingAssumptions(listed_fraction=1.0, strike_spacing=1.0)
    expected = int(round(2 * BOUNDS.strike_pct * 500.0))
    assert strikes_in_band(500.0, a) == expected


def test_wider_strike_spacing_means_fewer_strikes():
    dense = SizingAssumptions(listed_fraction=1.0, strike_spacing=1.0)
    sparse = SizingAssumptions(listed_fraction=1.0, strike_spacing=5.0)
    assert strikes_in_band(750.0, sparse) * 5 == strikes_in_band(750.0, dense)


def test_quote_rows_are_contracts_times_minutes_times_days():
    a = SizingAssumptions()
    est = estimate_year("SPY", 2026, a)
    assert est.quote_rows == (
        est.contracts_in_scope * MINUTES_PER_RTH_SESSION * TRADING_DAYS_PER_YEAR
    )
    # Calls and puts, so contracts is an even multiple of the strike count.
    assert est.contracts_in_scope == est.expirations_in_window * est.strikes_in_band * 2


def test_eod_is_one_row_per_contract_per_day():
    est = estimate_year("SPY", 2026, SizingAssumptions())
    assert est.eod_rows == est.contracts_in_scope * TRADING_DAYS_PER_YEAR
    # EOD must be dramatically cheaper than 1-minute data, or the whole
    # "start with EOD" strategy is pointless.
    assert est.eod_bytes < est.quote_bytes_low / 10


def test_modeled_assumptions_produce_a_range_measured_do_not():
    modeled = SizingAssumptions()
    lo, hi = modeled.bytes_per_quote_row_range()
    assert lo < hi

    measured = SizingAssumptions(bytes_per_quote_row=17.5, measured=True)
    lo, hi = measured.bytes_per_quote_row_range()
    assert lo == hi == 17.5


def test_report_totals_match_sum_of_years():
    report = build_report(("SPY", "QQQ"), 2024, 2026, SizingAssumptions())
    assert len(report.years) == 3 * 2
    totals = report.totals()
    assert totals["quote_rows"] == sum(y.quote_rows for y in report.years)

    recent = report.totals(since=2026)
    assert recent["quote_rows"] < totals["quote_rows"]


def test_assumptions_round_trip_through_disk(tmp_path):
    path = tmp_path / "measured.json"
    original = SizingAssumptions(
        bytes_per_quote_row=21.25,
        seconds_per_bulk_request=0.83,
        measured=True,
        measured_from="test",
    )
    save_assumptions(original, path)
    assert json.loads(path.read_text())["measured"] is True

    loaded = load_assumptions(path)
    assert loaded == original


def test_load_assumptions_falls_back_when_file_is_absent_or_corrupt(tmp_path):
    missing = tmp_path / "nope.json"
    assert load_assumptions(missing).measured is False

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert load_assumptions(corrupt).measured is False
