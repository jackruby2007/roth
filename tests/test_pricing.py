"""Tests for Black-Scholes pricing and greeks.

Pinned against textbook identities rather than remembered numbers, so these
stay meaningful without a reference implementation to compare against.
"""

from __future__ import annotations

import math

import pytest

from roth.pricing import greeks, implied_vol, norm_cdf, price

S, K, T, R, SIG, Q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0


def test_norm_cdf_known_points():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_put_call_parity():
    """C - P == S*e^-qT - K*e^-rT. The single strongest check on the pricer."""
    c = price(S, K, T, R, SIG, "C", Q)
    p = price(S, K, T, R, SIG, "P", Q)
    expected = S * math.exp(-Q * T) - K * math.exp(-R * T)
    assert (c - p) == pytest.approx(expected, abs=1e-8)


def test_put_call_parity_with_dividend_yield():
    q = 0.03
    c = price(S, 95.0, T, R, SIG, "C", q)
    p = price(S, 95.0, T, R, SIG, "P", q)
    expected = S * math.exp(-q * T) - 95.0 * math.exp(-R * T)
    assert (c - p) == pytest.approx(expected, abs=1e-8)


def test_price_is_at_least_intrinsic():
    deep_itm_call = price(150.0, 100.0, 0.5, R, SIG, "C")
    assert deep_itm_call >= 150.0 - 100.0 * math.exp(-R * 0.5) - 1e-9

    deep_itm_put = price(50.0, 100.0, 0.5, R, SIG, "P")
    assert deep_itm_put >= 100.0 * math.exp(-R * 0.5) - 50.0 - 1e-9


def test_at_expiry_price_is_intrinsic():
    assert price(110.0, 100.0, 0.0, R, SIG, "C") == pytest.approx(10.0)
    assert price(90.0, 100.0, 0.0, R, SIG, "C") == pytest.approx(0.0)
    assert price(90.0, 100.0, 0.0, R, SIG, "P") == pytest.approx(10.0)


def test_call_price_decreases_with_strike():
    prices = [price(S, k, T, R, SIG, "C") for k in range(80, 121, 5)]
    assert all(a > b for a, b in zip(prices, prices[1:], strict=False))


def test_price_increases_with_volatility():
    low = price(S, K, T, R, 0.10, "C")
    high = price(S, K, T, R, 0.40, "C")
    assert high > low


def test_call_delta_is_bounded_and_ordered():
    deep_itm = greeks(150.0, 100.0, T, R, SIG, "C")["delta"]
    atm = greeks(100.0, 100.0, T, R, SIG, "C")["delta"]
    deep_otm = greeks(60.0, 100.0, T, R, SIG, "C")["delta"]

    assert 0.0 <= deep_otm < atm < deep_itm <= 1.0
    assert deep_itm == pytest.approx(1.0, abs=0.02)
    assert deep_otm == pytest.approx(0.0, abs=0.02)


def test_put_delta_is_negative():
    d = greeks(100.0, 100.0, T, R, SIG, "P")["delta"]
    assert -1.0 <= d <= 0.0


def test_delta_parity():
    """call delta - put delta == e^-qT."""
    q = 0.02
    cd = greeks(S, K, T, R, SIG, "C", q)["delta"]
    pd_ = greeks(S, K, T, R, SIG, "P", q)["delta"]
    assert (cd - pd_) == pytest.approx(math.exp(-q * T), abs=1e-9)


def test_gamma_and_vega_peak_at_the_money_with_no_carry():
    """With zero rates and no dividend the forward equals spot, so the peak sits
    exactly at the money."""
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    gammas = [greeks(S, k, T, 0.0, SIG, "C", 0.0)["gamma"] for k in strikes]
    vegas = [greeks(S, k, T, 0.0, SIG, "C", 0.0)["vega"] for k in strikes]

    assert gammas.index(max(gammas)) == 2
    assert vegas.index(max(vegas)) == 2


def test_gamma_peak_tracks_the_forward_when_carry_is_positive():
    """With r=5% over a year the forward is near 105, and the gamma peak moves
    up with it. A peak pinned to spot would mean carry is being ignored."""
    forward = S * math.exp((R - Q) * T)
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    gammas = [greeks(S, k, T, R, SIG, "C", Q)["gamma"] for k in strikes]

    peak_strike = strikes[gammas.index(max(gammas))]
    assert peak_strike > S
    assert abs(peak_strike - forward) <= 10.0


def test_gamma_and_vega_match_between_calls_and_puts():
    c = greeks(S, K, T, R, SIG, "C")
    p = greeks(S, K, T, R, SIG, "P")
    assert c["gamma"] == pytest.approx(p["gamma"])
    assert c["vega"] == pytest.approx(p["vega"])


def test_long_option_theta_is_negative():
    assert greeks(S, K, T, R, SIG, "C")["theta"] < 0
    assert greeks(S, K, T, R, SIG, "P")["theta"] < 0


def test_delta_matches_a_numerical_derivative():
    """Analytic delta must agree with a finite difference of the price."""
    h = 0.01
    up = price(S + h, K, T, R, SIG, "C")
    down = price(S - h, K, T, R, SIG, "C")
    numeric = (up - down) / (2 * h)
    assert greeks(S, K, T, R, SIG, "C")["delta"] == pytest.approx(numeric, abs=1e-4)


def test_vega_matches_a_numerical_derivative():
    h = 1e-4
    up = price(S, K, T, R, SIG + h, "C")
    down = price(S, K, T, R, SIG - h, "C")
    numeric = (up - down) / (2 * h) / 100.0
    assert greeks(S, K, T, R, SIG, "C")["vega"] == pytest.approx(numeric, abs=1e-4)


def test_implied_vol_round_trips():
    for sigma in (0.08, 0.20, 0.55):
        target = price(S, K, T, R, sigma, "C")
        assert implied_vol(target, S, K, T, R, "C") == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_returns_none_for_impossible_prices():
    """An arbitrage-violating quote has no implied vol, and saying so is the
    honest answer -- better than returning a number that looks valid."""
    assert implied_vol(S + 10.0, S, K, T, R, "C") is None
    assert implied_vol(0.0, S, K, T, R, "C") is None
