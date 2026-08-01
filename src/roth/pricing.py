"""Black-Scholes pricing and greeks.

Used by the synthetic fixture generator to produce option chains that behave
like real ones. It is deliberately not used anywhere in the backtest path: the
backtest fills against quoted bid/ask only, never a model price. If a quote is
missing the trade is rejected, not priced.

No scipy dependency -- the normal CDF is `math.erf`, which is exact enough here
and keeps the install small.
"""

from __future__ import annotations

import math

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
    return d1, d1 - vol_t


def price(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    right: str,
    q: float = 0.0,
) -> float:
    """Black-Scholes price with continuous dividend yield.

    `t` is in years. At or past expiry, returns intrinsic value.
    """
    if t <= 0 or sigma <= 0:
        intrinsic = (s - k) if right == "C" else (k - s)
        return max(intrinsic, 0.0)

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)

    if right == "C":
        return s * disc_q * norm_cdf(d1) - k * disc_r * norm_cdf(d2)
    return k * disc_r * norm_cdf(-d2) - s * disc_q * norm_cdf(-d1)


def greeks(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    right: str,
    q: float = 0.0,
) -> dict[str, float]:
    """Delta, gamma, theta, vega, rho.

    Theta is per calendar day and vega per one volatility point, matching how
    both are conventionally quoted.
    """
    if t <= 0 or sigma <= 0:
        if right == "C":
            delta = 1.0 if s > k else 0.0
        else:
            delta = -1.0 if s < k else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    sqrt_t = math.sqrt(t)
    disc_r = math.exp(-r * t)
    disc_q = math.exp(-q * t)
    pdf_d1 = norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (s * sigma * sqrt_t)
    vega = s * disc_q * pdf_d1 * sqrt_t / 100.0

    common_theta = -(s * disc_q * pdf_d1 * sigma) / (2 * sqrt_t)
    if right == "C":
        delta = disc_q * norm_cdf(d1)
        theta = common_theta - r * k * disc_r * norm_cdf(d2) + q * s * disc_q * norm_cdf(d1)
        rho = k * t * disc_r * norm_cdf(d2) / 100.0
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = common_theta + r * k * disc_r * norm_cdf(-d2) - q * s * disc_q * norm_cdf(-d1)
        rho = -k * t * disc_r * norm_cdf(-d2) / 100.0

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta / 365.0,
        "vega": vega,
        "rho": rho,
    }


def implied_vol(
    target: float,
    s: float,
    k: float,
    t: float,
    r: float,
    right: str,
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """Invert Black-Scholes for volatility by bisection.

    Bisection rather than Newton: it cannot diverge, and speed is irrelevant
    here. Returns None when the target price is outside the achievable range,
    which is the honest answer for an arbitrage-violating quote.
    """
    if t <= 0:
        return None

    lo, hi = 1e-6, 5.0
    if price(s, k, t, r, hi, right, q) < target:
        return None
    if price(s, k, t, r, lo, right, q) > target:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        diff = price(s, k, t, r, mid, right, q) - target
        if abs(diff) < tol:
            return mid
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)
