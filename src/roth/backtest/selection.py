"""Contract selection.

Turns a strategy's intent -- "a 30-delta call about 14 days out" -- into a
specific listed contract, using only the chain as of the selection session.

Selection deliberately does not consider whether the resulting contract is
*tradeable*. That judgement belongs to the fill model, one step later, so that a
contract skipped for a wide spread is recorded as an unfillable signal rather
than quietly swapped for a different strike. Silently sliding to the next strike
because the intended one was untradeable would flatter the strategy and hide the
execution problem the harness exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from roth.backtest.fills import Right


@dataclass(frozen=True)
class ContractSpec:
    """What a strategy asks for."""

    right: Right
    target_dte: int
    # Exactly one of these should be set.
    target_delta: float | None = None
    target_moneyness: float | None = None  # 0.0 is at the money, +0.05 is 5% OTM

    min_dte: int | None = None
    max_dte: int | None = None


@dataclass(frozen=True)
class SelectedContract:
    right: Right
    strike: float
    expiration: date
    dte: int
    delta: float | None
    implied_vol: float | None
    underlying: float


def choose_expiration(
    chain: pd.DataFrame, day: date, spec: ContractSpec
) -> tuple[date | None, int]:
    """The listed expiration closest to the target days-to-expiry."""
    if chain.empty:
        return None, 0

    exps = pd.Series(sorted(chain["expiration_date"].unique()))
    dtes = (pd.to_datetime(exps) - pd.Timestamp(day)).dt.days

    lo = spec.min_dte if spec.min_dte is not None else 0
    hi = spec.max_dte if spec.max_dte is not None else 10_000
    eligible = (dtes >= lo) & (dtes <= hi)

    if not eligible.any():
        return None, 0

    candidates = exps[eligible]
    candidate_dtes = dtes[eligible]
    best = (candidate_dtes - spec.target_dte).abs().idxmin()
    return candidates.loc[best], int(candidate_dtes.loc[best])


def select_contract(
    chain: pd.DataFrame, day: date, spec: ContractSpec
) -> SelectedContract | None:
    """Pick the single contract matching `spec`, or None if none qualifies."""
    if chain.empty:
        return None

    expiration, dte = choose_expiration(chain, day, spec)
    if expiration is None:
        return None

    candidates = chain[
        (chain["expiration_date"] == expiration) & (chain["right"] == spec.right.value)
    ]
    if candidates.empty:
        return None

    if spec.target_delta is not None:
        if "delta" not in candidates.columns:
            return None
        # Puts carry negative delta; compare on magnitude so a strategy can ask
        # for "30 delta" without caring about the sign convention.
        distance = (candidates["delta"].abs() - abs(spec.target_delta)).abs()
    elif spec.target_moneyness is not None:
        underlying = float(candidates["underlying_close"].iloc[0])
        target_strike = underlying * (1 + spec.target_moneyness)
        distance = (candidates["strike_dollars"] - target_strike).abs()
    else:
        underlying = float(candidates["underlying_close"].iloc[0])
        distance = (candidates["strike_dollars"] - underlying).abs()

    distance = distance.dropna()
    if distance.empty:
        return None

    row = candidates.loc[distance.idxmin()]

    return SelectedContract(
        right=spec.right,
        strike=float(row["strike_dollars"]),
        expiration=expiration,
        dte=dte,
        delta=float(row["delta"]) if pd.notna(row.get("delta")) else None,
        implied_vol=float(row["implied_vol"]) if pd.notna(row.get("implied_vol")) else None,
        underlying=float(row["underlying_close"]),
    )


def select_vertical(
    chain: pd.DataFrame,
    day: date,
    spec: ContractSpec,
    width: float,
) -> tuple[SelectedContract, float] | None:
    """Select a vertical spread: the long leg from `spec`, short `width` away.

    Returns the long contract and the short strike. The short strike must
    actually be listed, otherwise the spread does not exist and None is
    returned -- rounding to a nearby strike would silently change the position's
    risk.
    """
    long_leg = select_contract(chain, day, spec)
    if long_leg is None:
        return None

    offset = width if spec.right is Right.CALL else -width
    short_strike = long_leg.strike + offset

    listed = chain[
        (chain["expiration_date"] == long_leg.expiration)
        & (chain["right"] == spec.right.value)
        & (chain["strike_dollars"] == short_strike)
    ]
    if listed.empty:
        return None

    return long_leg, short_strike
