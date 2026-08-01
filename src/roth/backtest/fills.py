"""Options fill model.

The rule that governs everything here: **never fill at mid**. Buys cross to the
ask, sells cross to the bid, using the actual historical NBBO at fill time.

Mid-price fills are the single most common way a backtest manufactures an edge
that does not exist. On a contract quoted 1.00 / 1.20, filling at 1.10 hands the
strategy six cents a side it would never have received. Over a few hundred
round trips that is the entire result.

When a quote is unusable -- missing, crossed, or spread wider than the
configured threshold -- the trade is **rejected**, not repaired. No synthetic
price, no interpolation, no falling back to the theoretical value. A strategy
that only works when bad quotes are quietly replaced does not work, and the
rejection count appears in every report so that it cannot be overlooked.

`use_mid=True` exists solely for the cost-sensitivity test: run a result both
ways and the difference is how much of the "edge" was really an execution
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from roth.config import COSTS, CostModel

CONTRACT_MULTIPLIER = 100


class Right(str, Enum):
    CALL = "C"
    PUT = "P"


class Action(str, Enum):
    OPEN = "open"
    CLOSE = "close"


class RejectReason(str, Enum):
    NO_CHAIN = "no_chain_for_session"
    NO_CONTRACT = "contract_not_listed"
    MISSING_QUOTE = "quote_missing_or_null"
    CROSSED = "crossed_market_bid_above_ask"
    ZERO_ASK = "non_positive_ask"
    WIDE_SPREAD = "spread_exceeds_threshold"

    def describe(self) -> str:
        return {
            RejectReason.NO_CHAIN: "no option chain on this session",
            RejectReason.NO_CONTRACT: "the contract was not listed",
            RejectReason.MISSING_QUOTE: "bid or ask was missing",
            RejectReason.CROSSED: "bid was above ask",
            RejectReason.ZERO_ASK: "ask was zero or negative",
            RejectReason.WIDE_SPREAD: "spread was wider than the configured limit",
        }[self]


@dataclass(frozen=True)
class Leg:
    """One leg of a structure.

    `quantity` is signed per structure unit: +1 long, -1 short. A vertical call
    spread is a +1 leg at the lower strike and a -1 leg at the higher.
    """

    right: Right
    strike: float
    expiration: date
    quantity: int = 1

    def occ_symbol(self, root: str) -> str:
        """OCC contract identifier, e.g. SPY240621C00500000.

        The industry-standard identifier, so a journalled trade can be matched
        against a broker statement or a third-party dataset without ambiguity
        about which contract was meant.
        """
        strike_thousandths = int(round(self.strike * 1000))
        return (
            f"{root.upper()}{self.expiration:%y%m%d}{self.right.value}"
            f"{strike_thousandths:08d}"
        )


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct_of_mid(self) -> float:
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return self.spread / mid


@dataclass
class LegFill:
    leg: Leg
    quote: Quote
    side: str  # "buy" or "sell"
    price: float  # per share, after slippage


@dataclass
class FillResult:
    """Outcome of attempting to fill a structure."""

    filled: bool
    action: Action
    day: date
    legs: list[LegFill] = field(default_factory=list)
    reject_reason: RejectReason | None = None
    reject_detail: str = ""

    # Per structure unit, in dollars (already multiplied by 100).
    price_per_unit: float = 0.0
    commission: float = 0.0
    contracts: int = 0

    @property
    def gross_value(self) -> float:
        """Market value of the position, positive for a debit structure."""
        return self.price_per_unit * self.contracts

    @property
    def net_cash(self) -> float:
        """Signed cash flow. Negative is money leaving the account.

        Opening a debit structure costs money; closing it returns money. Both
        pay commission.
        """
        direction = -1.0 if self.action is Action.OPEN else 1.0
        return direction * self.gross_value - self.commission


class FillModel:
    """Prices structures against historical NBBO, or rejects them."""

    def __init__(self, costs: CostModel = COSTS, use_mid: bool = False) -> None:
        self.costs = costs
        self.use_mid = use_mid

    # -- quote lookup ------------------------------------------------------

    def find_quote(
        self, chain: pd.DataFrame, leg: Leg
    ) -> tuple[Quote | None, RejectReason | None]:
        """Locate one contract's quote and validate it.

        Returns `(quote, None)` when tradeable, `(None, reason)` otherwise.
        """
        if chain is None or chain.empty:
            return None, RejectReason.NO_CHAIN

        match = chain[
            (chain["right"] == leg.right.value)
            & (chain["strike_dollars"] == leg.strike)
            & (chain["expiration_date"] == leg.expiration)
        ]
        if match.empty:
            return None, RejectReason.NO_CONTRACT

        row = match.iloc[0]
        bid, ask = row.get("bid"), row.get("ask")

        if pd.isna(bid) or pd.isna(ask):
            return None, RejectReason.MISSING_QUOTE

        bid, ask = float(bid), float(ask)

        if bid > ask:
            return None, RejectReason.CROSSED
        if ask <= 0:
            return None, RejectReason.ZERO_ASK

        quote = Quote(
            bid=bid,
            ask=ask,
            bid_size=float(row.get("bid_size") or 0),
            ask_size=float(row.get("ask_size") or 0),
        )

        if quote.spread_pct_of_mid > self.costs.max_spread_pct_of_mid:
            return None, RejectReason.WIDE_SPREAD

        return quote, None

    # -- pricing -----------------------------------------------------------

    def leg_price(self, quote: Quote, side: str) -> float:
        """Price one leg, per share.

        Buys pay the ask, sells receive the bid, and the configured extra
        slippage always moves the price against the trade.
        """
        slip = self.costs.extra_slippage_per_contract

        if self.use_mid:
            base = quote.mid
            return base + slip if side == "buy" else base - slip

        if side == "buy":
            return quote.ask + slip
        return quote.bid - slip

    @staticmethod
    def _side_for(leg: Leg, action: Action) -> str:
        """Which way a leg trades.

        Opening a long leg buys it; closing that same leg sells it. Getting this
        backwards would make every exit cross the wrong side of the spread.
        """
        going_long = leg.quantity > 0
        if action is Action.OPEN:
            return "buy" if going_long else "sell"
        return "sell" if going_long else "buy"

    def fill(
        self,
        chain: pd.DataFrame,
        legs: list[Leg],
        action: Action,
        day: date,
        contracts: int = 1,
    ) -> FillResult:
        """Attempt to fill a whole structure.

        All-or-nothing: if any leg is unfillable the structure is rejected. A
        partially filled vertical is a different position with a different risk
        profile, and silently taking one would misreport the strategy.
        """
        result = FillResult(filled=False, action=action, day=day, contracts=contracts)

        leg_fills: list[LegFill] = []
        for leg in legs:
            quote, reason = self.find_quote(chain, leg)
            if quote is None:
                result.reject_reason = reason
                result.reject_detail = (
                    f"{leg.right.value} {leg.strike:g} exp {leg.expiration}: "
                    f"{reason.describe()}"
                )
                return result

            side = self._side_for(leg, action)
            leg_fills.append(LegFill(leg=leg, quote=quote, side=side, price=self.leg_price(quote, side)))

        # Net *market value* of the structure per unit, signed by each leg's
        # position rather than by which way it happens to be trading.
        #
        # This is the subtle one. Long call: opening pays the ask, closing
        # receives the bid -- both are positive values of the same position.
        # Signing by trade side instead would make every exit negative, and the
        # P/L arithmetic downstream would double-count the entry cost.
        per_unit = sum(lf.leg.quantity * lf.price for lf in leg_fills)

        total_contracts = sum(abs(leg.quantity) for leg in legs) * contracts
        commission = self.costs.commission_per_contract * total_contracts

        result.filled = True
        result.legs = leg_fills
        result.price_per_unit = per_unit * CONTRACT_MULTIPLIER
        result.commission = commission
        return result


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------


def long_call(strike: float, expiration: date) -> list[Leg]:
    return [Leg(Right.CALL, strike, expiration, +1)]


def long_put(strike: float, expiration: date) -> list[Leg]:
    return [Leg(Right.PUT, strike, expiration, +1)]


def vertical_call_spread(long_strike: float, short_strike: float, expiration: date) -> list[Leg]:
    """Long the lower strike, short the higher. A debit spread."""
    return [
        Leg(Right.CALL, long_strike, expiration, +1),
        Leg(Right.CALL, short_strike, expiration, -1),
    ]


def vertical_put_spread(long_strike: float, short_strike: float, expiration: date) -> list[Leg]:
    """Long the higher strike, short the lower. A debit spread."""
    return [
        Leg(Right.PUT, long_strike, expiration, +1),
        Leg(Right.PUT, short_strike, expiration, -1),
    ]


STRUCTURES = {
    "long_call": long_call,
    "long_put": long_put,
    "vertical_call_spread": vertical_call_spread,
    "vertical_put_spread": vertical_put_spread,
}
