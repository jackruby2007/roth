"""Tests for the options fill model.

This is the component that decides whether a backtest is honest, so the tests
are correspondingly blunt: fills must cross the spread, and unusable quotes must
be refused rather than repaired.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from roth.backtest.fills import (
    Action,
    FillModel,
    Leg,
    Quote,
    RejectReason,
    Right,
    long_call,
    long_put,
    vertical_call_spread,
    vertical_put_spread,
)
from roth.config import CostModel

EXP = date(2024, 6, 21)
DAY = date(2024, 6, 3)

NO_COMMISSION = CostModel(commission_per_contract=0.0, extra_slippage_per_contract=0.0)


def chain(rows: list[dict]) -> pd.DataFrame:
    base = {
        "day": DAY,
        "expiration_date": EXP,
        "right": "C",
        "strike_dollars": 500.0,
        "bid": 1.00,
        "ask": 1.20,
        "bid_size": 10,
        "ask_size": 10,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


# -- the central rule: never fill at mid ------------------------------------


def test_buying_pays_the_ask():
    model = FillModel(costs=NO_COMMISSION)
    result = model.fill(chain([{}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert result.filled
    # 1.20 ask x 100, not the 1.10 mid.
    assert result.price_per_unit == pytest.approx(120.0)


def test_selling_receives_the_bid():
    model = FillModel(costs=NO_COMMISSION)
    result = model.fill(chain([{}]), long_call(500.0, EXP), Action.CLOSE, DAY)

    assert result.filled
    assert result.price_per_unit == pytest.approx(100.0)


def test_a_round_trip_at_an_unchanged_quote_loses_the_spread():
    """Buy and immediately sell with no market move: the loss is exactly the
    spread. If this ever comes out at zero, the model is filling at mid."""
    model = FillModel(costs=NO_COMMISSION)
    c = chain([{}])

    entry = model.fill(c, long_call(500.0, EXP), Action.OPEN, DAY)
    exit_ = model.fill(c, long_call(500.0, EXP), Action.CLOSE, DAY)

    assert exit_.price_per_unit - entry.price_per_unit == pytest.approx(-20.0)


def test_mid_mode_is_available_and_costs_nothing_round_trip():
    """The cost-sensitivity switch. Round-tripping at mid is free, which is
    exactly why it is not the default."""
    model = FillModel(costs=NO_COMMISSION, use_mid=True)
    c = chain([{}])

    entry = model.fill(c, long_call(500.0, EXP), Action.OPEN, DAY)
    exit_ = model.fill(c, long_call(500.0, EXP), Action.CLOSE, DAY)

    assert entry.price_per_unit == pytest.approx(110.0)
    assert exit_.price_per_unit - entry.price_per_unit == pytest.approx(0.0)


# -- rejections: never repair a bad quote -----------------------------------


def test_crossed_market_is_rejected():
    model = FillModel()
    result = model.fill(chain([{"bid": 1.50, "ask": 1.20}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.CROSSED


def test_absurd_spread_is_rejected():
    model = FillModel()
    # 0.05 / 2.05 is a spread of 2.00 on a mid of 1.05, far beyond 50%.
    result = model.fill(chain([{"bid": 0.05, "ask": 2.05}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.WIDE_SPREAD


def test_missing_quote_is_rejected():
    model = FillModel()
    result = model.fill(chain([{"bid": None}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.MISSING_QUOTE


def test_unlisted_contract_is_rejected():
    model = FillModel()
    result = model.fill(chain([{}]), long_call(999.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.NO_CONTRACT


def test_empty_chain_is_rejected():
    model = FillModel()
    result = model.fill(pd.DataFrame(), long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.NO_CHAIN


def test_zero_ask_is_rejected():
    model = FillModel()
    result = model.fill(chain([{"bid": 0.0, "ask": 0.0}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.ZERO_ASK


def test_a_rejected_fill_produces_no_price_at_all():
    """The point of rejecting: no synthetic price leaks through."""
    model = FillModel()
    result = model.fill(chain([{"bid": 1.50, "ask": 1.20}]), long_call(500.0, EXP), Action.OPEN, DAY)

    assert result.price_per_unit == 0.0
    assert result.legs == []
    assert result.reject_reason.describe()


def test_spread_threshold_is_configurable():
    tolerant = FillModel(costs=CostModel(max_spread_pct_of_mid=5.0))
    result = tolerant.fill(
        chain([{"bid": 0.05, "ask": 2.05}]), long_call(500.0, EXP), Action.OPEN, DAY
    )
    assert result.filled


# -- costs ------------------------------------------------------------------


def test_commission_is_per_contract_per_leg():
    model = FillModel(costs=CostModel(commission_per_contract=0.65))
    result = model.fill(chain([{}]), long_call(500.0, EXP), Action.OPEN, DAY, contracts=3)
    assert result.commission == pytest.approx(3 * 0.65)


def test_vertical_pays_commission_on_both_legs():
    c = chain([{"strike_dollars": 500.0}, {"strike_dollars": 505.0}])
    model = FillModel(costs=CostModel(commission_per_contract=0.65))
    result = model.fill(c, vertical_call_spread(500.0, 505.0, EXP), Action.OPEN, DAY, contracts=2)
    assert result.commission == pytest.approx(2 * 2 * 0.65)


def test_extra_slippage_always_hurts():
    slipped = FillModel(
        costs=CostModel(commission_per_contract=0.0, extra_slippage_per_contract=0.05)
    )
    entry = slipped.fill(chain([{}]), long_call(500.0, EXP), Action.OPEN, DAY)
    exit_ = slipped.fill(chain([{}]), long_call(500.0, EXP), Action.CLOSE, DAY)

    # Buy worse than the ask, sell worse than the bid.
    assert entry.price_per_unit == pytest.approx(125.0)
    assert exit_.price_per_unit == pytest.approx(95.0)


def test_slippage_hurts_in_mid_mode_too():
    slipped = FillModel(
        costs=CostModel(commission_per_contract=0.0, extra_slippage_per_contract=0.05),
        use_mid=True,
    )
    entry = slipped.fill(chain([{}]), long_call(500.0, EXP), Action.OPEN, DAY)
    assert entry.price_per_unit == pytest.approx(115.0)


def test_net_cash_signs_are_right():
    model = FillModel(costs=CostModel(commission_per_contract=0.65))
    entry = model.fill(chain([{}]), long_call(500.0, EXP), Action.OPEN, DAY)
    exit_ = model.fill(chain([{}]), long_call(500.0, EXP), Action.CLOSE, DAY)

    assert entry.net_cash < 0  # buying costs money
    assert exit_.net_cash > 0  # selling returns money


# -- structures -------------------------------------------------------------


def test_vertical_call_spread_is_a_debit():
    """Long the lower strike, short the higher. Opening pays ask on the long
    leg and receives bid on the short leg."""
    c = chain(
        [
            {"strike_dollars": 500.0, "bid": 5.00, "ask": 5.20},
            {"strike_dollars": 505.0, "bid": 2.00, "ask": 2.20},
        ]
    )
    model = FillModel(costs=NO_COMMISSION)
    result = model.fill(c, vertical_call_spread(500.0, 505.0, EXP), Action.OPEN, DAY)

    assert result.filled
    # 5.20 paid - 2.00 received = 3.20 debit.
    assert result.price_per_unit == pytest.approx(320.0)


def test_closing_a_vertical_crosses_the_other_way():
    c = chain(
        [
            {"strike_dollars": 500.0, "bid": 5.00, "ask": 5.20},
            {"strike_dollars": 505.0, "bid": 2.00, "ask": 2.20},
        ]
    )
    model = FillModel(costs=NO_COMMISSION)
    result = model.fill(c, vertical_call_spread(500.0, 505.0, EXP), Action.CLOSE, DAY)

    # Receive 5.00 on the long leg, pay 2.20 to buy back the short: 2.80.
    assert result.price_per_unit == pytest.approx(280.0)


def test_vertical_round_trip_loses_both_spreads():
    c = chain(
        [
            {"strike_dollars": 500.0, "bid": 5.00, "ask": 5.20},
            {"strike_dollars": 505.0, "bid": 2.00, "ask": 2.20},
        ]
    )
    model = FillModel(costs=NO_COMMISSION)
    legs = vertical_call_spread(500.0, 505.0, EXP)
    entry = model.fill(c, legs, Action.OPEN, DAY)
    exit_ = model.fill(c, legs, Action.CLOSE, DAY)

    assert exit_.price_per_unit - entry.price_per_unit == pytest.approx(-40.0)


def test_vertical_put_spread_structure():
    legs = vertical_put_spread(500.0, 495.0, EXP)
    assert legs[0].quantity == 1 and legs[0].strike == 500.0
    assert legs[1].quantity == -1 and legs[1].strike == 495.0
    assert all(leg.right is Right.PUT for leg in legs)


def test_a_vertical_is_all_or_nothing():
    """One unfillable leg rejects the whole structure. A half-filled vertical is
    a different position with different risk."""
    c = chain(
        [
            {"strike_dollars": 500.0, "bid": 5.00, "ask": 5.20},
            {"strike_dollars": 505.0, "bid": 3.00, "ask": 2.20},  # crossed
        ]
    )
    model = FillModel()
    result = model.fill(c, vertical_call_spread(500.0, 505.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.CROSSED


def test_long_put_uses_put_quotes():
    c = chain([{"right": "P", "bid": 2.00, "ask": 2.20}])
    model = FillModel(costs=NO_COMMISSION)
    result = model.fill(c, long_put(500.0, EXP), Action.OPEN, DAY)

    assert result.filled
    assert result.price_per_unit == pytest.approx(220.0)


def test_calls_and_puts_are_not_confused():
    """A chain holding only puts cannot fill a call order."""
    c = chain([{"right": "P"}])
    model = FillModel()
    result = model.fill(c, long_call(500.0, EXP), Action.OPEN, DAY)

    assert not result.filled
    assert result.reject_reason is RejectReason.NO_CONTRACT


# -- quote arithmetic -------------------------------------------------------


def test_quote_spread_metrics():
    q = Quote(bid=1.00, ask=1.20)
    assert q.mid == pytest.approx(1.10)
    assert q.spread == pytest.approx(0.20)
    assert q.spread_pct_of_mid == pytest.approx(0.20 / 1.10)


def test_zero_mid_spread_is_infinite_not_a_crash():
    assert Quote(bid=0.0, ask=0.0).spread_pct_of_mid == float("inf")


def test_side_selection_reverses_between_open_and_close():
    long_leg = Leg(Right.CALL, 500.0, EXP, +1)
    short_leg = Leg(Right.CALL, 505.0, EXP, -1)

    assert FillModel._side_for(long_leg, Action.OPEN) == "buy"
    assert FillModel._side_for(long_leg, Action.CLOSE) == "sell"
    assert FillModel._side_for(short_leg, Action.OPEN) == "sell"
    assert FillModel._side_for(short_leg, Action.CLOSE) == "buy"
