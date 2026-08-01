"""Backtest engine, fill model, and strategy interface."""

from roth.backtest.access import LookaheadError, TimeGate
from roth.backtest.engine import BacktestEngine, BacktestResult, ClosedTrade
from roth.backtest.fills import Action, FillModel, Leg, RejectReason, Right
from roth.backtest.selection import ContractSpec, select_contract
from roth.backtest.strategy import (
    CategoryRule,
    Context,
    Direction,
    ExitRules,
    PositionSizing,
    Rule,
    RuleResult,
    Strategy,
    ThresholdRule,
)

__all__ = [
    "Action",
    "BacktestEngine",
    "BacktestResult",
    "CategoryRule",
    "ClosedTrade",
    "ContractSpec",
    "Context",
    "Direction",
    "ExitRules",
    "FillModel",
    "Leg",
    "LookaheadError",
    "PositionSizing",
    "RejectReason",
    "Right",
    "Rule",
    "RuleResult",
    "Strategy",
    "ThresholdRule",
    "TimeGate",
    "select_contract",
]
