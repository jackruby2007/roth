"""Wiring between storage and the engine.

Loads features and chains, drops quarantined sessions, builds the gate, and runs
the strategy. Keeps the engine itself free of any knowledge about where data
lives.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from roth.backtest.access import TimeGate
from roth.backtest.engine import BacktestEngine, BacktestResult
from roth.backtest.strategy import Strategy
from roth.config import COSTS, CostModel
from roth.features.build import load_features
from roth.paths import RAW_OPTION_EOD
from roth.quality import load_quarantine
from roth.storage import read_dataset


def build_gate(
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    exclude_quarantined: bool = True,
) -> tuple[TimeGate | None, int]:
    """Build a gate for one symbol. Returns the gate and sessions excluded."""
    features = load_features(symbol)
    if features.empty:
        return None, 0

    if start is not None:
        features = features[features["day"] >= start]
    if end is not None:
        features = features[features["day"] <= end]
    if features.empty:
        return None, 0

    excluded = 0
    if exclude_quarantined:
        bad = {d for s, d in load_quarantine() if s == symbol}
        if bad:
            before = len(features)
            features = features[~features["day"].isin(bad)]
            excluded = before - len(features)

    if features.empty:
        return None, excluded

    chains = read_dataset(
        RAW_OPTION_EOD,
        symbols=[symbol],
        start=features["day"].min(),
        end=features["day"].max(),
    )

    return TimeGate(features=features, chains=chains, symbol=symbol), excluded


def run_backtest(
    strategy: Strategy,
    symbols: tuple[str, ...],
    start: date | None = None,
    end: date | None = None,
    costs: CostModel = COSTS,
    use_mid_fills: bool = False,
    exclude_quarantined: bool = True,
) -> BacktestResult:
    """Run one strategy across one or more symbols.

    Each symbol is walked independently; results are merged into a single
    record so that reporting sees one coherent set of trades.
    """
    from roth.data.synth import is_synthetic

    merged = BacktestResult(
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        symbols=tuple(symbols),
        start=None,
        end=None,
    )
    merged.used_mid_fills = use_mid_fills
    merged.synthetic_data = is_synthetic()

    for symbol in symbols:
        gate, excluded = build_gate(symbol, start, end, exclude_quarantined)
        merged.quarantined_sessions_excluded += excluded
        if gate is None:
            continue

        engine = BacktestEngine(strategy, costs=costs, use_mid_fills=use_mid_fills)
        single = engine.run(gate)

        offset = len(merged.signals)
        merged.signals.extend(single.signals)
        for t in single.trades:
            t.signal_id += offset
            merged.trades.append(t)
        for u in single.unfillable:
            u.signal_id += offset
            merged.unfillable.append(u)

        merged.sessions_processed += single.sessions_processed
        for attr in ("start", "end"):
            value = getattr(single, attr)
            current = getattr(merged, attr)
            if value is None:
                continue
            if current is None:
                setattr(merged, attr, value)
            elif attr == "start":
                setattr(merged, attr, min(current, value))
            else:
                setattr(merged, attr, max(current, value))

    merged.trades.sort(key=lambda t: (t.entry_day, t.symbol))
    return merged


def trades_to_frame(result: BacktestResult) -> pd.DataFrame:
    return result.trades_frame()
