"""Trade journal.

Every simulated trade is written to parquet, together with the full rule
evaluation for every candidate signal -- including the signals that failed and
the reason each rule rejected them.

The `signal_id` column is the join key between the two tables. Given a trade,
the rule evaluations explain exactly why it was taken; given a session with no
trade, they explain exactly why not. That is the trade-explanation feature, and
it costs nothing extra because the engine already records it.

Unfillable signals get their own table. They are not a footnote: a strategy that
only works when bad quotes are ignored does not work, so the rejections are
stored alongside the trades rather than summarised away.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from roth.backtest.engine import BacktestResult
from roth.paths import RULE_EVALS, TRADES, ensure_dirs
from roth.storage import write_derived

TRADES_TABLE = "trades"
RULE_EVALS_TABLE = "rule_evaluations"
UNFILLABLE_TABLE = "unfillable_signals"

# Columns the journal guarantees, in the order a reader expects them.
JOURNAL_COLUMNS: tuple[str, ...] = (
    "run_id",
    "entry_day",
    "exit_day",
    "signal_day",
    "symbol",
    "strategy_name",
    "strategy_version",
    "direction",
    "structure_type",
    "contract_ids",
    "strikes",
    "expiration",
    "contracts",
    "entry_price",
    "exit_price",
    "entry_bid",
    "entry_ask",
    "exit_bid",
    "exit_ask",
    "capital_at_risk",
    "gross_pnl",
    "commissions",
    "net_pnl",
    "return_pct",
    "r_multiple",
    "holding_days",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "exit_reason",
    "entry_delta",
    "entry_iv",
    "entry_underlying",
    "trend_direction",
    "trend_strength",
    "vol_bucket",
    "day_type",
    "signal_id",
)


def make_run_id(result: BacktestResult) -> str:
    """Identifier tying a journal row back to the run that produced it."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    fills = "mid" if result.used_mid_fills else "nbbo"
    origin = "synth" if result.synthetic_data else "real"
    return f"{result.strategy_name}_{result.strategy_version}_{fills}_{origin}_{stamp}"


def build_journal(result: BacktestResult, run_id: str | None = None) -> pd.DataFrame:
    """The trade journal for one run, with columns in guaranteed order."""
    df = result.trades_frame()
    if df.empty:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    df = df.copy()
    df["run_id"] = run_id or make_run_id(result)

    missing = [c for c in JOURNAL_COLUMNS if c not in df.columns]
    for col in missing:
        df[col] = None

    return df[list(JOURNAL_COLUMNS)].sort_values(["entry_day", "symbol"]).reset_index(drop=True)


def write_journal(result: BacktestResult, run_id: str | None = None) -> dict[str, int]:
    """Persist trades, rule evaluations and unfillable signals for one run."""
    ensure_dirs()
    run_id = run_id or make_run_id(result)

    journal = build_journal(result, run_id)
    write_derived(journal, TRADES, f"{TRADES_TABLE}_{run_id}")

    evals = result.rule_evaluations_frame()
    if not evals.empty:
        evals = evals.copy()
        evals["run_id"] = run_id
    write_derived(evals, RULE_EVALS, f"{RULE_EVALS_TABLE}_{run_id}")

    unfillable = pd.DataFrame([u.__dict__ for u in result.unfillable])
    if not unfillable.empty:
        unfillable["run_id"] = run_id
    write_derived(unfillable, TRADES, f"{UNFILLABLE_TABLE}_{run_id}")

    return {
        "trades": len(journal),
        "rule_evaluations": len(evals),
        "unfillable": len(unfillable),
    }


def explain_signal(result: BacktestResult, signal_id: int) -> str:
    """Human-readable account of why one signal was or was not taken."""
    if signal_id < 0 or signal_id >= len(result.signals):
        return f"No signal with id {signal_id}."

    decision = result.signals[signal_id]
    lines = [decision.explain()]

    trade = next((t for t in result.trades if t.signal_id == signal_id), None)
    if trade is not None:
        lines.append(
            f"  -> traded {trade.contract_ids} entered {trade.entry_day} at "
            f"{trade.entry_price:,.2f}, exited {trade.exit_day} at "
            f"{trade.exit_price:,.2f} ({trade.exit_reason}), net {trade.net_pnl:+,.2f}"
        )

    rejected = [u for u in result.unfillable if u.signal_id == signal_id]
    for u in rejected:
        lines.append(f"  -> UNFILLABLE on {u.day} ({u.action}): {u.detail}")

    if trade is None and not rejected and decision.passed:
        lines.append("  -> signal passed but no position was opened (one already held)")

    return "\n".join(lines)


def load_latest_journal() -> pd.DataFrame:
    """The most recently written trade journal, or an empty frame."""
    files = sorted(TRADES.glob(f"{TRADES_TABLE}_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])
