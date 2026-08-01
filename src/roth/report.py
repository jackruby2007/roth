"""Reporting.

One function takes a set of trades and produces every performance statistic the
harness reports. Text output to the terminal; CSV export for anything further.
No charting library, no dashboard.

Two numbers appear in every report and are not optional:

* **Unfillable signals.** How many entries or exits the fill model refused. A
  strategy that only works when bad quotes are ignored does not work, and this
  is where that shows.
* **Quarantined sessions.** How many days the quality layer excluded. A result
  computed over a dataset with holes in it needs that stated, not buried.

Risk statistics are computed from a daily equity curve rather than from
per-trade returns, because a strategy that is flat most of the time has very
different daily volatility from one that is always in the market, and per-trade
statistics hide that difference entirely.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Notional account size used to express returns as percentages. Position sizing
# in Phase 1 is a fixed contract count, so this scales the equity curve rather
# than driving position size.
DEFAULT_INITIAL_CAPITAL = 100_000.0


@dataclass
class PerformanceReport:
    strategy_name: str
    strategy_version: str
    symbols: tuple[str, ...]

    sample_count: int = 0
    first_trade: date | None = None
    last_trade: date | None = None

    win_rate: float = 0.0
    wins: int = 0
    losses: int = 0
    average_winner: float = 0.0
    average_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    profit_factor: float = 0.0

    gross_pnl: float = 0.0
    total_commissions: float = 0.0
    net_pnl: float = 0.0

    expectancy_r: float = 0.0
    expectancy_dollars: float = 0.0
    r_std: float = 0.0

    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    longest_losing_streak: int = 0

    average_holding_days: float = 0.0
    total_mfe: float = 0.0
    total_mae: float = 0.0

    # Context that qualifies every number above.
    #
    # Entry and exit rejections are counted separately on purpose. Lumping them
    # together and dividing by the candidate-signal count produces a badly
    # misleading "rejection rate": exits are attempted repeatedly over a
    # position's life, so a handful of stuck positions can inflate the combined
    # figure to look like most entries were refused.
    unfillable_signals: int = 0
    unfillable_entries: int = 0
    unfillable_exits: int = 0
    candidate_signals: int = 0
    quarantined_sessions: int = 0
    synthetic_data: bool = False
    used_mid_fills: bool = False

    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    exit_reason_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("equity_curve", None)
        return d


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def build_equity_curve(
    trades: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> pd.DataFrame:
    """Daily equity, with each trade's P/L applied on its exit day.

    Sessions between trades carry zero return rather than being dropped. That
    matters: a strategy holding one position a week has genuinely low daily
    volatility, and compressing the flat days away would inflate its Sharpe.
    """
    if trades.empty:
        return pd.DataFrame(columns=["day", "pnl", "equity", "drawdown"])

    by_day = (
        trades.groupby("exit_day", as_index=False)["net_pnl"]
        .sum()
        .rename(columns={"exit_day": "day", "net_pnl": "pnl"})
        .sort_values("day")
    )

    span = pd.date_range(
        pd.Timestamp(trades["entry_day"].min()),
        pd.Timestamp(trades["exit_day"].max()),
        freq="B",
    )
    curve = pd.DataFrame({"day": [d.date() for d in span]})
    curve = curve.merge(by_day, on="day", how="left")
    curve["pnl"] = curve["pnl"].fillna(0.0)

    curve["equity"] = initial_capital + curve["pnl"].cumsum()
    running_max = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] - running_max

    return curve


def _annualised_risk_stats(curve: pd.DataFrame) -> dict[str, float]:
    if curve.empty or len(curve) < 2:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0}

    returns = curve["equity"].pct_change().dropna()
    if returns.empty or returns.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = returns.mean() / returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)

    downside = returns[returns < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        sortino = 0.0
    else:
        sortino = returns.mean() / downside.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)

    start_equity = curve["equity"].iloc[0]
    end_equity = curve["equity"].iloc[-1]
    years = len(curve) / TRADING_DAYS_PER_YEAR

    max_dd = abs(curve["drawdown"].min())
    if max_dd > 0 and years > 0 and start_equity > 0:
        total_return = end_equity / start_equity
        annualised = total_return ** (1 / years) - 1 if total_return > 0 else -1.0
        calmar = annualised / (max_dd / start_equity)
    else:
        calmar = 0.0

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
    }


def longest_losing_streak(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    losing = (trades.sort_values("exit_day")["net_pnl"] <= 0).to_numpy()
    best = run = 0
    for is_loss in losing:
        run = run + 1 if is_loss else 0
        best = max(best, run)
    return int(best)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def build_report(
    trades: pd.DataFrame,
    strategy_name: str = "",
    strategy_version: str = "",
    symbols: tuple[str, ...] = (),
    unfillable_signals: int = 0,
    unfillable_entries: int = 0,
    unfillable_exits: int = 0,
    candidate_signals: int = 0,
    quarantined_sessions: int = 0,
    synthetic_data: bool = False,
    used_mid_fills: bool = False,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> PerformanceReport:
    """Every performance statistic, from a set of trades."""
    report = PerformanceReport(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        symbols=symbols,
        unfillable_signals=unfillable_signals,
        unfillable_entries=unfillable_entries,
        unfillable_exits=unfillable_exits,
        candidate_signals=candidate_signals,
        quarantined_sessions=quarantined_sessions,
        synthetic_data=synthetic_data,
        used_mid_fills=used_mid_fills,
    )

    if trades.empty:
        return report

    report.sample_count = len(trades)
    report.first_trade = trades["entry_day"].min()
    report.last_trade = trades["exit_day"].max()

    winners = trades[trades["net_pnl"] > 0]
    losers = trades[trades["net_pnl"] <= 0]

    report.wins = len(winners)
    report.losses = len(losers)
    report.win_rate = len(winners) / len(trades)
    report.average_winner = float(winners["net_pnl"].mean()) if len(winners) else 0.0
    report.average_loser = float(losers["net_pnl"].mean()) if len(losers) else 0.0
    report.largest_winner = float(trades["net_pnl"].max())
    report.largest_loser = float(trades["net_pnl"].min())

    gross_wins = float(winners["net_pnl"].sum()) if len(winners) else 0.0
    gross_losses = abs(float(losers["net_pnl"].sum())) if len(losers) else 0.0
    report.profit_factor = (
        gross_wins / gross_losses if gross_losses > 0 else (math.inf if gross_wins > 0 else 0.0)
    )

    report.gross_pnl = float(trades["gross_pnl"].sum())
    report.total_commissions = float(trades["commissions"].sum())
    report.net_pnl = float(trades["net_pnl"].sum())

    report.expectancy_r = float(trades["r_multiple"].mean())
    report.expectancy_dollars = float(trades["net_pnl"].mean())
    report.r_std = float(trades["r_multiple"].std(ddof=0)) if len(trades) > 1 else 0.0

    report.average_holding_days = float(trades["holding_days"].mean())
    report.total_mfe = float(trades["max_favorable_excursion"].sum())
    report.total_mae = float(trades["max_adverse_excursion"].sum())
    report.longest_losing_streak = longest_losing_streak(trades)

    curve = build_equity_curve(trades, initial_capital)
    report.equity_curve = curve
    report.max_drawdown = float(abs(curve["drawdown"].min())) if not curve.empty else 0.0
    report.max_drawdown_pct = report.max_drawdown / initial_capital * 100 if initial_capital else 0.0
    report.__dict__.update(_annualised_risk_stats(curve))

    report.exit_reason_counts = dict(trades["exit_reason"].value_counts())

    return report


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _fmt(value: float, kind: str = "money") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if kind == "money":
        return f"{value:+,.2f}"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "ratio":
        return "inf" if math.isinf(value) else f"{value:.2f}"
    return str(value)


def render_text(report: PerformanceReport, width: int = 74) -> str:
    """The full report as plain text."""
    line = "=" * width
    thin = "-" * width
    out: list[str] = []

    out.append(line)
    out.append(f" {report.strategy_name} v{report.strategy_version}")
    if report.symbols:
        out.append(f" {', '.join(report.symbols)}")
    out.append(line)

    if report.synthetic_data:
        out.append("")
        out.append(" !! SYNTHETIC DATA !!")
        out.append(" These numbers were computed from a generated price series.")
        out.append(" They demonstrate that the pipeline runs. They say nothing")
        out.append(" about whether this strategy has an edge.")
        out.append("")

    if report.used_mid_fills:
        out.append("")
        out.append(" !! MID-PRICE FILLS !!")
        out.append(" Fills were priced at the midpoint, which is not achievable.")
        out.append(" For cost-sensitivity comparison only.")
        out.append("")

    if report.sample_count == 0:
        out.append("")
        out.append(" No trades.")
        out.append("")
        out.append(f" Candidate signals   {report.candidate_signals:>12,}")
        out.append(f" Unfillable signals  {report.unfillable_signals:>12,}")
        out.append(line)
        return "\n".join(out)

    def row(label: str, value: str) -> str:
        return f" {label:<28}{value:>20}"

    out.append("")
    out.append(" SAMPLE")
    out.append(thin)
    out.append(row("Trades", f"{report.sample_count:,}"))
    out.append(row("Date range", f"{report.first_trade} to {report.last_trade}"))
    out.append(row("Average holding (sessions)", f"{report.average_holding_days:.1f}"))

    out.append("")
    out.append(" OUTCOME")
    out.append(thin)
    out.append(row("Win rate", _fmt(report.win_rate, "pct")))
    out.append(row("Wins / losses", f"{report.wins} / {report.losses}"))
    out.append(row("Average winner", _fmt(report.average_winner)))
    out.append(row("Average loser", _fmt(report.average_loser)))
    out.append(row("Largest winner", _fmt(report.largest_winner)))
    out.append(row("Largest loser", _fmt(report.largest_loser)))
    out.append(row("Profit factor", _fmt(report.profit_factor, "ratio")))

    out.append("")
    out.append(" PROFIT AND LOSS")
    out.append(thin)
    out.append(row("Gross P/L", _fmt(report.gross_pnl)))
    out.append(row("Commissions", _fmt(-abs(report.total_commissions))))
    out.append(row("Net P/L", _fmt(report.net_pnl)))
    out.append(row("Expectancy per trade", _fmt(report.expectancy_dollars)))
    out.append(row("Expectancy in R", f"{report.expectancy_r:+.3f}"))
    out.append(row("R standard deviation", f"{report.r_std:.3f}"))

    out.append("")
    out.append(" RISK")
    out.append(thin)
    out.append(row("Sharpe", _fmt(report.sharpe, "ratio")))
    out.append(row("Sortino", _fmt(report.sortino, "ratio")))
    out.append(row("Calmar", _fmt(report.calmar, "ratio")))
    out.append(row("Max drawdown", _fmt(-abs(report.max_drawdown))))
    out.append(row("Max drawdown %", f"{report.max_drawdown_pct:.2f}%"))
    out.append(row("Longest losing streak", f"{report.longest_losing_streak}"))
    out.append(row("Total MFE", _fmt(report.total_mfe)))
    out.append(row("Total MAE", _fmt(report.total_mae)))

    out.append("")
    out.append(" EXITS")
    out.append(thin)
    for reason, count in sorted(
        report.exit_reason_counts.items(), key=lambda kv: -kv[1]
    ):
        out.append(row(reason, f"{count:,}"))

    out.append("")
    out.append(" DATA QUALITY")
    out.append(thin)
    out.append(row("Candidate signals", f"{report.candidate_signals:,}"))
    out.append(row("Entries refused", f"{report.unfillable_entries:,}"))
    if report.candidate_signals:
        pct = report.unfillable_entries / report.candidate_signals
        out.append(row("  as % of candidates", _fmt(pct, "pct")))
    out.append(row("Exit attempts refused", f"{report.unfillable_exits:,}"))
    settled = report.exit_reason_counts.get("expired_settled_at_intrinsic", 0)
    if settled:
        out.append(row("Settled at intrinsic", f"{int(settled):,}"))
        if report.sample_count:
            out.append(
                row("  as % of trades", _fmt(settled / report.sample_count, "pct"))
            )
    out.append(row("Quarantined sessions", f"{report.quarantined_sessions:,}"))

    out.append("")
    out.append(" EQUITY CURVE")
    out.append(thin)
    out.append(_sparkline(report.equity_curve))

    out.append(line)
    return "\n".join(out)


def _sparkline(curve: pd.DataFrame, width: int = 58, height: int = 9) -> str:
    """A coarse ASCII equity curve.

    Deliberately crude. It is enough to see the shape without pulling in a
    charting library; anything more detailed belongs in the CSV export.
    """
    if curve.empty or len(curve) < 2:
        return " (not enough data)"

    equity = curve["equity"].to_numpy()
    idx = np.linspace(0, len(equity) - 1, min(width, len(equity))).astype(int)
    sampled = equity[idx]

    lo, hi = sampled.min(), sampled.max()
    if hi == lo:
        return " (flat)"

    rows = [[" "] * len(sampled) for _ in range(height)]
    for x, value in enumerate(sampled):
        y = int((value - lo) / (hi - lo) * (height - 1))
        rows[height - 1 - y][x] = "*"

    lines = []
    for i, r in enumerate(rows):
        if i == 0:
            label = f"{hi:>12,.0f} "
        elif i == height - 1:
            label = f"{lo:>12,.0f} "
        else:
            label = " " * 13
        lines.append(label + "".join(r))

    lines.append(" " * 13 + f"{curve['day'].iloc[0]}  ->  {curve['day'].iloc[-1]}")
    return "\n".join(lines)


def export_csv(report: PerformanceReport, trades: pd.DataFrame, directory) -> dict[str, str]:
    """Write trades, the equity curve, and the summary to CSV."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    stem = f"{report.strategy_name}_{report.strategy_version}"
    paths = {}

    trades_path = directory / f"{stem}_trades.csv"
    trades.to_csv(trades_path, index=False)
    paths["trades"] = str(trades_path)

    if not report.equity_curve.empty:
        curve_path = directory / f"{stem}_equity.csv"
        report.equity_curve.to_csv(curve_path, index=False)
        paths["equity"] = str(curve_path)

    summary_path = directory / f"{stem}_summary.csv"
    pd.DataFrame([report.to_dict()]).to_csv(summary_path, index=False)
    paths["summary"] = str(summary_path)

    return paths
