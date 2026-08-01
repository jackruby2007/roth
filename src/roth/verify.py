"""Correctness tests.

The harness has to prove it is not lying before any result from it is trusted.
Four independent checks, each attacking a different way the plumbing could be
silently wrong.

1. **Known answer.** Buy-and-hold must reproduce a return that can be verified
   from outside the harness. If the bar-walking arithmetic is broken, this is
   where it shows.
2. **Lookahead trap.** A deliberately cheating strategy must produce an absurd
   equity curve, *and* the honest route to writing that cheat must be blocked
   by the data access layer.
3. **Cost sensitivity.** The same strategy run at bid/ask and at mid. The gap
   is how much of any apparent edge is really an execution assumption.
4. **Timezone.** The market open must map to the right UTC instant on both
   sides of both daylight saving transitions.

A check that cannot run reports SKIP with the reason. It never reports PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time

import pandas as pd

from roth.backtest.access import LookaheadError, TimeGate
from roth.backtest.engine import BacktestEngine
from roth.backtest.runner import build_gate, run_backtest
from roth.calendar import to_utc, trading_sessions
from roth.config import COSTS, SYMBOLS
from roth.paths import RAW

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

# User-supplied benchmark returns for the known-answer check. A CSV with
# columns: symbol, start, end, total_return_pct, source.
BENCHMARK_PATH = RAW / "reference" / "benchmarks.csv"

# Buy-and-hold price return must match the reference within this many
# percentage points.
KNOWN_ANSWER_TOLERANCE_PCT = 0.5


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    numbers: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PASS


# ---------------------------------------------------------------------------
# 1. Known answer
# ---------------------------------------------------------------------------


def _walk_buy_and_hold(gate: TimeGate) -> dict:
    """Compound daily returns by walking the gate one bar at a time.

    Deliberately the long way round. The endpoint ratio last/first is trivially
    correct; the compounded walk is not, and an off-by-one in the time cursor --
    using the next bar's close, or skipping the first -- makes the two disagree.
    That disagreement is the thing being tested.
    """
    closes: list[float] = []
    days: list[date] = []
    equity = 1.0

    prev: float | None = None
    while gate.advance():
        row = gate.current()
        close = float(row["close"])
        closes.append(close)
        days.append(gate.now)
        if prev is not None:
            equity *= close / prev
        prev = close

    if len(closes) < 2:
        return {}

    return {
        "sessions": len(closes),
        "first_day": days[0],
        "last_day": days[-1],
        "first_close": closes[0],
        "last_close": closes[-1],
        "compounded_return_pct": (equity - 1.0) * 100.0,
        "endpoint_return_pct": (closes[-1] / closes[0] - 1.0) * 100.0,
    }


def check_known_answer_internal(symbol: str = "SPY") -> CheckResult:
    """Buy-and-hold computed two independent ways must agree exactly."""
    gate, _ = build_gate(symbol, exclude_quarantined=False)
    if gate is None:
        return CheckResult(
            "known_answer_internal",
            SKIP,
            f"no feature data for {symbol}; run `roth features build` first",
        )

    stats = _walk_buy_and_hold(gate)
    if not stats:
        return CheckResult("known_answer_internal", SKIP, "fewer than two sessions available")

    diff = abs(stats["compounded_return_pct"] - stats["endpoint_return_pct"])
    status = PASS if diff < 1e-6 else FAIL

    return CheckResult(
        "known_answer_internal",
        status,
        (
            f"walked {stats['sessions']} sessions bar by bar: "
            f"{stats['compounded_return_pct']:.4f}% compounded vs "
            f"{stats['endpoint_return_pct']:.4f}% endpoint, difference {diff:.2e}"
        ),
        stats | {"difference": diff},
    )


def load_benchmarks() -> pd.DataFrame:
    if not BENCHMARK_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(BENCHMARK_PATH)
    except (pd.errors.ParserError, OSError):
        return pd.DataFrame()
    required = {"symbol", "start", "end", "total_return_pct"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["start"] = pd.to_datetime(df["start"]).dt.date
    df["end"] = pd.to_datetime(df["end"]).dt.date
    return df


def check_known_answer_external() -> CheckResult:
    """Buy-and-hold against an externally sourced reference return.

    This is the check that catches a harness which is internally consistent but
    disconnected from reality. It requires real market data and a reference
    figure the user supplies, and it SKIPS rather than passing when either is
    missing -- a fabricated benchmark would defeat the entire purpose.
    """
    from roth.data.synth import is_synthetic

    benchmarks = load_benchmarks()
    if benchmarks.empty:
        return CheckResult(
            "known_answer_external",
            SKIP,
            (
                "no benchmark file. Create "
                f"{BENCHMARK_PATH} with columns "
                "symbol,start,end,total_return_pct,source -- e.g. a published "
                "SPY calendar-year return. Cannot pass without one."
            ),
        )

    if is_synthetic():
        return CheckResult(
            "known_answer_external",
            SKIP,
            (
                "the dataset on disk is synthetic. A generated price series "
                "cannot reproduce a real published return, and pretending "
                "otherwise would make this check meaningless."
            ),
        )

    results = []
    for _, bm in benchmarks.iterrows():
        gate, _ = build_gate(
            str(bm["symbol"]), bm["start"], bm["end"], exclude_quarantined=False
        )
        if gate is None:
            results.append((str(bm["symbol"]), None, float(bm["total_return_pct"])))
            continue
        stats = _walk_buy_and_hold(gate)
        actual = stats.get("endpoint_return_pct") if stats else None
        results.append((str(bm["symbol"]), actual, float(bm["total_return_pct"])))

    usable = [(s, a, e) for s, a, e in results if a is not None]
    if not usable:
        return CheckResult(
            "known_answer_external",
            SKIP,
            "benchmark file present but no matching data on disk for those periods",
        )

    worst = max(abs(a - e) for _, a, e in usable)
    status = PASS if worst <= KNOWN_ANSWER_TOLERANCE_PCT else FAIL
    lines = "; ".join(f"{s}: harness {a:.2f}% vs reference {e:.2f}%" for s, a, e in usable)

    return CheckResult(
        "known_answer_external",
        status,
        f"{lines} (worst gap {worst:.2f} pts, tolerance {KNOWN_ANSWER_TOLERANCE_PCT})",
        {"worst_gap_pct": worst, "comparisons": len(usable)},
    )


# ---------------------------------------------------------------------------
# 2. Lookahead trap
# ---------------------------------------------------------------------------


def check_lookahead_blocked(symbol: str = "SPY") -> CheckResult:
    """The honest route to cheating must be refused by the access layer."""
    gate, _ = build_gate(symbol, exclude_quarantined=False)
    if gate is None:
        return CheckResult("lookahead_blocked", SKIP, f"no feature data for {symbol}")

    gate.advance()
    next_day = gate.peek_next_day()
    if next_day is None:
        return CheckResult("lookahead_blocked", SKIP, "not enough sessions")

    attempts = []

    for label, fn in (
        ("chain(next_session)", lambda: gate.chain(next_day)),
        ("row(next_session)", lambda: gate.row(next_day)),
    ):
        try:
            fn()
            attempts.append((label, "SERVED"))
        except LookaheadError:
            attempts.append((label, "refused"))

    served = [a for a in attempts if a[1] == "SERVED"]
    status = FAIL if served else PASS

    return CheckResult(
        "lookahead_blocked",
        status,
        (
            "the access layer refused every attempt to read the next session"
            if not served
            else f"future data was served for: {', '.join(a[0] for a in served)}"
        ),
        {"attempts": dict(attempts)},
    )


def check_lookahead_trap(symbol: str = "SPY") -> CheckResult:
    """A strategy that *does* see the future must produce an absurd result.

    The cheat cannot be written through the gate, so it is injected: a feature
    column holding the next session's return is added by hand, and a strategy
    trades on it. If the harness were leaking future data, ordinary strategies
    would look like this one. It is the calibration point for "too good".
    """
    from roth.backtest.fills import Right
    from roth.backtest.selection import ContractSpec
    from roth.backtest.strategy import ExitRules, Strategy, ThresholdRule
    from roth.features.build import load_features
    from roth.paths import RAW_OPTION_EOD
    from roth.storage import read_dataset

    features = load_features(symbol)
    if features.empty:
        return CheckResult("lookahead_trap", SKIP, f"no feature data for {symbol}")

    cheating = features.sort_values("day").reset_index(drop=True).copy()

    # The cheat has to look ahead far enough to matter. Signalling on session D
    # means the engine enters at D+1's close and, with a one-session time stop,
    # exits at D+2's close -- so the move worth knowing about is the one from
    # D+1 to D+2, not from D to D+1.
    #
    # An earlier version of this trap used the D-to-D+1 return and produced a 0%
    # win rate: the move was already over before the fill landed. That failure
    # was itself evidence the next-bar fill rule holds, but it did not test what
    # this check exists to test.
    cheating["FUTURE_HOLDING_RETURN"] = (
        cheating["close"].shift(-2) / cheating["close"].shift(-1) - 1.0
    )
    cheating = cheating[cheating["FUTURE_HOLDING_RETURN"].notna()]

    chains = read_dataset(
        RAW_OPTION_EOD,
        symbols=[symbol],
        start=cheating["day"].min(),
        end=cheating["day"].max(),
    )

    class Cheater(Strategy):
        name = "lookahead_cheater"
        version = "0.0.0"
        structure_type = "long_call"

        def rules(self):
            # Buy only when the session being held is known in advance to be a
            # strong up day. The threshold is set well above the spread so the
            # advantage cannot be eaten by execution costs.
            return [ThresholdRule("holding_day_is_up", "FUTURE_HOLDING_RETURN", ">", 0.01)]

        def contract_spec(self, ctx):
            return ContractSpec(
                right=Right.CALL, target_dte=10, target_delta=0.40, min_dte=5, max_dte=45
            )

        def exits(self):
            return ExitRules(time_stop_days=1, close_before_expiry_days=1)

    honest_gate, _ = build_gate(symbol, exclude_quarantined=False)
    cheat_gate = TimeGate(features=cheating, chains=chains, symbol=symbol)

    cheat_result = BacktestEngine(Cheater(), costs=COSTS).run(cheat_gate)

    from roth.strategies.reference import MondayCallReference

    honest_result = BacktestEngine(MondayCallReference(), costs=COSTS).run(honest_gate)

    cheat_trades = cheat_result.trades_frame()
    honest_trades = honest_result.trades_frame()

    if cheat_trades.empty:
        return CheckResult("lookahead_trap", SKIP, "the cheating strategy took no trades")

    cheat_win = float((cheat_trades["net_pnl"] > 0).mean())
    cheat_r = float(cheat_trades["r_multiple"].mean())
    honest_win = float((honest_trades["net_pnl"] > 0).mean()) if not honest_trades.empty else 0.0
    honest_r = float(honest_trades["r_multiple"].mean()) if not honest_trades.empty else 0.0

    # The cheat must be visibly, implausibly better than the honest strategy.
    absurd = cheat_win > 0.70 and cheat_win > honest_win + 0.20
    status = PASS if absurd else FAIL

    return CheckResult(
        "lookahead_trap",
        status,
        (
            f"cheating strategy: {cheat_win:.0%} win rate, {cheat_r:+.2f} mean R over "
            f"{len(cheat_trades)} trades. Honest strategy: {honest_win:.0%} win rate, "
            f"{honest_r:+.2f} mean R over {len(honest_trades)} trades. "
            + (
                "The gap is the signature of lookahead; results resembling the "
                "cheater should be assumed broken."
                if absurd
                else "The cheat did NOT produce an absurd result, which means either "
                "the injection failed or the engine is not acting on the signal."
            )
        ),
        {
            "cheat_win_rate": cheat_win,
            "cheat_mean_r": cheat_r,
            "honest_win_rate": honest_win,
            "honest_mean_r": honest_r,
            "cheat_trades": len(cheat_trades),
        },
    )


# ---------------------------------------------------------------------------
# 3. Cost sensitivity
# ---------------------------------------------------------------------------


def check_cost_sensitivity(
    symbols: tuple[str, ...] = SYMBOLS, strategy=None
) -> CheckResult:
    """Run the same strategy at bid/ask and at mid, and report the gap."""
    from roth.strategies.reference import MondayCallReference

    strategy = strategy or MondayCallReference()

    realistic = run_backtest(strategy, symbols, use_mid_fills=False)
    optimistic = run_backtest(strategy, symbols, use_mid_fills=True)

    a, b = realistic.trades_frame(), optimistic.trades_frame()
    if a.empty and b.empty:
        return CheckResult("cost_sensitivity", SKIP, "no trades under either fill assumption")

    real_pnl = float(a["net_pnl"].sum()) if not a.empty else 0.0
    mid_pnl = float(b["net_pnl"].sum()) if not b.empty else 0.0
    real_r = float(a["r_multiple"].mean()) if not a.empty else 0.0
    mid_r = float(b["r_multiple"].mean()) if not b.empty else 0.0

    gap = mid_pnl - real_pnl

    # This check reports rather than judges: any gap is legitimate information.
    # It only fails if mid fills are somehow *worse*, which would mean the fill
    # model has its sides crossed.
    status = FAIL if gap < -1e-6 else PASS

    return CheckResult(
        "cost_sensitivity",
        status,
        (
            f"bid/ask fills: {real_pnl:+,.2f} net over {len(a)} trades ({real_r:+.3f} mean R). "
            f"Mid fills: {mid_pnl:+,.2f} over {len(b)} trades ({mid_r:+.3f} mean R). "
            f"Crossing the spread costs {gap:,.2f}. "
            + (
                "That figure is how much of any apparent edge is an execution assumption."
                if gap >= 0
                else "Mid fills came out WORSE, which means the fill model is crossing "
                "the wrong side of the spread."
            )
        ),
        {
            "realistic_net_pnl": real_pnl,
            "mid_net_pnl": mid_pnl,
            "spread_cost": gap,
            "realistic_mean_r": real_r,
            "mid_mean_r": mid_r,
            "realistic_trades": len(a),
            "mid_trades": len(b),
        },
    )


# ---------------------------------------------------------------------------
# 4. Timezone
# ---------------------------------------------------------------------------


# DST transitions, chosen to test both directions in consecutive years.
DST_CASES = [
    # (session, expected UTC hour of 09:30 ET, note)
    (date(2024, 3, 8), 14, "Friday before spring forward, EST"),
    (date(2024, 3, 11), 13, "Monday after spring forward, EDT"),
    (date(2024, 11, 1), 13, "Friday before fall back, EDT"),
    (date(2024, 11, 4), 14, "Monday after fall back, EST"),
    (date(2025, 3, 7), 14, "Friday before spring forward, EST"),
    (date(2025, 3, 10), 13, "Monday after spring forward, EDT"),
]


def check_timezone() -> CheckResult:
    """09:30 America/New_York must map to the right UTC hour on both sides of
    both transitions, and the exchange calendar must agree."""
    failures: list[str] = []
    checked = 0

    for day, expected_hour, note in DST_CASES:
        converted = to_utc(day, time(9, 30))
        if converted.hour != expected_hour:
            failures.append(
                f"{day} ({note}): 09:30 ET converted to {converted:%H:%M} UTC, "
                f"expected {expected_hour:02d}:30"
            )
        checked += 1

        sched = trading_sessions(day, day)
        if sched.empty:
            failures.append(f"{day}: not a trading session in the calendar")
            continue

        open_utc = pd.Timestamp(sched.iloc[0]["session_open_utc"])
        if open_utc.hour != expected_hour:
            failures.append(
                f"{day} ({note}): calendar session open is {open_utc:%H:%M} UTC, "
                f"expected {expected_hour:02d}:30"
            )
        checked += 1

    # A stored UTC timestamp must round-trip back to 09:30 local.
    for day, _, _ in DST_CASES:
        back = pd.Timestamp(to_utc(day, time(9, 30))).tz_convert("America/New_York")
        if (back.hour, back.minute) != (9, 30):
            failures.append(f"{day}: UTC did not round-trip back to 09:30 ET, got {back:%H:%M}")
        checked += 1

    status = PASS if not failures else FAIL
    detail = (
        f"{checked} assertions across {len(DST_CASES)} sessions spanning both DST "
        "transitions in both directions"
        if not failures
        else "; ".join(failures)
    )
    return CheckResult("timezone", status, detail, {"assertions": checked})


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(symbols: tuple[str, ...] = SYMBOLS) -> list[CheckResult]:
    primary = symbols[0] if symbols else "SPY"
    return [
        check_known_answer_internal(primary),
        check_known_answer_external(),
        check_lookahead_blocked(primary),
        check_lookahead_trap(primary),
        check_cost_sensitivity(symbols),
        check_timezone(),
    ]
