"""Single CLI entry point for the research harness.

Everything runs on demand through this command. There is no server, no
scheduler, and no background process. The only thing that ever needs to be
running is Theta Terminal, and only while `roth pilot` or `roth backfill` is
downloading.
"""

from __future__ import annotations

from datetime import date, datetime

import typer
from rich.console import Console
from rich.table import Table

from roth import config
from roth.paths import ensure_dirs, human_bytes, human_duration

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Options research harness for SPY/QQQ hypothesis testing.",
)
console = Console()


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.command()
def doctor() -> None:
    """Check that everything this harness needs is present and reachable."""
    from roth.data.thetadata import ThetaClient, ThetaError

    ensure_dirs()
    console.print("[bold]Environment check[/bold]\n")

    ok = True

    import sys

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        console.print(f"  [green]OK[/green]    Python {py}")
    else:
        console.print(f"  [red]FAIL[/red]  Python {py} (need 3.11 or newer)")
        ok = False

    for mod in ("duckdb", "pyarrow", "pandas", "polars", "pandas_market_calendars"):
        try:
            __import__(mod)
            console.print(f"  [green]OK[/green]    {mod}")
        except ImportError:
            console.print(f"  [red]FAIL[/red]  {mod} is not installed")
            ok = False

    console.print(f"\n  Data directory: {config.SYMBOLS} under [cyan]data/[/cyan]")

    console.print("\n[bold]Theta Terminal[/bold]\n")
    try:
        with ThetaClient() as client:
            info = client.check_connection()
        console.print(f"  [green]OK[/green]    Connected at {info['base_url']}")
        console.print(f"        Feed status: {info['status']}")
    except ThetaError as exc:
        console.print("  [yellow]NOT READY[/yellow]\n")
        for line in str(exc).splitlines():
            console.print(f"        {line}")
        console.print(
            "\n  This is only needed for downloads. Everything else works without it."
        )

    console.print()
    if ok:
        console.print("[green]Environment is ready.[/green]")
    else:
        console.print("[red]Environment has problems listed above.[/red]")
        raise typer.Exit(code=1)


@app.command()
def probe(
    root: str = typer.Option("SPY", help="Symbol to probe with."),
) -> None:
    """Test every ThetaData endpoint and write a report.

    Run this once, immediately after Theta Terminal connects for the first
    time. It checks every endpoint the harness uses, tries alternates where a
    path might differ, and discovers how far back your subscription can read.

    It writes one file. Send that file back and the client gets corrected from
    evidence rather than guesswork.
    """
    from roth.data.probe import run_probe

    console.print("[bold]Probing ThetaData[/bold]")
    console.print(f"Connecting to {config.THETA.base_url}\n")

    report, path = run_probe(root=root)

    for r in report.results:
        style = "green" if r.ok else "red"
        console.print(f"  [{style}]{r.summary()}[/{style}]")

    console.print(
        f"\n[bold]{len(report.passed)} ok, {len(report.failed)} failed[/bold]"
    )

    if report.notes:
        console.print("\n[bold]Notes[/bold]")
        for n in report.notes:
            console.print(f"  - {n}")

    console.print(f"\n[green]Report written to:[/green] {path}")

    if not report.passed:
        console.print(
            "\n[red]Nothing succeeded.[/red] The most likely cause is that Theta "
            "Terminal is not running or has not finished connecting.\n"
            "Run [cyan]roth doctor[/cyan] to check, then try again."
        )
        raise typer.Exit(code=1)

    console.print(
        "\n[bold]Send that file back.[/bold] It contains no credentials -- only "
        "endpoint paths, HTTP status codes, and column names."
    )


@app.command()
def pilot(
    symbol: str = typer.Option("SPY", help="Underlying to pilot."),
    start: str = typer.Option(None, help="Start date YYYY-MM-DD. Defaults to last full month."),
    end: str = typer.Option(None, help="End date YYYY-MM-DD. Defaults to last full month."),
) -> None:
    """Download one month of option quotes and measure size and download time.

    Run this before choosing a subscription tier. It answers the only two
    questions that matter for that decision: how much disk a month costs, and
    how long a month takes to pull.
    """
    from roth.data.pilot import persist_pilot, run_pilot
    from roth.data.thetadata import ThetaError

    console.print(f"[bold]Pilot download: {symbol}[/bold]")
    console.print(
        f"Bounds: expirations within {config.BOUNDS.max_dte} days, "
        f"strikes within +/-{config.BOUNDS.strike_pct:.0%} of spot, "
        f"{config.BOUNDS.interval_ms // 1000}s bars, "
        f"{'RTH only' if config.BOUNDS.rth_only else 'all hours'}\n"
    )

    try:
        result = run_pilot(symbol=symbol, start=_parse_date(start), end=_parse_date(end))
    except ThetaError as exc:
        console.print("[red]Pilot could not run.[/red]\n")
        for line in str(exc).splitlines():
            console.print(f"  {line}")
        raise typer.Exit(code=1) from exc

    persist_pilot(result)

    table = Table(title=f"Measured: {result.start} to {result.end}", show_header=False)
    table.add_column("", style="cyan")
    table.add_column("", justify="right")
    table.add_row("Trading days", f"{result.trading_days}")
    table.add_row("Expirations touched", f"{result.expirations_touched}")
    table.add_row("Rows written", f"{result.rows_written:,}")
    table.add_row("Disk footprint", human_bytes(result.bytes_on_disk))
    table.add_row("Bytes over the wire", human_bytes(result.bytes_over_wire))
    table.add_row("Wall clock", human_duration(result.wall_clock_seconds))
    table.add_row("Requests made", f"{result.requests_made:,}")
    table.add_row("Bytes per row", f"{result.bytes_per_row:.1f}")
    table.add_row("Seconds per request", f"{result.seconds_per_request:.2f}")
    console.print(table)

    if result.failures:
        console.print(f"\n[yellow]{len(result.failures)} failures during download:[/yellow]")
        for f in result.failures[:20]:
            console.print(f"  {f}")
        if len(result.failures) > 20:
            console.print(f"  ... and {len(result.failures) - 20} more")

    console.print(
        "\n[green]Measurements saved.[/green] Run [cyan]roth estimate[/cyan] to see the "
        "full-history extrapolation using these measured numbers."
    )


@app.command()
def estimate(
    first_year: int = typer.Option(2018, help="First year of the hypothetical backfill."),
    last_year: int = typer.Option(2026, help="Last year of the hypothetical backfill."),
) -> None:
    """Extrapolate full-history disk and download cost.

    Uses measured numbers from `roth pilot` if one has been run, otherwise a
    documented model. Which one is in play is stated in the output.
    """
    from roth.data.sizing import build_report, load_assumptions

    assumptions = load_assumptions()
    report = build_report(config.SYMBOLS, first_year, last_year, assumptions)

    if assumptions.measured:
        console.print(
            f"[green]Using MEASURED values[/green] from {assumptions.measured_from}\n"
        )
    else:
        console.print(
            "[yellow]Using MODELED values.[/yellow] No pilot download has been run, so "
            "these are arithmetic on stated assumptions, not measurements.\n"
            "Run [cyan]roth pilot[/cyan] against a live Theta Terminal to replace them.\n"
        )

    a = assumptions
    console.print("[bold]Assumptions[/bold]")
    console.print(f"  Strike spacing near the money    ${a.strike_spacing:.2f}")
    console.print(f"  Fraction of band actually quoted {a.listed_fraction:.0%}")
    lo, hi = a.bytes_per_quote_row_range()
    console.print(
        f"  Compressed bytes per quote row   {lo:.0f}"
        + (f" to {hi:.0f}" if hi != lo else " (measured)")
    )
    console.print(f"  Seconds per bulk request         {a.seconds_per_bulk_request:.2f}\n")

    table = Table(title=f"Per-year estimate, {first_year}-{last_year}")
    table.add_column("Year", justify="right")
    table.add_column("Sym")
    table.add_column("Exps", justify="right")
    table.add_column("Strikes", justify="right")
    table.add_column("Contracts", justify="right")
    table.add_column("1-min rows", justify="right")
    table.add_column("1-min disk", justify="right")
    table.add_column("EOD disk", justify="right")
    table.add_column("Download", justify="right")

    for y in report.years:
        disk = (
            human_bytes(y.quote_bytes_low)
            if y.quote_bytes_low == y.quote_bytes_high
            else f"{human_bytes(y.quote_bytes_low)}-{human_bytes(y.quote_bytes_high)}"
        )
        table.add_row(
            str(y.year),
            y.symbol,
            str(y.expirations_in_window),
            str(y.strikes_in_band),
            f"{y.contracts_in_scope:,}",
            f"{y.quote_rows / 1e6:,.0f}M",
            disk,
            human_bytes(y.eod_bytes),
            human_duration(y.download_seconds),
        )
    console.print(table)

    for label, since in (
        (f"Full {first_year}-{last_year}", None),
        ("Last 4 years only", max(first_year, last_year - 3)),
    ):
        t = report.totals(since)
        console.print(f"\n[bold]{label}[/bold]")
        console.print(f"  1-minute option quotes : {t['quote_rows'] / 1e9:,.1f} billion rows")
        console.print(
            f"  1-minute disk          : "
            f"{human_bytes(t['quote_bytes_low'])} to {human_bytes(t['quote_bytes_high'])}"
        )
        console.print(f"  EOD chains disk        : {human_bytes(t['eod_bytes'])}")
        console.print(f"  Bulk requests          : {t['bulk_requests']:,.0f}")
        console.print(f"  Download wall clock    : {human_duration(t['download_seconds'])}")

    console.print(
        "\n[dim]EOD chains are cheap and worth pulling for the full period regardless.\n"
        "The 1-minute quote data is what drives both the disk and the download cost.[/dim]"
    )


ingest_app = typer.Typer(no_args_is_help=True, help="Download data into the immutable raw store.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("calendar")
def ingest_calendar_cmd(
    start: str = typer.Option("2018-01-01", help="First date YYYY-MM-DD."),
    end: str = typer.Option(None, help="Last date YYYY-MM-DD. Defaults to one year ahead."),
) -> None:
    """Build the trading calendar. Needs no data feed and no subscription."""
    from roth.data.ingest import ingest_calendar

    start_d = _parse_date(start)
    end_d = _parse_date(end) or date.today().replace(year=date.today().year + 1)

    df, availability = ingest_calendar(start_d, end_d)
    console.print(f"[green]Wrote {len(df):,} trading sessions[/green] ({start_d} to {end_d})\n")

    table = Table(title="Calendar flags")
    table.add_column("Flag")
    table.add_column("Source")
    table.add_column("Days marked", justify="right")

    for name, label in (
        ("is_monthly_opex", "third Friday, computed"),
        ("is_quarterly_opex", "third Friday of Mar/Jun/Sep/Dec, computed"),
        ("is_opex_week", "computed"),
        ("is_quarter_end", "last session of quarter, computed"),
        ("is_early_close", "NYSE calendar"),
        ("is_short_week", "computed"),
    ):
        table.add_row(name, label, f"{int(df[name].sum()):,}")

    for avail in availability:
        flag = f"is_{avail.name}_day"
        if not avail.available:
            table.add_row(flag, f"[red]UNAVAILABLE ({avail.source})[/red]", "-")
        else:
            marked = int(df[flag].sum()) if flag in df else 0
            style = "yellow" if "rule" in avail.source else "green"
            table.add_row(flag, f"[{style}]{avail.source}[/{style}]", f"{marked:,}")

    console.print(table)

    missing = [a for a in availability if not a.available]
    if missing:
        console.print(
            "\n[yellow]Some event flags are unavailable.[/yellow] They are stored as null, "
            "not False, so no day is silently mislabelled as a non-event day."
        )
        for a in missing:
            console.print(f"  {a.name}: put a CSV with a 'date' column at\n    {a.path}")


@ingest_app.command("underlying")
def ingest_underlying_cmd(
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols by default."),
    start: str = typer.Option(None, help="First date YYYY-MM-DD."),
    end: str = typer.Option(None, help="Last date YYYY-MM-DD."),
    minute: bool = typer.Option(False, "--minute", help="Also pull 1-minute bars."),
) -> None:
    """Download underlying daily (and optionally 1-minute) bars."""
    from roth.data.ingest import (
        default_backfill_range,
        ingest_underlying_daily,
        ingest_underlying_minute,
        ingest_vix_daily,
    )
    from roth.data.thetadata import ThetaError

    d_start, d_end = default_backfill_range()
    d_start = _parse_date(start) or d_start
    d_end = _parse_date(end) or d_end
    symbols = (symbol,) if symbol else config.SYMBOLS

    try:
        for sym in symbols:
            console.print(f"[bold]{sym} daily[/bold] {d_start} to {d_end}")
            console.print(f"  {ingest_underlying_daily(sym, d_start, d_end).summary()}")
            if minute:
                console.print(f"[bold]{sym} 1-minute[/bold]")
                console.print(f"  {ingest_underlying_minute(sym, d_start, d_end).summary()}")
        console.print("[bold]VIX daily[/bold]")
        console.print(f"  {ingest_vix_daily(d_start, d_end).summary()}")
    except ThetaError as exc:
        _report_theta_error(exc)
        raise typer.Exit(code=1) from exc


@ingest_app.command("options-eod")
def ingest_options_eod_cmd(
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols by default."),
    start: str = typer.Option(None, help="First date YYYY-MM-DD."),
    end: str = typer.Option(None, help="Last date YYYY-MM-DD."),
) -> None:
    """Download end-of-day option chains, bounded to the configured strike and expiry window."""
    from roth.data.ingest import default_backfill_range, ingest_option_eod
    from roth.data.thetadata import ThetaError

    d_start, d_end = default_backfill_range()
    d_start = _parse_date(start) or d_start
    d_end = _parse_date(end) or d_end
    symbols = (symbol,) if symbol else config.SYMBOLS

    try:
        for sym in symbols:
            console.print(f"[bold]{sym} option EOD[/bold] {d_start} to {d_end}")
            result = ingest_option_eod(sym, d_start, d_end)
            console.print(f"  {result.summary()}")
            for f in result.failures[:10]:
                console.print(f"    [yellow]{f}[/yellow]")
    except ThetaError as exc:
        _report_theta_error(exc)
        raise typer.Exit(code=1) from exc


def _report_theta_error(exc: Exception) -> None:
    console.print("\n[red]Download could not run.[/red]\n")
    for line in str(exc).splitlines():
        console.print(f"  {line}")


@app.command()
def backtest(
    strategy: str = typer.Option(
        "reference_monday_call", help="Strategy name. See `roth strategies`."
    ),
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols."),
    start: str = typer.Option(None, help="First date YYYY-MM-DD."),
    end: str = typer.Option(None, help="Last date YYYY-MM-DD."),
    mid_fills: bool = typer.Option(
        False, "--mid-fills", help="Fill at mid instead of bid/ask. Comparison only."
    ),
    slippage: float = typer.Option(
        None, help="Extra slippage per contract, on top of crossing the spread."
    ),
    include_quarantined: bool = typer.Option(
        False, "--include-quarantined", help="Do not exclude quarantined sessions."
    ),
    csv: bool = typer.Option(False, "--csv", help="Also export CSV files."),
    explain: int = typer.Option(None, help="Explain one signal by id, then exit."),
) -> None:
    """Run a strategy and print the full performance report."""
    import dataclasses

    from roth.backtest.runner import run_backtest
    from roth.journal import explain_signal, make_run_id, write_journal
    from roth.report import build_report, export_csv, render_text
    from roth.strategies import STRATEGIES

    ensure_dirs()

    if strategy not in STRATEGIES:
        console.print(
            f"[red]Unknown strategy {strategy!r}.[/red] Available: "
            f"{', '.join(sorted(STRATEGIES))}"
        )
        raise typer.Exit(code=1)

    costs = config.COSTS
    if slippage is not None:
        costs = dataclasses.replace(costs, extra_slippage_per_contract=slippage)

    symbols = (symbol,) if symbol else config.SYMBOLS
    instance = STRATEGIES[strategy]()

    result = run_backtest(
        instance,
        symbols,
        start=_parse_date(start),
        end=_parse_date(end),
        costs=costs,
        use_mid_fills=mid_fills,
        exclude_quarantined=not include_quarantined,
    )

    if explain is not None:
        console.print(explain_signal(result, explain))
        return

    trades = result.trades_frame()
    perf = build_report(
        trades,
        strategy_name=result.strategy_name,
        strategy_version=result.strategy_version,
        symbols=result.symbols,
        unfillable_signals=result.unfillable_count,
        unfillable_entries=sum(1 for u in result.unfillable if u.action == "entry"),
        unfillable_exits=sum(1 for u in result.unfillable if u.action == "exit"),
        candidate_signals=result.candidate_signals,
        quarantined_sessions=result.quarantined_sessions_excluded,
        synthetic_data=result.synthetic_data,
        used_mid_fills=result.used_mid_fills,
    )

    console.print(render_text(perf), highlight=False)

    run_id = make_run_id(result)
    counts = write_journal(result, run_id)
    console.print(
        f"\nJournal written: {counts['trades']:,} trades, "
        f"{counts['rule_evaluations']:,} rule evaluations, "
        f"{counts['unfillable']:,} unfillable signals."
    )

    if not trades.empty:
        summary = result.rule_filter_summary()
        table = Table(title="Which rule does the filtering")
        table.add_column("Rule")
        table.add_column("Evaluations", justify="right")
        table.add_column("Failures", justify="right")
        table.add_column("Fail rate", justify="right")
        for _, r in summary.iterrows():
            table.add_row(
                r["rule_name"],
                f"{int(r['evaluations']):,}",
                f"{int(r['failures']):,}",
                f"{r['fail_rate']:.1%}",
            )
        console.print(table)

    if csv:
        from roth.paths import REPORTS

        paths = export_csv(perf, trades, REPORTS)
        console.print("\nCSV exported:")
        for k, v in paths.items():
            console.print(f"  {k}: {v}")

    console.print(
        f"\n[dim]Explain any signal with "
        f"[cyan]roth backtest --strategy {strategy} --explain <id>[/cyan][/dim]"
    )


@app.command()
def strategies() -> None:
    """List available strategies."""
    from roth.strategies import STRATEGIES

    table = Table(title="Strategies")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Structure")
    for name, cls in sorted(STRATEGIES.items()):
        table.add_row(name, cls.version, cls.structure_type)
    console.print(table)


@app.command()
def verify() -> None:
    """Run the correctness tests.

    The harness must prove it is not lying before any result from it is
    trusted. A check that cannot run reports SKIP, never PASS.
    """
    from roth.verify import FAIL, PASS, SKIP, run_all

    from roth.data.synth import is_synthetic

    ensure_dirs()

    if is_synthetic():
        console.print(
            "[yellow]SYNTHETIC DATA[/yellow] - checks that require real market data "
            "will report SKIP.\n"
        )

    results = run_all(config.SYMBOLS)

    table = Table(title="Correctness tests")
    table.add_column("Check")
    table.add_column("Result")
    for r in results:
        style = {PASS: "green", FAIL: "red", SKIP: "yellow"}[r.status]
        table.add_row(r.name, f"[{style}]{r.status}[/{style}]")
    console.print(table)

    console.print()
    for r in results:
        style = {PASS: "green", FAIL: "red", SKIP: "yellow"}[r.status]
        console.print(f"[{style}]{r.status}[/{style}] [bold]{r.name}[/bold]")
        for line in _wrap(r.detail):
            console.print(f"      {line}")
        console.print()

    failures = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]

    if failures:
        console.print(
            f"[red]{len(failures)} check(s) FAILED.[/red] Results from this harness "
            "should not be trusted until they pass."
        )
        raise typer.Exit(code=1)

    if skipped:
        console.print(
            f"[yellow]{len(results) - len(skipped)} passed, {len(skipped)} skipped.[/yellow]\n"
            "A skipped check has not verified anything. See the reasons above."
        )
    else:
        console.print("[green]All correctness tests passed.[/green]")


def _wrap(text: str, width: int = 88) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


features_app = typer.Typer(no_args_is_help=True, help="Build and verify the feature store.")
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build_cmd(
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols."),
) -> None:
    """Build the daily feature table.

    Every row contains only what was knowable by that session's close.
    """
    from roth.features.build import build_all

    symbols = (symbol,) if symbol else config.SYMBOLS
    counts = build_all(symbols)

    table = Table(title="Feature tables")
    table.add_column("Symbol")
    table.add_column("Sessions", justify="right")
    for sym, n in counts.items():
        table.add_row(sym, f"{n:,}")
    console.print(table)

    if all(v == 0 for v in counts.values()):
        console.print(
            "\n[yellow]No features built.[/yellow] There is no underlying data on disk."
        )
        raise typer.Exit(code=1)

    console.print(
        "\nRun [cyan]roth features verify[/cyan] to confirm no feature depends on "
        "future data."
    )


@features_app.command("verify")
def features_verify_cmd(
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols."),
    cutoffs: int = typer.Option(4, help="Number of truncation points to test."),
) -> None:
    """Verify causality by point-in-time invariance.

    Rebuilds the feature table with history truncated at several cutoff dates
    and confirms every value at or before each cutoff is identical to the
    full-history build. A feature that peeks forward cannot survive this.
    """
    from roth.features.causality import verify_causality

    symbols = (symbol,) if symbol else config.SYMBOLS
    failed = False

    for sym in symbols:
        console.print(f"[bold]{sym}[/bold] - rebuilding at {cutoffs} cutoff dates...")
        result = verify_causality(sym, n_cutoffs=cutoffs)

        if not result.cutoffs:
            console.print("  [yellow]no data[/yellow]")
            continue

        if result.passed:
            console.print(f"  [green]{result.summary()}[/green]")
        else:
            failed = True
            console.print(f"  [red]{result.summary()}[/red]\n")
            for v in result.violations[:15]:
                console.print(
                    f"    [red]{v.column}[/red] truncated at {v.cutoff}: "
                    f"{v.rows_differing} row(s) differ"
                )
                console.print(
                    f"      {v.example_day}: full history gave {v.full_value!r}, "
                    f"truncated gave {v.truncated_value!r}"
                )

    if failed:
        console.print(
            "\n[red]Causality verification failed.[/red] A feature above changes value "
            "when future data is removed, which means it is using that future data.\n"
            "Every backtest result depending on those columns is invalid until fixed."
        )
        raise typer.Exit(code=1)

    console.print("\n[green]All features are causal.[/green]")


@app.command()
def quality(
    start: str = typer.Option(None, help="First date YYYY-MM-DD."),
    end: str = typer.Option(None, help="Last date YYYY-MM-DD."),
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols."),
    detail: bool = typer.Option(False, "--detail", help="List every quarantined session."),
) -> None:
    """Run data quality checks and write the quarantine table.

    Runs before any research. Quarantined sessions are excluded from research by
    default, and the count appears in every report.
    """
    from roth.data.synth import is_synthetic
    from roth.quality import persist_quarantine, run_quality
    from roth.storage import read_manifest

    ensure_dirs()

    if read_manifest().empty:
        console.print(
            "[yellow]No data on disk.[/yellow] Nothing to check.\n"
            "Run [cyan]roth synth[/cyan] to generate a dataset, or ingest real data."
        )
        raise typer.Exit(code=1)

    if is_synthetic():
        console.print("[yellow]SYNTHETIC DATA[/yellow] - these findings describe fabricated data.\n")

    symbols = (symbol,) if symbol else config.SYMBOLS

    # Default to the span the data actually covers. Defaulting to a fixed early
    # date would report every session before the first download as "missing",
    # which is true but useless.
    manifest = read_manifest()
    d_start = _parse_date(start) or manifest["day"].min()
    d_end = _parse_date(end) or manifest["day"].max()
    console.print(f"[dim]Checking {d_start} to {d_end}[/dim]\n")

    report = run_quality(symbols, d_start, d_end)

    summary = report.by_check()
    if summary.empty:
        console.print("[green]No quality findings.[/green]")
    else:
        table = Table(title="Quality findings")
        table.add_column("Check")
        table.add_column("Severity")
        table.add_column("Sessions", justify="right")
        table.add_column("Occurrences", justify="right")
        for _, row in summary.iterrows():
            style = "red" if row["severity"] == "quarantine" else "yellow"
            table.add_row(
                row["check"],
                f"[{style}]{row['severity']}[/{style}]",
                f"{int(row['sessions']):,}",
                f"{int(row['occurrences']):,}",
            )
        console.print(table)

    qframe = persist_quarantine(report)
    n_quarantined = len(report.quarantined_days)

    console.print(
        f"\n[bold]Sessions quarantined:[/bold] {n_quarantined:,} "
        f"(excluded from research by default)"
    )
    console.print(f"[bold]Findings flagged, not excluded:[/bold] {len(report.flags):,}")
    console.print(
        "\n[dim]Flagged findings are real market conditions, not corrupt data. Wide\n"
        "spreads and isolated crossed quotes are handled by the fill model at trade\n"
        "time, which rejects the affected contract rather than the whole session.[/dim]"
    )

    if detail and not qframe.empty:
        dtable = Table(title="Quarantined sessions")
        dtable.add_column("Symbol")
        dtable.add_column("Day")
        dtable.add_column("Check")
        dtable.add_column("Detail")
        for _, row in qframe.head(80).iterrows():
            dtable.add_row(row["symbol"], str(row["day"]), row["check"], row["detail"])
        console.print(dtable)
        if len(qframe) > 80:
            console.print(f"[dim]... and {len(qframe) - 80} more[/dim]")


@app.command("synth")
def synth_cmd(
    start: str = typer.Option("2022-01-01", help="First date YYYY-MM-DD."),
    end: str = typer.Option("2024-12-31", help="Last date YYYY-MM-DD."),
    symbol: str = typer.Option(None, help="One symbol, or all configured symbols by default."),
    no_minute: bool = typer.Option(False, "--no-minute", help="Skip 1-minute bars."),
    no_options: bool = typer.Option(False, "--no-options", help="Skip option chains."),
    force: bool = typer.Option(False, "--force", help="Wipe existing raw data first."),
) -> None:
    """Generate a synthetic dataset so the pipeline can be exercised without a feed.

    The data is fabricated. It proves the plumbing runs; it says nothing about
    whether any strategy has edge.
    """
    import shutil

    from roth.data.synth import generate, is_synthetic
    from roth.paths import RAW
    from roth.storage import read_manifest

    if not read_manifest().empty:
        if not force:
            origin = "synthetic" if is_synthetic() else "REAL"
            console.print(
                f"[red]Raw data already exists[/red] and it is marked {origin}.\n"
                "Generating would mix fabricated data into it.\n"
                "Re-run with [cyan]--force[/cyan] to wipe the raw store first."
            )
            raise typer.Exit(code=1)
        if RAW.exists():
            shutil.rmtree(RAW)
        ensure_dirs()

    symbols = (symbol,) if symbol else config.SYMBOLS
    console.print(f"[bold]Generating synthetic data[/bold] for {', '.join(symbols)}\n")

    counts = generate(
        _parse_date(start),
        _parse_date(end),
        symbols=symbols,
        with_minute=not no_minute,
        with_options=not no_options,
    )

    table = Table(title="Generated")
    table.add_column("Dataset")
    table.add_column("Rows", justify="right")
    for k, v in counts.items():
        table.add_row(k, f"{v:,}")
    console.print(table)

    console.print(
        "\n[yellow]This data is fabricated.[/yellow] Every report built from it will "
        "carry a synthetic-data banner. Strategy results from it are meaningless."
    )


@app.command()
def status() -> None:
    """Show what is currently on disk."""
    from roth.data.synth import is_synthetic, synthetic_info
    from roth.storage import dataset_summary

    ensure_dirs()

    if is_synthetic():
        info = synthetic_info() or {}
        console.print(
            "[yellow]SYNTHETIC DATA[/yellow] - generated "
            f"{info.get('generated_at', 'unknown')}, "
            f"{info.get('sessions', '?')} sessions.\n"
            "Results computed from it demonstrate the pipeline runs, nothing more.\n"
        )

    df = dataset_summary(None)
    if df.empty:
        console.print(
            "[yellow]No data ingested yet.[/yellow]\n\n"
            "Start with [cyan]roth ingest calendar[/cyan], which needs no subscription."
        )
        return

    table = Table(title="Raw data on disk")
    for col in ("Dataset", "Symbol", "Days", "Rows", "Size", "First", "Last"):
        table.add_column(col, justify="right" if col in ("Days", "Rows", "Size") else "left")

    for _, row in df.iterrows():
        table.add_row(
            row["dataset"],
            row["symbol"],
            f"{int(row['days']):,}",
            f"{int(row['rows']):,}",
            human_bytes(int(row["bytes"])),
            str(row["first_day"]),
            str(row["last_day"]),
        )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
