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
def status() -> None:
    """Show what is currently on disk."""
    from roth.storage import dataset_summary

    ensure_dirs()
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
