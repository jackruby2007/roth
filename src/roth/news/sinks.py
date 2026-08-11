"""Where alerts go.

Four destinations, any combination, chosen with `--sink`:

    console   coloured terminal output (the default)
    desktop   a native OS notification, so it reaches you outside the terminal
    jsonl     an append-only log at data/news/stream.jsonl
    webhook   an HTTP POST, shaped for Slack and Discord incoming webhooks

The contract every sink shares is that it must not raise. A notification daemon
that is not running, a webhook returning 500, a full disk -- none of these are
reasons to lose the alert stream, so failures are counted and reported by the
runner instead of propagating. The console sink is the backstop and is always
safe to fall back to.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.table import Table

from roth.news.brief import (
    Brief,
    fmt_et,
    fmt_pct,
    fmt_price,
    render_alert_oneline,
    render_alert_text,
    render_brief_text,
    render_move_alert,
)
from roth.news.models import NewsItem, Quote


class Sink(Protocol):
    name: str
    failures: int

    def emit_item(self, item: NewsItem, quote: Quote | None) -> None: ...
    def emit_move(self, quote: Quote) -> None: ...
    def emit_brief(self, brief: Brief) -> None: ...
    def close(self) -> None: ...


@dataclass
class _BaseSink:
    failures: int = 0
    last_error: str | None = None

    def _fail(self, exc: Exception) -> None:
        self.failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"

    def close(self) -> None:  # pragma: no cover - most sinks need no teardown
        return None


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


@dataclass
class ConsoleSink(_BaseSink):
    name: str = "console"
    show_reasons: bool = False
    console: Console = field(default_factory=Console)

    def _colour(self, pct: float) -> str:
        if pct > 0:
            return "green"
        return "red" if pct < 0 else "white"

    def _print_url(self, url: str, indent: int) -> None:
        """Print a URL without letting rich break it across lines.

        Rich word-wraps at the console width, which inserts a hard newline into
        the middle of a long EDGAR path. That kills both click-through and
        copy-paste -- exactly when you most want to open the filing. `soft_wrap`
        hands the line to the terminal intact and lets it do the wrapping.
        """
        self.console.print(f"{' ' * indent}[blue dim]{url}[/]", soft_wrap=True)

    def emit_item(self, item: NewsItem, quote: Quote | None) -> None:
        try:
            move = ""
            if quote is not None:
                move = f" [{self._colour(quote.change_pct)}]{fmt_pct(quote.change_pct)}[/]"
            tag = "[bold yellow]SEC[/]" if item.source == "sec" else "[dim]news[/]"
            kind = f" [yellow]{item.kind}[/]" if item.source == "sec" and item.kind else ""

            self.console.print(
                f"{tag} [bold cyan]{item.symbol}[/]{move}{kind} "
                f"[dim]{fmt_et(item.published_utc)}[/] [dim]({item.score})[/]"
            )
            self.console.print(f"  {item.title}")
            if item.summary and item.source == "sec":
                self.console.print(f"  [dim]{item.summary}[/]")
            if item.url:
                self._print_url(item.url, indent=2)
            if self.show_reasons and item.reasons:
                self.console.print(f"  [dim]why: {'; '.join(item.reasons)}[/]")
            self.console.print()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def emit_move(self, quote: Quote) -> None:
        try:
            colour = self._colour(quote.change_pct)
            self.console.print(
                f"[bold magenta]MOVE[/] [bold cyan]{quote.symbol}[/] "
                f"[{colour}]{fmt_pct(quote.change_pct)}[/] "
                f"{fmt_price(quote.effective_price)} "
                f"[dim]{fmt_et(quote.as_of_utc)}[/]\n"
            )
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def emit_brief(self, brief: Brief) -> None:
        try:
            self.console.rule(f"[bold]{brief.title()}[/]")
            self.console.print(f"[dim]{brief.phase_note}[/]\n")

            table = Table(box=None, pad_edge=False, header_style="bold")
            table.add_column("Symbol", style="cyan")
            table.add_column("Price", justify="right")
            table.add_column("Change", justify="right")
            table.add_column("Prev close", justify="right")
            table.add_column("State", style="dim")
            table.add_column("News", justify="right", style="dim")

            for section in brief.movers:
                q = section.quote
                if q is None:
                    table.add_row(section.symbol, "-", "-", "-", "unavailable", str(section.headline_count))
                    continue
                table.add_row(
                    section.symbol,
                    fmt_price(q.effective_price),
                    f"[{self._colour(q.change_pct)}]{fmt_pct(q.change_pct)}[/]",
                    fmt_price(q.previous_close),
                    q.market_state.lower(),
                    str(section.headline_count),
                )
            self.console.print(table)
            self.console.print()

            for section in brief.movers:
                if not section.items:
                    continue
                move = fmt_pct(section.change_pct) if section.quote else ""
                colour = self._colour(section.change_pct)
                self.console.print(
                    f"[bold cyan]{section.symbol}[/] [{colour}]{move}[/]"
                )
                for item in section.items:
                    marker = "[bold yellow]SEC[/]" if item.source == "sec" else "   "
                    self.console.print(
                        f"  {marker} [dim]{fmt_et(item.published_utc)}[/] {item.title} [dim]({item.score})[/]"
                    )
                    if item.url:
                        self._print_url(item.url, indent=6)
                self.console.print()

            if brief.degraded:
                self.console.print("[yellow]Degraded sources[/]")
                for note in brief.degraded:
                    self.console.print(f"  [yellow]![/] {note}")
                self.console.print()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)


# ---------------------------------------------------------------------------
# Desktop notification
# ---------------------------------------------------------------------------


@dataclass
class DesktopSink(_BaseSink):
    """Native OS notification.

    Detection happens once at construction. If no mechanism is available the
    sink reports `unavailable` and silently does nothing thereafter -- the
    runner surfaces that at startup so it is a known limitation rather than a
    stream of alerts that quietly go nowhere.
    """

    name: str = "desktop"
    mechanism: str = ""

    def __post_init__(self) -> None:
        system = platform.system()
        if system == "Darwin" and shutil.which("osascript"):
            self.mechanism = "osascript"
        elif system == "Linux" and shutil.which("notify-send"):
            self.mechanism = "notify-send"
        elif system == "Windows" and shutil.which("powershell"):
            self.mechanism = "powershell"
        else:
            self.mechanism = ""

    @property
    def available(self) -> bool:
        return bool(self.mechanism)

    def _notify(self, title: str, body: str) -> None:
        if not self.available:
            return
        try:
            if self.mechanism == "osascript":
                # Quotes must be escaped or a headline containing one breaks
                # the AppleScript rather than the notification.
                script = (
                    f'display notification {_applescript_str(body)} '
                    f'with title {_applescript_str(title)}'
                )
                cmd = ["osascript", "-e", script]
            elif self.mechanism == "notify-send":
                cmd = ["notify-send", "--app-name=roth news", title, body]
            else:
                ps = (
                    "[Windows.UI.Notifications.ToastNotificationManager, "
                    "Windows.UI.Notifications, ContentType=WindowsRuntime] > $null; "
                    f"Write-Output {json.dumps(title + ': ' + body)}"
                )
                cmd = ["powershell", "-NoProfile", "-Command", ps]
            subprocess.run(cmd, check=False, capture_output=True, timeout=10)
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def emit_item(self, item: NewsItem, quote: Quote | None) -> None:
        self._notify(f"{item.symbol} news", render_alert_oneline(item, quote))

    def emit_move(self, quote: Quote) -> None:
        self._notify(f"{quote.symbol} move", render_move_alert(quote))

    def emit_brief(self, brief: Brief) -> None:
        movers = [s for s in brief.movers if s.quote][:3]
        summary = ", ".join(f"{s.symbol} {fmt_pct(s.change_pct)}" for s in movers)
        self._notify(brief.title(), summary or "no quotes available")


def _applescript_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


@dataclass
class JsonlSink(_BaseSink):
    """Append-only record of everything emitted.

    This is what makes the bot auditable after the fact: when a name moved and
    you want to know what the bot knew and when, the answer is in this file.
    """

    name: str = "jsonl"
    path: Path = Path("data/news/stream.jsonl")

    def __post_init__(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._fail(exc)

    def _write(self, record: dict) -> None:
        try:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            self._fail(exc)

    def emit_item(self, item: NewsItem, quote: Quote | None) -> None:
        record = {"type": "item", **item.to_dict()}
        if quote is not None:
            record["quote"] = quote.to_dict()
        self._write(record)

    def emit_move(self, quote: Quote) -> None:
        self._write({"type": "move", **quote.to_dict()})

    def emit_brief(self, brief: Brief) -> None:
        self._write(
            {
                "type": "brief",
                "generated_utc": brief.generated_utc.isoformat(),
                "phase": brief.phase,
                "degraded": brief.degraded,
                "symbols": [
                    {
                        "symbol": s.symbol,
                        "quote": s.quote.to_dict() if s.quote else None,
                        "items": [i.to_dict() for i in s.items],
                    }
                    for s in brief.symbols
                ],
            }
        )


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


@dataclass
class WebhookSink(_BaseSink):
    """POST to a Slack or Discord incoming webhook.

    Both accept a JSON body with a `content`/`text` field, so one payload
    carrying both keys works for either without configuration.
    """

    name: str = "webhook"
    url: str = ""
    timeout: float = 10.0

    def _post(self, text: str) -> None:
        if not self.url:
            return
        try:
            import httpx

            # Discord rejects anything over 2000 characters outright.
            body = text if len(text) <= 1900 else text[:1897] + "..."
            httpx.post(
                self.url,
                json={"text": body, "content": body},
                timeout=self.timeout,
            ).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._fail(exc)

    def emit_item(self, item: NewsItem, quote: Quote | None) -> None:
        self._post(render_alert_text(item, quote))

    def emit_move(self, quote: Quote) -> None:
        self._post(render_move_alert(quote))

    def emit_brief(self, brief: Brief) -> None:
        self._post(render_brief_text(brief))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

SINK_NAMES = ("console", "desktop", "jsonl", "webhook")


def build_sinks(
    names: tuple[str, ...],
    webhook_url: str | None = None,
    jsonl_path: Path | None = None,
    show_reasons: bool = False,
    console: Console | None = None,
) -> list[Sink]:
    """Construct the requested sinks, skipping any that cannot work."""
    from roth.paths import NEWS
    from roth.news.config import STREAM_FILENAME

    sinks: list[Sink] = []
    for name in names:
        if name == "console":
            sinks.append(ConsoleSink(show_reasons=show_reasons, console=console or Console()))
        elif name == "desktop":
            sinks.append(DesktopSink())
        elif name == "jsonl":
            sinks.append(JsonlSink(path=jsonl_path or (NEWS / STREAM_FILENAME)))
        elif name == "webhook":
            if not webhook_url:
                raise ValueError("--sink webhook requires --webhook-url")
            sinks.append(WebhookSink(url=webhook_url))
        else:
            raise ValueError(f"Unknown sink {name!r}. Known: {', '.join(SINK_NAMES)}")
    return sinks
