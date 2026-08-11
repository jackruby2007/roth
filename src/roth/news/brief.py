"""Assembling and rendering what you actually read.

Two output shapes, because two situations:

* **The brief** -- one screen, all eight names, ordered by how much they moved.
  Built before the open so the day starts with a picture rather than a backlog,
  and rebuildable on demand at any hour.
* **The alert** -- one item, arriving mid-session, with just enough context to
  decide whether to look further.

Rendering is separated from assembly so the same brief can go to a terminal, a
webhook, or a file without the assembly running three times.
"""

from __future__ import annotations

import zoneinfo
from dataclasses import dataclass, field
from datetime import datetime, timezone

from roth.news.config import MARKET_TZ
from roth.news.models import NewsItem, Quote

ET = zoneinfo.ZoneInfo(MARKET_TZ)


def to_et(when: datetime) -> datetime:
    return when.astimezone(ET)


def fmt_et(when: datetime, with_date: bool = False) -> str:
    stamp = to_et(when)
    return stamp.strftime("%a %d %b %H:%M ET") if with_date else stamp.strftime("%H:%M ET")


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


def fmt_price(value: float) -> str:
    return f"${value:,.2f}"


@dataclass
class SymbolBrief:
    symbol: str
    quote: Quote | None = None
    items: list[NewsItem] = field(default_factory=list)

    @property
    def change_pct(self) -> float:
        return self.quote.change_pct if self.quote else 0.0

    @property
    def headline_count(self) -> int:
        return len(self.items)

    @property
    def top_score(self) -> int:
        return max((i.score for i in self.items), default=0)

    def price_line(self) -> str:
        if self.quote is None:
            return f"{self.symbol:<6} price unavailable"
        q = self.quote
        state = {
            "PRE": "pre-market",
            "POST": "after hours",
            "REGULAR": "",
            "CLOSED": "at close",
        }.get(q.market_state, q.market_state.lower())
        suffix = f"  ({state})" if state else ""
        return (
            f"{self.symbol:<6} {fmt_price(q.effective_price):>11}  "
            f"{fmt_pct(q.change_pct):>8}  prev {fmt_price(q.previous_close)}{suffix}"
        )


@dataclass
class Brief:
    """A whole-watchlist snapshot at a moment in time."""

    generated_utc: datetime
    phase: str
    phase_note: str
    symbols: list[SymbolBrief] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    @property
    def movers(self) -> list[SymbolBrief]:
        """Biggest absolute move first. That is the order you want to read."""
        return sorted(self.symbols, key=lambda s: abs(s.change_pct), reverse=True)

    def title(self) -> str:
        label = {
            "premarket": "Pre-market brief",
            "regular": "Session brief",
            "afterhours": "After-hours brief",
            "overnight": "Overnight brief",
            "closed": "Market closed brief",
        }.get(self.phase, "Brief")
        return f"{label} - {fmt_et(self.generated_utc, with_date=True)}"


def build_brief(
    phase: str,
    phase_note: str,
    quotes: dict[str, Quote],
    items: list[NewsItem],
    symbols: tuple[str, ...],
    max_items_per_symbol: int = 5,
    min_score: int = 0,
    degraded: list[str] | None = None,
    generated_utc: datetime | None = None,
) -> Brief:
    """Group scored items and quotes into a per-symbol picture."""
    by_symbol: dict[str, list[NewsItem]] = {s: [] for s in symbols}
    for item in items:
        if item.symbol in by_symbol and item.score >= min_score:
            by_symbol[item.symbol].append(item)

    sections = []
    for symbol in symbols:
        ranked = sorted(
            by_symbol[symbol], key=lambda i: (i.score, i.published_utc), reverse=True
        )
        sections.append(
            SymbolBrief(
                symbol=symbol,
                quote=quotes.get(symbol),
                items=ranked[:max_items_per_symbol],
            )
        )

    return Brief(
        generated_utc=generated_utc or datetime.now(timezone.utc),
        phase=phase,
        phase_note=phase_note,
        symbols=sections,
        degraded=list(degraded or []),
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_brief_text(brief: Brief, show_scores: bool = False) -> str:
    """Plain text, for webhooks, files, and terminals without colour."""
    lines: list[str] = [brief.title(), "=" * len(brief.title()), "", brief.phase_note, ""]

    lines.append("Prices")
    lines.append("-" * 6)
    for section in brief.movers:
        lines.append("  " + section.price_line())
    lines.append("")

    any_news = False
    for section in brief.movers:
        if not section.items:
            continue
        any_news = True
        move = fmt_pct(section.change_pct) if section.quote else "n/a"
        heading = f"{section.symbol}  {move}"
        lines.append(heading)
        lines.append("-" * len(heading))
        for item in section.items:
            marker = "[SEC]" if item.source == "sec" else "     "
            score = f" ({item.score})" if show_scores else ""
            lines.append(f"  {marker} {fmt_et(item.published_utc)}  {item.title}{score}")
            if item.url:
                lines.append(f"          {item.url}")
        lines.append("")

    if not any_news:
        lines.append("No qualifying news for any watchlist name in this window.")
        lines.append("")

    if brief.degraded:
        lines.append("Degraded sources")
        lines.append("-" * 16)
        for note in brief.degraded:
            lines.append(f"  ! {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_alert_text(item: NewsItem, quote: Quote | None = None, show_reasons: bool = False) -> str:
    """One item, formatted for an interrupt."""
    move = f"  {fmt_pct(quote.change_pct)} {fmt_price(quote.effective_price)}" if quote else ""
    head = f"[{item.symbol}]{move}  {fmt_et(item.published_utc)}"
    lines = [head, item.title]
    if item.summary:
        lines.append(item.summary)
    if item.url:
        lines.append(item.url)
    if show_reasons and item.reasons:
        lines.append(f"score {item.score}: " + "; ".join(item.reasons))
    return "\n".join(lines)


def render_alert_oneline(item: NewsItem, quote: Quote | None = None) -> str:
    """Under 200 characters, for desktop notifications and push."""
    move = f" {fmt_pct(quote.change_pct)}" if quote else ""
    prefix = f"{item.symbol}{move}"
    if item.source == "sec":
        prefix = f"{item.symbol} {item.kind}{move}"
    body = f"{prefix}: {item.title}"
    return body[:197] + "..." if len(body) > 200 else body


def render_move_alert(quote: Quote) -> str:
    direction = "up" if quote.change_pct > 0 else "down"
    state = {"PRE": " pre-market", "POST": " after hours"}.get(quote.market_state, "")
    return (
        f"{quote.symbol} {direction} {abs(quote.change_pct):.2f}%{state} "
        f"at {fmt_price(quote.effective_price)} (prev close {fmt_price(quote.previous_close)})"
    )
