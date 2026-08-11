"""Tests for the poll loop, its sinks, and brief assembly.

The sources are faked here. What is under test is the behaviour that decides
whether the bot is usable across a whole day: priming on the first cycle,
degrading rather than crashing when a feed dies, respecting the per-phase alert
threshold, and never sleeping through the opening bell.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from roth.news.brief import build_brief, render_alert_oneline, render_brief_text
from roth.news.config import CADENCE, RuntimeOptions, SourceToggles
from roth.news.dedupe import SeenStore
from roth.news.models import NewsItem, Quote, SourceResult
from roth.news.runner import NewsRunner
from roth.news.session import SessionClock
from roth.news.sinks import JsonlSink

SYMBOLS = ("NVDA", "TSLA")

# Wednesday 2026-08-12, 15:00 UTC == 11:00 ET, mid-session.
MIDSESSION = datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc)
# 12:30 UTC == 08:30 ET, after the pre-open brief time.
PREOPEN = datetime(2026, 8, 12, 12, 30, tzinfo=timezone.utc)


def make_item(symbol="NVDA", title="Nvidia cuts guidance for Q4", source="yahoo", url=None):
    return NewsItem(
        symbol=symbol,
        source=source,
        title=title,
        url=url or f"https://reuters.com/{abs(hash(title)) % 10**8}",
        published_utc=MIDSESSION - timedelta(minutes=5),
        kind="8-K" if source == "sec" else "headline",
        tags=("sec", "form:8-K", "item:2.02") if source == "sec" else ("headline",),
    )


def make_quote(symbol="NVDA", price=108.0, prev=100.0, state="REGULAR"):
    return Quote(
        symbol=symbol, price=price, previous_close=prev, market_state=state,
        as_of_utc=MIDSESSION,
    )


@dataclass
class FakeSource:
    name: str = "fake"
    items: list = field(default_factory=list)
    error: str | None = None
    calls: int = 0

    def fetch(self, http, symbols):
        self.calls += 1
        return SourceResult(source=self.name, items=list(self.items), error=self.error)


@dataclass
class FakeQuoteSource:
    name: str = "quotes"
    quotes: list = field(default_factory=list)
    error: str | None = None

    def fetch(self, http, symbols):
        return SourceResult(source=self.name, quotes=list(self.quotes), error=self.error)


@dataclass
class RecordingSink:
    name: str = "recording"
    failures: int = 0
    items: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    briefs: list = field(default_factory=list)

    def emit_item(self, item, quote):
        self.items.append((item, quote))

    def emit_move(self, quote):
        self.moves.append(quote)

    def emit_brief(self, brief):
        self.briefs.append(brief)

    def close(self):
        return None


@pytest.fixture
def runner_factory(tmp_path):
    def build(items=None, quotes=None, source_error=None, quote_error=None, **opt_kwargs):
        options = RuntimeOptions(
            symbols=SYMBOLS,
            toggles=SourceToggles(sec_edgar=False, yahoo_rss=False, quotes=False),
            **opt_kwargs,
        )
        sink = RecordingSink()
        runner = NewsRunner(
            options,
            [sink],
            clock=SessionClock(),
            http=None,
            store=SeenStore(path=tmp_path / "seen.json"),
        )
        runner.sources = [FakeSource(items=items or [], error=source_error)]
        runner.quote_source = FakeQuoteSource(quotes=quotes or [], error=quote_error)
        return runner, sink

    return build


# -- priming ----------------------------------------------------------------


def test_the_first_cycle_briefs_instead_of_alerting(runner_factory):
    """Starting at 11:00 must not fire forty alerts for the morning's news."""
    runner, sink = runner_factory(items=[make_item(), make_item(title="Nvidia halts trading")])
    stats = runner.poll_once(now_utc=MIDSESSION)

    assert stats.primed is True
    assert sink.items == []
    assert len(sink.briefs) == 1


def test_items_seen_while_priming_never_alert_afterwards(runner_factory):
    runner, sink = runner_factory(items=[make_item()])
    runner.poll_once(now_utc=MIDSESSION)
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))
    assert sink.items == []


def test_an_item_arriving_after_priming_does_alert(runner_factory):
    runner, sink = runner_factory(items=[make_item()])
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items.append(make_item(title="Nvidia halts trading after outage"))
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))

    assert len(sink.items) == 1
    assert "halts trading" in sink.items[0][0].title


# -- thresholds -------------------------------------------------------------


def test_low_scoring_noise_is_not_alerted_mid_session(runner_factory):
    runner, sink = runner_factory(items=[])
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items = [make_item(title="3 reasons Nvidia stock is a buy right now")]
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))
    assert sink.items == []


def test_an_earnings_filing_clears_the_mid_session_threshold(runner_factory):
    runner, sink = runner_factory(items=[])
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items = [make_item(source="sec", title="NVDA: 8-K - results")]
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))
    assert len(sink.items) == 1


def test_an_explicit_min_score_overrides_the_phase_default(runner_factory):
    runner, sink = runner_factory(items=[], min_score=0)
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items = [make_item(title="3 reasons Nvidia stock is a buy right now")]
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))
    assert len(sink.items) == 1


def test_the_alert_carries_the_symbols_current_quote(runner_factory):
    runner, sink = runner_factory(items=[], quotes=[make_quote()])
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items = [make_item(source="sec")]
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))

    _, quote = sink.items[0]
    assert quote is not None and quote.symbol == "NVDA"


# -- price moves ------------------------------------------------------------


def test_a_big_move_alerts_even_with_no_headline(runner_factory):
    runner, sink = runner_factory(quotes=[make_quote(price=108.0, prev=100.0)])
    runner.poll_once(now_utc=MIDSESSION)
    runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))
    assert len(sink.moves) == 1


def test_a_standing_move_does_not_re_alert_every_poll(runner_factory):
    runner, sink = runner_factory(quotes=[make_quote(price=108.0, prev=100.0)])
    for i in range(4):
        runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=i))
    assert len(sink.moves) == 1


def test_moves_are_not_alerted_overnight(runner_factory):
    """An overnight quote is the previous close restated."""
    runner, _ = runner_factory(quotes=[make_quote(price=108.0, prev=100.0)])
    overnight = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)  # 00:00 ET
    stats = runner.poll_once(now_utc=overnight)
    assert stats.move_alerts == 0


# -- degradation ------------------------------------------------------------


def test_a_failing_source_degrades_rather_than_raising(runner_factory):
    runner, sink = runner_factory(items=[], source_error="NVDA: HTTP 503")
    stats = runner.poll_once(now_utc=MIDSESSION)
    assert stats.degraded and "503" in stats.degraded[0]
    assert len(sink.briefs) == 1


def test_a_failing_quote_source_does_not_stop_news_alerts(runner_factory):
    runner, sink = runner_factory(items=[], quote_error="all: HTTP 429")
    runner.poll_once(now_utc=MIDSESSION)

    runner.sources[0].items = [make_item(source="sec")]
    stats = runner.poll_once(now_utc=MIDSESSION + timedelta(minutes=1))

    assert len(sink.items) == 1
    assert sink.items[0][1] is None  # no quote, alert still sent
    assert any("quotes" in d for d in stats.degraded)


def test_the_degraded_note_appears_in_the_brief(runner_factory):
    runner, sink = runner_factory(items=[], source_error="NVDA: HTTP 503")
    runner.poll_once(now_utc=MIDSESSION)
    assert sink.briefs[0].degraded


# -- the pre-open brief -----------------------------------------------------


def test_the_pre_open_brief_fires_once_per_session_day(runner_factory):
    runner, sink = runner_factory(items=[make_item()])
    runner.poll_once(now_utc=PREOPEN)                          # primes, briefs
    runner.poll_once(now_utc=PREOPEN + timedelta(minutes=1))   # scheduled brief
    runner.poll_once(now_utc=PREOPEN + timedelta(minutes=2))   # no second one
    assert len(sink.briefs) == 2


def test_no_pre_open_brief_before_the_scheduled_time(runner_factory):
    runner, sink = runner_factory(items=[])
    early = datetime(2026, 8, 12, 11, 0, tzinfo=timezone.utc)  # 07:00 ET
    runner.poll_once(now_utc=early)
    runner.poll_once(now_utc=early + timedelta(minutes=1))
    assert len(sink.briefs) == 1  # the priming brief only


# -- sleep ------------------------------------------------------------------


def test_sleep_uses_the_phase_cadence(runner_factory):
    runner, _ = runner_factory()
    assert runner.sleep_seconds(MIDSESSION) == CADENCE.regular


def test_sleep_never_overshoots_the_opening_bell(runner_factory):
    """At 09:20 ET the overnight cadence would sleep past the open."""
    runner, _ = runner_factory()
    just_before_open = datetime(2026, 8, 12, 13, 20, tzinfo=timezone.utc)
    assert runner.sleep_seconds(just_before_open) <= 10 * 60 + 1


def test_sleep_is_slower_when_the_market_is_closed(runner_factory):
    runner, _ = runner_factory()
    saturday = datetime(2026, 8, 15, 18, 0, tzinfo=timezone.utc)
    assert runner.sleep_seconds(saturday) > CADENCE.regular


def test_a_cadence_override_is_respected(runner_factory):
    runner, _ = runner_factory(cadence_override=5)
    assert runner.sleep_seconds(MIDSESSION) == 5


# -- brief assembly and rendering ------------------------------------------


def test_the_brief_orders_symbols_by_absolute_move():
    brief = build_brief(
        phase="premarket",
        phase_note="pre-market",
        quotes={
            "NVDA": make_quote("NVDA", 101.0, 100.0),
            "TSLA": make_quote("TSLA", 90.0, 100.0),
        },
        items=[],
        symbols=SYMBOLS,
    )
    assert [s.symbol for s in brief.movers] == ["TSLA", "NVDA"]


def test_the_brief_drops_items_below_the_threshold():
    from roth.news.score import apply_scores

    items = apply_scores([make_item(source="sec"), make_item(title="5 stocks to buy now")])
    brief = build_brief("regular", "", {}, items, SYMBOLS, min_score=40)
    assert all(i.score >= 40 for s in brief.symbols for i in s.items)


def test_a_symbol_without_a_quote_still_renders():
    brief = build_brief("regular", "", {}, [], SYMBOLS)
    assert "price unavailable" in render_brief_text(brief)


def test_rendering_an_empty_brief_says_so_rather_than_looking_broken():
    brief = build_brief("regular", "session", {"NVDA": make_quote()}, [], SYMBOLS)
    assert "No qualifying news" in render_brief_text(brief)


def test_the_one_line_alert_fits_a_notification():
    item = make_item(title="x" * 400)
    assert len(render_alert_oneline(item, make_quote())) <= 200


def test_the_one_line_alert_names_the_form_for_a_filing():
    assert render_alert_oneline(make_item(source="sec"), None).startswith("NVDA 8-K")


# -- sinks ------------------------------------------------------------------


def test_the_jsonl_sink_appends_one_record_per_alert(tmp_path):
    import json

    path = tmp_path / "stream.jsonl"
    sink = JsonlSink(path=path)
    sink.emit_item(make_item(), make_quote())
    sink.emit_move(make_quote())

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in records] == ["item", "move"]
    assert records[0]["symbol"] == "NVDA"


def test_a_sink_write_failure_is_counted_rather_than_raised(tmp_path):
    sink = JsonlSink(path=tmp_path / "nonexistent-dir" / "x" / "stream.jsonl")
    sink.path = tmp_path / "no" / "such" / "dir" / "stream.jsonl"
    sink.emit_move(make_quote())
    assert sink.failures == 1


def test_an_injected_empty_seen_store_is_not_silently_replaced(tmp_path):
    """`SeenStore` defines `__len__`, so an empty one is falsy.

    Written after `store or SeenStore.load(...)` quietly swapped the throwaway
    store that `news brief` and `--replay` rely on for the real persistent one,
    which made both of them hide exactly the items they exist to show.
    """
    store = SeenStore(path=tmp_path / "injected.json")
    assert not store  # the falsy empty store that caused the bug

    options = RuntimeOptions(
        symbols=SYMBOLS, toggles=SourceToggles(sec_edgar=False, yahoo_rss=False, quotes=False)
    )
    runner = NewsRunner(options, [RecordingSink()], store=store, http=None)
    assert runner.store is store


def test_the_runner_reports_stats_through_the_callback(runner_factory, tmp_path):
    seen = []
    options = RuntimeOptions(
        symbols=SYMBOLS, toggles=SourceToggles(sec_edgar=False, yahoo_rss=False, quotes=False)
    )
    runner = NewsRunner(
        options, [RecordingSink()], http=None,
        store=SeenStore(path=tmp_path / "seen.json"), on_cycle=seen.append,
    )
    runner.sources = [FakeSource(items=[make_item()])]
    runner.quote_source = None
    runner.poll_once(now_utc=MIDSESSION)

    assert len(seen) == 1 and seen[0].fetched == 1
