"""The loop.

One cycle is: work out what phase of the day it is, poll every enabled source,
score what came back, drop what has already been said, and emit the rest to
every sink. Then sleep for as long as the phase allows -- but never past the
next phase boundary, so the bot is awake at 09:30 rather than eleven minutes
into the session.

Three behaviours here are the difference between a bot you keep running and one
you turn off after two days:

* **The first cycle primes, it does not alert.** Starting the bot at 11:00
  should not fire forty alerts for headlines from 09:00. The first cycle marks
  everything seen and prints a brief instead, so you get the state of the world
  once and only genuine arrivals after that.
* **A dead source degrades, it does not crash.** Sources return errors as
  values. The loop reports them, keeps the others running, and stops repeating
  the same complaint every minute.
* **Sleep is interruptible and phase-aware.** Ctrl-C returns immediately and
  the seen-store is flushed on the way out.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from roth.news.brief import Brief, build_brief
from roth.news.config import (
    ALERTS,
    CADENCE,
    PREOPEN_BRIEF_AT,
    SEEN_FILENAME,
    RuntimeOptions,
)
from roth.news.dedupe import SeenStore
from roth.news.http import NewsHttp
from roth.news.models import NewsItem, Quote
from roth.news.score import apply_scores
from roth.news.session import SessionClock
from roth.news.sinks import Sink
from roth.news.sources import build_sources
from roth.news.sources.quotes import YahooQuotes
from roth.paths import NEWS


@dataclass
class CycleStats:
    """What one poll did. Printed by `--verbose` and used by the tests."""

    started_utc: datetime
    phase: str
    fetched: int = 0
    new_items: int = 0
    alerted: int = 0
    move_alerts: int = 0
    quotes: int = 0
    degraded: list[str] = field(default_factory=list)
    primed: bool = False


class NewsRunner:
    """Owns the poll loop and everything that has to survive across cycles."""

    def __init__(
        self,
        options: RuntimeOptions,
        sinks: list[Sink],
        clock: SessionClock | None = None,
        http: NewsHttp | None = None,
        store: SeenStore | None = None,
        on_cycle=None,
    ) -> None:
        self.options = options
        self.sinks = sinks
        # Explicit None checks, not `or`. `SeenStore` defines `__len__`, so a
        # freshly constructed empty store is falsy -- and `store or default`
        # would silently discard exactly the throwaway store that `news brief`
        # and `--replay` pass in to bypass deduplication.
        self.clock = SessionClock() if clock is None else clock
        self.http = NewsHttp() if http is None else http
        self.store = SeenStore.load(NEWS / SEEN_FILENAME) if store is None else store
        self.on_cycle = on_cycle

        self.sources = build_sources(options.toggles)
        self.quote_source = YahooQuotes() if options.toggles.quotes else None

        self._stop = threading.Event()
        self._primed = False
        self._last_brief_day: date | None = None
        # A source that fails every minute should complain once, not sixty
        # times an hour. Tracks the last error text reported per source.
        self._reported_errors: dict[str, str] = {}
        self.latest_quotes: dict[str, Quote] = {}

    # -- control -----------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread, or a platform without SIGTERM.
                pass

    # -- one cycle ---------------------------------------------------------

    def poll_once(self, now_utc: datetime | None = None, prime: bool | None = None) -> CycleStats:
        now_utc = now_utc or datetime.now(timezone.utc)
        state = self.clock.state(now_utc)
        stats = CycleStats(started_utc=now_utc, phase=state.phase)

        priming = self._primed is False if prime is None else prime

        items: list[NewsItem] = []
        for source in self.sources:
            result = source.fetch(self.http, self.options.symbols)
            items.extend(result.items)
            stats.fetched += len(result.items)
            if not result.ok:
                note = f"{result.source}: {result.error}"
                stats.degraded.append(note)

        if self.quote_source is not None:
            qresult = self.quote_source.fetch(self.http, self.options.symbols)
            for quote in qresult.quotes:
                self.latest_quotes[quote.symbol] = quote
            stats.quotes = len(qresult.quotes)
            if not qresult.ok:
                stats.degraded.append(f"quotes: {qresult.error}")

        scored = apply_scores(items)
        fresh = self.store.filter_new(scored)
        stats.new_items = len(fresh)

        threshold = (
            self.options.min_score
            if self.options.min_score is not None
            else ALERTS.min_score(state.phase)
        )

        if priming:
            # Everything above has been marked seen. Show the picture once.
            stats.primed = True
            brief = self._make_brief(state, scored, stats.degraded, min_score=threshold)
            self._emit_brief(brief)
            self._primed = True
            self._note_errors(stats.degraded)
            self._finish(stats)
            return stats

        for item in fresh:
            if item.score < threshold:
                continue
            self._emit_item(item, self.latest_quotes.get(item.symbol))
            stats.alerted += 1

        stats.move_alerts = self._emit_move_alerts(state.phase)
        self._maybe_emit_scheduled_brief(state, scored, stats)
        self._note_errors(stats.degraded)
        self._finish(stats)
        return stats

    def _finish(self, stats: CycleStats) -> None:
        try:
            self.store.save()
        except OSError as exc:
            stats.degraded.append(f"seen-store: could not save: {exc}")
        if self.on_cycle is not None:
            self.on_cycle(stats)

    # -- emission ----------------------------------------------------------

    def _emit_item(self, item: NewsItem, quote: Quote | None) -> None:
        for sink in self.sinks:
            sink.emit_item(item, quote)

    def _emit_brief(self, brief: Brief) -> None:
        for sink in self.sinks:
            sink.emit_brief(brief)

    def _emit_move_alerts(self, phase: str) -> int:
        """Alert on a big move even when no headline explains it.

        Only during hours when a price is actually forming. An overnight quote
        is the previous close restated, and alerting on it would fire the same
        move every fifteen minutes until 04:00.
        """
        if phase not in ("premarket", "regular", "afterhours"):
            return 0

        count = 0
        for symbol, quote in self.latest_quotes.items():
            if self.store.should_alert_move(
                symbol, quote.change_pct, ALERTS.move_alert_pct, ALERTS.move_realert_step_pct
            ):
                for sink in self.sinks:
                    sink.emit_move(quote)
                self.store.mark_move(symbol, quote.change_pct)
                count += 1
        return count

    def _make_brief(self, state, scored: list[NewsItem], degraded: list[str], min_score: int) -> Brief:
        return build_brief(
            phase=state.phase,
            phase_note=state.describe(),
            quotes=self.latest_quotes,
            items=scored,
            symbols=self.options.symbols,
            min_score=min_score,
            degraded=degraded,
            generated_utc=state.now_utc,
        )

    def _maybe_emit_scheduled_brief(self, state, scored: list[NewsItem], stats: CycleStats) -> None:
        """The pre-open brief, once per trading day.

        Fires on the first cycle at or after 08:00 ET on a session day, which
        is late enough for the overnight tape to be complete and early enough
        to still be preparation rather than commentary.
        """
        if state.session is None or state.phase not in ("premarket", "overnight"):
            return
        day = state.session.day
        if self._last_brief_day == day:
            return

        import zoneinfo

        et_now = state.now_utc.astimezone(zoneinfo.ZoneInfo("America/New_York"))
        if et_now.date() != day or et_now.time() < PREOPEN_BRIEF_AT:
            return

        brief = self._make_brief(state, scored, stats.degraded, min_score=ALERTS.min_score_premarket)
        self._emit_brief(brief)
        self._last_brief_day = day

    def _note_errors(self, degraded: list[str]) -> None:
        """Report each distinct source failure once, and its recovery once."""
        seen_now: dict[str, str] = {}
        for note in degraded:
            source, _, detail = note.partition(": ")
            seen_now[source] = detail

        for source, detail in seen_now.items():
            if self._reported_errors.get(source) != detail:
                self._reported_errors[source] = detail
        for source in list(self._reported_errors):
            if source not in seen_now:
                self._reported_errors.pop(source, None)

    # -- loop --------------------------------------------------------------

    def sleep_seconds(self, now_utc: datetime | None = None) -> float:
        """How long to sleep, given the phase and the next phase boundary.

        Capped at the boundary so the bot is awake for the open rather than
        arriving whenever the overnight cadence happens to land.
        """
        state = self.clock.state(now_utc or datetime.now(timezone.utc))
        base = (
            self.options.cadence_override
            if self.options.cadence_override is not None
            else CADENCE.for_phase(state.phase)
        )
        to_change = state.seconds_to_next_change
        if to_change is None:
            return float(base)
        # One second past the boundary, so the next cycle is unambiguously in
        # the new phase rather than racing the clock at exactly 09:30:00.
        return float(max(1.0, min(base, to_change + 1.0)))

    def run(self) -> None:
        self.install_signal_handlers()
        try:
            while not self._stop.is_set():
                self.poll_once()
                if self.options.once:
                    return
                self._stop.wait(self.sleep_seconds())
        finally:
            self.close()

    def close(self) -> None:
        try:
            self.store.save()
        except OSError:
            pass
        self.http.close()
        for sink in self.sinks:
            sink.close()


def run_once(options: RuntimeOptions, sinks: list[Sink], prime: bool = False) -> CycleStats:
    """A single poll, used by `roth news once` and `roth news brief`."""
    runner = NewsRunner(options, sinks)
    try:
        return runner.poll_once(prime=prime)
    finally:
        runner.close()


def wait_for_market_open(clock: SessionClock, poll: float = 1.0) -> None:  # pragma: no cover
    """Block until the opening bell. Kept for interactive use."""
    while True:
        state = clock.state()
        if state.phase == "regular":
            return
        time.sleep(min(poll, max(1.0, state.seconds_to_open or poll)))
