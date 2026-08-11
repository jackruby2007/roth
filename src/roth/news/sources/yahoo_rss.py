"""Yahoo Finance per-symbol headline feed.

Free, no key, one request per symbol. It is not an official API and carries no
uptime promise, which is why the bot treats it as one input among several and
why `roth news doctor` reports its health separately: if this feed goes dark,
SEC filings still arrive and the bot says the headline stream is degraded
rather than pretending the day was quiet.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from roth.news.config import ticker as lookup_ticker
from roth.news.http import FetchError, NewsHttp
from roth.news.models import NewsItem, SourceResult
from roth.news.sources.rss import FeedParseError, parse_feed

FEED_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"


def parse_headlines(
    body: str,
    symbol: str,
    since: datetime | None = None,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Turn one symbol's RSS body into news items.

    Entries with no timestamp are stamped with the current time rather than
    dropped: an undated headline is still news, and the dedupe store stops it
    from being re-emitted on the next poll.
    """
    now = now or datetime.now(timezone.utc)
    out: list[NewsItem] = []

    for entry in parse_feed(body):
        published = entry.published_utc or now
        # A feed clock running fast would otherwise park an item in the future
        # and keep it at the top of every brief.
        if published > now + timedelta(minutes=5):
            published = now
        if since is not None and published < since:
            continue
        out.append(
            NewsItem(
                symbol=symbol,
                source="yahoo",
                title=entry.title,
                url=entry.link,
                published_utc=published,
                summary=entry.summary,
                kind="headline",
                tags=("headline",),
            )
        )
    return out


class YahooRss:
    name = "yahoo"

    def __init__(self, lookback_hours: int = 48) -> None:
        self.lookback_hours = lookback_hours

    def fetch(self, http: NewsHttp, symbols: tuple[str, ...]) -> SourceResult:
        started = time.monotonic()
        result = SourceResult(source=self.name)
        since = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        failures: list[str] = []

        for symbol in symbols:
            lookup_ticker(symbol)  # reject anything not on the watchlist
            try:
                body = http.get_text(
                    FEED_URL, params={"s": symbol, "region": "US", "lang": "en-US"}
                )
                result.items.extend(parse_headlines(body, symbol, since=since))
            except (FetchError, FeedParseError) as exc:
                failures.append(f"{symbol}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a source must never raise
                failures.append(f"{symbol}: unexpected {type(exc).__name__}: {exc}")

        if failures:
            result.error = "; ".join(failures)
        result.elapsed_seconds = time.monotonic() - started
        return result
