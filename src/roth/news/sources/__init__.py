"""News and price sources.

Every source implements `fetch(http, symbols) -> SourceResult` and is expected
never to raise: a source that cannot reach its feed returns a result carrying
an error string. The runner reports degraded sources and keeps going, because
one dead RSS feed must not stop SEC filings from arriving.
"""

from __future__ import annotations

from typing import Protocol

from roth.news.http import NewsHttp
from roth.news.models import SourceResult


class Source(Protocol):
    name: str

    def fetch(self, http: NewsHttp, symbols: tuple[str, ...]) -> SourceResult: ...


from roth.news.sources.quotes import YahooQuotes  # noqa: E402
from roth.news.sources.sec_edgar import SecEdgar  # noqa: E402
from roth.news.sources.yahoo_rss import YahooRss  # noqa: E402

__all__ = ["Source", "SecEdgar", "YahooRss", "YahooQuotes", "build_sources"]


def build_sources(toggles=None) -> list[Source]:
    """The enabled sources, in the order their results should be read."""
    from roth.news.config import SOURCES

    toggles = toggles or SOURCES
    sources: list[Source] = []
    # SEC first: it is the only authoritative source here, and when a filing
    # and a press rewrite describe the same event the filing should win.
    if toggles.sec_edgar:
        sources.append(SecEdgar(forms=toggles.sec_forms))
    if toggles.yahoo_rss:
        sources.append(YahooRss())
    return sources
