"""The types that move between sources, scoring, and sinks.

Everything is timezone-aware and stored in UTC. Display conversion to
America/New_York happens in the renderer and nowhere else, which is the same
rule the research harness follows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Tracking parameters that change per-fetch and would otherwise make the same
# article look like a new one every poll.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "guccounter",
    "guce_referrer", "guce_referrer_sig", "yptr", ".tsrc", "soc_src", "soc_trk",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_url(url: str) -> str:
    """Strip tracking noise so the same article hashes to the same id.

    Yahoo in particular appends a per-request `guccounter` and re-orders query
    parameters, so a naive hash of the raw URL re-alerts the same headline on
    every single poll.
    """
    if not url:
        return ""
    parts = urlsplit(url.strip())
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() not in _TRACKING_PARAMS]
    kept.sort()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, urlencode(kept), ""))


def normalize_title(title: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation.

    Used for the fallback identity of items whose URL is absent or unstable,
    and for near-duplicate detection across syndicating outlets.
    """
    text = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class NewsItem:
    """One thing that happened, attributable to one symbol.

    An article mentioning three watchlist names becomes three items. That is
    intentional: alerts are per-symbol, and per-symbol is how you read them.
    """

    symbol: str
    source: str
    title: str
    url: str
    published_utc: datetime
    summary: str = ""
    # Free-form source-specific detail, e.g. the SEC form type.
    kind: str = ""
    tags: tuple[str, ...] = ()
    score: int = 0
    reasons: tuple[str, ...] = ()

    @property
    def item_id(self) -> str:
        """Stable identity across restarts and across polls.

        Deliberately does not include the timestamp: Yahoo restates publication
        times on edit, and including it would re-alert the same story.
        """
        basis = canonical_url(self.url) or normalize_title(self.title)
        digest = hashlib.sha256(f"{self.symbol}|{self.source}|{basis}".encode()).hexdigest()
        return digest[:16]

    @property
    def dedupe_key(self) -> str:
        """Cross-source identity, so a wire story syndicated by four outlets
        alerts once. Title-based, because the URLs genuinely differ."""
        title = normalize_title(self.title)
        return hashlib.sha256(f"{self.symbol}|{title}".encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "symbol": self.symbol,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "published_utc": self.published_utc.isoformat(),
            "summary": self.summary,
            "kind": self.kind,
            "tags": list(self.tags),
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Quote:
    """A price snapshot, including the extended-hours print when there is one."""

    symbol: str
    price: float
    previous_close: float
    currency: str = "USD"
    market_state: str = ""
    as_of_utc: datetime = field(default_factory=utcnow)
    # Set only outside the regular session; None during it.
    extended_price: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: int | None = None

    @property
    def effective_price(self) -> float:
        return self.extended_price if self.extended_price is not None else self.price

    @property
    def change(self) -> float:
        return self.effective_price - self.previous_close

    @property
    def change_pct(self) -> float:
        if not self.previous_close:
            return 0.0
        return 100.0 * self.change / self.previous_close

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "extended_price": self.extended_price,
            "previous_close": self.previous_close,
            "change_pct": round(self.change_pct, 4),
            "market_state": self.market_state,
            "as_of_utc": self.as_of_utc.isoformat(),
        }


@dataclass
class SourceResult:
    """What one source returned on one poll.

    Failure is a value here, not an exception, because a single dead feed must
    never take down the loop -- it degrades that symbol's coverage and says so.
    """

    source: str
    items: list[NewsItem] = field(default_factory=list)
    quotes: list[Quote] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None
