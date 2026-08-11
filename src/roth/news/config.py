"""Watchlist, session windows, and polling cadence for the news bot.

Everything tunable lives here. The CIK numbers are the one piece of data in
this file that cannot be guessed: SEC EDGAR is keyed on CIK, not ticker, and a
wrong CIK silently returns another company's filings rather than an error. They
are pinned as constants and cross-checked against SEC's own ticker mapping by
`roth news doctor`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time

MARKET_TZ = "America/New_York"


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ticker:
    symbol: str
    name: str
    # SEC Central Index Key, zero-padded to 10 digits as EDGAR expects.
    cik: str
    # Extra phrases that indicate news is about this company even when the
    # ticker is absent from the headline.
    aliases: tuple[str, ...] = ()


WATCHLIST: tuple[Ticker, ...] = (
    Ticker("NVDA", "NVIDIA Corporation", "0001045810", ("nvidia", "jensen huang")),
    Ticker("TSLA", "Tesla, Inc.", "0001318605", ("tesla", "elon musk")),
    Ticker("META", "Meta Platforms, Inc.", "0001326801", ("meta platforms", "facebook", "instagram")),
    Ticker("AAPL", "Apple Inc.", "0000320193", ("apple", "iphone", "tim cook")),
    Ticker("AMZN", "Amazon.com, Inc.", "0001018724", ("amazon", "aws", "andy jassy")),
    Ticker("MSFT", "Microsoft Corporation", "0000789019", ("microsoft", "azure", "satya nadella")),
    Ticker("AVGO", "Broadcom Inc.", "0001730168", ("broadcom", "hock tan", "vmware")),
    Ticker("GOOGL", "Alphabet Inc.", "0001652044", ("alphabet", "google", "sundar pichai", "deepmind")),
)

SYMBOLS: tuple[str, ...] = tuple(t.symbol for t in WATCHLIST)

BY_SYMBOL: dict[str, Ticker] = {t.symbol: t for t in WATCHLIST}


def ticker(symbol: str) -> Ticker:
    try:
        return BY_SYMBOL[symbol.upper()]
    except KeyError:
        raise KeyError(
            f"{symbol!r} is not on the watchlist. Known: {', '.join(SYMBOLS)}"
        ) from None


# ---------------------------------------------------------------------------
# Session windows
# ---------------------------------------------------------------------------
#
# The regular session open and close come from the exchange calendar, never
# from these constants, because early closes move the close to 13:00 and DST
# moves both in UTC. What the calendar does not give is the extended-hours
# window, which is a broker convention rather than an exchange session. Those
# two are pinned here.

PREMARKET_OPEN = time(4, 0)
AFTERHOURS_CLOSE = time(20, 0)

# How long before the open the bot switches from "overnight" to the louder
# pre-market cadence, and starts building the open-prep brief.
PREOPEN_BRIEF_AT = time(8, 0)


# ---------------------------------------------------------------------------
# Polling cadence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cadence:
    """Seconds between polls in each phase of the day.

    These are chosen against the two rate limits that actually bind: SEC EDGAR
    asks for no more than 10 requests/second sustained, and Yahoo's RSS starts
    returning 429 well before that. One poll cycle is roughly two requests per
    symbol, so a 60-second cycle on eight symbols is ~0.27 req/s. There is a
    lot of headroom; the limits are not the reason these are not lower.
    """

    premarket: int = 60
    regular: int = 60
    afterhours: int = 180
    overnight: int = 900
    closed: int = 1800

    def for_phase(self, phase: str) -> int:
        return {
            "premarket": self.premarket,
            "regular": self.regular,
            "afterhours": self.afterhours,
            "overnight": self.overnight,
            "closed": self.closed,
        }[phase]


CADENCE = Cadence()


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertPolicy:
    """What is loud enough to interrupt you.

    Materiality runs 0-100. The default floor is deliberately high during the
    regular session: eight mega-caps generate a constant drizzle of syndicated
    rewrites, and an alert stream you learn to ignore is worse than none.
    """

    min_score_premarket: int = 25
    min_score_regular: int = 40
    min_score_afterhours: int = 30

    # Price move (percent, absolute) that is itself worth an alert, checked
    # once per poll against the previous close.
    move_alert_pct: float = 2.0

    # Never re-alert the same move until it has extended by this much again.
    move_realert_step_pct: float = 1.0

    def min_score(self, phase: str) -> int:
        if phase == "premarket":
            return self.min_score_premarket
        if phase == "regular":
            return self.min_score_regular
        return self.min_score_afterhours


ALERTS = AlertPolicy()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
#
# SEC's access policy requires a declared User-Agent carrying a real contact
# address, and blocks unidentified clients. There is no sensible default for
# somebody else's email, so it is read from the environment and the SEC source
# refuses to run without it rather than getting the whole IP blocked.

SEC_CONTACT_ENV = "ROTH_SEC_CONTACT"

USER_AGENT_TEMPLATE = "roth-news/0.1 ({contact})"

REQUEST_TIMEOUT_SECONDS = 20.0
MAX_RETRIES = 3


def sec_contact() -> str | None:
    value = os.environ.get(SEC_CONTACT_ENV, "").strip()
    return value or None


def user_agent() -> str:
    return USER_AGENT_TEMPLATE.format(contact=sec_contact() or "contact-not-set")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceToggles:
    """Which sources run. All default sources are free and need no API key."""

    sec_edgar: bool = True
    yahoo_rss: bool = True
    quotes: bool = True

    # Forms worth waking up for. Form 4 (insider transactions) is included
    # because a cluster of them around news is informative, but it is scored
    # low so it does not fire an alert on its own.
    sec_forms: tuple[str, ...] = ("8-K", "10-Q", "10-K", "SC 13D", "SC 13G", "4")


SOURCES = SourceToggles()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
#
# The seen-store has to outlive the process, otherwise every restart replays
# the whole feed as new. Kept under data/news/ so it never mixes with the
# harness's immutable raw store.

STATE_DIRNAME = "news"
SEEN_FILENAME = "seen.json"
STREAM_FILENAME = "stream.jsonl"

# Items older than this are dropped from the seen-store on save. Feeds never
# reach back this far, so anything older can no longer be re-emitted as new.
SEEN_RETENTION_DAYS = 14


@dataclass
class RuntimeOptions:
    """Per-run overrides, set from CLI flags."""

    symbols: tuple[str, ...] = SYMBOLS
    min_score: int | None = None
    sinks: tuple[str, ...] = ("console",)
    webhook_url: str | None = None
    once: bool = False
    include_extended_hours: bool = True
    cadence_override: int | None = None
    toggles: SourceToggles = field(default_factory=lambda: SOURCES)
