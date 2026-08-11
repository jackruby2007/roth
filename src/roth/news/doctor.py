"""End-to-end verification of every source, run on the machine that will poll.

This command carries more weight than a normal health check. The bot was built
in an environment whose egress policy blocks every finance and news host, so no
source could be exercised against its live endpoint during development: the
parsers are covered by recorded fixtures, and this is what proves the live
endpoints still match those fixtures.

Run it before trusting the bot, and again whenever the output looks wrong. It
checks reachability, payload shape, and the two things that fail silently
rather than loudly:

* whether the pinned CIK constants still match SEC's own ticker mapping, and
* which timezone reading EDGAR's `acceptanceDateTime` actually needs today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from roth.news.config import SEC_CONTACT_ENV, WATCHLIST, sec_contact
from roth.news.http import NewsHttp
from roth.news.sources.quotes import CHART_URL, parse_chart
from roth.news.sources.sec_edgar import (
    SUBMISSIONS_URL,
    _parse_acceptance,
    fetch_official_cik_map,
    parse_submissions,
)
from roth.news.sources.yahoo_rss import FEED_URL, parse_headlines

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    lines: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def _check_sec_contact() -> Check:
    contact = sec_contact()
    if contact and "@" in contact:
        return Check("SEC contact", OK, f"{SEC_CONTACT_ENV}={contact}")
    if contact:
        return Check(
            "SEC contact", WARN,
            f"{SEC_CONTACT_ENV} is set but does not look like an email address",
        )
    return Check(
        "SEC contact", FAIL,
        f"{SEC_CONTACT_ENV} is not set",
        [
            "SEC requires a contact address in the User-Agent and blocks clients",
            "that do not send one. Set it before running the bot:",
            "",
            "    export ROTH_SEC_CONTACT='you@example.com'",
        ],
    )


def _check_cik_map(http: NewsHttp) -> Check:
    """The pinned CIKs against SEC's authoritative mapping.

    A wrong CIK is the single most dangerous error in this bot, because EDGAR
    answers it successfully with another company's filings. Nothing downstream
    would notice.
    """
    try:
        official = fetch_official_cik_map(http)
    except Exception as exc:  # noqa: BLE001
        return Check("CIK constants", WARN, f"could not fetch SEC ticker map: {exc}")

    mismatches, missing = [], []
    for tkr in WATCHLIST:
        actual = official.get(tkr.symbol)
        if actual is None:
            missing.append(tkr.symbol)
        elif actual != tkr.cik:
            mismatches.append(f"{tkr.symbol}: pinned {tkr.cik}, SEC says {actual}")

    if mismatches:
        return Check(
            "CIK constants", FAIL,
            f"{len(mismatches)} of {len(WATCHLIST)} do not match SEC",
            [*mismatches, "", "Correct them in src/roth/news/config.py before polling."],
        )
    if missing:
        return Check(
            "CIK constants", WARN,
            f"not in SEC's map: {', '.join(missing)}",
            ["SEC's ticker file omits some share classes; verify these by hand."],
        )
    return Check("CIK constants", OK, f"all {len(WATCHLIST)} match SEC's ticker map")


def _check_sec_filings(http: NewsHttp, symbol: str = "AAPL") -> Check:
    tkr = next(t for t in WATCHLIST if t.symbol == symbol)
    try:
        payload = http.get_json(SUBMISSIONS_URL.format(cik=tkr.cik))
    except Exception as exc:  # noqa: BLE001
        return Check("SEC filings", FAIL, f"{symbol}: {exc}")

    name = payload.get("name", "?")
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    if not forms:
        return Check("SEC filings", WARN, f"{symbol}: payload had no recent filings")

    items = parse_submissions(payload, tkr, ("8-K", "10-Q", "10-K", "4"), since=None)
    lines = [f"EDGAR name: {name}", f"recent filings in payload: {len(forms)}"]

    # Which reading of acceptanceDateTime the live data needs today.
    raw = (recent.get("acceptanceDateTime") or [""])[0]
    if raw:
        parsed, reading = _parse_acceptance(raw)
        lines.append(f"latest acceptanceDateTime: {raw!r} read as {reading} -> {parsed.isoformat()}")

    for item in items[:3]:
        lines.append(f"  {item.published_utc.isoformat()}  {item.kind:<6} {item.title}")

    if name.upper().split()[0] not in tkr.name.upper() and tkr.symbol != "GOOGL":
        return Check(
            "SEC filings", WARN,
            f"{symbol}: EDGAR returned {name!r}, expected {tkr.name!r}", lines,
        )
    return Check("SEC filings", OK, f"{symbol}: {len(items)} parsed filings", lines)


def _check_yahoo_rss(http: NewsHttp, symbol: str = "AAPL") -> Check:
    try:
        body = http.get_text(FEED_URL, params={"s": symbol, "region": "US", "lang": "en-US"})
    except Exception as exc:  # noqa: BLE001
        return Check(
            "Yahoo headlines", FAIL, f"{symbol}: {exc}",
            ["Headlines will be unavailable. SEC filings still work;",
             "the bot reports the stream as degraded rather than as quiet."],
        )

    try:
        items = parse_headlines(body, symbol)
    except Exception as exc:  # noqa: BLE001
        return Check("Yahoo headlines", FAIL, f"{symbol}: could not parse feed: {exc}",
                     [f"first 200 bytes: {body[:200]!r}"])

    if not items:
        return Check("Yahoo headlines", WARN, f"{symbol}: feed parsed but returned no entries",
                     [f"first 200 bytes: {body[:200]!r}"])

    lines = [f"  {i.published_utc.isoformat()}  {i.title[:80]}" for i in items[:3]]
    return Check("Yahoo headlines", OK, f"{symbol}: {len(items)} entries", lines)


def _check_quotes(http: NewsHttp, symbol: str = "AAPL") -> Check:
    try:
        payload = http.get_json(
            CHART_URL.format(symbol=symbol),
            params={"range": "5d", "interval": "5m", "includePrePost": "true"},
        )
    except Exception as exc:  # noqa: BLE001
        return Check("Quotes", FAIL, f"{symbol}: {exc}",
                     ["Alerts will carry no price context and move alerts will not fire."])

    meta = ((payload.get("chart") or {}).get("result") or [{}])[0].get("meta") or {}
    try:
        quote = parse_chart(payload, symbol)
    except Exception as exc:  # noqa: BLE001
        return Check("Quotes", FAIL, f"{symbol}: {exc}",
                     [f"meta keys present: {', '.join(sorted(meta))}"])

    return Check(
        "Quotes", OK,
        f"{symbol}: {quote.effective_price:.2f} ({quote.change_pct:+.2f}%) state={quote.market_state}",
        [
            f"previous close {quote.previous_close:.2f}, as of {quote.as_of_utc.isoformat()}",
            f"meta keys present: {', '.join(sorted(meta))}",
        ],
    )


def _check_state_dir() -> Check:
    from roth.paths import NEWS

    try:
        NEWS.mkdir(parents=True, exist_ok=True)
        probe = NEWS / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return Check("State directory", FAIL, f"{NEWS} is not writable: {exc}",
                     ["Without it the bot re-alerts everything on every restart."])
    return Check("State directory", OK, str(NEWS))


def _check_desktop() -> Check:
    from roth.news.sinks import DesktopSink

    sink = DesktopSink()
    if sink.available:
        return Check("Desktop notifications", OK, f"via {sink.mechanism}")
    return Check(
        "Desktop notifications", WARN, "no notification mechanism found",
        ["`--sink desktop` will do nothing on this machine.",
         "Linux needs notify-send (libnotify-bin); macOS uses osascript."],
    )


def run_checks(symbol: str = "AAPL", skip_network: bool = False) -> list[Check]:
    """Every check, in the order a failure should be read."""
    checks = [_check_sec_contact(), _check_state_dir(), _check_desktop()]
    if skip_network:
        return checks

    with NewsHttp() as http:
        checks.append(_check_cik_map(http))
        checks.append(_check_sec_filings(http, symbol))
        checks.append(_check_yahoo_rss(http, symbol))
        checks.append(_check_quotes(http, symbol))
    return checks


def summarize(checks: list[Check]) -> tuple[int, int, int]:
    ok = sum(1 for c in checks if c.status == OK)
    warn = sum(1 for c in checks if c.status == WARN)
    fail = sum(1 for c in checks if c.status == FAIL)
    return ok, warn, fail
