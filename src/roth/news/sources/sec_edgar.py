"""SEC EDGAR filings, via the structured submissions API.

This is the only authoritative source in the bot. Everything else is somebody's
description of an event; a filing *is* the event, timestamped by the SEC at the
moment it was accepted. For the kind of move that happens at 16:05 ET, the 8-K
lands here before any wire story exists.

`data.sec.gov/submissions/CIK##########.json` returns every recent filing for a
company in one request, which is why it is used instead of the per-form Atom
feeds: eight requests per poll covers the whole watchlist and every form type.

Two traps are handled explicitly:

* **CIK, not ticker.** EDGAR is keyed on CIK. A wrong CIK returns a different
  company's filings and never errors, so the numbers are pinned as constants
  and `roth news doctor` cross-checks them against SEC's own ticker mapping.
* **`acceptanceDateTime` carries a `Z` suffix but is Eastern Time.** Trusting
  the suffix backdates every filing by four or five hours, which silently
  reorders the day. See `_parse_acceptance`.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone

from roth.news.config import Ticker, ticker as lookup_ticker
from roth.news.http import FetchError, NewsHttp
from roth.news.models import NewsItem, SourceResult

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"

# 8-K item codes worth naming in an alert. The SEC's full list is long; these
# are the ones that move a mega-cap.
ITEM_LABELS: dict[str, str] = {
    "1.01": "material definitive agreement",
    "1.02": "termination of material agreement",
    "1.03": "bankruptcy or receivership",
    "2.01": "completion of acquisition or disposition",
    "2.02": "results of operations (earnings)",
    "2.03": "material direct financial obligation",
    "2.04": "acceleration of financial obligation",
    "2.05": "costs associated with exit or disposal",
    "2.06": "material impairment",
    "3.01": "delisting or listing standard failure",
    "3.02": "unregistered sale of equity",
    "4.01": "change in certifying accountant",
    "4.02": "non-reliance on previously issued financials",
    "5.01": "change in control",
    "5.02": "director or officer change",
    "5.03": "amendment to bylaws or fiscal year",
    "5.07": "shareholder vote results",
    "7.01": "Reg FD disclosure",
    "8.01": "other events",
    "9.01": "financial statements and exhibits",
}

_ITEM_CODE = re.compile(r"\b(\d\.\d{2})\b")


def _parse_acceptance(raw: str, now: datetime | None = None) -> tuple[datetime, str]:
    """Turn EDGAR's `acceptanceDateTime` into a real UTC instant.

    EDGAR emits `2026-08-10T16:31:22.000Z` for a filing accepted at 16:31
    *Eastern*, not 16:31 UTC. Taking the suffix at face value moves every
    filing four or five hours into the past, which puts an after-close 8-K
    before the close it followed.

    Rather than hardcode a belief about SEC's formatting, both readings are
    tested against the clock. Eastern is later in absolute terms than UTC for
    the same wall time, so a filing accepted minutes ago reads as several hours
    in the future if it is Eastern-interpreted wrongly. Anything landing in the
    future is therefore genuinely UTC. Returns the instant and which reading
    was used, so `doctor` can report it.
    """
    import zoneinfo

    now = now or datetime.now(timezone.utc)
    text = raw.strip().rstrip("Z")
    try:
        naive = datetime.fromisoformat(text)
    except ValueError:
        raise FetchError(f"Unparseable acceptanceDateTime: {raw!r}") from None
    naive = naive.replace(tzinfo=None)

    eastern = naive.replace(tzinfo=zoneinfo.ZoneInfo("America/New_York")).astimezone(timezone.utc)
    if eastern <= now + timedelta(minutes=2):
        return eastern, "eastern"
    return naive.replace(tzinfo=timezone.utc), "utc"


def _filing_urls(cik: str, accession: str, primary_document: str) -> tuple[str, str]:
    """(document url, filing index url) for one accession number."""
    cik_int = str(int(cik))
    nodash = accession.replace("-", "")
    index = f"{ARCHIVE_BASE}/{cik_int}/{nodash}/{accession}-index.htm"
    doc = f"{ARCHIVE_BASE}/{cik_int}/{nodash}/{primary_document}" if primary_document else index
    return doc, index


def describe_items(raw_items: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split an 8-K `items` string into codes and human labels."""
    codes = tuple(dict.fromkeys(_ITEM_CODE.findall(raw_items or "")))
    labels = tuple(ITEM_LABELS.get(c, f"item {c}") for c in codes)
    return codes, labels


def parse_submissions(
    payload: dict,
    tkr: Ticker,
    forms: tuple[str, ...],
    since: datetime | None = None,
    now: datetime | None = None,
) -> list[NewsItem]:
    """Turn one company's submissions payload into news items.

    Pure, so the whole parser is exercised by fixtures without a network.
    """
    recent = (payload.get("filings") or {}).get("recent") or {}
    if not recent:
        return []

    accession = recent.get("accessionNumber") or []
    form = recent.get("form") or []
    filing_date = recent.get("filingDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    primary = recent.get("primaryDocument") or []
    description = recent.get("primaryDocDescription") or []
    items = recent.get("items") or []

    wanted = {f.upper() for f in forms}
    out: list[NewsItem] = []

    # The arrays are parallel and ordered newest-first, but a short array is a
    # real possibility on a company with few filings, so every read is bounded.
    for i in range(len(accession)):
        form_type = (form[i] if i < len(form) else "") or ""
        # 8-K/A and 10-Q/A are amendments to a wanted form and matter as much.
        base_form = form_type.upper().removesuffix("/A")
        if base_form not in wanted:
            continue

        raw_accept = acceptance[i] if i < len(acceptance) else ""
        if raw_accept:
            published, _ = _parse_acceptance(raw_accept, now=now)
        elif i < len(filing_date) and filing_date[i]:
            # No acceptance time: fall back to the filing date at the close of
            # the EDGAR business day, which is late enough not to claim the
            # filing arrived before it did.
            import zoneinfo

            day = datetime.fromisoformat(filing_date[i])
            published = day.replace(
                hour=17, minute=30, tzinfo=zoneinfo.ZoneInfo("America/New_York")
            ).astimezone(timezone.utc)
        else:
            continue

        if since is not None and published < since:
            continue

        doc_url, index_url = _filing_urls(
            tkr.cik, accession[i], primary[i] if i < len(primary) else ""
        )

        codes, labels = describe_items(items[i] if i < len(items) else "")
        desc = (description[i] if i < len(description) else "") or ""

        title = f"{tkr.symbol}: {form_type} filed"
        if labels:
            title = f"{tkr.symbol}: {form_type} - {labels[0]}"
        elif desc:
            title = f"{tkr.symbol}: {form_type} - {desc}"

        summary_bits = [b for b in (desc, ", ".join(labels)) if b]
        out.append(
            NewsItem(
                symbol=tkr.symbol,
                source="sec",
                title=title,
                url=doc_url,
                published_utc=published,
                summary=" | ".join(summary_bits),
                kind=form_type,
                tags=("sec", f"form:{base_form}", *(f"item:{c}" for c in codes)),
            )
        )

    return out


class SecEdgar:
    """Polls EDGAR submissions for every watchlist company."""

    name = "sec"

    def __init__(self, forms: tuple[str, ...], lookback_hours: int = 48) -> None:
        self.forms = forms
        self.lookback_hours = lookback_hours
        self.acceptance_reading: str | None = None

    def fetch(self, http: NewsHttp, symbols: tuple[str, ...]) -> SourceResult:
        started = time.monotonic()
        result = SourceResult(source=self.name)
        since = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        failures: list[str] = []

        for symbol in symbols:
            tkr = lookup_ticker(symbol)
            try:
                payload = http.get_json(SUBMISSIONS_URL.format(cik=tkr.cik))
                result.items.extend(
                    parse_submissions(payload, tkr, self.forms, since=since)
                )
            except FetchError as exc:
                failures.append(f"{symbol}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a source must never raise
                failures.append(f"{symbol}: unexpected {type(exc).__name__}: {exc}")

        if failures:
            result.error = "; ".join(failures)
        result.elapsed_seconds = time.monotonic() - started
        return result


def fetch_official_cik_map(http: NewsHttp) -> dict[str, str]:
    """SEC's own ticker-to-CIK mapping, used by `doctor` to verify constants."""
    payload = http.get_json(COMPANY_TICKERS_URL)
    out: dict[str, str] = {}
    # Shaped as {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    rows = payload.values() if isinstance(payload, dict) else payload
    for row in rows:
        try:
            out[str(row["ticker"]).upper()] = f"{int(row['cik_str']):010d}"
        except (KeyError, TypeError, ValueError):
            continue
    return out
