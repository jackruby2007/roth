"""A small RSS 2.0 / Atom parser.

Deliberately stdlib-only. `feedparser` would do this too, but it is a large
dependency for a job that is a few dozen lines when the feeds are known, and
this bot polls the same three feed shapes forever.

Parsing is defensive throughout: a feed that returns an HTML error page, an
empty body, or entries missing timestamps must produce zero items rather than
an exception, because that decision is what keeps one broken feed from taking
down the whole poll loop.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

ATOM_NS = "{http://www.w3.org/2005/Atom}"

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FeedEntry:
    title: str
    link: str
    published_utc: datetime | None
    summary: str
    source_name: str = ""


class FeedParseError(ValueError):
    """The body was not a feed at all."""


def strip_html(text: str, limit: int = 400) -> str:
    """Feed summaries arrive as escaped HTML fragments; alerts want prose."""
    if not text:
        return ""
    clean = _WS_RE.sub(" ", _TAG_RE.sub(" ", html.unescape(text))).strip()
    return clean[: limit - 1] + "…" if len(clean) > limit else clean


def parse_datetime(raw: str | None) -> datetime | None:
    """RFC 822 (RSS) or ISO 8601 (Atom), normalised to UTC.

    A feed timestamp without an offset is read as UTC. That is a guess, but the
    alternative -- dropping the item -- loses real news, and every feed this
    bot reads does send an offset.
    """
    if not raw:
        return None
    raw = raw.strip()
    for parser in (_rfc822, _iso8601):
        value = parser(raw)
        if value is not None:
            return value
    return None


def _rfc822(raw: str) -> datetime | None:
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso8601(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _text(node, *names: str) -> str:
    for name in names:
        found = node.find(name)
        if found is not None and (found.text or "").strip():
            return (found.text or "").strip()
    return ""


def parse_feed(body: str) -> list[FeedEntry]:
    """Entries from an RSS 2.0 or Atom document, newest ordering preserved."""
    if not body or not body.strip():
        return []
    try:
        root = ElementTree.fromstring(body.strip())
    except ElementTree.ParseError as exc:
        snippet = body.strip()[:120].replace("\n", " ")
        raise FeedParseError(f"not XML: {snippet!r}") from exc

    entries = _parse_rss(root)
    if not entries:
        entries = _parse_atom(root)
    return entries


def _parse_rss(root) -> list[FeedEntry]:
    out: list[FeedEntry] = []
    for item in root.iter("item"):
        title = strip_html(_text(item, "title"), limit=300)
        if not title:
            continue
        out.append(
            FeedEntry(
                title=title,
                link=_text(item, "link", "guid"),
                published_utc=parse_datetime(_text(item, "pubDate", "date")),
                summary=strip_html(_text(item, "description")),
                source_name=_text(item, "source"),
            )
        )
    return out


def _parse_atom(root) -> list[FeedEntry]:
    out: list[FeedEntry] = []
    for entry in root.iter(f"{ATOM_NS}entry"):
        title = strip_html(_text(entry, f"{ATOM_NS}title"), limit=300)
        if not title:
            continue

        # Atom puts the URL in an attribute, preferring rel="alternate".
        link = ""
        for node in entry.findall(f"{ATOM_NS}link"):
            rel = node.get("rel", "alternate")
            if rel == "alternate":
                link = node.get("href", "")
                break
            link = link or node.get("href", "")

        out.append(
            FeedEntry(
                title=title,
                link=link,
                published_utc=parse_datetime(
                    _text(entry, f"{ATOM_NS}updated", f"{ATOM_NS}published")
                ),
                summary=strip_html(
                    _text(entry, f"{ATOM_NS}summary", f"{ATOM_NS}content")
                ),
            )
        )
    return out
