"""How loud is this, on a scale of nothing to drop everything.

Eight mega-caps produce an enormous amount of text per day, and almost none of
it is news. The great majority is syndicated rewrites, "3 reasons to buy"
content farming, and analyst notes restating last week. An alert stream that
includes all of it is an alert stream you stop reading by Wednesday, which is
strictly worse than no bot at all.

So every item gets a 0-100 materiality score and the runner only interrupts you
above a threshold that varies by session phase -- low before the open, when you
are reading everything anyway; high mid-session, when an interruption costs
attention you are spending elsewhere.

The score is deliberately transparent rather than clever: every contribution
carries a reason string, and `roth news once --explain` prints them. A scoring
model you cannot interrogate is one you cannot correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

# ---------------------------------------------------------------------------
# SEC forms
# ---------------------------------------------------------------------------
#
# A filing is an event rather than a description of one, so filings start high.
# The 8-K item code says far more than the form type does: item 2.02 is the
# earnings release, item 8.01 is frequently a press release about nothing.

ITEM_SCORES: dict[str, int] = {
    "1.03": 100,  # bankruptcy
    "4.02": 95,   # previously issued financials cannot be relied upon
    "2.02": 92,   # results of operations
    "5.01": 90,   # change in control
    "2.06": 82,   # material impairment
    "3.01": 82,   # delisting / listing standard failure
    "5.02": 78,   # director or officer change
    "2.01": 76,   # completed acquisition or disposition
    "1.01": 72,   # material definitive agreement
    "1.02": 70,   # termination of a material agreement
    "2.05": 66,   # exit or disposal costs
    "2.03": 62,   # material financial obligation
    "2.04": 62,   # acceleration of an obligation
    "4.01": 60,   # change of auditor
    "3.02": 55,   # unregistered equity sale
    "5.03": 45,   # bylaws or fiscal year amendment
    "7.01": 58,   # Reg FD disclosure
    "5.07": 40,   # shareholder vote results
    "8.01": 55,   # other events
    "9.01": 30,   # exhibits only
}

FORM_SCORES: dict[str, int] = {
    "8-K": 70,
    "10-K": 74,
    "10-Q": 72,
    "SC 13D": 70,   # activist stake
    "SC 13G": 42,   # passive stake
    "4": 22,        # insider transaction
}


# ---------------------------------------------------------------------------
# Headline keywords
# ---------------------------------------------------------------------------

HEADLINE_BASE = 18

KEYWORD_RULES: tuple[tuple[str, int, str], ...] = (
    (r"\b(bankrupt|chapter 11|insolven)", 55, "distress"),
    (r"\b(halt(ed|s)? trading|trading halt|circuit breaker)", 50, "trading halt"),
    (r"\b(guidance|outlook|forecast)\b.{0,30}\b(cut|slash|lower|rais|hik|boost)", 42, "guidance change"),
    (r"\b(cuts?|slashes|lowers|raises|lifts|hikes)\b.{0,20}\bguidance\b", 42, "guidance change"),
    (r"\b(acqui(re|res|sition)|merger|takeover|buyout|to buy)\b", 40, "M&A"),
    (r"\b(recall|breach|hack|outage|defect|fire|explosion|crash)\b", 38, "incident"),
    (r"\b(sues?|lawsuit|antitrust|subpoena|probe|investigat|doj|ftc|sec charges)\b", 36, "legal or regulatory"),
    (r"\b(ceo|cfo|coo|president|chairman)\b.{0,30}\b(resign|steps? down|depart|ousted|fired|to leave)", 36, "executive change"),
    (r"\b(earnings|results|revenue|eps)\b.{0,25}\b(beat|miss|top|fell short)", 35, "earnings surprise"),
    (r"\b(beats?|misses)\b.{0,20}\b(estimates?|expectations?|consensus)", 35, "earnings surprise"),
    (r"\b(export (ban|curb|control|restriction)|sanction|tariff|blacklist)\b", 34, "policy"),
    (r"\b(buyback|repurchase|dividend|stock split)\b", 30, "capital return"),
    (r"\b(downgrade[sd]?|upgrade[sd]?)\b", 26, "rating change"),
    (r"\bprice target\b", 20, "price target"),
    (r"\b(layoffs?|job cuts|restructur|workforce reduction)\b", 24, "restructuring"),
    (r"\b(partnership|deal|contract|order)\b.{0,20}\b(billion|bn)\b", 30, "large contract"),
    (r"\b(delay|postpone|pause[sd]?|scrapp?ed|cancel)", 22, "delay or cancellation"),
    (r"\b(launch|unveil|announce)\w*\b", 14, "product news"),
    (r"\b(earnings|quarterly results)\b", 22, "earnings"),
    (r"\b(record high|all-time high|plunge|plummet|soar|surge|tumble)\b", 18, "price action"),
)

# Content farming and evergreen listicles. These are the bulk of the volume and
# essentially never the reason a stock moved.
NOISE_RULES: tuple[tuple[str, int, str], ...] = (
    (r"\b(\d+|three|five|two|seven|ten)\s+(top|best|great|reasons?|stocks?|things|charts)\b", 40, "listicle"),
    (r"\b(should you|is it time to|why .{0,40}\bis a (buy|sell)|better buy)\b", 40, "opinion piece"),
    (r"\b(if you (had )?invested|would be worth|turned \$)\b", 45, "if-you-invested piece"),
    (r"\b(prediction|forecast for 20\d\d|where will .{0,30}be in)\b", 35, "speculation"),
    (r"\b(here'?s (how|why|what)|what to know|explained)\b", 22, "explainer"),
    (r"\b(motley fool|zacks|jim cramer|cramer says)\b", 35, "commentary"),
    (r"\bstocks? to (buy|watch)\b", 40, "listicle"),
    (r"\b(dividend|retirement) (stocks?|portfolio)\b", 30, "evergreen"),
)

# Outlets that break news, and outlets that recycle it.
DOMAIN_BONUS: dict[str, int] = {
    "reuters.com": 16,
    "bloomberg.com": 16,
    "wsj.com": 16,
    "ft.com": 14,
    "cnbc.com": 12,
    "barrons.com": 10,
    "apnews.com": 12,
    "theinformation.com": 14,
    "axios.com": 8,
    "techcrunch.com": 6,
    "businesswire.com": 14,
    "prnewswire.com": 14,
    "globenewswire.com": 14,
}

DOMAIN_PENALTY: dict[str, int] = {
    "fool.com": 32,
    "zacks.com": 28,
    "investorplace.com": 28,
    "24/7wallst.com": 28,
    "247wallst.com": 28,
    "benzinga.com": 12,
    "simplywall.st": 25,
    "gurufocus.com": 25,
    "insidermonkey.com": 25,
    "talkmarkets.com": 25,
}

_COMPILED_KEYWORDS = [(re.compile(p, re.I), w, label) for p, w, label in KEYWORD_RULES]
_COMPILED_NOISE = [(re.compile(p, re.I), w, label) for p, w, label in NOISE_RULES]


@dataclass(frozen=True)
class Scored:
    score: int
    reasons: tuple[str, ...]


def _domain(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def score_filing(form: str, item_codes: tuple[str, ...]) -> Scored:
    """Score an SEC filing from its form type and 8-K item codes."""
    base_form = (form or "").upper().removesuffix("/A")
    reasons: list[str] = []

    score = FORM_SCORES.get(base_form, 50)
    reasons.append(f"form {base_form or 'unknown'} (+{score})")

    if item_codes:
        best = max(item_codes, key=lambda c: ITEM_SCORES.get(c, 0))
        item_score = ITEM_SCORES.get(best, 0)
        if item_score:
            # The item code is more informative than the form, so it replaces
            # the form score rather than stacking on top of it.
            reasons.append(f"8-K item {best} (={item_score})")
            score = item_score

    if (form or "").upper().endswith("/A"):
        # An amendment restates something already seen.
        score -= 10
        reasons.append("amendment (-10)")

    return Scored(max(0, min(100, score)), tuple(reasons))


def score_headline(title: str, summary: str = "", url: str = "") -> Scored:
    """Score a headline from its wording and its publisher."""
    text = f"{title} {summary}".strip()
    reasons: list[str] = [f"headline base (+{HEADLINE_BASE})"]
    score = HEADLINE_BASE

    # Only the strongest signal counts. Summing every match lets a long summary
    # brushing three topics outrank a filing.
    best_weight, best_label = 0, ""
    for pattern, weight, label in _COMPILED_KEYWORDS:
        if weight > best_weight and pattern.search(text):
            best_weight, best_label = weight, label
    if best_weight:
        score += best_weight
        reasons.append(f"{best_label} (+{best_weight})")

    # Noise penalties do stack: a listicle by a content farm is twice damned.
    for pattern, weight, label in _COMPILED_NOISE:
        if pattern.search(title):
            score -= weight
            reasons.append(f"{label} (-{weight})")

    domain = _domain(url)
    for host, bonus in DOMAIN_BONUS.items():
        if domain == host or domain.endswith(f".{host}"):
            score += bonus
            reasons.append(f"{host} (+{bonus})")
            break
    for host, penalty in DOMAIN_PENALTY.items():
        if domain == host or domain.endswith(f".{host}"):
            score -= penalty
            reasons.append(f"{host} (-{penalty})")
            break

    return Scored(max(0, min(100, score)), tuple(reasons))


def score_item(item) -> Scored:
    """Dispatch on source. Returns a new score; callers attach it to the item."""
    if item.source == "sec":
        codes = tuple(t.split(":", 1)[1] for t in item.tags if t.startswith("item:"))
        return score_filing(item.kind, codes)
    return score_headline(item.title, item.summary, item.url)


def apply_scores(items):
    """Return the items with `score` and `reasons` populated, highest first."""
    from dataclasses import replace

    scored = [replace(i, score=(s := score_item(i)).score, reasons=s.reasons) for i in items]
    scored.sort(key=lambda i: (i.score, i.published_utc), reverse=True)
    return scored
