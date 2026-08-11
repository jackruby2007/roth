"""Tests for materiality scoring.

The point of the score is to keep the alert stream readable, so the tests are
mostly comparisons rather than pinned numbers: an earnings 8-K must outrank a
Form 4, a Reuters scoop must outrank a Motley Fool listicle. Pinning exact
values would make the thresholds impossible to tune without a red test suite.
"""

from __future__ import annotations

from datetime import datetime, timezone

from roth.news.models import NewsItem
from roth.news.score import apply_scores, score_filing, score_headline, score_item

NOW = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)


def headline(title: str, url: str = "https://example.com/a", summary: str = "") -> NewsItem:
    return NewsItem(
        symbol="NVDA", source="yahoo", title=title, url=url,
        published_utc=NOW, summary=summary, kind="headline",
    )


def filing(form: str, *codes: str) -> NewsItem:
    return NewsItem(
        symbol="NVDA", source="sec", title=f"NVDA: {form}", url="https://sec.gov/x",
        published_utc=NOW, kind=form, tags=("sec", f"form:{form}", *(f"item:{c}" for c in codes)),
    )


# -- filings ----------------------------------------------------------------


def test_an_earnings_8k_outranks_an_exhibits_only_8k():
    assert score_filing("8-K", ("2.02",)).score > score_filing("8-K", ("9.01",)).score


def test_the_item_code_overrides_the_form_score():
    """Item 1.03 is a bankruptcy notice. It must not be diluted by averaging
    against the generic 8-K score."""
    assert score_filing("8-K", ("1.03",)).score == 100


def test_the_strongest_item_wins_when_several_are_filed():
    assert score_filing("8-K", ("9.01", "2.02")).score == score_filing("8-K", ("2.02",)).score


def test_an_insider_form_4_scores_below_the_regular_session_threshold():
    """Form 4s arrive constantly and must not interrupt the session on their
    own, while still being visible in the brief."""
    from roth.news.config import ALERTS

    assert score_filing("4", ()).score < ALERTS.min_score_regular


def test_an_amendment_scores_below_the_original():
    assert score_filing("8-K/A", ("2.02",)).score < score_filing("8-K", ("2.02",)).score


def test_an_activist_stake_outranks_a_passive_one():
    assert score_filing("SC 13D", ()).score > score_filing("SC 13G", ()).score


def test_an_unknown_form_still_gets_a_usable_score():
    assert 0 < score_filing("S-8", ()).score <= 100


def test_filing_reasons_are_populated():
    assert score_filing("8-K", ("2.02",)).reasons


# -- headlines --------------------------------------------------------------


def test_a_guidance_cut_outranks_a_product_launch():
    assert (
        score_headline("Nvidia cuts guidance for the fourth quarter").score
        > score_headline("Nvidia unveils a new developer toolkit").score
    )


def test_a_listicle_is_pushed_below_the_regular_threshold():
    from roth.news.config import ALERTS

    assert (
        score_headline("3 reasons Nvidia stock is a buy right now").score
        < ALERTS.min_score_regular
    )


def test_an_if_you_invested_piece_scores_near_zero():
    assert score_headline("If you invested $1,000 in Nvidia in 2015").score <= 10


def test_a_reuters_scoop_outranks_the_same_story_from_a_content_farm():
    title = "Nvidia beats estimates and raises guidance"
    assert (
        score_headline(title, url="https://www.reuters.com/x").score
        > score_headline(title, url="https://www.fool.com/x").score
    )


def test_a_press_release_wire_counts_as_a_credible_domain():
    title = "Broadcom announces quarterly results"
    assert (
        score_headline(title, url="https://www.businesswire.com/x").score
        > score_headline(title, url="https://unknown-blog.example/x").score
    )


def test_subdomains_of_a_known_publisher_still_match():
    title = "Nvidia beats estimates"
    assert (
        score_headline(title, url="https://finance.reuters.com/x").score
        > score_headline(title, url="https://unknown-blog.example/x").score
    )


def test_only_the_strongest_keyword_counts():
    """A summary brushing several topics must not outrank a single hard fact."""
    broad = score_headline(
        "Nvidia news",
        summary="launch, partnership, price target, upgrade, dividend, earnings",
    )
    assert broad.score < score_filing("8-K", ("2.02",)).score


def test_noise_penalties_stack():
    """A listicle from a content farm is penalised twice.

    The headline carries a real keyword so the comparison happens above the
    floor -- two heavy penalties both clamp to zero and would compare equal.
    """
    title = "Nvidia to acquire Arm for $40 billion - here's why"
    one = score_headline(title)
    both = score_headline(title, url="https://www.fool.com/x")
    assert 0 < both.score < one.score
    assert any("explainer" in r for r in one.reasons)
    assert any("fool.com" in r for r in both.reasons)


def test_stacked_penalties_clamp_at_zero_rather_than_going_negative():
    scored = score_headline(
        "3 reasons Nvidia stock is a buy right now", url="https://www.fool.com/x"
    )
    assert scored.score == 0
    assert len(scored.reasons) >= 3  # base, listicle, domain


def test_scores_are_clamped_to_the_range():
    assert 0 <= score_headline("If you invested $1,000, here's why 5 stocks to buy").score <= 100
    assert 0 <= score_headline("Nvidia files for bankruptcy amid chapter 11").score <= 100


def test_headline_reasons_explain_the_score():
    reasons = score_headline("Nvidia cuts guidance", url="https://www.reuters.com/x").reasons
    assert any("guidance" in r for r in reasons)
    assert any("reuters.com" in r for r in reasons)


# -- dispatch and ordering --------------------------------------------------


def test_score_item_dispatches_on_source():
    assert score_item(filing("8-K", "2.02")).score == score_filing("8-K", ("2.02",)).score
    assert score_item(headline("Nvidia cuts guidance")).score == (
        score_headline("Nvidia cuts guidance", "", "https://example.com/a").score
    )


def test_apply_scores_sorts_highest_first_and_leaves_inputs_untouched():
    items = [headline("3 reasons Nvidia stock is a buy"), filing("8-K", "2.02")]
    scored = apply_scores(items)
    assert scored[0].kind == "8-K"
    assert scored[0].score > scored[1].score
    # The originals are frozen dataclasses and must not have been mutated.
    assert all(i.score == 0 for i in items)
