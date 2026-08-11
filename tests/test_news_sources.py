"""Parser tests for every news source.

These matter more than usual. The environment this bot was built in blocks
outbound access to every finance and news host, so no parser could be run
against its live endpoint during development. The fixtures below are the
recorded shapes the parsers were written against, and `roth news doctor` is
what confirms the live endpoints still match them.

So each fixture is kept deliberately faithful to the real payload -- including
the parts that are awkward: EDGAR's ragged parallel arrays and its Eastern-time
`Z` suffix, Yahoo's escaped HTML and tracking parameters, and the null padding
in a chart response.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from roth.news.config import ticker
from roth.news.models import canonical_url
from roth.news.sources.quotes import QuoteParseError, parse_chart
from roth.news.sources.rss import FeedParseError, parse_feed, parse_datetime, strip_html
from roth.news.sources.sec_edgar import (
    _filing_urls,
    _parse_acceptance,
    describe_items,
    fetch_official_cik_map,
    parse_submissions,
)
from roth.news.sources.yahoo_rss import parse_headlines

NVDA = ticker("NVDA")


# ===========================================================================
# SEC EDGAR
# ===========================================================================


def submissions_payload(**overrides) -> dict:
    """A faithful slice of data.sec.gov/submissions/CIK0001045810.json.

    The arrays are parallel and ordered newest first, exactly as EDGAR sends
    them. `items` is populated only for 8-K rows; every other form leaves it
    empty, which is why the parser must index rather than zip.
    """
    recent = {
        "accessionNumber": [
            "0001045810-26-000123",
            "0001045810-26-000122",
            "0001045810-26-000121",
            "0001045810-26-000120",
        ],
        "filingDate": ["2026-08-10", "2026-08-10", "2026-08-07", "2026-08-06"],
        "reportDate": ["2026-08-10", "", "2026-07-27", ""],
        "acceptanceDateTime": [
            "2026-08-10T16:31:22.000Z",
            "2026-08-10T18:02:11.000Z",
            "2026-08-07T16:05:44.000Z",
            "2026-08-06T09:30:01.000Z",
        ],
        "form": ["8-K", "4", "10-Q", "SC 13G/A"],
        "items": ["2.02,9.01", "", "", ""],
        "primaryDocument": [
            "nvda-20260810.htm",
            "xslF345X05/wk-form4.xml",
            "nvda-20260727.htm",
            "sc13ga.htm",
        ],
        "primaryDocDescription": ["8-K", "FORM 4", "10-Q", "SC 13G/A"],
    }
    recent.update(overrides)
    return {"cik": "1045810", "name": "NVIDIA CORP", "filings": {"recent": recent}}


def test_only_requested_forms_are_returned():
    items = parse_submissions(submissions_payload(), NVDA, ("8-K",), since=None)
    assert [i.kind for i in items] == ["8-K"]


def test_amendments_count_as_their_base_form():
    """SC 13G/A is an amendment to SC 13G and is just as much news."""
    items = parse_submissions(submissions_payload(), NVDA, ("SC 13G",), since=None)
    assert [i.kind for i in items] == ["SC 13G/A"]


def test_eight_k_item_codes_become_tags_and_a_readable_title():
    items = parse_submissions(submissions_payload(), NVDA, ("8-K",), since=None)
    item = items[0]
    assert "item:2.02" in item.tags
    assert "item:9.01" in item.tags
    assert "results of operations" in item.title


def test_filing_url_drops_leading_zeros_and_accession_dashes():
    doc, index = _filing_urls("0001045810", "0001045810-26-000123", "nvda-20260810.htm")
    assert doc == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000123/nvda-20260810.htm"
    )
    assert index.endswith("0001045810-26-000123-index.htm")


def test_filing_without_a_primary_document_falls_back_to_the_index():
    doc, index = _filing_urls("0001045810", "0001045810-26-000123", "")
    assert doc == index


def test_since_filter_excludes_older_filings():
    since = datetime(2026, 8, 8, tzinfo=timezone.utc)
    items = parse_submissions(
        submissions_payload(), NVDA, ("8-K", "4", "10-Q", "SC 13G"), since=since
    )
    assert all(i.published_utc >= since for i in items)
    assert "10-Q" not in [i.kind for i in items]


def test_ragged_arrays_do_not_crash_the_parser():
    """A company with few filings can return arrays of differing length."""
    payload = submissions_payload(items=[], primaryDocDescription=[], primaryDocument=[])
    items = parse_submissions(payload, NVDA, ("8-K",), since=None)
    assert len(items) == 1
    assert items[0].url.endswith("-index.htm")


def test_empty_payload_yields_nothing():
    assert parse_submissions({}, NVDA, ("8-K",)) == []
    assert parse_submissions({"filings": {"recent": {}}}, NVDA, ("8-K",)) == []


def test_missing_acceptance_time_falls_back_to_the_filing_date():
    payload = submissions_payload(acceptanceDateTime=["", "", "", ""])
    items = parse_submissions(payload, NVDA, ("8-K",), since=None)
    # 17:30 ET on the filing date, i.e. 21:30 UTC in August.
    assert items[0].published_utc == datetime(2026, 8, 10, 21, 30, tzinfo=timezone.utc)


def test_describe_items_maps_codes_to_labels():
    codes, labels = describe_items("2.02,9.01")
    assert codes == ("2.02", "9.01")
    assert labels[0] == "results of operations (earnings)"


def test_describe_items_tolerates_prose_and_duplicates():
    codes, _ = describe_items("Item 5.02 Departure of Directors; Item 5.02 again")
    assert codes == ("5.02",)


# -- the acceptanceDateTime trap -------------------------------------------


def test_acceptance_time_is_read_as_eastern_despite_the_z_suffix():
    """EDGAR stamps Eastern wall time and appends `Z`.

    A filing accepted at 16:31 ET on an August day is 20:31 UTC. Believing the
    suffix would place it at 16:31 UTC -- before the 16:00 ET close it actually
    followed.
    """
    now = datetime(2026, 8, 10, 23, 0, tzinfo=timezone.utc)
    parsed, reading = _parse_acceptance("2026-08-10T16:31:22.000Z", now=now)
    assert reading == "eastern"
    assert parsed == datetime(2026, 8, 10, 20, 31, 22, tzinfo=timezone.utc)


def test_acceptance_time_falls_back_to_utc_when_eastern_would_be_in_the_future():
    """The self-correcting guard: if SEC ever starts sending real UTC, the
    Eastern reading lands hours ahead of now and is rejected."""
    now = datetime(2026, 8, 10, 16, 32, tzinfo=timezone.utc)
    parsed, reading = _parse_acceptance("2026-08-10T16:31:22.000Z", now=now)
    assert reading == "utc"
    assert parsed == datetime(2026, 8, 10, 16, 31, 22, tzinfo=timezone.utc)


def test_acceptance_time_handles_the_winter_offset():
    now = datetime(2026, 1, 15, 23, 0, tzinfo=timezone.utc)
    parsed, reading = _parse_acceptance("2026-01-15T16:31:22.000Z", now=now)
    assert reading == "eastern"
    assert parsed == datetime(2026, 1, 15, 21, 31, 22, tzinfo=timezone.utc)


def test_unparseable_acceptance_time_raises():
    from roth.news.http import FetchError

    with pytest.raises(FetchError):
        _parse_acceptance("not a timestamp")


# -- the CIK map ------------------------------------------------------------


class _FakeHttp:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self, url, params=None, headers=None):
        return self.payload


def test_official_cik_map_is_zero_padded_to_ten_digits():
    payload = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    }
    mapping = fetch_official_cik_map(_FakeHttp(payload))
    assert mapping["AAPL"] == "0000320193"
    assert mapping["NVDA"] == "0001045810"


def test_official_cik_map_skips_malformed_rows():
    payload = {"0": {"ticker": "AAPL"}, "1": {"cik_str": 1045810, "ticker": "NVDA"}}
    assert fetch_official_cik_map(_FakeHttp(payload)) == {"NVDA": "0001045810"}


# ===========================================================================
# RSS
# ===========================================================================

YAHOO_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Yahoo! Finance: NVDA News</title>
    <link>https://finance.yahoo.com/quote/NVDA</link>
    <item>
      <title>Nvidia tops estimates, raises guidance</title>
      <link>https://www.reuters.com/technology/nvidia-q2-2026-08-10/?utm_source=rss&amp;guccounter=1</link>
      <pubDate>Mon, 10 Aug 2026 20:31:00 +0000</pubDate>
      <description>&lt;p&gt;Nvidia reported &lt;b&gt;record&lt;/b&gt; data centre revenue.&lt;/p&gt;</description>
      <guid isPermaLink="false">nvda-2026-08-10-earnings</guid>
    </item>
    <item>
      <title>3 reasons Nvidia stock is a buy right now</title>
      <link>https://www.fool.com/investing/2026/08/10/nvidia-buy/</link>
      <pubDate>Mon, 10 Aug 2026 18:00:00 +0000</pubDate>
      <description>Our analyst weighs in.</description>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Filings</title>
  <entry>
    <title>8-K - NVIDIA CORP</title>
    <link rel="self" href="https://example.com/self"/>
    <link rel="alternate" href="https://www.sec.gov/Archives/edgar/data/1045810/x.htm"/>
    <updated>2026-08-10T16:31:22-04:00</updated>
    <summary type="html">Filed &lt;b&gt;today&lt;/b&gt;</summary>
  </entry>
</feed>
"""


def test_rss_entries_are_parsed_with_html_stripped():
    entries = parse_feed(YAHOO_RSS)
    assert len(entries) == 2
    assert entries[0].title == "Nvidia tops estimates, raises guidance"
    assert entries[0].summary == "Nvidia reported record data centre revenue."
    assert entries[0].published_utc == datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc)


def test_atom_entries_prefer_the_alternate_link():
    entries = parse_feed(ATOM_FEED)
    assert len(entries) == 1
    assert entries[0].link.endswith("/x.htm")
    assert entries[0].published_utc == datetime(2026, 8, 10, 20, 31, 22, tzinfo=timezone.utc)


def test_a_non_xml_body_raises_rather_than_returning_junk():
    with pytest.raises(FeedParseError):
        parse_feed("<html><body>502 Bad Gateway</body></html>error")


def test_an_empty_body_is_simply_no_entries():
    assert parse_feed("") == []
    assert parse_feed("   ") == []


def test_entries_without_a_title_are_dropped():
    feed = '<?xml version="1.0"?><rss><channel><item><link>x</link></item></channel></rss>'
    assert parse_feed(feed) == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Mon, 10 Aug 2026 20:31:00 +0000", datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc)),
        ("Mon, 10 Aug 2026 16:31:00 -0400", datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc)),
        ("2026-08-10T20:31:00Z", datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc)),
        ("2026-08-10T16:31:00-04:00", datetime(2026, 8, 10, 20, 31, tzinfo=timezone.utc)),
    ],
)
def test_feed_timestamps_normalise_to_utc(raw, expected):
    assert parse_datetime(raw) == expected


def test_unparseable_timestamps_return_none_rather_than_raising():
    assert parse_datetime("last tuesday") is None
    assert parse_datetime("") is None
    assert parse_datetime(None) is None


def test_strip_html_truncates_long_summaries():
    assert strip_html("<p>" + "x" * 500 + "</p>", limit=50).endswith("…")


# -- the Yahoo headline source ---------------------------------------------


def test_headlines_become_news_items():
    items = parse_headlines(YAHOO_RSS, "NVDA", since=None)
    assert len(items) == 2
    assert items[0].symbol == "NVDA"
    assert items[0].source == "yahoo"


def test_tracking_parameters_do_not_change_an_items_identity():
    """Yahoo appends a fresh `guccounter` per request. Without canonicalising
    it, the same article alerts on every single poll."""
    first = parse_headlines(YAHOO_RSS, "NVDA", since=None)[0]
    second = parse_headlines(
        YAHOO_RSS.replace("guccounter=1", "guccounter=2"), "NVDA", since=None
    )[0]
    assert first.item_id == second.item_id


def test_canonical_url_strips_tracking_and_normalises_host():
    assert canonical_url("https://WWW.Example.com/a/?utm_source=x&b=2") == (
        "https://example.com/a?b=2"
    )


def test_undated_entries_are_stamped_now_rather_than_dropped():
    feed = (
        '<?xml version="1.0"?><rss><channel><item>'
        "<title>Breaking</title><link>https://x.com/a</link>"
        "</item></channel></rss>"
    )
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    items = parse_headlines(feed, "NVDA", since=None, now=now)
    assert items[0].published_utc == now


def test_a_feed_clock_running_fast_is_clamped_to_now():
    """Otherwise a future-dated item pins itself to the top of every brief."""
    now = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    feed = YAHOO_RSS.replace(
        "Mon, 10 Aug 2026 20:31:00 +0000", "Tue, 11 Aug 2026 23:00:00 +0000"
    )
    items = parse_headlines(feed, "NVDA", since=None, now=now)
    assert max(i.published_utc for i in items) <= now


# ===========================================================================
# Quotes
# ===========================================================================


def chart_payload(last_index: int = 2, **meta_overrides) -> dict:
    """A slice of query1.finance.yahoo.com/v8/finance/chart/NVDA.

    Bars run across the pre-market and regular windows for 2026-08-11, with
    Yahoo's characteristic null padding on bars that had no trades.
    """
    def epoch(text: str) -> int:
        return int(datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp())

    timestamps = [
        epoch("2026-08-11T12:00"),  # 08:00 ET, pre-market
        epoch("2026-08-11T13:00"),  # 09:00 ET, pre-market
        epoch("2026-08-11T14:00"),  # 10:00 ET, regular session
        epoch("2026-08-11T21:00"),  # 17:00 ET, post market
    ]
    closes = [101.0, 102.5, 108.0, 110.0]

    meta = {
        "currency": "USD",
        "symbol": "NVDA",
        "regularMarketPrice": 108.0,
        "chartPreviousClose": 100.0,
        "previousClose": 100.0,
        "regularMarketDayHigh": 109.0,
        "regularMarketDayLow": 99.5,
        "regularMarketVolume": 41_000_000,
        "currentTradingPeriod": {
            "pre": {"start": epoch("2026-08-11T08:00"), "end": epoch("2026-08-11T13:30")},
            "regular": {"start": epoch("2026-08-11T13:30"), "end": epoch("2026-08-11T20:00")},
            "post": {"start": epoch("2026-08-11T20:00"), "end": epoch("2026-08-12T00:00")},
        },
    }
    meta.update(meta_overrides)

    # Everything after `last_index` is null, the way Yahoo pads a partial day.
    padded = [c if i <= last_index else None for i, c in enumerate(closes)]
    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": timestamps,
                    "indicators": {"quote": [{"close": padded}]},
                }
            ],
            "error": None,
        }
    }


def test_regular_session_quote_uses_the_regular_price():
    quote = parse_chart(chart_payload(last_index=2), "NVDA")
    assert quote.market_state == "REGULAR"
    assert quote.extended_price is None
    assert quote.effective_price == 108.0
    assert quote.change_pct == pytest.approx(8.0)


def test_premarket_quote_reports_the_extended_print():
    """During pre-market the regular price is yesterday's. The extended print
    is the number that matters, and the move must be measured from it."""
    quote = parse_chart(chart_payload(last_index=1), "NVDA")
    assert quote.market_state == "PRE"
    assert quote.extended_price == 102.5
    assert quote.effective_price == 102.5
    assert quote.change_pct == pytest.approx(2.5)


def test_after_hours_quote_reports_the_post_market_print():
    quote = parse_chart(chart_payload(last_index=3), "NVDA")
    assert quote.market_state == "POST"
    assert quote.effective_price == 110.0
    assert quote.change_pct == pytest.approx(10.0)


def test_null_padded_bars_are_skipped_when_finding_the_last_print():
    payload = chart_payload(last_index=2)
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [101.0, None, None, None]
    quote = parse_chart(payload, "NVDA")
    assert quote.market_state == "PRE"
    assert quote.extended_price == 101.0


def test_a_chart_error_is_surfaced():
    payload = {"chart": {"result": None, "error": {"description": "No data found"}}}
    with pytest.raises(QuoteParseError, match="No data found"):
        parse_chart(payload, "NVDA")


def test_a_missing_previous_close_refuses_rather_than_inventing_a_move():
    payload = chart_payload()
    payload["chart"]["result"][0]["meta"].pop("previousClose")
    payload["chart"]["result"][0]["meta"].pop("chartPreviousClose")
    with pytest.raises(QuoteParseError, match="previous close"):
        parse_chart(payload, "NVDA")


def test_an_empty_result_list_raises():
    with pytest.raises(QuoteParseError):
        parse_chart({"chart": {"result": [], "error": None}}, "NVDA")


def test_zero_and_nan_prices_are_treated_as_missing():
    payload = chart_payload()
    payload["chart"]["result"][0]["meta"]["regularMarketPrice"] = 0
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"] = [0, None, None, None]
    with pytest.raises(QuoteParseError, match="no usable price"):
        parse_chart(payload, "NVDA")


def test_the_fixture_round_trips_as_json():
    """Guards against a fixture that only works because it is a Python dict."""
    assert parse_chart(json.loads(json.dumps(chart_payload())), "NVDA").symbol == "NVDA"
