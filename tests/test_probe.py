"""Tests for the ThetaData endpoint probe.

The probe is the one piece of code guaranteed to run against an API nobody here
has seen. It therefore has to survive anything the real server does -- wrong
paths, auth errors, empty bodies, plain-text responses -- without crashing,
because a crash costs the user a round trip.

A fake Theta Terminal is stood up with httpx's MockTransport so all of that can
be exercised offline.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest

from roth.data.probe import ProbeResult, ThetaProbe, render_report

EXPIRATIONS = "20240119\n20240216\n20240315\n20241220\n"
STRIKES = "480000\n490000\n500000\n510000\n"
EOD_CSV = "ms_of_day,open,high,low,close,volume,date\n34200000,470.1,472.0,469.5,471.2,1000,20240110\n"
CHAIN_CSV = (
    "ms_of_day,bid,ask,strike,right,expiration,date\n"
    "34200000,1.00,1.20,500000,C,20240119,20240110\n"
)


def fake_terminal(handler) -> ThetaProbe:
    """A probe wired to a fake terminal rather than a real one."""
    probe = ThetaProbe(root="SPY")
    probe.client = httpx.Client(
        base_url="http://fake", transport=httpx.MockTransport(handler), timeout=5.0
    )
    return probe


def healthy(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/v2/system"):
        return httpx.Response(200, text="CONNECTED")
    if path == "/v2/list/expirations":
        return httpx.Response(200, text=EXPIRATIONS)
    if path == "/v2/list/strikes":
        return httpx.Response(200, text=STRIKES)
    if path.startswith("/v2/list/roots"):
        return httpx.Response(200, text="SPY\nQQQ\n")
    if "bulk_hist" in path:
        return httpx.Response(200, text=CHAIN_CSV)
    if "hist/stock" in path or "hist/index" in path or "hist/option" in path:
        return httpx.Response(200, text=EOD_CSV)
    return httpx.Response(404, text="not found")


# -- happy path -------------------------------------------------------------


def test_probe_completes_against_a_healthy_terminal():
    with fake_terminal(healthy) as probe:
        report = probe.run()

    assert report.passed
    assert not [r for r in report.failed if "history depth" not in r.label]


def test_probe_discovers_an_expiration_and_strike_to_use():
    with fake_terminal(healthy) as probe:
        probe.run()

    assert probe.sample_expiration in {
        date(2024, 1, 19),
        date(2024, 2, 16),
        date(2024, 3, 15),
        date(2024, 12, 20),
    }
    assert probe.sample_strike in {480.0, 490.0, 500.0, 510.0}


def test_probe_records_column_names():
    """The column names are what let the client's parsing be corrected."""
    with fake_terminal(healthy) as probe:
        report = probe.run()

    eod = next(r for r in report.results if r.label == "stock EOD")
    assert "close" in eod.header_line
    assert eod.sample_row


# -- it must survive a hostile server --------------------------------------


def test_probe_survives_a_server_that_404s_everything():
    def all_404(request):
        return httpx.Response(404, text="nope")

    with fake_terminal(all_404) as probe:
        report = probe.run()

    assert report.results
    assert not report.passed
    assert all(r.status == 404 for r in report.results)


def test_probe_survives_auth_errors():
    """A free tier refusing paid endpoints must be reported, not crash."""

    def paywalled(request):
        if "bulk_hist" in request.url.path:
            return httpx.Response(472, text="subscription required")
        return healthy(request)

    with fake_terminal(paywalled) as probe:
        report = probe.run()

    blocked = [r for r in report.results if "bulk" in r.label and not r.ok]
    assert blocked
    assert all(r.status == 472 for r in blocked)
    # Everything else still ran.
    assert any(r.ok for r in report.results)


def test_probe_survives_empty_bodies():
    def empty(request):
        return httpx.Response(200, text="")

    with fake_terminal(empty) as probe:
        report = probe.run()

    assert report.results
    assert any("empty body" in r.error for r in report.results)


def test_probe_survives_connection_refused():
    def refuse(request):
        raise httpx.ConnectError("connection refused")

    with fake_terminal(refuse) as probe:
        report = probe.run()

    assert report.results
    assert all(not r.ok for r in report.results)
    assert all("could not connect" in r.error for r in report.results)


def test_probe_survives_a_timeout():
    def slow(request):
        raise httpx.ReadTimeout("too slow")

    with fake_terminal(slow) as probe:
        report = probe.run()

    assert all("timed out" in r.error for r in report.results)


def test_probe_survives_an_unexpected_exception():
    """Anything at all must be caught. A crash costs the user a round trip."""

    def explode(request):
        raise ValueError("something nobody anticipated")

    with fake_terminal(explode) as probe:
        report = probe.run()

    assert report.results
    assert any("ValueError" in r.error for r in report.results)


# -- alternate path discovery ----------------------------------------------


def test_probe_finds_a_working_alternate_path():
    """If the configured path is wrong, the probe must locate the right one
    rather than just reporting a failure."""

    def only_alt_greeks(request):
        path = request.url.path
        if path == "/v2/bulk_hist/option/eod_greeks":
            return httpx.Response(404, text="wrong")
        if path == "/v2/bulk_hist/option/greeks":
            return httpx.Response(200, text=CHAIN_CSV)
        return healthy(request)

    with fake_terminal(only_alt_greeks) as probe:
        report = probe.run()

    assert any("works" in n and "greeks" in n for n in report.notes)


# -- history depth discovery ------------------------------------------------


def test_history_depth_stops_at_the_subscription_limit():
    """Free tiers cut history off. The probe finds where, by walking back."""

    def two_years_only(request):
        if "hist/stock" in request.url.path:
            start = request.url.params.get("start_date", "")
            if start and int(start[:4]) < date.today().year - 2:
                return httpx.Response(472, text="outside subscription window")
        return healthy(request)

    with fake_terminal(two_years_only) as probe:
        report = probe.run()

    depth = report.history_depth
    assert depth
    assert depth["1y"]["ok"]
    # It stopped rather than probing all the way to 20 years.
    assert not depth[max(depth, key=lambda k: int(k[:-1]))]["ok"]
    assert any("history stops" in n for n in report.notes)


# -- report rendering -------------------------------------------------------


def test_report_renders_without_credentials():
    with fake_terminal(healthy) as probe:
        report = probe.run()

    text = render_report(report)
    assert "ThetaData endpoint probe" in text
    assert "SUMMARY" in text
    assert "DETAIL" in text
    # The probe never sees credentials, but assert the obvious markers anyway.
    for secret in ("password", "api_key", "token", "Authorization"):
        assert secret.lower() not in text.lower()


def test_report_renders_even_when_everything_failed():
    def refuse(request):
        raise httpx.ConnectError("refused")

    with fake_terminal(refuse) as probe:
        report = probe.run()

    text = render_report(report)
    assert "0 ok" in text


# -- parsers ----------------------------------------------------------------


def test_parse_dates_ignores_junk():
    r = ProbeResult("t", "GET", "/x", {})
    r.body_lines = ["20240119", "not-a-date", "", "20241220", "999"]
    assert ThetaProbe._parse_dates(r) == [date(2024, 1, 19), date(2024, 12, 20)]


def test_parse_dates_rejects_impossible_dates():
    r = ProbeResult("t", "GET", "/x", {})
    r.body_lines = ["20241332", "20240119"]
    assert ThetaProbe._parse_dates(r) == [date(2024, 1, 19)]


def test_parse_strikes_scales_by_a_thousand():
    r = ProbeResult("t", "GET", "/x", {})
    r.body_lines = ["500000", "62500"]
    assert ThetaProbe._parse_strikes(r) == [62.5, 500.0]


def test_parse_strikes_drops_non_positive_values():
    r = ProbeResult("t", "GET", "/x", {})
    r.body_lines = ["-1", "0", "500000"]
    assert ThetaProbe._parse_strikes(r) == [500.0]


@pytest.mark.parametrize("body", ["", "   ", "\n\n"])
def test_parsers_handle_empty_bodies(body):
    r = ProbeResult("t", "GET", "/x", {})
    r.body_lines = [ln for ln in body.splitlines() if ln.strip()]
    assert ThetaProbe._parse_dates(r) == []
    assert ThetaProbe._parse_strikes(r) == []
