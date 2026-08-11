"""CLI smoke tests.

These exist because a dependency mismatch between typer and click broke
`roth --help` at runtime while every unit test still passed. Anything the user
is told to type should be exercised here.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from roth.cli import app

runner = CliRunner()

COMMANDS = [
    [],
    ["doctor"],
    ["status"],
    ["estimate"],
    ["pilot"],
    ["probe"],
    ["ingest"],
    ["ingest", "calendar"],
    ["ingest", "underlying"],
    ["ingest", "options-eod"],
    ["synth"],
    ["quality"],
    ["features"],
    ["features", "build"],
    ["features", "verify"],
    ["backtest"],
    ["strategies"],
    ["verify"],
    ["news"],
    ["news", "watch"],
    ["news", "once"],
    ["news", "brief"],
    ["news", "doctor"],
    ["news", "symbols"],
]


@pytest.mark.parametrize("cmd", COMMANDS, ids=lambda c: " ".join(c) or "root")
def test_help_renders(cmd):
    """Help must render for every command without raising."""
    result = runner.invoke(app, [*cmd, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_estimate_runs_without_data_or_network(tmp_path, monkeypatch):
    monkeypatch.setattr("roth.data.sizing.ASSUMPTIONS_PATH", tmp_path / "none.json")
    result = runner.invoke(app, ["estimate", "--first-year", "2025", "--last-year", "2026"])
    assert result.exit_code == 0, result.output
    # With no pilot on file it must say so rather than presenting model output
    # as measurement.
    assert "MODELED" in result.output


def test_status_is_honest_when_nothing_is_ingested(tmp_path, monkeypatch):
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", tmp_path / "manifest.jsonl")
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "No data ingested yet" in result.output


def test_ingest_calendar_works_offline(tmp_path, monkeypatch):
    """The calendar needs no feed and no subscription, so it must run anywhere."""
    monkeypatch.setattr("roth.paths.RAW_CALENDAR", tmp_path)
    monkeypatch.setattr("roth.data.ingest.RAW_CALENDAR", tmp_path)

    result = runner.invoke(
        app, ["ingest", "calendar", "--start", "2026-01-01", "--end", "2026-06-30"]
    )
    assert result.exit_code == 0, result.output
    assert "trading sessions" in result.output
    assert (tmp_path / "trading_calendar.parquet").exists()


def test_download_commands_fail_cleanly_without_theta_terminal():
    """No stack trace, and a message that says what to do."""
    for cmd in (["ingest", "underlying"], ["ingest", "options-eod"], ["pilot"]):
        result = runner.invoke(app, cmd)
        assert result.exit_code == 1, f"{cmd}: {result.output}"
        assert "Theta Terminal" in result.output
        assert "Traceback" not in result.output


def test_news_symbols_lists_the_whole_watchlist():
    result = runner.invoke(app, ["news", "symbols"])
    assert result.exit_code == 0, result.output
    for symbol in ("NVDA", "TSLA", "META", "AAPL", "AMZN", "MSFT", "AVGO", "GOOGL"):
        assert symbol in result.output


def test_news_doctor_offline_fails_loudly_without_an_sec_contact(monkeypatch):
    """SEC blocks clients that do not identify themselves, so a missing
    contact address must be a failure rather than a warning."""
    monkeypatch.delenv("ROTH_SEC_CONTACT", raising=False)
    result = runner.invoke(app, ["news", "doctor", "--offline"])
    assert result.exit_code == 1
    assert "ROTH_SEC_CONTACT" in result.output


def test_news_doctor_offline_passes_once_a_contact_is_set(monkeypatch):
    monkeypatch.setenv("ROTH_SEC_CONTACT", "someone@example.com")
    result = runner.invoke(app, ["news", "doctor", "--offline"])
    assert result.exit_code == 0, result.output


def test_news_rejects_a_symbol_outside_the_watchlist():
    result = runner.invoke(app, ["news", "once", "--symbols", "GME"])
    assert result.exit_code != 0
    assert "GME" in str(result.exception) or "GME" in result.output


def test_news_webhook_sink_requires_a_url():
    result = runner.invoke(app, ["news", "once", "--sink", "webhook"])
    assert result.exit_code == 2
    assert "webhook-url" in result.output
