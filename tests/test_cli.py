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
    ["ingest"],
    ["ingest", "calendar"],
    ["ingest", "underlying"],
    ["ingest", "options-eod"],
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
