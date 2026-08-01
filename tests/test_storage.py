"""Tests for the storage layer.

The immutability rule is the point of this module, so most of these tests are
about refusing to do things.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from roth.storage import (
    ImmutableRawError,
    already_downloaded,
    append_manifest,
    ManifestEntry,
    partition_path,
    read_manifest,
    write_derived,
    write_raw,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"day": [date(2026, 6, 1)] * 3, "close": [1.0, 2.0, 3.0]})


def test_partition_path_is_hive_style():
    p = partition_path(pd.io.common.Path("/tmp/x"), "SPY", date(2026, 6, 1))
    assert p.parts[-3:] == ("symbol=SPY", "year=2026", "2026-06-01.parquet")


def test_write_raw_then_read_back(tmp_path, frame, monkeypatch):
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", tmp_path / "manifest.jsonl")
    target = write_raw(frame, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))
    assert target.exists()
    assert len(pd.read_parquet(target)) == 3


def test_raw_refuses_to_overwrite(tmp_path, frame, monkeypatch):
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", tmp_path / "manifest.jsonl")
    write_raw(frame, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))

    with pytest.raises(ImmutableRawError, match="immutable"):
        write_raw(frame, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))


def test_raw_overwrite_requires_explicit_replace(tmp_path, frame, monkeypatch):
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", tmp_path / "manifest.jsonl")
    write_raw(frame, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))

    bigger = pd.concat([frame, frame], ignore_index=True)
    target = write_raw(bigger, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1), replace=True)
    assert len(pd.read_parquet(target)) == 6


def test_derived_always_overwrites(tmp_path, frame):
    write_derived(frame, tmp_path / "der", "features")
    doubled = pd.concat([frame, frame], ignore_index=True)
    target = write_derived(doubled, tmp_path / "der", "features")
    assert len(pd.read_parquet(target)) == 6


def test_manifest_records_every_write(tmp_path, frame, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", manifest)

    for day in (date(2026, 6, 1), date(2026, 6, 2)):
        write_raw(frame, tmp_path / "ds", "test_ds", "SPY", day)

    df = read_manifest(manifest)
    assert len(df) == 2
    assert set(df["day"]) == {date(2026, 6, 1), date(2026, 6, 2)}
    assert df["rows"].sum() == 6


def test_already_downloaded_drives_resumable_backfill(tmp_path, frame, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", manifest)

    write_raw(frame, tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))
    write_raw(frame, tmp_path / "ds", "test_ds", "QQQ", date(2026, 6, 2))

    spy = already_downloaded("test_ds", "SPY", manifest)
    assert spy == {date(2026, 6, 1)}
    # Symbols must not bleed into each other.
    assert date(2026, 6, 2) not in spy


def test_manifest_survives_a_corrupt_line(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    append_manifest(
        ManifestEntry("ds", "SPY", date(2026, 6, 1), 3, 100, pd.Timestamp.utcnow().to_pydatetime()),
        manifest,
    )
    with manifest.open("a") as fh:
        fh.write("{ this is not json\n")

    df = read_manifest(manifest)
    assert len(df) == 1


def test_missing_manifest_reads_as_empty(tmp_path):
    assert read_manifest(tmp_path / "nope.jsonl").empty
    assert already_downloaded("ds", "SPY", tmp_path / "nope.jsonl") == set()


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    """A failure mid-write must not leave a readable half-written parquet."""
    monkeypatch.setattr("roth.storage.MANIFEST_PATH", tmp_path / "manifest.jsonl")

    class Boom(pd.DataFrame):
        def to_parquet(self, *a, **k):  # noqa: ANN002, ANN003
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_raw(Boom({"a": [1]}), tmp_path / "ds", "test_ds", "SPY", date(2026, 6, 1))

    target = partition_path(tmp_path / "ds", "SPY", date(2026, 6, 1))
    assert not target.exists()
    assert not list((tmp_path / "ds").rglob("*.tmp"))
