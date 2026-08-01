"""Storage layer.

Parquet files on local disk, queried with DuckDB. DuckDB is a library, not a
service: there is nothing to start and nothing to keep running.

Two rules are enforced here rather than left to convention:

1. **Raw is immutable.** `write_raw` refuses to overwrite an existing partition.
   Re-downloading the same day is a no-op unless the caller explicitly asks to
   replace it, which is a deliberate, logged act.
2. **Derived is disposable.** `write_derived` always overwrites. Nothing in
   `derived/` is ever hand-edited; it is rebuilt from `raw/`.

A manifest records every raw partition that has landed, so a backfill can be
interrupted and resumed without re-downloading what is already on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from roth.paths import DERIVED, RAW, ensure_dirs

MANIFEST_PATH = RAW / "_manifest.jsonl"

# zstd gives roughly 15-25% better compression than snappy on quote data at a
# read-speed cost that is invisible next to the disk it saves.
COMPRESSION = "zstd"


class ImmutableRawError(RuntimeError):
    """Raised on an attempt to overwrite raw data that already exists."""


# ---------------------------------------------------------------------------
# Partition paths
# ---------------------------------------------------------------------------


def partition_path(root: Path, symbol: str, day: date, suffix: str = "parquet") -> Path:
    """Hive-style partition path: <root>/symbol=SPY/year=2026/2026-06-01.parquet

    DuckDB reads the symbol and year straight out of the directory names, so a
    query filtered to one symbol never touches another symbol's files.
    """
    return root / f"symbol={symbol}" / f"year={day.year}" / f"{day.isoformat()}.{suffix}"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    symbol: str
    day: date
    rows: int
    bytes: int
    written_at: datetime

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "symbol": self.symbol,
                "day": self.day.isoformat(),
                "rows": self.rows,
                "bytes": self.bytes,
                "written_at": self.written_at.isoformat(),
            }
        )


def append_manifest(entry: ManifestEntry, path: Path | None = None) -> None:
    # Resolved at call time, not bound as a default, so the location stays
    # redirectable (tests, alternate data roots).
    path = path or MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(entry.to_json() + "\n")


def read_manifest(path: Path | None = None) -> pd.DataFrame:
    """Everything that has landed in raw/, as a frame. Empty if nothing has."""
    path = path or MANIFEST_PATH
    columns = ["dataset", "symbol", "day", "rows", "bytes", "written_at"]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def already_downloaded(dataset: str, symbol: str, path: Path | None = None) -> set[date]:
    """Days already present for this dataset and symbol, for resumable backfill."""
    df = read_manifest(path)
    if df.empty:
        return set()
    mask = (df["dataset"] == dataset) & (df["symbol"] == symbol)
    return set(df.loc[mask, "day"])


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write via a temp file and rename, so an interrupted run never leaves a
    half-written parquet file that later reads treat as real data."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_parquet(tmp, engine="pyarrow", compression=COMPRESSION, index=False)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_raw(
    df: pd.DataFrame,
    dataset_dir: Path,
    dataset_name: str,
    symbol: str,
    day: date,
    replace: bool = False,
) -> Path:
    """Write one day of raw data. Refuses to overwrite unless `replace=True`.

    This is the immutability rule. Raw data is what the provider gave us; if it
    is wrong, that fact belongs in the quarantine table, not in an edit.
    """
    ensure_dirs()
    target = partition_path(dataset_dir, symbol, day)

    if target.exists() and not replace:
        raise ImmutableRawError(
            f"Raw data already exists for {dataset_name} {symbol} {day}:\n  {target}\n"
            "Raw data is immutable. Pass replace=True only if the original download "
            "is known to be corrupt."
        )

    _atomic_write_parquet(df, target)

    append_manifest(
        ManifestEntry(
            dataset=dataset_name,
            symbol=symbol,
            day=day,
            rows=len(df),
            bytes=target.stat().st_size,
            written_at=datetime.now(timezone.utc),
        )
    )
    return target


def write_derived(df: pd.DataFrame, dataset_dir: Path, name: str) -> Path:
    """Write a derived table. Always overwrites; derived data is disposable."""
    ensure_dirs()
    target = dataset_dir / f"{name}.parquet"
    _atomic_write_parquet(df, target)
    return target


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def connect() -> duckdb.DuckDBPyConnection:
    """An in-process DuckDB connection.

    Deliberately in-memory: there is no database file to corrupt, migrate, or
    back up. The parquet files on disk are the database.
    """
    con = duckdb.connect(":memory:")
    con.execute("SET TimeZone='UTC'")
    return con


def _glob(dataset_dir: Path, symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return [str(dataset_dir / "**" / "*.parquet")]
    return [str(dataset_dir / f"symbol={s}" / "**" / "*.parquet") for s in symbols]


def read_dataset(
    dataset_dir: Path,
    symbols: Iterable[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    columns: str = "*",
) -> pd.DataFrame:
    """Read a partitioned dataset into a frame.

    `start` and `end` filter on the partition's `day` column when present. This
    is a convenience for reporting and inspection. Research code must not use
    it to reach across time -- see `roth.access.TimeGate`, which is the only
    sanctioned way for a backtest to read bars.
    """
    patterns = _glob(dataset_dir, symbols)
    existing = [p for p in patterns if list(Path(p.split("**")[0]).rglob("*.parquet"))]
    if not existing:
        return pd.DataFrame()

    con = connect()
    try:
        union = " UNION ALL ".join(
            f"SELECT {columns} FROM read_parquet('{p}', hive_partitioning=true)" for p in existing
        )
        sql = f"SELECT * FROM ({union})"
        clauses = []
        if start is not None:
            clauses.append(f"day >= DATE '{start.isoformat()}'")
        if end is not None:
            clauses.append(f"day <= DATE '{end.isoformat()}'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def dataset_summary(path: Path | None = None) -> pd.DataFrame:
    """Per-symbol row counts and date coverage, straight from the manifest."""
    df = read_manifest(path)
    if df.empty:
        return df
    return (
        df.groupby(["dataset", "symbol"], as_index=False)
        .agg(
            days=("day", "nunique"),
            rows=("rows", "sum"),
            bytes=("bytes", "sum"),
            first_day=("day", "min"),
            last_day=("day", "max"),
        )
        .sort_values(["dataset", "symbol"])
    )


__all__ = [
    "DERIVED",
    "RAW",
    "ImmutableRawError",
    "ManifestEntry",
    "already_downloaded",
    "connect",
    "dataset_summary",
    "partition_path",
    "read_dataset",
    "read_manifest",
    "write_derived",
    "write_raw",
]
