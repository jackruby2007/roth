"""Filesystem layout for the research harness.

Everything is files on local disk. There is no database server. The only rule
that matters here: `RAW` is immutable. Once a download lands there it is never
edited or overwritten in place. `DERIVED` is disposable and is always rebuilt
from `RAW`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    env = os.environ.get("ROTH_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # src/roth/paths.py -> src/roth -> src -> project root
    return Path(__file__).resolve().parents[2]


ROOT = _project_root()
DATA = ROOT / "data"

# Immutable landing zone for downloaded data. Never edited after write.
RAW = DATA / "raw"
RAW_OPTION_QUOTES = RAW / "option_quotes"
RAW_OPTION_EOD = RAW / "option_eod"
RAW_UNDERLYING_MINUTE = RAW / "underlying_minute"
RAW_UNDERLYING_DAILY = RAW / "underlying_daily"
RAW_VIX = RAW / "vix"
RAW_CALENDAR = RAW / "calendar"

# Rebuildable outputs. Safe to delete at any time.
DERIVED = DATA / "derived"
FEATURES = DERIVED / "features"
QUARANTINE = DERIVED / "quarantine"
TRADES = DERIVED / "trades"
RULE_EVALS = DERIVED / "rule_evals"

REPORTS = DATA / "reports"
PILOT = DATA / "pilot"

ALL_DIRS = [
    RAW,
    RAW_OPTION_QUOTES,
    RAW_OPTION_EOD,
    RAW_UNDERLYING_MINUTE,
    RAW_UNDERLYING_DAILY,
    RAW_VIX,
    RAW_CALENDAR,
    DERIVED,
    FEATURES,
    QUARANTINE,
    TRADES,
    RULE_EVALS,
    REPORTS,
    PILOT,
]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(path: Path) -> int:
    """Total bytes on disk under `path`. Returns 0 if it does not exist."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,.0f} B"
        n /= 1024.0
    return f"{n:,.1f} TB"


def human_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:,.1f} sec"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:,.1f} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:,.1f} hours"
    return f"{hours / 24:,.1f} days"
