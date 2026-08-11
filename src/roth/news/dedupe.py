"""Remembering what has already been said.

Two kinds of duplicate have to be suppressed, and they need different keys:

* **The same item, seen again.** Every poll re-reads the whole feed, so without
  this the bot re-alerts every headline every sixty seconds. Keyed on
  `item_id`, which is derived from a canonicalised URL so Yahoo's per-request
  tracking parameters do not make one article look like sixty.
* **The same story from a second outlet.** One Reuters report reaches four
  aggregators within a minute. Keyed on `dedupe_key`, the normalised title.

The store persists to disk, because the failure mode of an in-memory-only store
is the worst one available: restart the bot at 09:25 and it alerts the entire
overnight feed as breaking news at the open.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from roth.news.config import SEEN_RETENTION_DAYS

STORE_VERSION = 1


@dataclass
class SeenStore:
    """Persistent record of everything already emitted.

    Loading a corrupt or unreadable store is not fatal: it starts empty and
    says so. The cost is one duplicated round of alerts; the alternative is a
    bot that will not start.
    """

    path: Path
    retention_days: int = SEEN_RETENTION_DAYS
    items: dict[str, str] = field(default_factory=dict)
    stories: dict[str, str] = field(default_factory=dict)
    # symbol -> the move percentage most recently alerted on.
    alerted_moves: dict[str, float] = field(default_factory=dict)
    load_error: str | None = None
    _dirty: bool = False

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path, retention_days: int = SEEN_RETENTION_DAYS) -> SeenStore:
        store = cls(path=path, retention_days=retention_days)
        if not path.exists():
            return store
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            store.load_error = f"could not read {path}: {exc}"
            return store

        if not isinstance(payload, dict) or payload.get("version") != STORE_VERSION:
            store.load_error = f"{path} has an unrecognised format; starting fresh"
            return store

        store.items = {k: v for k, v in (payload.get("items") or {}).items() if isinstance(v, str)}
        store.stories = {
            k: v for k, v in (payload.get("stories") or {}).items() if isinstance(v, str)
        }
        store.alerted_moves = {
            k: float(v)
            for k, v in (payload.get("alerted_moves") or {}).items()
            if isinstance(v, (int, float))
        }
        return store

    def save(self) -> None:
        """Atomic write. A half-written store is worse than a missing one."""
        if not self._dirty:
            return
        self.prune()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STORE_VERSION,
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "items": self.items,
            "stories": self.stories,
            "alerted_moves": self.alerted_moves,
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._dirty = False

    # -- queries -----------------------------------------------------------

    def is_new(self, item) -> bool:
        """True when neither this item nor its story has been emitted before."""
        return item.item_id not in self.items and item.dedupe_key not in self.stories

    def mark(self, item, when: datetime | None = None) -> None:
        stamp = (when or datetime.now(timezone.utc)).isoformat()
        self.items[item.item_id] = stamp
        self.stories.setdefault(item.dedupe_key, stamp)
        self._dirty = True

    def filter_new(self, items: list) -> list:
        """New items only, marked as seen. Order is preserved.

        Marking happens here rather than at the sink so that two items from the
        same poll describing one story collapse to one alert.
        """
        out = []
        for item in items:
            if self.is_new(item):
                self.mark(item)
                out.append(item)
        return out

    # -- price moves -------------------------------------------------------

    def should_alert_move(self, symbol: str, change_pct: float, threshold: float, step: float) -> bool:
        """Whether a price move is worth an alert given what was already said.

        A stock that opens down 4% is down 4% all morning. Alerting once per
        poll on a standing condition is how a bot trains you to mute it, so a
        move only re-alerts once it has extended by `step` beyond the last one
        it fired on -- or reversed through zero into a move the other way.
        """
        if abs(change_pct) < threshold:
            return False
        previous = self.alerted_moves.get(symbol)
        if previous is None:
            return True
        if (previous > 0) != (change_pct > 0):
            return True
        return abs(change_pct) >= abs(previous) + step

    def mark_move(self, symbol: str, change_pct: float) -> None:
        self.alerted_moves[symbol] = change_pct
        self._dirty = True

    # -- maintenance -------------------------------------------------------

    def prune(self, now: datetime | None = None) -> int:
        """Drop entries older than the retention window.

        Safe because no feed reaches back that far, so a pruned item can never
        reappear and be mistaken for new.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.retention_days)
        removed = 0
        for bucket in (self.items, self.stories):
            for key, raw in list(bucket.items()):
                try:
                    stamp = datetime.fromisoformat(raw)
                except ValueError:
                    bucket.pop(key, None)
                    removed += 1
                    continue
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                if stamp < cutoff:
                    bucket.pop(key, None)
                    removed += 1
        if removed:
            self._dirty = True
        return removed

    def __len__(self) -> int:
        return len(self.items)
