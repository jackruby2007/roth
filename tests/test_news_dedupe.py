"""Tests for the seen-store.

Every failure mode here is a bot you turn off: re-alerting the same headline
every minute, replaying the overnight feed after a restart, or repeating the
same standing price move until you mute it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from roth.news.dedupe import SeenStore
from roth.news.models import NewsItem

NOW = datetime(2026, 8, 11, 14, 0, tzinfo=timezone.utc)


def item(title: str, url: str = "https://example.com/a", symbol: str = "NVDA") -> NewsItem:
    return NewsItem(
        symbol=symbol, source="yahoo", title=title, url=url, published_utc=NOW
    )


# -- item identity ----------------------------------------------------------


def test_the_same_item_is_only_emitted_once(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    assert store.filter_new([item("Nvidia beats")]) != []
    assert store.filter_new([item("Nvidia beats")]) == []


def test_a_repeated_url_with_different_tracking_params_is_the_same_item(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.filter_new([item("Nvidia beats", "https://x.com/a?utm_source=rss")])
    assert store.filter_new([item("Nvidia beats", "https://x.com/a?utm_source=email")]) == []


def test_the_same_story_from_a_second_outlet_is_suppressed(tmp_path):
    """One wire report reaches several aggregators within a minute."""
    store = SeenStore(path=tmp_path / "seen.json")
    store.filter_new([item("Nvidia beats estimates", "https://reuters.com/a")])
    assert store.filter_new([item("Nvidia beats estimates", "https://cnbc.com/b")]) == []


def test_the_same_headline_about_a_different_symbol_is_not_suppressed(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.filter_new([item("Chip stocks rally", symbol="NVDA")])
    assert store.filter_new([item("Chip stocks rally", symbol="AVGO")]) != []


def test_duplicates_within_a_single_batch_collapse_to_one(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    batch = [
        item("Nvidia beats estimates", "https://reuters.com/a"),
        item("Nvidia beats estimates", "https://cnbc.com/b"),
    ]
    assert len(store.filter_new(batch)) == 1


# -- persistence ------------------------------------------------------------


def test_the_store_survives_a_restart(tmp_path):
    """The failure this prevents: restart at 09:25 and alert the whole
    overnight feed as breaking news at the open."""
    path = tmp_path / "seen.json"
    first = SeenStore(path=path)
    first.filter_new([item("Nvidia beats")])
    first.save()

    second = SeenStore.load(path)
    assert second.load_error is None
    assert second.filter_new([item("Nvidia beats")]) == []


def test_loading_a_missing_store_starts_empty_without_error(tmp_path):
    store = SeenStore.load(tmp_path / "absent.json")
    assert len(store) == 0
    assert store.load_error is None


def test_a_corrupt_store_starts_fresh_and_reports_why(tmp_path):
    """Refusing to start is a worse outcome than one duplicated round."""
    path = tmp_path / "seen.json"
    path.write_text("{not json")
    store = SeenStore.load(path)
    assert store.load_error is not None
    assert store.filter_new([item("Nvidia beats")]) != []


def test_a_store_from_a_future_version_is_not_misread(tmp_path):
    path = tmp_path / "seen.json"
    path.write_text('{"version": 999, "items": {"x": "y"}}')
    store = SeenStore.load(path)
    assert store.load_error is not None
    assert len(store) == 0


def test_saving_is_atomic_and_leaves_no_temp_files(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.filter_new([item("Nvidia beats")])
    store.save()
    assert list(tmp_path.glob("*.tmp")) == []


def test_saving_a_clean_store_writes_nothing(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.save()
    assert not (tmp_path / "seen.json").exists()


# -- pruning ----------------------------------------------------------------


def test_entries_older_than_the_retention_window_are_pruned(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json", retention_days=7)
    store.items["old"] = (NOW - timedelta(days=30)).isoformat()
    store.items["recent"] = (NOW - timedelta(days=1)).isoformat()
    store.prune(now=NOW)
    assert "old" not in store.items
    assert "recent" in store.items


def test_pruning_discards_unparseable_timestamps(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.items["bad"] = "not a timestamp"
    store.prune(now=NOW)
    assert "bad" not in store.items


def test_naive_timestamps_in_an_old_store_are_treated_as_utc(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json", retention_days=7)
    store.items["naive"] = (NOW - timedelta(days=1)).replace(tzinfo=None).isoformat()
    store.prune(now=NOW)
    assert "naive" in store.items


# -- price moves ------------------------------------------------------------


def test_a_move_beyond_the_threshold_alerts_once(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    assert store.should_alert_move("NVDA", -4.0, threshold=2.0, step=1.0)
    store.mark_move("NVDA", -4.0)
    assert not store.should_alert_move("NVDA", -4.1, threshold=2.0, step=1.0)


def test_a_move_below_the_threshold_never_alerts(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    assert not store.should_alert_move("NVDA", 1.5, threshold=2.0, step=1.0)


def test_a_move_re_alerts_once_it_extends_by_the_step(tmp_path):
    store = SeenStore(path=tmp_path / "seen.json")
    store.mark_move("NVDA", -4.0)
    assert not store.should_alert_move("NVDA", -4.9, threshold=2.0, step=1.0)
    assert store.should_alert_move("NVDA", -5.0, threshold=2.0, step=1.0)


def test_a_reversal_through_zero_alerts_again(tmp_path):
    """Down four then up three is genuinely new information."""
    store = SeenStore(path=tmp_path / "seen.json")
    store.mark_move("NVDA", -4.0)
    assert store.should_alert_move("NVDA", 3.0, threshold=2.0, step=1.0)


def test_move_state_persists_across_a_restart(tmp_path):
    path = tmp_path / "seen.json"
    first = SeenStore(path=path)
    first.mark_move("NVDA", -4.0)
    first.save()
    assert not SeenStore.load(path).should_alert_move("NVDA", -4.2, threshold=2.0, step=1.0)
