"""Tests for pilot-download plumbing that does not need a network.

The CSV page handling matters more than it looks: ThetaData repeats the header
on every page, and silently concatenating pages would inject header rows into
the data as strings.
"""

from __future__ import annotations

from datetime import date

from roth.data.pilot import _parse_csv_pages, nyse_sessions
from roth.paths import human_bytes, human_duration

HEADER = "ms_of_day,bid_size,bid,ask_size,ask,date,root,expiration,strike,right"


def _row(ms: int, strike_x1000: int) -> str:
    return f"{ms},10,5.10,12,5.20,20260601,SPY,20260619,{strike_x1000},C"


def test_single_page_parses():
    page = "\n".join([HEADER, _row(34200000, 740000), _row(34260000, 740000)])
    df = _parse_csv_pages([page])
    assert len(df) == 2
    assert list(df.columns) == HEADER.split(",")
    assert df["strike"].tolist() == [740000, 740000]


def test_repeated_headers_across_pages_are_stripped():
    page1 = "\n".join([HEADER, _row(34200000, 740000)])
    page2 = "\n".join([HEADER, _row(34260000, 741000)])
    page3 = "\n".join([HEADER, _row(34320000, 742000)])

    df = _parse_csv_pages([page1, page2, page3])

    assert len(df) == 3
    # If a header row leaked in, this column would be object dtype holding the
    # literal string "strike".
    assert df["strike"].tolist() == [740000, 741000, 742000]
    assert str(df["strike"].dtype).startswith("int")


def test_empty_and_headeronly_pages_are_ignored():
    page1 = "\n".join([HEADER, _row(34200000, 740000)])
    assert len(_parse_csv_pages([page1, "", HEADER, "   "])) == 1


def test_no_pages_gives_empty_frame():
    assert _parse_csv_pages([]).empty
    assert _parse_csv_pages(["", "  "]).empty


def test_nyse_sessions_excludes_weekends_and_holidays():
    # July 2026: July 4 falls on a Saturday, observed Friday July 3.
    sessions = nyse_sessions(date(2026, 7, 1), date(2026, 7, 10))
    assert date(2026, 7, 4) not in sessions  # Saturday
    assert date(2026, 7, 5) not in sessions  # Sunday
    assert date(2026, 7, 3) not in sessions  # observed Independence Day
    assert date(2026, 7, 1) in sessions
    assert date(2026, 7, 6) in sessions


def test_a_normal_month_has_roughly_21_sessions():
    sessions = nyse_sessions(date(2026, 6, 1), date(2026, 6, 30))
    assert 19 <= len(sessions) <= 23


def test_human_bytes_and_duration_are_readable():
    assert human_bytes(512) == "512 B"
    assert human_bytes(1024 * 1024).startswith("1.0 MB")
    assert human_bytes(5 * 1024**3).startswith("5.0 GB")

    assert human_duration(45).endswith("sec")
    assert human_duration(600).endswith("min")
    assert human_duration(3600 * 5).endswith("hours")
    assert human_duration(3600 * 24 * 4).endswith("days")
