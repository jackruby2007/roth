"""Price context, including the pre- and post-market print.

News without a price is half the story: an 8-K matters differently when the
stock is already down four percent on it. This source exists to attach that
number to every brief and to fire its own alert when a name moves hard with no
headline attached -- which is itself information.

Yahoo's chart endpoint is used because it needs no key and no cookie. It is
unofficial. Every field read from it is treated as optional, and `roth news
doctor` prints exactly which fields came back so a silent shape change shows up
as a diagnostic rather than as wrong numbers.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from roth.news.config import ticker as lookup_ticker
from roth.news.http import FetchError, NewsHttp
from roth.news.models import Quote, SourceResult

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


class QuoteParseError(ValueError):
    """The payload came back, but not in a shape carrying a usable price."""


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # Yahoo pads gaps with nulls, but a zero price is also meaningless here.
    return out if out == out and out > 0 else None


def _period_bounds(meta: dict, name: str) -> tuple[int, int] | None:
    period = (meta.get("currentTradingPeriod") or {}).get(name)
    if not isinstance(period, dict):
        return None
    start, end = period.get("start"), period.get("end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return int(start), int(end)
    return None


def parse_chart(payload: dict, symbol: str) -> Quote:
    """Extract a `Quote` from one chart response.

    Pure, so the parser is covered by fixtures rather than by hitting Yahoo.
    """
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        detail = error.get("description") if isinstance(error, dict) else error
        raise QuoteParseError(f"{symbol}: chart error: {detail}")

    results = chart.get("result") or []
    if not results:
        raise QuoteParseError(f"{symbol}: chart returned no result")

    result = results[0] or {}
    meta = result.get("meta") or {}

    previous_close = _num(meta.get("previousClose")) or _num(meta.get("chartPreviousClose"))
    regular_price = _num(meta.get("regularMarketPrice"))

    # Walk the bars to find the most recent real print and when it happened.
    timestamps = result.get("timestamp") or []
    quote_block = ((result.get("indicators") or {}).get("quote") or [{}])[0] or {}
    closes = quote_block.get("close") or []

    last_price: float | None = None
    last_ts: int | None = None
    for i in range(min(len(timestamps), len(closes)) - 1, -1, -1):
        candidate = _num(closes[i])
        if candidate is not None:
            last_price, last_ts = candidate, int(timestamps[i])
            break

    if regular_price is None and last_price is None:
        raise QuoteParseError(f"{symbol}: no usable price in chart payload")
    if previous_close is None:
        # Without a reference close there is no percentage move to report, and
        # inventing one would be worse than saying so.
        raise QuoteParseError(f"{symbol}: no previous close in chart payload")

    price = regular_price if regular_price is not None else last_price

    # Classify the last print. A bar outside the regular window is an extended
    # hours trade and must not be compared against the regular price.
    state = "CLOSED"
    extended: float | None = None
    if last_ts is not None:
        pre = _period_bounds(meta, "pre")
        regular = _period_bounds(meta, "regular")
        post = _period_bounds(meta, "post")
        if regular and regular[0] <= last_ts < regular[1]:
            state = "REGULAR"
        elif pre and pre[0] <= last_ts < pre[1]:
            state, extended = "PRE", last_price
        elif post and post[0] <= last_ts < post[1]:
            state, extended = "POST", last_price

    return Quote(
        symbol=symbol,
        price=price if price is not None else 0.0,
        previous_close=previous_close,
        currency=str(meta.get("currency") or "USD"),
        market_state=state,
        as_of_utc=(
            datetime.fromtimestamp(last_ts, tz=timezone.utc)
            if last_ts is not None
            else datetime.now(timezone.utc)
        ),
        extended_price=extended,
        day_high=_num(meta.get("regularMarketDayHigh")),
        day_low=_num(meta.get("regularMarketDayLow")),
        volume=int(meta["regularMarketVolume"])
        if isinstance(meta.get("regularMarketVolume"), (int, float))
        else None,
    )


class YahooQuotes:
    """Polls a price snapshot for every watchlist symbol."""

    name = "quotes"

    def fetch(self, http: NewsHttp, symbols: tuple[str, ...]) -> SourceResult:
        started = time.monotonic()
        result = SourceResult(source=self.name)
        failures: list[str] = []

        for symbol in symbols:
            lookup_ticker(symbol)
            try:
                payload = http.get_json(
                    CHART_URL.format(symbol=symbol),
                    params={"range": "5d", "interval": "5m", "includePrePost": "true"},
                )
                result.quotes.append(parse_chart(payload, symbol))
            except (FetchError, QuoteParseError) as exc:
                failures.append(f"{symbol}: {exc}")
            except Exception as exc:  # noqa: BLE001 - a source must never raise
                failures.append(f"{symbol}: unexpected {type(exc).__name__}: {exc}")

        if failures:
            result.error = "; ".join(failures)
        result.elapsed_seconds = time.monotonic() - started
        return result
