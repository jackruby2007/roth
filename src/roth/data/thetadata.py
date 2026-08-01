"""ThetaData client.

Talks to a locally running Theta Terminal over its REST API. Theta Terminal is
the only process in this entire system that ever needs to be running, and only
while a download is in progress.

Design notes:

* Bulk endpoints are used wherever they exist. Pulling one contract at a time
  turns a one-hour download into a multi-day download.
* Responses are requested as CSV. It is roughly 3x smaller on the wire than
  ThetaData's JSON envelope and parses faster.
* Pagination is driven by the `Next-Page` response header, which ThetaData sets
  to `null` on the final page.
* Nothing here writes to disk. Callers decide where bytes land, which keeps the
  immutability rule enforceable in one place.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

import httpx

from roth.config import THETA, ThetaConfig


class ThetaError(RuntimeError):
    """Raised for any ThetaData failure that the user needs to see verbatim."""


class ThetaNotRunning(ThetaError):
    """Theta Terminal is not reachable on the configured host and port."""


class ThetaNoData(ThetaError):
    """Theta Terminal answered, but has no data for this request."""


def _fmt_date(d: date) -> str:
    """ThetaData wants dates as YYYYMMDD integers."""
    return d.strftime("%Y%m%d")


@dataclass
class ThetaResponse:
    """One page of a ThetaData response."""

    text: str
    bytes_received: int
    next_page: str | None


class ThetaClient:
    """Thin, synchronous ThetaData REST client.

    Usage::

        with ThetaClient() as client:
            client.check_connection()
            for page in client.bulk_option_quotes("SPY", date(2026, 6, 1), ...):
                ...
    """

    def __init__(self, cfg: ThetaConfig = THETA) -> None:
        self.cfg = cfg
        self._client = httpx.Client(
            base_url=cfg.base_url,
            timeout=httpx.Timeout(cfg.timeout_seconds),
            # Theta Terminal is strictly local. Never route it through a proxy.
            trust_env=False,
        )
        self.bytes_received = 0
        self.requests_made = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> ThetaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- low level ---------------------------------------------------------

    def _get(self, path: str, params: dict[str, object] | None = None) -> ThetaResponse:
        """One HTTP GET with retries on transient failures."""
        params = dict(params or {})
        params.setdefault("use_csv", "true")

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.get(path, params=params)
            except httpx.ConnectError as exc:
                raise ThetaNotRunning(
                    f"Could not reach Theta Terminal at {self.cfg.base_url}.\n"
                    "Theta Terminal must be running before a download can start.\n"
                    "Start it, wait until it prints CONNECTED, then re-run this command."
                ) from exc
            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                time.sleep(2**attempt)
                continue

            self.requests_made += 1

            # ThetaData signals "no data for this request" with 472. That is a
            # normal outcome (e.g. a strike that did not exist yet), not a bug.
            if resp.status_code == 472:
                raise ThetaNoData(f"No data for {path} {params}")

            # 429 / 474 mean the terminal is busy or rate limiting. Back off.
            if resp.status_code in (429, 474, 500, 502, 503, 504):
                last_exc = ThetaError(f"HTTP {resp.status_code} from {path}: {resp.text[:400]}")
                time.sleep(2**attempt)
                continue

            if resp.status_code != 200:
                raise ThetaError(
                    f"ThetaData returned HTTP {resp.status_code} for {path}\n"
                    f"params: {params}\n"
                    f"body: {resp.text[:1000]}"
                )

            body = resp.content
            self.bytes_received += len(body)

            next_page = resp.headers.get("Next-Page")
            if next_page in (None, "null", ""):
                next_page = None

            return ThetaResponse(
                text=body.decode("utf-8", errors="replace"),
                bytes_received=len(body),
                next_page=next_page,
            )

        raise ThetaError(
            f"ThetaData request to {path} failed after {self.cfg.max_retries} attempts."
        ) from last_exc

    def _paged(self, path: str, params: dict[str, object]) -> Iterator[ThetaResponse]:
        """Yield every page of a request, following the Next-Page header."""
        page = self._get(path, params)
        yield page
        while page.next_page:
            # Next-Page is an absolute URL already carrying its own query string.
            url = page.next_page
            last_exc: Exception | None = None
            for attempt in range(self.cfg.max_retries):
                try:
                    resp = self._client.get(url)
                    break
                except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                    last_exc = exc
                    time.sleep(2**attempt)
            else:
                raise ThetaError(f"Paging failed for {url}") from last_exc

            self.requests_made += 1
            if resp.status_code != 200:
                raise ThetaError(f"Paging returned HTTP {resp.status_code} for {url}")

            self.bytes_received += len(resp.content)
            nxt = resp.headers.get("Next-Page")
            if nxt in (None, "null", ""):
                nxt = None
            page = ThetaResponse(
                text=resp.content.decode("utf-8", errors="replace"),
                bytes_received=len(resp.content),
                next_page=nxt,
            )
            yield page

    # -- health ------------------------------------------------------------

    def check_connection(self) -> dict[str, str]:
        """Confirm Theta Terminal is up and report what it says about itself.

        Raises ThetaNotRunning with a plain-English message if it is not.
        """
        try:
            resp = self._client.get("/v2/system/mdds/status")
        except httpx.ConnectError as exc:
            raise ThetaNotRunning(
                f"Theta Terminal is not running (nothing listening on {self.cfg.base_url}).\n"
                "Start Theta Terminal, wait for it to report CONNECTED, then try again."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ThetaNotRunning(
                f"Theta Terminal at {self.cfg.base_url} accepted a connection but did not "
                "respond in time. It may still be starting up."
            ) from exc

        status = resp.text.strip()
        if resp.status_code != 200 or "CONNECTED" not in status.upper():
            raise ThetaNotRunning(
                f"Theta Terminal is running but not connected to the data feed.\n"
                f"It reported: {status!r}\n"
                "Check that your ThetaData subscription is active and credentials are correct."
            )
        return {"status": status, "base_url": self.cfg.base_url}

    # -- reference data ----------------------------------------------------

    def list_expirations(self, root: str) -> list[date]:
        """Every expiration ThetaData knows about for this root, ascending."""
        page = self._get("/v2/list/expirations", {"root": root})
        out: list[date] = []
        for line in page.text.splitlines():
            line = line.strip()
            if not line or not line.isdigit():
                continue
            out.append(date(int(line[:4]), int(line[4:6]), int(line[6:8])))
        return sorted(out)

    def list_strikes(self, root: str, exp: date) -> list[float]:
        """Strikes listed for one expiration. ThetaData returns strikes x1000."""
        page = self._get("/v2/list/strikes", {"root": root, "exp": _fmt_date(exp)})
        out: list[float] = []
        for line in page.text.splitlines():
            line = line.strip()
            if not line or not line.lstrip("-").isdigit():
                continue
            out.append(int(line) / 1000.0)
        return sorted(out)

    # -- underlying --------------------------------------------------------

    def stock_ohlc(
        self,
        root: str,
        start: date,
        end: date,
        interval_ms: int,
        rth_only: bool = True,
    ) -> Iterator[ThetaResponse]:
        """OHLCV bars for the underlying. interval_ms=0 requests daily bars."""
        params: dict[str, object] = {
            "root": root,
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
            "ivl": interval_ms,
            "rth": "true" if rth_only else "false",
        }
        endpoint = "/v2/hist/stock/eod" if interval_ms == 0 else "/v2/hist/stock/ohlc"
        if interval_ms == 0:
            params.pop("ivl")
            params.pop("rth")
        yield from self._paged(endpoint, params)

    # -- options -----------------------------------------------------------

    def bulk_option_eod(self, root: str, exp: date, start: date, end: date):
        """End-of-day greeks/quote snapshot for every strike of one expiration.

        This is the cheap dataset: one row per contract per day.
        """
        params: dict[str, object] = {
            "root": root,
            "exp": _fmt_date(exp),
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
        }
        yield from self._paged("/v2/bulk_hist/option/eod", params)

    def bulk_option_quotes(
        self,
        root: str,
        exp: date,
        start: date,
        end: date,
        interval_ms: int,
        rth_only: bool = True,
    ) -> Iterator[ThetaResponse]:
        """Intraday NBBO quotes for every strike of one expiration.

        This is the expensive dataset and the one the pilot download measures.
        """
        params: dict[str, object] = {
            "root": root,
            "exp": _fmt_date(exp),
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
            "ivl": interval_ms,
            "rth": "true" if rth_only else "false",
        }
        yield from self._paged("/v2/bulk_hist/option/quote", params)

    def bulk_option_greeks(
        self,
        root: str,
        exp: date,
        start: date,
        end: date,
        interval_ms: int,
        rth_only: bool = True,
    ) -> Iterator[ThetaResponse]:
        """Intraday greeks and implied volatility for every strike."""
        params: dict[str, object] = {
            "root": root,
            "exp": _fmt_date(exp),
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
            "ivl": interval_ms,
            "rth": "true" if rth_only else "false",
        }
        yield from self._paged("/v2/bulk_hist/option/greeks", params)
