"""One HTTP client for every news source.

Three things live here because getting them wrong in eight different places is
how a polling bot gets itself IP-banned:

* a declared User-Agent, which SEC requires and which several feeds use to
  decide whether to answer at all,
* a token-bucket rate limiter shared across sources, and
* retry with exponential backoff that treats 429 and 5xx as transient and
  everything else as final.

Unlike the ThetaData client, this one *does* honour proxy environment
variables: these are public internet endpoints and the caller's network may
require a proxy to reach them.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

import httpx

from roth.news.config import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, user_agent


class FetchError(RuntimeError):
    """A request that failed in a way the caller should report, not retry."""


class RateLimited(FetchError):
    """The remote asked us to slow down and kept asking."""


@dataclass
class _Bucket:
    """Token bucket. `rate` tokens accrue per second, capped at `capacity`."""

    rate: float
    capacity: float
    tokens: float
    updated: float


class RateLimiter:
    """Shared across every source so the process has one global budget.

    SEC publishes a limit of 10 requests/second. The default here is 4, which
    is well inside it and leaves room for the retry path to spend a few extra
    requests without crossing the line.
    """

    def __init__(self, rate: float = 4.0, capacity: float = 8.0) -> None:
        self._bucket = _Bucket(rate, capacity, capacity, time.monotonic())
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                b = self._bucket
                b.tokens = min(b.capacity, b.tokens + (now - b.updated) * b.rate)
                b.updated = now
                if b.tokens >= 1.0:
                    b.tokens -= 1.0
                    return
                wait = (1.0 - b.tokens) / b.rate
            time.sleep(wait)


_LIMITER = RateLimiter()

# Status codes worth trying again. Everything else -- 401, 403, 404 -- means
# retrying will fail identically, so it is surfaced immediately.
_RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


class NewsHttp:
    """Synchronous HTTP with retries, shared by all sources.

    Usage::

        with NewsHttp() as http:
            text = http.get_text("https://example.com/feed.xml")
    """

    def __init__(
        self,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.max_retries = max_retries
        self.limiter = limiter or _LIMITER
        self.requests_made = 0
        self.bytes_received = 0
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={
                "User-Agent": user_agent(),
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
        )

    def __enter__(self) -> NewsHttp:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> httpx.Response:
        last: Exception | None = None

        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                resp = self._client.get(url, params=params, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                self._backoff(attempt)
                continue

            self.requests_made += 1
            self.bytes_received += len(resp.content)

            if resp.status_code in _RETRYABLE:
                last = FetchError(f"HTTP {resp.status_code} from {url}")
                # Honour Retry-After when the server sets it; it is usually
                # far more accurate than our own backoff guess.
                delay = _retry_after(resp)
                self._backoff(attempt, override=delay)
                continue

            if resp.status_code >= 400:
                raise FetchError(f"HTTP {resp.status_code} from {url}")

            return resp

        if isinstance(last, FetchError) and "429" in str(last):
            raise RateLimited(str(last)) from last
        raise FetchError(f"{url} failed after {self.max_retries} attempts: {last}") from last

    def get_text(self, url: str, params: dict | None = None, headers: dict | None = None) -> str:
        return self.get(url, params=params, headers=headers).text

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = self.get(url, params=params, headers=headers)
        try:
            return resp.json()
        except ValueError as exc:
            body = resp.text[:200].replace("\n", " ")
            raise FetchError(f"{url} returned non-JSON: {body!r}") from exc

    def _backoff(self, attempt: int, override: float | None = None) -> None:
        if override is not None:
            time.sleep(min(override, 30.0))
            return
        # Full jitter. Without it, eight symbols failing at once retry in
        # lockstep and hit the same limit again together.
        time.sleep(random.uniform(0, min(2**attempt, 8.0)))


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
