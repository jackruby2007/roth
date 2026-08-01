"""ThetaData endpoint probe.

The client in `thetadata.py` was written without access to ThetaData's docs or
API, so its endpoint paths and response shapes are informed guesses. This module
tests all of them against a live Theta Terminal in one pass, records exactly
what happened, and writes a report.

It also probes the *alternate* paths each endpoint might live at, so if a guess
is wrong the probe finds the right one rather than just reporting a failure.

Two other things it discovers empirically, because neither could be looked up:

* What the account's subscription tier actually permits.
* How far back history reaches, by walking back year by year until data stops.

The point is a single round trip: run this once, send the report, and the client
gets corrected from evidence instead of from more guessing.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta

import httpx

from roth.config import THETA
from roth.paths import REPORTS, ensure_dirs

MAX_BODY_CHARS = 400

# Body lines retained for parsing follow-up probe parameters.
MAX_KEPT_LINES = 5000


@dataclass
class ProbeResult:
    label: str
    method: str
    path: str
    params: dict
    status: int | None = None
    ok: bool = False
    error: str = ""
    header_line: str = ""
    row_count: int = 0
    sample_row: str = ""
    next_page: str | None = None
    elapsed_ms: float = 0.0
    # A bounded slice of the body, kept so list endpoints can be parsed for
    # follow-up probes without holding a whole chain in memory.
    body_lines: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"OK    {self.label}  ({self.row_count} rows, {self.elapsed_ms:.0f}ms)"
        code = self.status if self.status is not None else "---"
        return f"FAIL  {self.label}  [{code}] {self.error[:120]}"


@dataclass
class ProbeReport:
    results: list[ProbeResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    history_depth: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    @property
    def passed(self) -> list[ProbeResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> list[ProbeResult]:
        return [r for r in self.results if not r.ok]


def _fmt(d: date) -> str:
    return d.strftime("%Y%m%d")


class ThetaProbe:
    """Runs every probe against a live Theta Terminal."""

    def __init__(self, root: str = "SPY", timeout: float = 60.0) -> None:
        self.root = root
        self.base = THETA.base_url
        self.client = httpx.Client(
            base_url=self.base, timeout=httpx.Timeout(timeout), trust_env=False
        )
        self.report = ProbeReport()
        # Filled in by discovery, then reused by later probes.
        self.sample_expiration: date | None = None
        self.sample_strike: float | None = None
        self.sample_session: date | None = None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> ThetaProbe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- one request -------------------------------------------------------

    def probe(self, label: str, path: str, params: dict | None = None) -> ProbeResult:
        import time

        params = dict(params or {})
        params.setdefault("use_csv", "true")
        result = ProbeResult(label=label, method="GET", path=path, params=params)

        started = time.monotonic()
        try:
            resp = self.client.get(path, params=params)
        except httpx.ConnectError as exc:
            result.error = f"could not connect: {exc}"
            self.report.results.append(result)
            return result
        except httpx.ReadTimeout:
            result.error = "timed out"
            self.report.results.append(result)
            return result
        except Exception as exc:  # noqa: BLE001 - the probe must never crash
            result.error = f"{type(exc).__name__}: {exc}"
            self.report.results.append(result)
            return result

        result.elapsed_ms = (time.monotonic() - started) * 1000
        result.status = resp.status_code
        nxt = resp.headers.get("Next-Page")
        result.next_page = None if nxt in (None, "null", "") else nxt

        text = resp.text
        if resp.status_code != 200:
            result.error = text[:MAX_BODY_CHARS].replace("\n", " ")
            self.report.results.append(result)
            return result

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            result.error = "HTTP 200 but empty body"
            self.report.results.append(result)
            return result

        result.header_line = lines[0][:MAX_BODY_CHARS]
        result.row_count = max(0, len(lines) - 1)
        if len(lines) > 1:
            result.sample_row = lines[1][:MAX_BODY_CHARS]
        result.body_lines = [ln.strip() for ln in lines[:MAX_KEPT_LINES]]
        result.ok = result.row_count > 0 or "," in lines[0]

        self.report.results.append(result)
        return result

    @staticmethod
    def _parse_dates(result: ProbeResult) -> list[date]:
        """Pull YYYYMMDD values out of a list endpoint's body."""
        out: list[date] = []
        for line in result.body_lines:
            token = line.split(",")[0].strip()
            if len(token) == 8 and token.isdigit():
                try:
                    out.append(date(int(token[:4]), int(token[4:6]), int(token[6:8])))
                except ValueError:
                    continue
        return sorted(out)

    @staticmethod
    def _parse_strikes(result: ProbeResult) -> list[float]:
        """Pull strike values out of a list endpoint's body.

        ThetaData reports strikes multiplied by 1000.
        """
        out: list[float] = []
        for line in result.body_lines:
            token = line.split(",")[0].strip()
            if token.isdigit() and int(token) > 0:
                out.append(int(token) / 1000.0)
        return sorted(out)

    def probe_alternates(self, label: str, paths: list[str], params: dict) -> ProbeResult | None:
        """Try several candidate paths; stop at the first that works.

        This is what turns a wrong guess into a discovery rather than a dead
        end.
        """
        first_failure = None
        for i, path in enumerate(paths):
            tag = label if i == 0 else f"{label} [alt: {path}]"
            result = self.probe(tag, path, params)
            if result.ok:
                if i > 0:
                    self.report.notes.append(
                        f"{label}: the configured path failed but {path} works. "
                        "The client should be pointed at this one."
                    )
                return result
            first_failure = first_failure or result
        return first_failure

    # -- the probes --------------------------------------------------------

    def run(self) -> ProbeReport:
        self.report.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "base_url": self.base,
            "probe_root": self.root,
            "probe_run_at": date.today().isoformat(),
        }

        self._probe_status()
        self._probe_reference()
        self._probe_underlying()
        self._probe_options()
        self._probe_history_depth()
        return self.report

    def _probe_status(self) -> None:
        for label, path in (
            ("terminal status", "/v2/system/mdds/status"),
            ("terminal version", "/v2/system/terminal/version"),
        ):
            r = self.probe(label, path)
            # These return plain text, not CSV, so judge them on status alone.
            r.ok = r.status == 200

    def _probe_reference(self) -> None:
        roots = self.probe_alternates(
            "list option roots", ["/v2/list/roots/option", "/v2/list/roots"], {"sec": "OPTION"}
        )
        if roots and roots.ok:
            self.report.notes.append(f"root list returned {roots.row_count} symbols")

        exps = self.probe_alternates(
            "list expirations", ["/v2/list/expirations"], {"root": self.root}
        )
        if exps and exps.ok:
            parsed = self._parse_dates(exps)
            if parsed:
                today = date.today()
                future = [d for d in parsed if d >= today]
                # Prefer a near-dated expiration that has already passed, so
                # historical queries against it are certain to have data.
                past = [d for d in parsed if d < today]
                self.sample_expiration = (
                    past[-1] if past else (future[0] if future else parsed[-1])
                )
                self.report.notes.append(
                    f"expirations: {len(parsed)} listed, "
                    f"{min(parsed)} to {max(parsed)}; probing with "
                    f"{self.sample_expiration}"
                )

        if self.sample_expiration:
            strikes = self.probe(
                "list strikes",
                "/v2/list/strikes",
                {"root": self.root, "exp": _fmt(self.sample_expiration)},
            )
            if strikes.ok:
                values = self._parse_strikes(strikes)
                # Take a middle strike: the extremes of a chain are the least
                # likely to carry a usable quote.
                self.sample_strike = values[len(values) // 2] if values else None
                self.report.notes.append(
                    f"strikes: {strikes.row_count} listed for {self.sample_expiration}"
                    + (f", probing with {self.sample_strike:g}" if self.sample_strike else "")
                )

    def _recent_session(self) -> date:
        """A weekday roughly a week back, safe for historical queries."""
        d = date.today() - timedelta(days=7)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    def _probe_underlying(self) -> None:
        session = self._recent_session()
        self.sample_session = session
        week_ago = session - timedelta(days=7)

        self.probe_alternates(
            "stock EOD",
            ["/v2/hist/stock/eod"],
            {"root": self.root, "start_date": _fmt(week_ago), "end_date": _fmt(session)},
        )

        self.probe_alternates(
            "stock 1-minute OHLC",
            ["/v2/hist/stock/ohlc"],
            {
                "root": self.root,
                "start_date": _fmt(session),
                "end_date": _fmt(session),
                "ivl": 60_000,
                "rth": "true",
            },
        )

        self.probe(
            "VIX EOD",
            "/v2/hist/stock/eod",
            {"root": "VIX", "start_date": _fmt(week_ago), "end_date": _fmt(session)},
        )

        self.probe(
            "index EOD (VIX as index)",
            "/v2/hist/index/eod",
            {"root": "VIX", "start_date": _fmt(week_ago), "end_date": _fmt(session)},
        )

    def _probe_options(self) -> None:
        if not self.sample_expiration or not self.sample_session:
            self.report.notes.append(
                "skipped option probes: no expiration was discovered to probe with"
            )
            return

        exp = _fmt(self.sample_expiration)
        session = self.sample_session
        # Query a session before the expiration, otherwise the contract is dead.
        query_day = min(session, self.sample_expiration - timedelta(days=3))
        while query_day.weekday() >= 5:
            query_day -= timedelta(days=1)

        common = {
            "root": self.root,
            "exp": exp,
            "start_date": _fmt(query_day),
            "end_date": _fmt(query_day),
        }

        self.probe_alternates(
            "bulk option EOD",
            ["/v2/bulk_hist/option/eod", "/v2/bulk_hist/option/eod_greeks"],
            dict(common),
        )

        self.probe_alternates(
            "bulk option EOD greeks",
            [
                "/v2/bulk_hist/option/eod_greeks",
                "/v2/bulk_hist/option/greeks",
                "/v2/bulk_hist/option/trade_greeks",
            ],
            dict(common),
        )

        self.probe_alternates(
            "bulk option 1-min quotes",
            ["/v2/bulk_hist/option/quote"],
            dict(common) | {"ivl": 60_000, "rth": "true"},
        )

        self.probe_alternates(
            "bulk option 1-min greeks",
            ["/v2/bulk_hist/option/greeks"],
            dict(common) | {"ivl": 60_000, "rth": "true"},
        )

        if self.sample_strike is not None:
            self.probe(
                "single-contract EOD",
                "/v2/hist/option/eod",
                dict(common)
                | {"strike": int(self.sample_strike * 1000), "right": "C"},
            )

    def _probe_history_depth(self) -> None:
        """Walk back year by year until the account stops returning data.

        This is how the subscription tier's real history limit gets discovered,
        rather than assumed from a pricing page.
        """
        today = date.today()
        for years_back in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
            probe_day = today - timedelta(days=365 * years_back)
            while probe_day.weekday() >= 5:
                probe_day -= timedelta(days=1)

            r = self.probe(
                f"history depth: {years_back}y back ({probe_day})",
                "/v2/hist/stock/eod",
                {
                    "root": self.root,
                    "start_date": _fmt(probe_day),
                    "end_date": _fmt(probe_day + timedelta(days=5)),
                },
            )
            self.report.history_depth[f"{years_back}y"] = {
                "date": probe_day.isoformat(),
                "status": r.status,
                "rows": r.row_count,
                "ok": r.ok,
            }
            if not r.ok:
                self.report.notes.append(
                    f"stock history stops before {probe_day} "
                    f"({years_back} years back): HTTP {r.status}"
                )
                break


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def render_report(report: ProbeReport) -> str:
    out: list[str] = []
    out.append("=" * 78)
    out.append(" ThetaData endpoint probe")
    out.append("=" * 78)

    out.append("")
    out.append("ENVIRONMENT")
    for k, v in report.environment.items():
        out.append(f"  {k}: {v}")

    out.append("")
    out.append(f"SUMMARY: {len(report.passed)} ok, {len(report.failed)} failed")
    out.append("-" * 78)
    for r in report.results:
        out.append("  " + r.summary())

    if report.notes:
        out.append("")
        out.append("NOTES")
        out.append("-" * 78)
        for n in report.notes:
            out.append(f"  - {n}")

    if report.history_depth:
        out.append("")
        out.append("HISTORY DEPTH (how far back this account can read)")
        out.append("-" * 78)
        for k, v in report.history_depth.items():
            mark = "ok " if v["ok"] else "NO "
            out.append(f"  {mark} {k:>4}  {v['date']}  HTTP {v['status']}  {v['rows']} rows")

    out.append("")
    out.append("DETAIL")
    out.append("=" * 78)
    for r in report.results:
        out.append("")
        out.append(f"[{'OK' if r.ok else 'FAIL'}] {r.label}")
        out.append(f"  path   : {r.path}")
        out.append(f"  params : {json.dumps(r.params, default=str)}")
        out.append(f"  status : {r.status}")
        if r.error:
            out.append(f"  error  : {r.error}")
        if r.header_line:
            out.append(f"  columns: {r.header_line}")
        if r.sample_row:
            out.append(f"  row    : {r.sample_row}")
        if r.next_page:
            out.append("  paged  : yes")

    out.append("")
    out.append("=" * 78)
    return "\n".join(out)


def run_probe(root: str = "SPY") -> tuple[ProbeReport, str]:
    """Run every probe and write the report to disk."""
    ensure_dirs()
    with ThetaProbe(root=root) as probe:
        report = probe.run()

    text = render_report(report)
    path = REPORTS / "thetadata_probe.txt"
    path.write_text(text)
    return report, str(path)
