# Phase 1 status

## Order of work

| # | Step | Status |
| --- | --- | --- |
| 1 | Environment setup and pilot data download | **Environment done. Pilot blocked — see below.** |
| 2 | Storage layer, ingestion, raw data backfill | Not started |
| 3 | Data quality layer | Not started |
| 4 | Feature store with causality enforcement | Not started |
| 5 | Backtest engine and options fill model | Not started |
| 6 | Correctness tests | Not started |
| 7 | Strategy interface, trade journal, reporting | Not started |
| 8 | Reference strategy end to end | Not started |

## Step 1: what is done

- Python 3.11 project managed by `uv`, all dependency versions pinned in
  `pyproject.toml`.
- Package layout under `src/roth/`, single CLI entry point (`roth`).
- Filesystem layout with the raw/derived split enforced in `paths.py`.
- Ingestion bounds and cost model centralised in `config.py`.
- ThetaData REST client (`data/thetadata.py`) using bulk endpoints, CSV
  transport, header-driven pagination, and retry with backoff.
- `roth pilot` — downloads one month of SPY option quotes at the real bounds
  and measures disk footprint and wall-clock time.
- `roth estimate` — extrapolates full-history cost, using pilot measurements
  when present and clearly labelled model arithmetic when not.
- `roth doctor` — environment and Theta Terminal health check.
- 16 tests covering the size model and the CSV/calendar plumbing.

## Step 1: what is blocked

The pilot download cannot run from the current development container. Every
ThetaData host is refused at the network egress policy:

```
www.thetadata.net:443        403 at proxy
http-docs.thetadata.us:443   403 at proxy
download-stable.thetadata.us 403 at proxy
```

This is an organisation network policy, not a credentials problem. It also
means Theta Terminal cannot be downloaded or run here.

Consequence: the two numbers the pilot is supposed to *measure* are currently
*modelled* instead. `roth estimate` says so every time it runs.

The pilot itself is written and ready. It runs on any machine that can reach
ThetaData and has Theta Terminal running.

## Non-goals for Phase 1

Out of scope, not to be built, noted here so they are not lost:

- Automated daily data updates or any scheduled job
- Strategy discovery / combinatorial search
- Machine learning of any kind
- Web dashboards or a UI
- Strategy leaderboard, versioning, or portfolio simulation
- SPX (feed quirks, deferred)
- Any live broker connection

## Architecture constraints

Hard requirements, chosen to keep the ops burden near zero:

- No server processes. No Postgres, no Redis, no Docker, no web server.
- Storage is parquet files on local disk, queried with DuckDB as a library.
- Everything runs on demand via one CLI entry point.
- Pure Python 3.11+, pinned versions.
- Theta Terminal runs only during a download, never after.
