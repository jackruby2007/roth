# roth

A research harness for testing options trading hypotheses on SPY and QQQ.

It is not a trading bot, not a dashboard, and not a live system. It answers one
question at a time: **does this specific setup have a positive expectancy after
realistic execution costs?**

## How it runs

Everything runs on demand through one command. There is no database server, no
web server, no Docker, and no scheduled job. Storage is parquet files on local
disk, queried with DuckDB, which is a library rather than a service.

The only process that ever needs to be running is ThetaData's Theta Terminal,
and only while a download is in progress. Never after.

## Setup

```
uv sync --extra dev
```

Then confirm everything is in place:

```
uv run roth doctor
```

## Commands available today

| Command | What it does |
| --- | --- |
| `roth doctor` | Checks Python, dependencies, and whether Theta Terminal is reachable. |
| `roth pilot` | Downloads one month of real option quotes and measures disk footprint and download time. |
| `roth estimate` | Extrapolates the full backfill cost, using pilot measurements if they exist. |

## The pilot download

Before paying for years of history, run:

```
uv run roth pilot
uv run roth estimate
```

`roth pilot` pulls one calendar month of SPY option quotes at the exact bounds
the real backfill would use, then records two measured numbers: bytes per row
on disk, and seconds per request. `roth estimate` extrapolates those to the
full history.

Until a pilot has run, `roth estimate` prints **MODELED** numbers, which are
arithmetic on stated assumptions rather than measurements. It says so in the
output, and it lists every assumption it used.

## Ingestion bounds

Full option chains at 1-minute resolution across all strikes and expirations is
a size bomb. The bounds in `src/roth/config.py` keep it finite:

- Expirations within **60 days** of the trade date
- Strikes within **+/-10%** of spot
- Regular trading hours only
- 1-minute resolution

These are enforced at download time, not at query time.

## Data rules

- `data/raw/` is immutable. Once a download lands there it is never edited or
  overwritten.
- `data/derived/` is disposable and always rebuilt from raw.
- Everything is stored as UTC and displayed as America/New_York.

## Project status

Phase 1, step 1 of 8. See `docs/PHASE1.md` for the full plan and where things
stand.
