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

## Quick start

Without a data subscription, the whole pipeline can be exercised on a generated
dataset:

```
uv run roth synth            # generate a synthetic dataset
uv run roth quality          # check it, write the quarantine table
uv run roth features build   # build the feature store
uv run roth features verify  # prove no feature sees the future
uv run roth verify           # run the correctness tests
uv run roth backtest         # run the reference strategy, print the report
```

The synthetic data is fabricated and every report built on it says so. It
proves the plumbing runs; it says nothing about whether a strategy has an edge.

## Commands

| Command | What it does |
| --- | --- |
| `roth doctor` | Checks dependencies and whether Theta Terminal is reachable. |
| `roth pilot` | Downloads one month of real option quotes and measures disk footprint and download time. |
| `roth estimate` | Extrapolates the full backfill cost, using pilot measurements if they exist. |
| `roth ingest calendar` | Builds the trading calendar. Needs no subscription. |
| `roth ingest underlying` | Downloads underlying daily and 1-minute bars. |
| `roth ingest options-eod` | Downloads EOD option chains within the configured bounds. |
| `roth synth` | Generates a synthetic dataset for pipeline verification. |
| `roth quality` | Runs data quality checks and writes the quarantine table. |
| `roth features build` | Builds the feature store. |
| `roth features verify` | Proves no feature depends on future data. |
| `roth verify` | Runs the four correctness tests. |
| `roth backtest` | Runs a strategy and prints the full performance report. |
| `roth strategies` | Lists available strategies. |
| `roth status` | Shows what is currently on disk. |

## Adding a hypothesis

Write one strategy class and nothing else. See `docs/PHASE1.md` for the shape.
Every rule evaluation is stored for every candidate signal, failures included,
so `roth backtest --explain <signal_id>` explains any decision the engine made.

## What makes the results trustworthy

- **Fills never happen at mid.** Buys cross to the ask, sells to the bid,
  against the historical NBBO. Unusable quotes are rejected, never repaired.
- **Lookahead is structurally blocked.** Every read goes through a forward-only
  cursor that raises on any attempt to read the future.
- **Features are verified causal**, by rebuilding the table with history
  truncated and requiring every earlier value to be identical.
- **`roth verify`** runs a known-answer test, a lookahead trap, a
  cost-sensitivity comparison, and a timezone check.

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
