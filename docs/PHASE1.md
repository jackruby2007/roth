# Phase 1 status

## Order of work

| # | Step | Status |
| --- | --- | --- |
| 1 | Environment setup and pilot data download | Environment done. Pilot blocked, see below. |
| 2 | Storage layer, ingestion, raw data backfill | Code done. Backfill blocked. |
| 3 | Data quality layer | Done |
| 4 | Feature store with causality enforcement | Done |
| 5 | Backtest engine and options fill model | Done |
| 6 | Correctness tests | Done, 5 pass / 1 skip |
| 7 | Strategy interface, trade journal, reporting | Done |
| 8 | Reference strategy end to end | Done |

## Definition of done

| Requirement | Status |
| --- | --- |
| User runs one command and gets a full report on the reference strategy | `roth backtest` |
| All four correctness tests pass | 5 pass, 1 skips pending real data |
| Data quality report shows the state of the dataset | `roth quality` |
| No background process required except a fresh download | Nothing to run |
| Adding a hypothesis means writing one strategy class and nothing else | One class in `src/roth/strategies/` |

## The blocker

Every ThetaData host is refused by this environment's network egress policy:

```
www.thetadata.net:443        403 at proxy
http-docs.thetadata.us:443   403 at proxy
download-stable.thetadata.us 403 at proxy
```

This is an organisation network policy, not a credentials problem. Theta
Terminal cannot be downloaded or run here either.

Consequences, all of which are stated wherever they matter rather than being
quietly absorbed:

* The pilot download's two measurements are modelled instead of measured.
  `roth estimate` prints MODELED every time until a pilot runs.
* Everything from step 3 onward was built and verified against a **synthetic**
  dataset. `roth synth` generates it; every dataset it writes drops a
  `_SYNTHETIC_DATA` marker, and every report built on one carries a banner.
* The external known-answer check reports SKIP. It refuses to pass on
  generated data, because a fabricated series cannot reproduce a real
  published return.
* The ThetaData client is written but has never spoken to the live API. Its
  v2 endpoint paths could not be verified against the docs, so they are
  collected at the top of `data/thetadata.py` for easy correction.

## Commands

| Command | What it does |
| --- | --- |
| `roth doctor` | Checks dependencies and Theta Terminal reachability |
| `roth pilot` | Downloads one month and measures disk and wall clock |
| `roth estimate` | Extrapolates full backfill cost |
| `roth ingest calendar` | Builds the trading calendar. No subscription needed |
| `roth ingest underlying` | Downloads underlying daily and 1-minute bars |
| `roth ingest options-eod` | Downloads EOD option chains within the bounds |
| `roth synth` | Generates a synthetic dataset for pipeline verification |
| `roth quality` | Runs data quality checks, writes the quarantine table |
| `roth features build` | Builds the feature store |
| `roth features verify` | Proves no feature depends on future data |
| `roth verify` | Runs the four correctness tests |
| `roth backtest` | Runs a strategy and prints the full report |
| `roth strategies` | Lists available strategies |
| `roth status` | Shows what is on disk |

## Adding a hypothesis

Write one class in `src/roth/strategies/`, register it in that package's
`STRATEGIES` dict, and run `roth backtest --strategy <name>`. Nothing else.

```python
class MyHypothesis(Strategy):
    name = "my_hypothesis"
    version = "1.0.0"
    structure_type = "vertical_call_spread"

    def rules(self):
        return [
            ThresholdRule("rsi_oversold", "rsi_14", "<", 30),
            CategoryRule("calm_regime", "vol_bucket", {"low", "mid"}),
        ]

    def contract_spec(self, ctx):
        return ContractSpec(right=Right.CALL, target_dte=30, target_delta=0.30)

    def exits(self):
        return ExitRules(profit_target_pct=0.50, stop_loss_pct=0.50, time_stop_days=10)
```

Every rule evaluation is stored for every candidate signal, failures included,
so `roth backtest --explain <signal_id>` explains any decision.

## Bugs the harness caught in itself

Kept as a record of what the checks are worth.

| Found by | Bug |
| --- | --- |
| Causality verification | `is_quarter_end` and `is_short_week` were computed from the loaded data range, so the last session in any slice looked like a quarter end and labels changed depending on when the table was built |
| Running the quality layer | Minute-bar completeness measured against a constant 390, which would have quarantined every early close in real data |
| Running the quality layer | The generator emitted 390 bars on early closes too, so the half-day path would never have been exercised |
| Smoke-running the engine | Net price was signed by trade side rather than the leg's signed quantity, making every exit value negative and double-counting entry cost |
| Smoke-running the engine | A position whose exit fill was rejected stayed open past expiry, its contract left the bounded chain, and it blocked every later signal while logging a rejection each session |
| Lookahead trap | `sessions_held` was incremented on the entry session, so a one-session time stop exited at the same close it entered and every longer time stop was off by one |
| Reading the first report | Entry and exit rejections were summed and divided by candidate signals, reporting a 48% rejection rate when the true entry rejection rate was 0.7% |
| CLI smoke test | typer 0.15.1 is incompatible with click >= 8.2, which crashed `roth --help` while every unit test passed |

## Non-goals for Phase 1

Out of scope, recorded so they are not lost:

- Automated daily data updates or any scheduled job
- Strategy discovery / combinatorial search
- Machine learning of any kind
- Web dashboards or a UI
- Strategy leaderboard, versioning, or portfolio simulation
- SPX (feed quirks, deferred)
- Any live broker connection

## Architecture constraints

- No server processes. No Postgres, no Redis, no Docker, no web server.
- Storage is parquet files on local disk, queried with DuckDB as a library.
- Everything runs on demand via one CLI entry point.
- Pure Python 3.11+, pinned versions.
- Theta Terminal runs only during a download, never after.
