# The live news bot

Continuous news monitoring for eight large-cap names — NVDA, TSLA, META, AAPL,
AMZN, MSFT, AVGO, GOOGL — through the pre-market, the session, and after hours.

It is a separate subsystem from the research harness. The harness is on-demand,
offline, and reproducible. This is a long-running process that talks to the
public internet. They share the trading calendar and nothing else, and nothing
the bot collects is ever written into `data/raw/` — a live feed is not
reproducible, and the harness's entire claim rests on being so.

---

## Setup

One environment variable is required. SEC's access policy requires a contact
address in the User-Agent and blocks clients that do not send one:

```
export ROTH_SEC_CONTACT='you@example.com'
```

Put it in your shell profile so it survives a reboot. Then verify everything:

```
uv run roth news doctor
```

**Run `doctor` before trusting the bot.** It is not a formality here. This code
was written in an environment whose egress policy blocked every finance and
news host, so no source could be exercised against its live endpoint during
development. The parsers are covered by recorded fixtures in
`tests/test_news_sources.py`; `doctor` is what proves the live endpoints still
match those fixtures. It checks reachability and payload shape for all three
sources, verifies the pinned CIK constants against SEC's own ticker map, and
reports which timezone reading EDGAR's `acceptanceDateTime` needs today.

---

## Commands

| Command | What it does |
| --- | --- |
| `roth news watch` | Poll all day. Runs until Ctrl-C. |
| `roth news brief` | Build the brief now: prices, moves, and the news behind them. |
| `roth news once` | One poll, emit what is new, exit. Suitable for cron. |
| `roth news doctor` | Verify every source end to end on this machine. |
| `roth news symbols` | The watchlist and the CIK each symbol resolves to. |

The two you will actually use:

```
uv run roth news brief          # over coffee, before the open
uv run roth news watch          # leave running all day
```

### Useful flags

```
--symbols NVDA,TSLA             # a subset of the watchlist
--min-score 60                  # quieter; overrides the per-phase default
--sink desktop --sink console   # repeatable; console is the default
--sink webhook --webhook-url ...# Slack or Discord incoming webhook
--explain                       # print why each item scored as it did
--replay                        # (once) show what the bot would have said
```

---

## What a day looks like

| Phase | Window (ET) | Poll interval | Alert floor |
| --- | --- | --- | --- |
| overnight | 20:00 – 04:00 | 15 min | — |
| pre-market | 04:00 – 09:30 | 60 s | 25 |
| regular | 09:30 – close | 60 s | 40 |
| after hours | close – 20:00 | 3 min | 30 |
| closed | weekends, holidays | 30 min | — |

The regular session's open and close come from the exchange calendar, not from
constants, so holidays, half-days (13:00 closes), and both daylight-saving
transitions are handled without special cases. The bot never sleeps past a
phase boundary: at 09:20 it wakes for the bell rather than eleven minutes into
the session.

**The pre-open brief** fires once per session day, on the first poll at or
after 08:00 ET — late enough for the overnight tape to be complete, early
enough to still be preparation.

**The first cycle primes rather than alerts.** Starting the bot at 11:00 prints
a brief and marks everything already published as seen. Without this, starting
mid-session fires forty alerts for headlines you have already read.

---

## Sources

| Source | Cost | Authority | What it gives |
| --- | --- | --- | --- |
| SEC EDGAR submissions | free, no key | **authoritative** | 8-K, 10-Q, 10-K, SC 13D/G, Form 4 |
| Yahoo Finance RSS | free, no key | unofficial | per-symbol headlines |
| Yahoo chart API | free, no key | unofficial | price, previous close, pre/post print |

EDGAR is the only authoritative source. Everything else is somebody's
description of an event; a filing *is* the event, timestamped by the SEC when
it was accepted. For the 16:05 ET earnings 8-K, it lands here before any wire
story exists.

The two Yahoo endpoints are unofficial and carry no uptime promise. That is why
a dead source **degrades** rather than crashing: the bot reports the headline
stream as unavailable and keeps delivering filings, instead of implying the day
was quiet.

### Two traps handled explicitly

**EDGAR is keyed on CIK, not ticker.** A wrong CIK returns another company's
filings and never errors — nothing downstream would notice. The CIKs are pinned
as constants in `src/roth/news/config.py` and cross-checked against SEC's own
ticker map by `doctor`.

**`acceptanceDateTime` carries a `Z` suffix but is Eastern time.** Believing the
suffix backdates every filing by four or five hours, putting an after-close 8-K
before the close it followed. The parser tests both readings against the clock
and takes the one that is not in the future, so it stays correct even if SEC
changes the format.

---

## Scoring

Eight mega-caps produce an enormous amount of text per day and almost none of
it is news. An alert stream that includes the "3 reasons to buy" content farming
is one you stop reading by Wednesday, which is worse than no bot at all.

Every item gets a 0–100 materiality score:

- **Filings** score from the form and, for an 8-K, the item code — which says
  far more than the form does. Item 2.02 (earnings) is 92; item 9.01 (exhibits
  only) is 30; a Form 4 is 22, visible in the brief but never loud enough to
  interrupt the session on its own.
- **Headlines** score from wording and publisher. Guidance changes, M&A,
  recalls, and executive departures score high; listicles, "if you invested
  $1,000" pieces, and known content farms are penalised hard, often to zero.

Run with `--explain` to see every contribution as a reason string. A scoring
model you cannot interrogate is one you cannot correct — so if the bot is too
loud or too quiet, `--explain` tells you which rule to change, and the rules are
plain tables at the top of `src/roth/news/score.py`.

---

## Duplicate suppression

Two kinds of duplicate, two different keys, both persisted to
`data/news/seen.json`:

- **The same item, seen again.** Keyed on a canonicalised URL, because Yahoo
  appends a fresh tracking parameter on every request — without this the same
  article alerts every sixty seconds.
- **The same story from a second outlet.** Keyed on the normalised title, so one
  Reuters report reaching four aggregators alerts once.

Persistence matters more than it looks: the failure mode of an in-memory store
is restarting at 09:25 and alerting the entire overnight feed as breaking news
at the open.

Price moves are suppressed the same way. A stock that opens down 4% is down 4%
all morning, so a move re-alerts only once it extends by another 1% or reverses
through zero.

---

## Leaving it running

`watch` is a foreground process. Pick whichever fits your machine.

**A terminal multiplexer**, simplest:

```
tmux new -s news 'uv run roth news watch --sink console --sink desktop'
```

**launchd (macOS)** — `~/Library/LaunchAgents/com.roth.news.plist`, with
`ROTH_SEC_CONTACT` in `EnvironmentVariables` and `KeepAlive` set, running
`uv run --directory /path/to/roth roth news watch --sink desktop --sink jsonl`.

**systemd (Linux)** — a user unit with `Restart=always` and
`Environment=ROTH_SEC_CONTACT=you@example.com`.

**cron**, if you would rather not keep a process alive. `once` is built for it:

```
*/2 13-20 * * 1-5  cd /path/to/roth && uv run roth news once --sink desktop
55  11    * * 1-5  cd /path/to/roth && uv run roth news brief --sink desktop
```

Note those hours are **UTC**, and the US market's UTC offset changes twice a
year. The `watch` process handles that itself; a crontab does not.

---

## Limits worth knowing

**This is polling, not a real-time feed.** Alerts arrive within one poll
interval — up to 60 seconds during the session. SEC filings surface quickly
after acceptance; Yahoo's headline feed already lags the wire by minutes on top
of that. If you need sub-second latency, you need a paid streaming feed, and
this is the wrong architecture for it.

**Yahoo can break without warning.** Both Yahoo endpoints are unofficial. If
they change shape or start rate-limiting, `doctor` will say so plainly, and the
bot degrades to filings-only rather than going quiet.

**FMP news is not used.** The account this was built against returns
`ACCESS DENIED` for every news endpoint — they require the Starter plan or
above. If you upgrade, FMP would be a good keyed source to add alongside Yahoo;
the `Source` protocol in `src/roth/news/sources/__init__.py` is the extension
point, and it is one class with one `fetch` method.

**Coverage is not exhaustive.** Two feeds and one filings API will miss things —
a story that breaks on a podcast, a regulator's own site, or a post on X will
not appear until an outlet writes it up.

**Nothing here is investment advice, and the bot never trades.** It reads public
sources and prints them. Materiality scores are heuristics about *attention*,
not about direction or magnitude.
