# First run with real data

You have a ThetaData account. This is what to do next, in order.

Everything here runs on **your** machine. The development environment this was
built in cannot reach ThetaData — their servers are blocked by its network
policy — so the first contact with real data has to happen on yours.

If any step produces an error, copy the whole error and send it back. That is
the only thing you should ever have to do.

---

## Step 1 — Install three things

**uv** (manages Python; you do not need Python installed first)

| Your machine | Command |
| --- | --- |
| macOS / Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

**git** — Windows does not ship with it. macOS and most Linux systems already
have it.

| Your machine | Where |
| --- | --- |
| Windows | Download from git-scm.com, run the installer, accept the defaults |
| macOS | Already present, or `xcode-select --install` |
| Linux | `sudo apt install git` |

**Java 21 or newer** — Theta Terminal is a Java program and will not start
without it.

| Your machine | Where |
| --- | --- |
| macOS | `brew install openjdk@21`, or download from adoptium.net |
| Windows | Download from adoptium.net, run the installer |
| Linux | `sudo apt install openjdk-21-jre` |

Check it worked:

```
java -version
```

You want to see `21` or higher. If you see "command not found", Java did not
install correctly — send that back.

---

## Step 2 — Get the code and install it

```
git clone https://github.com/jackruby2007/roth.git
cd roth
git checkout claude/options-research-harness-h163q1
uv sync --extra dev
```

Then:

```
uv run roth doctor
```

Expected: every dependency shows `OK`, and Theta Terminal shows `NOT READY`.
That is correct at this point — the terminal is not running yet.

---

## Step 3 — Start Theta Terminal

Download it from your ThetaData account dashboard. It is a `.jar` file.

Put it in the `roth` folder you just cloned. That keeps the command short, and
`.jar` files are excluded from version control so it will not be committed.

Then run:

```
java -jar ThetaTerminal.jar YOUR_EMAIL YOUR_PASSWORD
```

Use the email and password you signed up with.

**Leave this window open.** The terminal must keep running for the whole
download. Wait until it prints `CONNECTED` before continuing.

This is the only process in the entire system that ever needs to be running,
and only while data is downloading. Once the download finishes you can close
it and never think about it again.

In a **second** terminal window, in the `roth` folder:

```
uv run roth doctor
```

Theta Terminal should now show `OK`.

---

## Step 4 — Probe the API

```
uv run roth probe
```

This is the important one. The ThetaData client in this harness was written
without access to their documentation or their API, so its endpoint paths are
educated guesses. The probe tests every one of them, tries alternatives where a
path might differ, and works out how far back your free account can actually
read.

It writes a file:

```
data/reports/thetadata_probe.txt
```

**Send that file back.** It contains no credentials — only endpoint paths, HTTP
status codes, and column names.

Expect some failures. That is the point of running it. One probe run tells me
everything needed to fix the client in one pass instead of a dozen.

Do not continue past this step until the client has been corrected. Steps 5 and
6 will fail in confusing ways if the endpoints are wrong.

---

## Step 5 — The pilot download

Once the client is corrected:

```
uv run roth pilot
uv run roth estimate
```

`pilot` downloads one month of real SPY option data at the exact bounds a full
backfill would use, and measures two things: how much disk a month costs, and
how long a month takes to pull. `estimate` extrapolates those to the full
history.

Send back what those print. Those are the real numbers behind the subscription
decision — the ones currently marked MODELED.

---

## Step 6 — Real data end to end

```
uv run roth ingest calendar
uv run roth ingest underlying
uv run roth ingest options-eod
uv run roth quality
uv run roth features build
uv run roth features verify
uv run roth verify
uv run roth backtest
```

The synthetic dataset gets replaced by real data along the way. Two things to
watch for:

- `roth quality` will report findings the synthetic data never had. That is
  expected and is what the quarantine table is for.
- `roth verify` will still SKIP the external known-answer check until a
  benchmark file exists. See below.

---

## Step 7 — The benchmark file

This is the check that proves the harness is connected to reality rather than
merely self-consistent. Create:

```
data/raw/reference/benchmarks.csv
```

with this content:

```
symbol,start,end,total_return_pct,source
SPY,2023-01-03,2023-12-29,<published figure>,<where you got it>
```

Use a published SPY price return for a calendar year from a source you trust.
Then `roth verify` compares the harness's own buy-and-hold calculation against
it.

If those two numbers disagree, something is wrong and no other result from the
harness should be believed until it is fixed. That is the whole point of the
check, and it is why it refuses to pass on generated data.

---

## What could go wrong

| Symptom | What it means |
| --- | --- |
| `roth doctor` says Theta Terminal is not running | The terminal window closed, or has not printed CONNECTED yet |
| Probe shows many 472 responses | The free tier does not cover those endpoints. Useful information — send it |
| Probe shows 404 everywhere | The endpoint paths are wrong. Exactly what the probe exists to find |
| Download is far slower than estimated | Free tiers are rate limited. The pilot measures this honestly |
| `roth verify` fails after real data lands | Send the output. A failure here is worth more than a pass |
