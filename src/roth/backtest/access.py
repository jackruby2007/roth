"""Gated data access.

The engine walks forward through time and *cannot* read a row it has not
reached. That is enforced here, in the data access layer, rather than left to
each strategy's good behaviour.

Every read goes through a `TimeGate`. The gate holds a cursor, and any request
for data at or after the cursor's next session raises `LookaheadError`. A
strategy that tries to peek does not get a subtly optimistic backtest -- it gets
an exception with the offending date in the message.

The one way to see more data is `advance()`, which the engine alone calls. This
is also what makes the fill rule mechanical rather than aspirational: a signal
is evaluated while the cursor sits on session D, and the engine must advance to
D+1 before a fill can be priced, because the D+1 chain is unreadable until then.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


class LookaheadError(RuntimeError):
    """Raised when code attempts to read data from the future.

    This is always a bug in the strategy or the engine, never a data problem.
    """


class TimeGate:
    """A forward-only cursor over one symbol's features and option chains."""

    def __init__(
        self,
        features: pd.DataFrame,
        chains: pd.DataFrame | None = None,
        symbol: str = "",
    ) -> None:
        if features.empty:
            raise ValueError("TimeGate needs at least one session of features")

        self.symbol = symbol or str(features["symbol"].iloc[0])

        self._features = features.sort_values("day").reset_index(drop=True)
        self._days: list[date] = list(self._features["day"])
        self._day_index = {d: i for i, d in enumerate(self._days)}

        # Chains are indexed by day once, up front. Slicing a grouped dict is
        # far cheaper than filtering a million-row frame on every bar.
        self._chains_by_day: dict[date, pd.DataFrame] = {}
        if chains is not None and not chains.empty:
            for day, grp in chains.groupby("day"):
                self._chains_by_day[day] = grp.reset_index(drop=True)

        self._cursor = -1

    # -- cursor ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._days)

    @property
    def started(self) -> bool:
        return self._cursor >= 0

    @property
    def now(self) -> date:
        """The session the cursor currently sits on."""
        if self._cursor < 0:
            raise LookaheadError("The clock has not started; call advance() first.")
        return self._days[self._cursor]

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._days) - 1

    def advance(self) -> bool:
        """Move to the next session. Returns False when there are none left."""
        if self.exhausted:
            return False
        self._cursor += 1
        return True

    def peek_next_day(self) -> date | None:
        """The date of the next session, without revealing any of its data.

        Knowing that a session exists is not the same as seeing its prices, and
        the engine needs the former to schedule a fill.
        """
        if self.exhausted:
            return None
        return self._days[self._cursor + 1]

    # -- guarded reads -----------------------------------------------------

    def _guard(self, day: date) -> None:
        if self._cursor < 0:
            raise LookaheadError("The clock has not started; call advance() first.")
        if day > self.now:
            raise LookaheadError(
                f"Attempted to read {self.symbol} data for {day} while the clock is "
                f"on {self.now}. This is lookahead: the value would not have been "
                "knowable at signal time."
            )

    def current(self) -> pd.Series:
        """The feature row for the current session."""
        if self._cursor < 0:
            raise LookaheadError("The clock has not started; call advance() first.")
        return self._features.iloc[self._cursor]

    def row(self, day: date) -> pd.Series | None:
        """The feature row for a past or current session."""
        self._guard(day)
        idx = self._day_index.get(day)
        return None if idx is None else self._features.iloc[idx]

    def history(self, lookback: int | None = None) -> pd.DataFrame:
        """Every session up to and including the current one.

        Never includes the future. `lookback` limits how far back it reaches.
        """
        if self._cursor < 0:
            raise LookaheadError("The clock has not started; call advance() first.")
        start = 0 if lookback is None else max(0, self._cursor + 1 - lookback)
        return self._features.iloc[start : self._cursor + 1]

    def chain(self, day: date | None = None) -> pd.DataFrame:
        """The option chain for a past or current session.

        Requesting a future session raises rather than returning empty, because
        an empty frame would be silently mistaken for "no contracts listed".
        """
        day = self.now if day is None else day
        self._guard(day)
        return self._chains_by_day.get(day, pd.DataFrame())

    def has_chain(self, day: date) -> bool:
        self._guard(day)
        return day in self._chains_by_day

    # -- helpers -----------------------------------------------------------

    def sessions_remaining(self) -> int:
        return len(self._days) - self._cursor - 1

    def day_offset(self, day: date) -> int | None:
        """Index distance from the cursor to `day`, or None if unknown."""
        idx = self._day_index.get(day)
        return None if idx is None else idx - self._cursor
