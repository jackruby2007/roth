"""Causality verification.

The rule is that a feature stamped at time T may use only information available
at or before T. Comments and good intentions do not enforce that. This does.

**The test: point-in-time invariance.** Build the feature table twice. Once with
the full history, and once with the data truncated at some cutoff date T. Every
value for every day at or before T must be *identical* between the two builds.

If a feature peeks forward -- a centred rolling window, a `bfill`, a normalise
against the full-sample mean, a percentile computed over the whole series -- the
truncated build cannot reproduce it, and the comparison fails. That makes the
check sensitive to the entire class of lookahead bugs rather than the specific
ones someone thought to look for.

Run it after touching anything in the feature store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

# Values closer than this are treated as equal, to absorb float noise from
# different-length input arrays flowing through the same arithmetic.
TOLERANCE = 1e-9


@dataclass
class ColumnViolation:
    column: str
    cutoff: date
    rows_differing: int
    max_abs_diff: float
    example_day: date | None
    full_value: object = None
    truncated_value: object = None


@dataclass
class CausalityResult:
    symbol: str
    cutoffs: list[date]
    columns_checked: int = 0
    rows_compared: int = 0
    violations: list[ColumnViolation] = field(default_factory=list)
    skipped_columns: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        if self.passed:
            return (
                f"{self.symbol}: PASS -- {self.columns_checked} columns verified across "
                f"{len(self.cutoffs)} cutoffs, {self.rows_compared:,} row-comparisons"
            )
        cols = sorted({v.column for v in self.violations})
        return (
            f"{self.symbol}: FAIL -- {len(cols)} column(s) depend on future data: "
            f"{', '.join(cols)}"
        )


def _compare(
    full: pd.Series, truncated: pd.Series, column: str, cutoff: date
) -> ColumnViolation | None:
    """Compare one column between the two builds, ignoring shared NaNs."""
    both_na = full.isna() & truncated.isna()
    one_na = full.isna() ^ truncated.isna()

    if one_na.any():
        day = one_na[one_na].index[0]
        return ColumnViolation(
            column=column,
            cutoff=cutoff,
            rows_differing=int(one_na.sum()),
            max_abs_diff=float("nan"),
            example_day=day,
            full_value=full.loc[day],
            truncated_value=truncated.loc[day],
        )

    comparable = ~both_na
    if not comparable.any():
        return None

    a, b = full[comparable], truncated[comparable]

    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        diff = (a.astype(float) - b.astype(float)).abs()
        bad = diff > TOLERANCE
        if bad.any():
            day = diff.idxmax()
            return ColumnViolation(
                column=column,
                cutoff=cutoff,
                rows_differing=int(bad.sum()),
                max_abs_diff=float(diff.max()),
                example_day=day,
                full_value=a.loc[day],
                truncated_value=b.loc[day],
            )
        return None

    bad = a.astype(str) != b.astype(str)
    if bad.any():
        day = bad[bad].index[0]
        return ColumnViolation(
            column=column,
            cutoff=cutoff,
            rows_differing=int(bad.sum()),
            max_abs_diff=float("nan"),
            example_day=day,
            full_value=a.loc[day],
            truncated_value=b.loc[day],
        )
    return None


def verify_causality(
    symbol: str,
    cutoffs: list[date] | None = None,
    n_cutoffs: int = 4,
    builder=None,
) -> CausalityResult:
    """Verify that no feature in `symbol`'s table depends on future data.

    `builder` defaults to the real feature builder and exists so tests can point
    this at a deliberately broken one and confirm the check actually fires.
    """
    if builder is None:
        from roth.features.build import build_daily_features as builder  # noqa: N813

    full = builder(symbol)
    if full.empty:
        return CausalityResult(symbol=symbol, cutoffs=[])

    full = full.set_index("day").sort_index()
    days = list(full.index)

    if cutoffs is None:
        # Spread the cutoffs across the back half, where enough history exists
        # for the long-window features to have values worth comparing.
        lo = max(len(days) // 2, 1)
        positions = np.linspace(lo, len(days) - 2, n_cutoffs).astype(int)
        cutoffs = sorted({days[p] for p in positions})

    result = CausalityResult(symbol=symbol, cutoffs=list(cutoffs))

    for cutoff in cutoffs:
        truncated = builder(symbol, None, cutoff)
        if truncated.empty:
            continue
        truncated = truncated.set_index("day").sort_index()

        overlap = full.index[full.index <= cutoff].intersection(truncated.index)
        if len(overlap) == 0:
            continue

        shared = [c for c in full.columns if c in truncated.columns and c != "symbol"]
        result.columns_checked = max(result.columns_checked, len(shared))
        result.rows_compared += len(overlap) * len(shared)

        for column in shared:
            violation = _compare(
                full.loc[overlap, column], truncated.loc[overlap, column], column, cutoff
            )
            if violation is not None:
                result.violations.append(violation)

    return result
