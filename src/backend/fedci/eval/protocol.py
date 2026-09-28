"""Evaluation protocol shared by all questions (docs/question.md, "Evaluation protocol"):

1. Chronological walk-forward, expanding window, split at whole-meeting granularity.
   Never random folds, never firms from one meeting on both sides.
2. Uncertainty by meeting: resample whole meetings, never rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    k: int
    train: pd.DatetimeIndex   # meeting dates
    test: pd.DatetimeIndex

    def split(self, df: pd.DataFrame, meeting_col: str = "announcement_date") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Assign whole rows of a panel to train/test by their meeting."""
        m = pd.to_datetime(df[meeting_col])
        return df[m.isin(self.train)], df[m.isin(self.test)]


def walk_forward(meetings, min_train: int, test_size: int, step: int | None = None, embargo: int = 0) -> list[Fold]:
    """Expanding-window folds over sorted meeting dates.

    min_train  meetings in the first training window
    test_size  meetings per test block
    step       how far the window advances (default test_size: non-overlapping test blocks)
    embargo    meetings dropped between train and test (use when outcome windows, e.g. t+20
               reversal, could reach the next meeting)
    """
    m = pd.DatetimeIndex(sorted(set(pd.to_datetime(meetings))))
    step = step or test_size
    if min_train - embargo < 1:
        raise ValueError("min_train must exceed embargo")
    folds, start, k = [], min_train, 0
    while start < len(m):
        folds.append(Fold(k, m[: start - embargo], m[start : start + test_size]))
        start += step
        k += 1
    return folds


def meeting_bootstrap(
    df: pd.DataFrame,
    stat: Callable[[pd.DataFrame], float],
    meeting_col: str = "announcement_date",
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict:
    """Block bootstrap resampling whole meetings. Returns estimate, se, and percentile CI.
    n_meetings -- not row count -- is the sample size to report."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(meeting_col)}
    keys = list(groups)
    draws = np.array([
        stat(pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))], ignore_index=True))
        for _ in range(n)
    ])
    a = (1 - level) / 2
    return dict(
        estimate=float(stat(df)), se=float(np.nanstd(draws, ddof=1)),
        ci_low=float(np.nanquantile(draws, a)), ci_high=float(np.nanquantile(draws, 1 - a)),
        n_meetings=len(keys), n_rows=len(df),
    )
