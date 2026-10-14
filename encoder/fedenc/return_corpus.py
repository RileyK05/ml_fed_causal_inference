"""Dated, lazy next-trading-day return examples. No future values enter history."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedcore.q3.contracts import L, RET
from fedcore.q3.real import PretrainCorpus, pretrain_corpus


@dataclass(frozen=True)
class ReturnCorpus:
    source: PretrainCorpus
    rows: np.ndarray
    excluded_dates: pd.DatetimeIndex

    def __len__(self):
        return len(self.rows)

    @property
    def history_end_dates(self):
        return self.source.end_dates[self.rows]

    @property
    def target_dates(self):
        return self.source.calendar[self.source.end[self.rows] + 1]

    def batch(self, rows):
        indices = self.rows[np.asarray(rows, dtype=np.int64)]
        history, mask = self.source.windows(indices)
        target = self.source.channels[self.source.firm[indices], self.source.end[indices] + 1, RET]
        return history, mask, target

    def split(self, val_frac=0.1, *, purge_history=False):
        """Last fraction of unique target dates; firms on one date stay together."""
        if not 0 < val_frac < 1:
            raise ValueError("val_frac must be between zero and one")
        dates = self.target_dates
        unique = dates.unique().sort_values()
        if len(unique) < 2:
            raise ValueError("need at least two ordinary target dates")
        n_val = min(max(1, int(np.ceil(len(unique) * val_frac))), len(unique) - 1)
        boundary = unique[-n_val]
        train, val = np.flatnonzero(dates < boundary), np.flatnonzero(dates >= boundary)
        if purge_history:
            # A reconstruction label is anywhere inside a validation history.
            # Apply the same conservative purge to both objectives for comparison.
            first_history_day = self.source.calendar[self.source.end[self.rows[val]].min() - (L - 1)]
            train = train[dates[train] < first_history_day]
            if not len(train):
                raise ValueError("no training examples after purging validation history overlap")
        return train, val


def from_history_corpus(source: PretrainCorpus, excluded_dates) -> ReturnCorpus:
    """Require the next trading day's observed return and its date strictly before cutoff.

    Past FOMC observations may occur inside histories; only target days are excluded.
    Gaps are never filled and a missing next-day target is never skipped to a later day.
    """
    excluded = pd.DatetimeIndex(excluded_dates).normalize().unique().sort_values()
    if excluded.hasnans:
        raise ValueError("excluded_dates contains NaT")
    next_day = source.end + 1
    in_grid = next_day < len(source.calendar)
    rows = np.flatnonzero(in_grid)
    next_day = next_day[rows]
    dates = source.calendar[next_day]
    target = source.channels[source.firm[rows], next_day, RET]
    eligible = (
        (dates < source.cutoff)
        & ~dates.normalize().isin(excluded)
        & source.observed[source.firm[rows], next_day]
        & np.isfinite(target)
    )
    rows = rows[eligible]
    rows = rows[np.argsort(source.end[rows], kind="stable")]
    if not len(rows):
        raise ValueError("no eligible ordinary next-day targets before cutoff")
    return ReturnCorpus(source, rows, excluded)


def real_return_corpus(cutoff, *, stride=1, min_observed=126) -> ReturnCorpus:
    """All ordinary target days by default, with scheduled and unscheduled FOMC excluded."""
    from fedcore.data import load

    if stride < 1 or min_observed < 1 or min_observed > L:
        raise ValueError("stride must be positive and min_observed must be in 1..252")
    # usmpd_statements includes unscheduled actions, unlike meeting_table's default.
    statements = load("usmpd_statements")
    source = pretrain_corpus(cutoff, stride=stride, min_observed=min_observed)
    return from_history_corpus(source, pd.to_datetime(statements["Date"]))
