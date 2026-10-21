import numpy as np
import pandas as pd

from fedcore.q3.shock_days import shock_days


def test_shock_days_threshold_and_fomc_exclusion():
    cal = pd.bdate_range("2000-01-03", periods=10)
    y2 = pd.Series([5.00, 5.10, 5.12, 5.00, 5.01, 4.90, 4.91, 5.05, 5.06, 5.20], index=cal)
    # changes (bp): +10, +2, -12, +1, -11, +1, +14, +1, +14
    fomc = pd.DatetimeIndex([cal[3]])                     # statement on day 3 -> day 3 and day 4 excluded
    got = shock_days(y2, fomc, threshold_bp=5, start=cal[0], end=cal[-1])
    assert list(got.index) == [cal[1], cal[5], cal[7], cal[9]]
    assert np.allclose(got.to_numpy(), [10, -11, 14, 14])


def test_shock_days_window_and_missing_values():
    cal = pd.bdate_range("2000-01-03", periods=6)
    y2 = pd.Series([5.0, np.nan, 5.2, 5.0, 5.3, 5.31], index=cal)
    got = shock_days(y2, pd.DatetimeIndex([]), threshold_bp=5, start=cal[3], end=cal[-1])
    assert list(got.index) == [cal[3], cal[4]]             # a gap is dropped, not filled; day 2's change spans it
