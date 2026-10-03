import pandas as pd

from fedcore.protocol import meeting_bootstrap, walk_forward

MEETINGS = pd.date_range("2000-01-01", periods=50, freq="45D")


def test_walk_forward_is_chronological_and_non_overlapping():
    folds = walk_forward(MEETINGS, min_train=20, test_size=5)
    assert len(folds) == 6
    for f in folds:
        assert f.train.max() < f.test.min()
    tests = [d for f in folds for d in f.test]
    assert len(tests) == len(set(tests)) == 30


def test_embargo_drops_meetings_before_test():
    f = walk_forward(MEETINGS, min_train=20, test_size=5, embargo=2)[0]
    assert len(f.train) == 18 and f.test[0] == MEETINGS[20]


def test_split_never_splits_a_meeting():
    panel = pd.DataFrame({"announcement_date": MEETINGS.repeat(3), "firm": list("abc") * 50})
    f = walk_forward(MEETINGS, min_train=20, test_size=5)[0]
    tr, te = f.split(panel)
    assert set(tr.announcement_date).isdisjoint(te.announcement_date)
    assert len(te) == 15


def test_meeting_bootstrap_counts_meetings_not_rows():
    panel = pd.DataFrame({"announcement_date": MEETINGS.repeat(10), "y": range(500)})
    out = meeting_bootstrap(panel, lambda d: d.y.mean(), n=200)
    assert out["n_meetings"] == 50 and out["n_rows"] == 500
    assert out["ci_low"] < out["estimate"] < out["ci_high"]
