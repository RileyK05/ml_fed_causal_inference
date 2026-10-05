import pandas as pd
import pytest

from fedcore import data


def test_catalog_all_datasets_pass_integrity():
    failures = {n: data.check(d) for n, d in data.catalog().items() if data.check(d)}
    assert not failures


def test_load_parses_dates_and_filters_inclusive():
    df = data.load("etf_returns", start="2024-01-02", end="2024-01-31")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].min() == pd.Timestamp("2024-01-02")
    assert df["date"].max() <= pd.Timestamp("2024-01-31")


def test_na_strings_become_missing():
    mps = data.load("mps_surprises")
    assert mps.loc[mps.Date == "1994-02-04", "PC"].isna().all()
    assert pd.api.types.is_float_dtype(mps["PC"])


def test_sql_views_match_pandas():
    for name in ["fomc_meetings", "mps_surprises", "etf_returns"]:
        n_sql = data.query(f'SELECT COUNT(*) AS n FROM "{name}"').n.iloc[0]
        assert n_sql == len(data.load(name))
    assert set(data.tables()) == {n for n, d in data.catalog().items() if d.exists}


def test_meetings_spine_excludes_unscheduled():
    all_ = data.meetings(scheduled_only=False)
    sched = data.meetings()
    assert sched["use_for_surprise"].all()
    assert len(sched) < len(all_)
    assert sched.announcement_date.is_unique


def test_event_panel_shape_and_no_fill():
    p = data.event_panel("etf_returns", pre=2, post=3, columns=["SPY"], long=False)
    assert set(p.rel_day) == set(range(-2, 4))
    day0 = p[p.rel_day == 0]
    assert (day0.date == day0.announcement_date).all()
    # rel_day +1 is the next trading day, strictly after the announcement
    assert (p[p.rel_day == 1].date > p[p.rel_day == 1].announcement_date).all()


def test_event_panel_rejects_non_daily():
    with pytest.raises(ValueError):
        data.event_panel("fomc_meetings")


def test_sql_connection_is_read_only(tmp_path):
    from fedcore.data import db

    out = tmp_path / "x.csv"
    with db.connect() as con:
        for bad in (
            f"COPY (SELECT 1) TO '{out.as_posix()}'",
            f"ATTACH '{(tmp_path / 'a.db').as_posix()}'",
            f"SELECT * FROM read_csv('{__file__.replace(chr(92), '/')}')",
            "SET enable_external_access=true",
            "INSTALL httpfs",
        ):
            with pytest.raises(Exception):
                con.execute(bad)
        for view in db.tables():  # every catalog view still resolves under the lock
            con.execute(f'SELECT count(*) FROM "{view}"').fetchone()
    assert not out.exists()
