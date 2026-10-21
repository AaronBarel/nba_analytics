import numpy as np
import pandas as pd
import pytest

from nba_analytics.players import projections as pr


def test_parse_awards():
    out = pr.parse_awards(pd.Series(["MVP-3,AS,NBA1,DEF2", "DPOY-5", None, "AS,NBA3"]))
    assert out.loc[0].to_dict() == {"all_star": 1, "all_nba": 3, "all_def": 1,
                                    "mvp_votes": 1, "dpoy_votes": 0}
    assert out.loc[1, "dpoy_votes"] == 1 and out.loc[2].sum() == 0
    assert out.loc[3, "all_nba"] == 1


def _seasons(rows):
    """rows: (player_id, season, mp, bpm, age, pos)"""
    df = pd.DataFrame(rows, columns=["player_id", "season", "mp", "bpm", "age", "pos"])
    for col in pr.LAG_STATS + list(pr.RATE_TARGETS):
        if col not in df:
            df[col] = 1.0
    df["player"] = df["player_id"].str.upper()
    df["team"] = "BOS"
    return df


SEASONS = _seasons([
    ("a", 2005, 2000, 4.0, 25, "PG"), ("a", 2006, 1000, 2.0, 26, "PG"),
    ("a", 2007, 500, 1.0, 27, "PG"), ("a", 2008, 1500, 3.0, 28, "PG"),
    ("b", 2005, 1000, 0.0, 30, "C"),                       # gone after 2005
    ("c", 2007, 800, -1.0, 22, "C"), ("c", 2009, 900, 0.0, 24, "C"),  # missed 2008
])


@pytest.fixture
def frame(monkeypatch):
    monkeypatch.setattr(pr, "career_start", lambda last: pd.Series({"a": 2005, "b": 2005,
                                                                    "c": 2007}))
    return pr.build_frame(SEASONS, last=2009).set_index(["player_id", "season"])


def test_build_frame_lags_and_rows(frame):
    assert frame.at[("a", 2008), "bpm_1"] == 1.0 and frame.at[("a", 2008), "bpm_3"] == 4.0
    assert frame.at[("a", 2008), "bpm"] == 3.0  # target
    assert frame.at[("a", 2008), "age"] == 28 and frame.at[("a", 2008), "career_season"] == 4
    assert ("b", 2008) in frame.index and ("b", 2009) not in frame.index  # 3-season window
    assert np.isnan(frame.at[("c", 2009), "bpm_1"]) and frame.at[("c", 2009), "bpm_2"] == -1
    assert frame.at[("c", 2009), "age"] == 24
    # The unplayed season only includes players active in the last season
    assert ("c", 2010) in frame.index and ("a", 2010) not in frame.index


def test_marcel_weights_minutes_and_regresses(frame, monkeypatch):
    monkeypatch.setitem(pr.REGRESS_MINUTES, "bpm", 500)
    curves = pd.DataFrame(0.0, index=range(pr.AGE_RANGE[0], pr.AGE_RANGE[1] + 1),
                          columns=["games_frac", "mp_per_g", *pr.RATE_TARGETS])
    f = frame.reset_index()
    out = pr.marcel(f, SEASONS, curves)
    i = f.index[(f["player_id"] == "a") & (f["season"] == 2008)][0]
    num = 1.0 * 500 * 1.0 + 0.8 * 1000 * 2.0 + 0.6 * 2000 * 4.0
    den = 1.0 * 500 + 0.8 * 1000 + 0.6 * 2000
    guard_mean_2007 = 1.0  # only guard in 2007 is a, with BPM 1.0
    assert out.at[i, "bpm"] == pytest.approx((num + 500 * guard_mean_2007) / (den + 500))
