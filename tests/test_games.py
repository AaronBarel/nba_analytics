import numpy as np
import pandas as pd
import pytest

from nba_analytics.teams.games import elo, game_table
from nba_analytics.teams.roster import REPLACEMENT_BPM, _allocate
from nba_analytics.teams.schedule import haversine_km, schedule_features, utc_offset, venue


def test_haversine_new_york_to_los_angeles():
    assert haversine_km(40.75, -73.99, 34.04, -118.27) == pytest.approx(3940, rel=0.01)


def test_utc_offset_handles_phoenix_and_daylight_saving():
    assert utc_offset("PHO", "2025-01-10") == utc_offset("DEN", "2025-01-10") == -7
    assert utc_offset("PHO", "2025-04-01") == utc_offset("LAL", "2025-04-01") == -7
    assert utc_offset("DEN", "2025-04-01") == -6


def test_venue_neutral_and_bubble():
    assert venue("2025-01-23", "IND", neutral=True) == "Paris"
    assert venue("2020-08-01", "LAL", neutral=False) == "Orlando bubble"
    assert venue("2025-01-21", "IND", neutral=False) == "IND"


def _logs():
    rows = [  # team, date, location, opponent, pts, opp_pts
        ("BOS", "2025-01-01", None, "LAL", 110, 100),
        ("LAL", "2025-01-01", "@", "BOS", 100, 110),
        ("LAL", "2025-01-02", "@", "NYK", 95, 105),
        ("NYK", "2025-01-02", None, "LAL", 105, 95),
        ("IND", "2025-01-23", "N", "SAS", 136, 98),
        ("SAS", "2025-01-23", "N", "IND", 98, 136),
    ]
    df = pd.DataFrame(rows, columns=["team_id", "date", "game_location", "opp_name_abbr",
                                     "pts", "opp_pts"])
    return df.assign(season=2025, poss=100.0)


def test_game_table_one_row_per_game_from_home_side():
    g = game_table(_logs())
    assert len(g) == 3
    paris = g[g["venue"] == "Paris"].iloc[0]
    assert paris["neutral"] and paris["home"] == "IND" and paris["margin"] == 38
    assert g.loc[g["home"] == "BOS", "home_win"].item() == 1


def test_schedule_back_to_back_and_travel():
    g = game_table(_logs())
    tg = pd.concat([g[["season", "date", "venue"]].assign(team_id=g[side])
                    for side in ["home", "away"]])
    s = schedule_features(tg).set_index(["team_id", "date"])
    lal = s.loc[("LAL", "2025-01-02")]
    assert lal["back_to_back"] == 1
    assert lal["travel_km"] == pytest.approx(300, rel=0.1)  # Boston to New York
    assert s.loc[("LAL", "2025-01-01"), "tz_shift"] == 3  # Los Angeles to Boston


def test_elo_is_zero_sum_and_favors_home():
    g = game_table(_logs())
    e = elo(g, k=20, home_adv=100, carryover=1.0)
    assert e["elo_prob"].iloc[0] > 0.5  # equal ratings plus home court
    assert e["elo_prob"].iloc[-1] == pytest.approx(0.5, abs=0.05)  # neutral, little history


def test_allocate_fills_240_minutes_then_replacement():
    assert _allocate(np.array([40.0] * 6), np.full(6, 2.0)) == pytest.approx(10.0)
    short = _allocate(np.array([30.0] * 4), np.zeros(4))
    assert short == pytest.approx(120 * REPLACEMENT_BPM / 48)
