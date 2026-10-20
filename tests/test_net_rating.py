import numpy as np
import pandas as pd
import pytest

from nba_analytics.teams.net_rating import (
    consecutive_seasons,
    first_n_above,
    possessions,
    r_squared,
    r_squared_by_games,
    split_at,
)


def test_possessions_matches_hand_calculation():
    box = {"fga": 80, "fta": 20, "orb": 10, "drb": 30, "fg": 40, "tov": 12}
    g = pd.DataFrame([{**box, **{f"opp_{k}": v for k, v in box.items()}}])
    # 80 + 0.4*20 - 1.07 * (10/40) * (80-40) + 12 = 89.3 for each side
    assert possessions(g).item() == pytest.approx(89.3)


def test_consecutive_seasons_follows_relocations():
    df = pd.DataFrame({
        "franchise": ["OKC", "OKC", "BOS"], "season": [2008, 2009, 2008],
        "team_id": ["SEA", "OKC", "BOS"], "team": ["Sonics", "Thunder", "Celtics"],
        "shortened": [False, False, False], "net_rtg": [-8.0, -6.0, 11.0],
        "win_pct": [0.24, 0.28, 0.80],
    })
    out = consecutive_seasons(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert (row["team_id_prev"], row["team_id_next"]) == ("SEA", "OKC")
    assert (row["net_rtg_prev"], row["net_rtg_next"]) == (-8.0, -6.0)


def _logs(margins_by_team):
    """Game logs with 100 possessions per game, so net rating = point margin."""
    rows = []
    for team, margins in margins_by_team.items():
        for i, m in enumerate(margins, start=1):
            rows.append({"season": 2025, "team_id": team, "game_num": i, "pts": 100 + m,
                         "opp_pts": 100, "poss": 100.0, "win": int(m > 0)})
    return pd.DataFrame(rows)


def test_split_at_separates_first_n_from_rest():
    out = split_at(_logs({"A": [10, -2, 4, 4]}), 2).iloc[0]
    assert out["net_rtg_first"] == pytest.approx(4.0)
    assert out["net_rtg_rest"] == pytest.approx(4.0)
    assert out["win_pct_first"] == 0.5
    assert out["win_pct_rest"] == 1.0


def test_r_squared_by_games_and_first_crossing():
    rng = np.random.default_rng(1)
    strength = rng.normal(0, 5, 30)
    # Each team plays its true strength plus noise, so early results become predictive
    logs = _logs({f"T{i}": s + rng.normal(0, 12, 82) for i, s in enumerate(strength)})
    curves = r_squared_by_games(logs, max_n=60, n_boot=50)
    net = curves[(curves.predictor == "net_rtg") & (curves.target == "net_rtg")]
    assert net["r2"].iloc[-1] > net["r2"].iloc[0]
    assert (net["lo"] <= net["r2"]).all() and (net["r2"] <= net["hi"]).all()
    crossings = first_n_above(curves, 0.5)
    assert len(crossings) == 4


def test_r_squared_perfect_fit():
    assert r_squared([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
