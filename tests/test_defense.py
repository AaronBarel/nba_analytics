import pandas as pd
import pytest

from nba_analytics.players.defense import build_defense, on_off_drtg


def _on_off(rows):
    """rows: (team, player, on_mp, drtg_on, drtg_off)"""
    out = []
    for team, pid, mp, on, off in rows:
        out += [
            {"season": 2026, "team_id": team, "player_id": pid, "split_id": "On Court",
             "mp": mp, "opp_off_rtg": on},
            {"season": 2026, "team_id": team, "player_id": pid, "split_id": "Off Court",
             "mp": 1000, "opp_off_rtg": off},
            {"season": 2026, "team_id": team, "player_id": pid, "split_id": "On - Off",
             "mp": None, "opp_off_rtg": on - off},
        ]
    return pd.DataFrame(out)


def test_on_off_drtg_weights_traded_players_by_minutes():
    df = _on_off([("LAC", "a", 300, 110, 114), ("CLE", "a", 100, 112, 108),
                  ("BOS", "b", 500, 105, 110)])
    out = on_off_drtg(df).set_index("player_id")["drtg_on_off"]
    assert out["a"] == pytest.approx((300 * -4 + 100 * 4) / 400)
    assert out["b"] == pytest.approx(-5)


PLAYERS = [
    # id, pos, mpg, games, obpm, ts, stl, blk, drb, dbpm, dws, def_rtg, charges, first_season
    ("star", "PG", 36, 80, 6.0, 0.62, 1.5, 0.5, 6.0, 0.5, 2.0, 114, 2, 2016),
    ("stopper", "SG", 32, 75, -2.0, 0.53, 3.0, 1.2, 8.0, 2.5, 4.0, 108, 25, 2015),
    ("rookie", "PG", 31, 70, -3.0, 0.51, 1.8, 0.4, 5.0, 0.0, 1.0, 116, 3, 2025),
    ("bench", "SG", 14, 60, -1.0, 0.55, 1.2, 0.3, 4.0, -1.0, 0.5, 117, 1, 2018),
    ("backup", "PG", 18, 70, 1.0, 0.58, 1.0, 0.2, 4.5, -0.5, 1.0, 115, 4, 2019),
    ("short", "SG", 30, 10, 0.0, 0.55, 2.0, 0.5, 5.0, 0.0, 0.5, 112, 0, 2020),
] + [  # more bench guards so the top-quartile minutes cutoff (9 qualifiers) is 31 MPG
    (f"bench{i}", "SG", mpg, 60, -1.0, 0.55, 1.2, 0.3, 4.0, -1.0, 0.5, 117, 1, 2018)
    for i, mpg in enumerate([10, 12, 15, 16])
]


def _tables():
    adv, poss, pg, pbp = [], [], [], []
    for pid, pos, mpg, g, obpm, ts, stl, blk, drb, dbpm, dws, drtg, ch, _ in PLAYERS:
        base = {"season": 2026, "player_id": pid, "team_name_abbr": "BOS"}
        adv.append({**base, "name_display": pid, "age": 25, "pos": pos, "games": g,
                    "mp": mpg * g, "ts_pct": ts, "usg_pct": 20.0, "obpm": obpm,
                    "dbpm": dbpm, "dws": dws})
        poss.append({**base, "stl_per_poss": stl, "blk_per_poss": blk, "drb_per_poss": drb,
                     "def_rtg": drtg})
        pg.append({**base, "mp_per_g": mpg})
        pbp.append({**base, "drawn_offensive": ch})
    on_off = _on_off([("BOS", p[0], p[2] * p[3], 110 + (p[11] - 113) / 2, 112)
                      for p in PLAYERS])
    first = pd.Series({p[0]: p[13] for p in PLAYERS})
    return [pd.DataFrame(t) for t in (poss, adv, pg, pbp)] + [on_off, first]


def test_build_defense_scores_and_flags():
    df = build_defense(*_tables(), min_games=20, min_minutes=500).set_index("player_id")
    assert "short" not in df.index  # fewer than 20 games
    assert set(df["pos_group"]) == {"G"}
    assert df["defense_index"].idxmax() == "stopper"
    # Heavy minutes = top quartile of guard MPG; below-average offense
    assert df.at["stopper", "hidden_defender"]
    assert df.at["rookie", "hidden_defender"] and df.at["rookie", "rookie_scale"]
    assert df.at["rookie", "career_season"] == 2
    assert not df.at["star", "hidden_defender"]  # good offense
    assert not df.at["bench", "hidden_defender"]  # light minutes
    assert not df.at["stopper", "rookie_scale"]
