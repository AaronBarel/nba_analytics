import pandas as pd
import pytest

from nba_analytics.players.gleague import STATS, link_players, pre_debut_features, translation_table

NBA = pd.DataFrame({
    "player_id": ["harrijo01", "cookda02", "smithch01", "smithch05", "jonesbo01"],
    "player": ["Joe Harris", "Daequan Cook", "Chris Smith", "Chris Smith", "Bobby Jones"],
    "season": [2015, 2008, 2013, 2014, 1985],
})


def test_link_players():
    gl = pd.DataFrame({
        "gleague_id": ["harrijo01d", "cookda01d", "smithch01d", "jonesbo01d", "abmasma01d"],
        "player": ["Jovan Harris", "Daequan Cook", "Chris Smith", "Bobby Jones", "Max Abmas"],
        "season": [2014, 2008, 2013, 2009, 2025],
    })
    out = link_players(gl, NBA).set_index("gleague_id")
    assert pd.isna(out.at["harrijo01d", "player_id"])            # same id, different person
    assert out.at["cookda01d", "player_id"] == "cookda02"         # name match, different id
    assert out.at["cookda01d", "link"] == "name"
    assert out.at["smithch01d", "player_id"] == "smithch01"       # id + name beats the namesake
    assert out.at["smithch01d", "link"] == "id"
    assert pd.isna(out.at["jonesbo01d", "player_id"])            # namesake from another era
    assert pd.isna(out.at["abmasma01d", "link"])


def test_pre_debut_features_uses_last_season_before_debut():
    gl = pd.DataFrame({"player_id": ["a", "a", "a", None], "season": [2018, 2019, 2020, 2019],
                       "mp": [300, 900, 1200, 500], "pts_per_36": [10, 20, 30, 5],
                       "ts_pct": 55.0, "usg_pct": 20.0, "ast_pct": 10.0, "trb_pct": 8.0,
                       "per": 15.0, "ws_per_48": 0.1})
    out = pre_debut_features(gl, pd.Series({"a": 2020, "b": 2021})).set_index("player_id")
    assert out.at["a", "gl_pre_mp"] == 900 and out.at["a", "gl_pre_pts_per_36"] == 20
    assert out.at["b", "gl_pre_mp"] == 0 and pd.isna(out.at["b", "gl_pre_ts_pct"])


def test_translation_table_weights_by_harmonic_minutes():
    pairs = pd.DataFrame({"mp_gl": [1000, 1000], "mp_nba": [1000, 250]})
    for stat in STATS:
        pairs[f"{stat}_gl"], pairs[f"{stat}_nba"] = [20.0, 10.0], [10.0, 6.0]
    row = translation_table(pairs).set_index("stat").loc["PTS per 36"]
    w = [1000, 400]  # harmonic means of (1000, 1000) and (1000, 250)
    assert row["G League"] == pytest.approx((20 * w[0] + 10 * w[1]) / sum(w))
    assert row["NBA"] == pytest.approx((10 * w[0] + 6 * w[1]) / sum(w))
    assert row["r"] == pytest.approx(1.0)
