import pandas as pd
import pytest

from nba_analytics.data.bbref import parse_table
from nba_analytics.players.usage_efficiency import build_usage_ts, season_rows


def _adv(rows):
    cols = ["season", "player_id", "name_display", "age", "team_name_abbr", "pos",
            "games", "ts_pct", "usg_pct"]
    return pd.DataFrame(rows, columns=cols)


def _pg(rows):
    return pd.DataFrame(rows, columns=["player_id", "team_name_abbr", "mp_per_g"])


ADV = _adv([
    (2026, "a", "Traded Guard", 25, "2TM", "PG", 60, 0.600, 25.0),
    (2026, "a", "Traded Guard", 25, "LAC", "PG", 40, 0.590, 26.0),
    (2026, "a", "Traded Guard", 25, "CLE", "PG", 20, 0.620, 23.0),
    (2026, "b", "Starter Guard", 27, "BOS", "SG", 70, 0.560, 20.0),
    (2026, "c", "Few Games", 22, "NYK", "SG", 19, 0.700, 15.0),
    (2026, "d", "Few Minutes", 30, "MIA", "C", 50, 0.650, 12.0),
    (2026, "e", "Big Man", 28, "DEN", "C", 75, 0.640, 30.0),
])
PG = _pg([
    ("a", "2TM", 30.0), ("a", "LAC", 31.0), ("a", "CLE", 28.0),
    ("b", "BOS", 32.0), ("c", "NYK", 20.0), ("d", "MIA", 5.9), ("e", "DEN", 34.0),
])


def test_season_rows_keeps_combined_row_with_last_team():
    out = season_rows(ADV)
    traded = out[out["player_id"] == "a"]
    assert len(traded) == 1
    assert traded["games"].item() == 60
    assert traded["team"].item() == "CLE"


def test_season_rows_handles_old_tot_label():
    df = ADV.replace({"team_name_abbr": {"2TM": "TOT"}})
    assert season_rows(df).set_index("player_id").at["a", "team"] == "CLE"


def test_filters_games_and_minutes():
    out = build_usage_ts(ADV, PG)
    assert set(out["player_id"]) == {"a", "b", "e"}


def test_positional_and_group_differences():
    out = build_usage_ts(ADV, PG).set_index("player_id")
    # PG and SG each have one qualifier, so the position difference is zero
    assert out.at["a", "ts_diff_pos"] == pytest.approx(0)
    # Guard group mean = (60 + 56) / 2 = 58
    assert out.at["a", "ts_diff_group"] == pytest.approx(2.0)
    assert out.at["b", "ts_diff_group"] == pytest.approx(-2.0)


def test_parse_table_reads_data_stat_and_commented_tables():
    html = """<html><body><!--
    <table id="t"><tbody>
      <tr><td data-stat="name_display" data-append-csv="x01">X</td><td data-stat="games">5</td></tr>
      <tr class="thead"><td data-stat="games">G</td></tr>
    </tbody></table> --></body></html>"""
    df = parse_table(html, "t")
    assert df.to_dict("records") == [{"name_display": "X", "player_id": "x01", "games": 5}]
