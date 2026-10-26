import numpy as np
import pandas as pd
import pytest

from nba_analytics import live
from nba_analytics.data import bbref, injuries
from nba_analytics.players.props import _player_history
from nba_analytics.teams.games import COUNTS, blended_counts, elo, game_table, strength_features
from nba_analytics.teams.roster import expected_minutes


def test_report_stamp_is_eastern_twelve_hour_time():
    ts = pd.Timestamp("2026-10-22 17:00", tz=injuries.EASTERN).to_pydatetime()
    assert injuries.stamp(ts) == "2026-10-22_05_00PM"


def test_current_season_turns_over_in_august():
    assert live.current_season("2026-10-23") == 2027
    assert live.current_season("2027-04-10") == 2027
    assert live.current_season("2027-08-01") == 2028


def _rosters():
    return pd.DataFrame({"team_id": ["DAL", "DAL", "DEN", "MEM"],
                         "player_id": ["jacksja02", "doncilu01", "jokicni01", "jacksja03"],
                         "player": ["Jaren Jackson Jr.", "Luka Dončić", "Nikola Jokić",
                                    "Jaren Jackson Jr."]})


def test_match_names_handles_suffixes_accents_and_teams():
    report = pd.DataFrame({"team_id": ["DAL", "MEM", "DEN", "DEN", "DAL"],
                           "player": ["Jackson Jr., Jaren", "Jackson Jr., Jaren",
                                      "Jokic, Nikola", "Nobody, Some", None]})
    ids = live.match_names(report, _rosters())
    assert ids.tolist() == ["jacksja02", "jacksja03", "jokicni01", None, None]


def test_availability_prefers_report_and_falls_back_to_list():
    slate = pd.DataFrame({"date": ["2026-10-23"], "home": ["DAL"], "away": ["DEN"]})
    report = pd.DataFrame({"game_date": "2026-10-23", "team_id": ["DAL", "DEN"],
                           "player": ["Doncic, Luka", None],
                           "status": ["Questionable", "Not Yet Submitted"]})
    listed = pd.DataFrame({"player_id": ["jokicni01", "jacksja02"],
                           "status": ["Day To Day", "Out For Season"]})
    a = live.availability(slate, _rosters(), report, listed).set_index("player_id")
    assert a.loc["doncilu01", "injury_status"] == "Questionable"
    assert a.loc["jacksja02", "injury_status"] == "Available"  # report covers DAL
    assert a.loc["jokicni01", "injury_status"] == "Questionable"  # DEN not submitted: list
    assert a.loc["jokicni01", "source"] == "Basketball-Reference list"


def _logs(scheduled: bool):
    rows = [("BOS", "2025-01-01", None, "LAL", 110), ("LAL", "2025-01-01", "@", "BOS", 100),
            ("BOS", "2025-01-03", "@", "LAL", 90), ("LAL", "2025-01-03", None, "BOS", 120)]
    if scheduled:
        rows += [("BOS", "2025-01-05", None, "LAL", None),
                 ("LAL", "2025-01-05", "@", "BOS", None)]
    df = pd.DataFrame(rows, columns=["team_id", "date", "game_location", "opp_name_abbr", "pts"])
    opp = df.set_index(["date", "team_id"])["pts"]
    df["opp_pts"] = [opp[(d, o)] for d, o in zip(df["date"], df["opp_name_abbr"])]
    for c in COUNTS:
        if c not in df:
            df[c] = np.where(df["pts"].notna(), 10.0, np.nan)
    df["poss"] = np.where(df["pts"].notna(), 100.0, np.nan)
    return df.assign(season=2025, franchise=df["team_id"])


def test_scheduled_games_change_nothing_before_them_and_see_every_result():
    before, after = strength_features(_logs(False)), strength_features(_logs(True))
    pd.testing.assert_frame_equal(after.iloc[[0, 1, 3, 4]].reset_index(drop=True),
                                  before.reset_index(drop=True))
    nxt = after[after["date"] == "2025-01-05"].set_index("team_id")
    assert nxt.loc["BOS", "games_played"] == 2
    assert nxt.loc["BOS", "form_net_rtg"] == pytest.approx(-10.0)  # (110 + 90 - 100 - 120) / 2
    b = blended_counts(_logs(True), ["pts"]).set_index(["team_id", "date"])
    assert not b["pts"].isna().any()


def test_elo_scheduled_game_moves_no_ratings():
    g = game_table(_logs(True))
    e = elo(g, k=20, home_adv=100, carryover=1.0)
    assert e["elo_prob"].notna().all()
    assert np.isfinite(e.to_numpy()).all()


def test_upcoming_rows_get_history_from_played_games():
    df = pd.DataFrame({
        "season": 2027, "player_id": "a", "team_id": "BOS",
        "date": ["2026-10-20", "2026-10-22", "2026-10-24"], "mp": [30.0, 20.0, np.nan],
        "started": [1, 0, 0], "available": True, "pts": [30.0, 10.0, np.nan],
        "trb": 5.0, "ast": 2.0, "stl": 1.0, "blk": 0.0, "fg3": 1.0})
    values = pd.DataFrame({"player_id": ["a"], "season": [2027], "bpm": [1.0], "mpg": [30.0]})
    h = _player_history(expected_minutes(df, values))
    last = h.iloc[-1]
    assert last["games_td"] == 2 and last["mp_td"] == 50 and last["pts_td"] == 40
    assert last["mpg_recent"] == 25 and last["pts_rate_recent"] == pytest.approx(0.8)
    assert last["expected_mp"] == pytest.approx((10 * 30 + 50) / 12)


def test_box_score_parses_players_inactives_and_totals(monkeypatch):
    def table(team, names):
        rows = "".join(
            f'<tr><th data-stat="player" data-append-csv="{pid}">{pid}</th>'
            f'<td data-stat="mp">{mp}</td><td data-stat="pts">{pts}</td>'
            f'<td data-stat="reason">{why}</td></tr>' for pid, mp, pts, why in names)
        return (f'<table id="box-{team}-game-basic"><tbody>{rows}</tbody><tfoot><tr>'
                f'<th data-stat="player">Team Totals</th><td data-stat="pts">100</td>'
                f'</tr></tfoot></table>')
    starters = [(f"s{i}", "30:00", 10, "") for i in range(5)]
    html = (table("BOS", [*starters, ("b1", "12:30", 4, ""), ("b2", "", "", "Did Not Play")])
            + table("DET", starters)
            + '<div><strong>Inactive:&nbsp;</strong><span><strong>BOS</strong></span>'
              '<a href="/players/c/cartede02.html">D</a></div>')
    monkeypatch.setattr(bbref, "fetch", lambda *a, **k: html)
    players, totals = bbref.box_score("/boxscores/x.html")
    bos = players[players["team_id"] == "BOS"].set_index("player_id")
    assert bos["started"].sum() == 5 and bos.loc["b1", "mp"] == 12.5
    assert bos.loc["b2", "status"] == "Did Not Play"
    assert bos.loc["cartede02", "status"] == "Inactive"
    assert totals.set_index("team_id")["pts"].to_dict() == {"BOS": 100, "DET": 100}


def test_totals_models_refits_a_stale_cache(monkeypatch, tmp_path):
    import pickle

    from nba_analytics.teams import totals

    monkeypatch.setattr(live, "PROCESSED_DIR", tmp_path)
    path = tmp_path / f"totals_models_{live.FIRST_TRAIN}_2026.pkl"
    path.write_bytes(pickle.dumps({"margin": None, "margin_sd": 13.0}))  # an older fit
    fresh = dict.fromkeys(totals.MODEL_KEYS, 1.0)
    monkeypatch.setattr(live.games, "game_dataset", lambda first, last: None)
    monkeypatch.setattr(totals, "total_features", lambda g: g)
    monkeypatch.setattr(totals, "fit", lambda train: fresh)
    assert live.totals_models(2026) == fresh
    assert pickle.loads(path.read_bytes()) == fresh


def test_props_models_refit_when_the_threes_prior_changed(monkeypatch, tmp_path):
    import pickle

    from nba_analytics.players import props

    monkeypatch.setattr(live, "PROCESSED_DIR", tmp_path)
    path = tmp_path / f"props_models_{live.FIRST_TRAIN}_2026.pkl"
    path.write_bytes(pickle.dumps({"rates": dict.fromkeys(props.STATS)}))  # league-rate prior
    fresh = {"rates": dict.fromkeys(props.STATS), "fg3_prior": props.FG3_PRIOR}
    monkeypatch.setattr(props, "props_dataset", lambda first, last: pd.DataFrame({"mp": [30.0]}))
    monkeypatch.setattr(props, "fit", lambda train: fresh)
    assert live._props_models(2026) == fresh
