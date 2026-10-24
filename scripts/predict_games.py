"""Predict the next day of games (or a given date) and save the predictions.

    uv run python scripts/predict_games.py              # next date with unplayed games
    uv run python scripts/predict_games.py 2026-10-23

Downloads new box scores, the playing teams' rosters and the latest official
injury report (a few minutes the first time each day), prints win
probabilities, margins, totals and the top projected scorers, and saves both
tables under ``data/predictions/<date>/``.
"""

import sys
import warnings

import pandas as pd

from nba_analytics import live

warnings.filterwarnings("ignore", category=FutureWarning)


def main(target: str | None) -> None:
    today = live.eastern_today()
    if target is None:
        sched = live.schedule(live.current_season(today), today)
        target = sched.loc[sched["box_score"].isna() & (sched["date"] >= today), "date"].min()
    data = live.build(target, today)
    games = live.predict_games(data)
    players = live.predict_players(data)
    live.save(data, games, players)

    print(f"{target}: {len(games)} games, injury report {data['stamp']}\n")
    out = games.assign(
        matchup=games["away"] + games["neutral"].map({True: " vs ", False: " @ "}) + games["home"],
        home_win=(100 * games["home_win_prob"]).round(1),
        margin=games["home_margin"].round(1), total=games["total_pred"].round(1),
        elo=(100 * games["elo_prob"]).round(1))
    print(out[["start_time", "matchup", "home_win", "margin", "total", "elo"]]
          .to_string(index=False))
    missed = live.unmatched(data["report"][data["report"]["game_date"] == target], data["rosters"])
    if len(missed):
        print("\nInjury report names not on a roster:", ", ".join(missed["player"]))
    top = players.nlargest(15, "pts_pred")
    print("\nTop projected scorers")
    print(pd.DataFrame({
        "player": top["player"], "team": top["team_id"], "opp": top["opp_id"],
        "status": top["injury_status"], "min": top["mp_pred"].round(1),
        "pts": top["pts_pred"].round(1),
        "80% range": top["pts_lo"].astype(int).astype(str) + "-" + top["pts_hi"].astype(int).astype(str),
        "reb": top["trb_pred"].round(1), "ast": top["ast_pred"].round(1),
        "3pm": top["fg3_pred"].round(1),
    }).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
