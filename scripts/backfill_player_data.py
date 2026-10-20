"""Download (or refresh from cache) player season tables and team on/off splits.

    uv run python scripts/backfill_player_data.py 2023 2026

Per-game tables go back to 2000-01 so career length (rookie-scale flag) is known.
Per-100, advanced, play-by-play and on/off pages are fetched for the requested
seasons: about 34 pages per season at ~3.5 s each.
"""

import sys

from nba_analytics.data.bbref import player_season_table, team_on_off, team_season_table

HISTORY_START = 2001


def main(first: int, last: int) -> None:
    for season in range(HISTORY_START, last + 1):
        player_season_table(season, "per_game")
    for season in range(first, last + 1):
        for kind in ("per_poss", "advanced", "play-by-play"):
            player_season_table(season, kind)
        for team_id in team_season_table(season)["team_id"]:
            team_on_off(team_id, season)
        print(f"{season}: done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
