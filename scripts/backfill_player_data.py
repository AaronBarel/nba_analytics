"""Download (or refresh from cache) player season tables and team on/off splits.

    uv run python scripts/backfill_player_data.py 2023 2026

Per-game tables go back to 2000-01 so career length (rookie-scale flag) is known.
Per-100, advanced, play-by-play and on/off pages are fetched for the requested
seasons: about 34 pages per season at ~3.5 s each.

    uv run python scripts/backfill_player_data.py 2005 2026 --season-tables

fetches only the per-100, advanced and team tables (3 pages per season), which
is all the season projections need.
"""

import sys

from nba_analytics.data.bbref import player_season_table, team_on_off, team_season_table

HISTORY_START = 2001


def main(first: int, last: int, season_tables_only: bool = False) -> None:
    for season in range(HISTORY_START, last + 1):
        player_season_table(season, "per_game")
    for season in range(first, last + 1):
        for kind in ("per_poss", "advanced"):
            player_season_table(season, kind)
        teams = team_season_table(season)["team_id"]
        if not season_tables_only:
            player_season_table(season, "play-by-play")
            for team_id in teams:
                team_on_off(team_id, season)
        print(f"{season}: done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), "--season-tables" in sys.argv)
