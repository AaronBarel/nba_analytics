"""Download (or refresh from cache) team seasons and game logs.

    uv run python scripts/backfill_team_data.py 2001 2026

About 30 pages per season at ~3.5 s each, so a full backfill takes ~45 minutes
the first time and seconds after that.
"""

import sys

from nba_analytics.data.bbref import team_game_log, team_season_table


def main(first: int, last: int) -> None:
    for season in range(first, last + 1):
        teams = team_season_table(season)["team_id"]
        for team_id in teams:
            team_game_log(team_id, season)
        print(f"{season}: {len(teams)} teams", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
