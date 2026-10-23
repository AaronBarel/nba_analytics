"""Download every player's regular-season game log pages into the HTML cache.

    uv run python scripts/backfill_player_gamelogs.py 2014 2026

One page per player and season (~530 a season) at ~3.5 s each, so about 30
minutes a season the first time. Newest seasons go first. Pages are only
downloaded here; ``teams.roster.gamelogs`` parses them.
"""

import sys

import requests

from nba_analytics.data.bbref import fetch, player_season_table


def main(first: int, last: int) -> None:
    for season in range(last, first - 1, -1):
        players = player_season_table(season, "totals")["player_id"].unique()
        for pid in players:
            try:
                fetch(f"/players/{pid[0]}/{pid}/gamelog/{season}")
            except requests.HTTPError as e:  # a few listed player-seasons have no page
                if e.response.status_code != 404:
                    raise
                print(f"  no game log: {pid} {season}", flush=True)
        print(f"{season}: {len(players)} players", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
