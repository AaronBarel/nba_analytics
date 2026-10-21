"""Download (or refresh from cache) NBA player pages and their college pages.

    uv run python scripts/backfill_college_data.py 2010 2026

Covers every player whose first NBA season (from the per-game tables) falls in
the range, plus the picks in the draft held at the end of the last season
(they have no NBA stats yet but get rookie projections). About 2 pages per college player at ~3.5 s each, so a full
backfill takes a few hours the first time and seconds after that.
"""

import sys

from nba_analytics.data.bbref import draft_class, player_bio, player_page
from nba_analytics.data.cbb import college_page
from nba_analytics.players.defense import career_start


def main(first: int, last: int) -> None:
    start = career_start(last)
    players = set(start[(start >= first) & (start <= last)].index)
    players |= set(draft_class(last)["player_id"].dropna())
    print(f"{len(players)} players debuting {first}-{last} or drafted in {last}", flush=True)
    for i, player_id in enumerate(sorted(players), start=1):
        bio = player_bio(player_page(player_id))
        if bio["cbb_path"]:
            college_page(bio["cbb_path"])
        if i % 50 == 0:
            print(f"{i}/{len(players)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
