"""Download (or refresh from cache) G League totals and advanced tables.

    uv run python scripts/backfill_gleague_data.py 2008 2026

Two pages per season at ~3.5 s each.
"""

import sys

from nba_analytics.data.gleague import gleague_table


def main(first: int, last: int) -> None:
    for season in range(first, last + 1):
        for kind in ("totals", "advanced"):
            gleague_table(season, kind)
        print(f"{season}: done", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
