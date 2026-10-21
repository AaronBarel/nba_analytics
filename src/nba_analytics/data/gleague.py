"""Basketball-Reference G League (formerly D-League) season tables.

G League player ids look like NBA ids plus a trailing 'd' ('abmasma01d'), but
they are assigned separately: 'harrijo01d' is Jovan Harris while NBA
'harrijo01' is Joe Harris. ``nba_id`` gives the candidate; ``players.gleague``
only accepts it when the names agree.
"""

from __future__ import annotations

import pandas as pd

from nba_analytics.data.bbref import fetch, parse_table

# Regular season and, from 2021-22, the Showcase Cup (early-season games played
# separately from the regular-season standings). Playoffs are left out.
PARTS = {"reg": "Regular season", "sc": "Showcase Cup"}


def gleague_table(season: int, kind: str = "totals", part: str = "reg",
                  refresh: bool = False) -> pd.DataFrame:
    """One G League table ('totals' or 'advanced') for a season (end year).

    Empty when the season has no such part (e.g. no Showcase Cup before 2022).
    """
    html = fetch(f"/gleague/years/gleague_{season}_{kind}.html", refresh=refresh)
    try:
        df = parse_table(html, f"{kind}_stats_{part}")
    except ValueError:
        return pd.DataFrame()
    df = df[df["player_id"].notna()].copy()
    df.insert(0, "season", season)
    df.insert(1, "part", part)
    return df.reset_index(drop=True)


def nba_id(gleague_id: pd.Series) -> pd.Series:
    return gleague_id.str.removesuffix("d")
