"""Sports-Reference college basketball player pages.

Shares the Basketball-Reference fetcher, so the cache format and request
spacing (which covers both sites) are the same.
"""

from __future__ import annotations

import pandas as pd

from nba_analytics.config import CACHE_DIR
from nba_analytics.data.bbref import fetch, parse_table

BASE_URL = "https://www.sports-reference.com"

# College player tables: kind -> table id
TABLES = {
    "per_game": "players_per_game",
    "per_poss": "players_per_poss",
    "per_min": "players_per_min",
    "advanced": "players_advanced",
    "totals": "players_totals",
}


def college_page(cbb_path: str, refresh: bool = False) -> str:
    """HTML for a path like '/cbb/players/cooper-flagg-1.html'."""
    return fetch(cbb_path, cache_dir=CACHE_DIR / "cbb", refresh=refresh, base_url=BASE_URL)


def college_table(cbb_path: str, kind: str) -> pd.DataFrame:
    """One row per college season. Per-100 and advanced stats start in 2009-10.

    ``season`` is the end year (2024-25 -> 2025), matching the NBA tables.
    Placeholder rows such as "Did not play - transfer" are dropped.
    """
    try:
        df = parse_table(college_page(cbb_path), TABLES[kind])
    except ValueError:  # older players have no per-100 / advanced tables
        return pd.DataFrame()
    if "games" not in df:  # listed but never played (e.g. ruled ineligible)
        return pd.DataFrame()
    df = df[df["year_id"].notna() & df["games"].notna()].copy()
    df.insert(0, "season", df["year_id"].str[:4].astype(int) + 1)
    df.insert(0, "cbb_path", cbb_path)
    return df.reset_index(drop=True)
