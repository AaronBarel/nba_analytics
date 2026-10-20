"""Basketball-Reference scraper with an on-disk HTML cache.

Basketball-Reference allows roughly 20 requests per minute, so requests are
spaced out and every page is cached. Tables are parsed by each cell's
``data-stat`` attribute, which is more stable than the displayed headers.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from nba_analytics.config import CACHE_DIR

BASE_URL = "https://www.basketball-reference.com"
MIN_REQUEST_INTERVAL = 3.5  # seconds
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) nba-analytics research"

# Season-level player tables: page suffix -> table id
PLAYER_TABLES = {
    "per_game": "per_game_stats",
    "totals": "totals_stats",
    "per_poss": "per_poss",
    "advanced": "advanced",
}

_last_request = 0.0


def fetch(path: str, cache_dir: Path = CACHE_DIR / "bbref", refresh: bool = False) -> str:
    """Return the HTML for ``path`` (e.g. '/leagues/NBA_2026_advanced.html')."""
    global _last_request
    cache_file = cache_dir / path.strip("/").replace("/", "__")
    if cache_file.exists() and not refresh:
        return cache_file.read_text()

    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(BASE_URL + path, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request = time.monotonic()
    resp.raise_for_status()
    resp.encoding = "utf-8"  # no charset in the headers; requests would guess Latin-1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    return resp.text


def parse_table(html: str, table_id: str) -> pd.DataFrame:
    """Parse a stats table into a DataFrame keyed by ``data-stat`` names.

    Some tables are shipped inside HTML comments and only rendered by
    JavaScript, so comments are searched when the table isn't found directly.
    The player id (e.g. 'jamesle01') is added as ``player_id`` when present.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is None:
        for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
            if f'id="{table_id}"' in comment:
                table = BeautifulSoup(comment, "lxml").find("table", id=table_id)
                break
    if table is None:
        raise ValueError(f"table {table_id!r} not found")

    records = []
    for row in table.tbody.find_all("tr"):
        if "thead" in (row.get("class") or []):
            continue  # repeated header rows
        record = {}
        for cell in row.find_all(["th", "td"]):
            stat = cell.get("data-stat")
            if stat is None:
                continue
            record[stat] = cell.get_text(strip=True)
            if cell.get("data-append-csv"):
                record["player_id"] = cell["data-append-csv"]
        if record:
            records.append(record)

    df = pd.DataFrame.from_records(records).replace("", pd.NA)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Keep text columns as text; only convert when nothing non-empty is lost.
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def player_season_table(season: int, kind: str, refresh: bool = False) -> pd.DataFrame:
    """One of the league-wide player tables for a season (end year, e.g. 2026).

    Rows are as published: a traded player has a combined row ('2TM', '3TM',
    ...) followed by one row per team in chronological order.
    """
    html = fetch(f"/leagues/NBA_{season}_{kind}.html", refresh=refresh)
    df = parse_table(html, PLAYER_TABLES[kind])
    df = df[df["player_id"].notna()].reset_index(drop=True)
    df.insert(0, "season", season)
    return df
