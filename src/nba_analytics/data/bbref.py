"""Basketball-Reference scraper with an on-disk HTML cache.

Basketball-Reference allows roughly 20 requests per minute, so requests are
spaced out and every page is cached. Tables are parsed by each cell's
``data-stat`` attribute, which is more stable than the displayed headers.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from nba_analytics.config import CACHE_DIR

BASE_URL = "https://www.basketball-reference.com"
MIN_REQUEST_INTERVAL = 3.5  # seconds
RETRY_BACKOFF = (10, 30, 60, 0)  # seconds to wait after each failed attempt
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) nba-analytics research"

# Season-level player tables: page suffix -> table id
PLAYER_TABLES = {
    "per_game": "per_game_stats",
    "totals": "totals_stats",
    "per_poss": "per_poss",
    "advanced": "advanced",
}

TEAM_HREF = re.compile(r"/teams/([A-Z]{3})/")

_last_request = 0.0


def fetch(path: str, cache_dir: Path = CACHE_DIR / "bbref", refresh: bool = False) -> str:
    """Return the HTML for ``path`` (e.g. '/leagues/NBA_2026_advanced.html')."""
    global _last_request
    cache_file = cache_dir / path.strip("/").replace("/", "__")
    if cache_file.exists() and not refresh:
        return cache_file.read_text()

    for attempt, backoff in enumerate(RETRY_BACKOFF):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(BASE_URL + path, headers={"User-Agent": USER_AGENT}, timeout=30)
        except (requests.ConnectionError, requests.Timeout):
            resp = None
        _last_request = time.monotonic()
        # Retry network failures and server errors. A 429 means we are rate-limited
        # (Basketball-Reference blocks for an hour), so fail fast instead.
        if resp is not None and resp.status_code < 500:
            break
        if attempt < len(RETRY_BACKOFF) - 1:
            time.sleep(backoff)
    if resp is None:
        raise ConnectionError(f"could not reach {BASE_URL + path}")
    resp.raise_for_status()
    resp.encoding = "utf-8"  # no charset in the headers; requests would guess Latin-1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    return resp.text


def parse_table(html: str, table_id: str) -> pd.DataFrame:
    """Parse a stats table into a DataFrame keyed by ``data-stat`` names.

    Some tables are shipped inside HTML comments and only rendered by
    JavaScript, so comments are searched when the table isn't found directly.
    The player id (e.g. 'jamesle01') is added as ``player_id`` and the first
    linked team (e.g. 'BOS') as ``team_id`` when present.
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
            link = cell.find("a", href=TEAM_HREF)
            if link is not None and "team_id" not in record:
                record["team_id"] = TEAM_HREF.search(link["href"]).group(1)
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


def team_season_table(season: int, refresh: bool = False) -> pd.DataFrame:
    """League-wide advanced team table: W/L, ORtg, DRtg, NRtg, pace, Four Factors."""
    html = fetch(f"/leagues/NBA_{season}.html", refresh=refresh)
    df = parse_table(html, "advanced-team")
    df = df[df["team_id"].notna()].reset_index(drop=True)
    df["team"] = df["team"].str.rstrip("*")  # '*' marks playoff teams
    df.insert(0, "season", season)
    return df.drop(columns=[c for c in df.columns if c.startswith("DUMMY")])


def team_game_log(team_id: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """Regular-season game log with team and opponent box score totals."""
    html = fetch(f"/teams/{team_id}/{season}/gamelog/", refresh=refresh)
    df = parse_table(html, "team_game_log_reg")
    df = df[df["team_game_num_season"].notna()].reset_index(drop=True)
    df = df.drop(columns="team_id")  # the only linked team here is the opponent
    df.insert(0, "season", season)
    df.insert(1, "team_id", team_id)
    return df
