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
    "play-by-play": "pbp_stats",
}

TEAM_HREF = re.compile(r"/teams/([A-Z]{3})/")
PLAYER_HREF = re.compile(r"/players/[a-z]/([a-z0-9]+)\.html")

_last_request = 0.0


def fetch(path: str, cache_dir: Path = CACHE_DIR / "bbref", refresh: bool = False,
          base_url: str = BASE_URL, max_age: float | None = None) -> str:
    """Return the HTML for ``path`` (e.g. '/leagues/NBA_2026_advanced.html').

    ``base_url`` lets other Sports-Reference sites share the cache logic and the
    request spacing, which is enforced across all of them. ``max_age`` (hours)
    re-downloads a cached page older than that, for pages that change.
    """
    global _last_request
    cache_file = cache_dir / path.strip("/").replace("/", "__")
    if max_age is not None and cache_file.exists():
        refresh |= time.time() - cache_file.stat().st_mtime > max_age * 3600
    if cache_file.exists() and not refresh:
        return cache_file.read_text()

    for attempt, backoff in enumerate(RETRY_BACKOFF):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(base_url + path, headers={"User-Agent": USER_AGENT}, timeout=30)
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
        raise ConnectionError(f"could not reach {base_url + path}")
    resp.raise_for_status()
    resp.encoding = "utf-8"  # no charset in the headers; requests would guess Latin-1

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(resp.text)
    return resp.text


def parse_table(html: str, table_id: str) -> pd.DataFrame:
    """Parse a stats table into a DataFrame keyed by ``data-stat`` names.

    Some tables are shipped inside HTML comments and only rendered by
    JavaScript, so comments are searched when the table isn't found directly.
    The player id (e.g. 'jamesle01', from ``data-append-csv`` or the player
    link) is added as ``player_id`` and the first
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
            elif stat == "player" and (plink := cell.find("a", href=PLAYER_HREF)):
                record["player_id"] = PLAYER_HREF.search(plink["href"]).group(1)
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


def team_on_off(team_id: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """Team stats with each player on and off the court (2000-01 onward).

    Three rows per player: 'On Court', 'Off Court' and 'On − Off'. Columns
    prefixed ``opp_`` are the opponent's numbers, so ``opp_off_rtg`` is the
    team's defensive rating for that split.
    """
    html = fetch(f"/teams/{team_id}/{season}/on-off/", refresh=refresh)
    df = parse_table(html, "on_off")
    df["player_id"] = df["player_id"].ffill()  # the On − Off row has no player link
    df["split_id"] = df["split_id"].str.replace("\u2212", "-")
    df["mp"] = pd.to_numeric(df["mp"], errors="coerce")  # On − Off row holds a share, e.g. 66%
    df = df.drop(columns=["player", "team_id"], errors="ignore")
    df.insert(0, "season", season)
    df.insert(1, "team_id", team_id)
    return df


CBB_HREF = re.compile(r"https?://www\.sports-reference\.com(/cbb/players/[a-z0-9-]+\.html)")
_ORD = r"(\d+)(?:st|nd|rd|th)"
DRAFT = re.compile(rf"{_ORD} round \({_ORD} pick, {_ORD} overall\), (\d{{4}}) NBA Draft")


def player_page(player_id: str, refresh: bool = False) -> str:
    return fetch(f"/players/{player_id[0]}/{player_id}.html", refresh=refresh)


def player_bio(html: str) -> dict:
    """Height, weight, birth date, draft slot, recruiting rank and college page path.

    Missing fields are None: undrafted players have no draft slot, players who
    skipped college have no college path, and older players have no RSCI rank.
    """
    soup = BeautifulSoup(html, "lxml")
    meta = soup.find(id="meta")
    text = " ".join(meta.get_text(" ").split()) if meta else ""
    birth = soup.find(attrs={"data-birth": True})
    cbb = CBB_HREF.search(html)
    draft = DRAFT.search(text)
    height = re.search(r"\((\d+)\s*cm", text)
    weight = re.search(r"(\d+)\s*kg\)", text)
    rsci = re.search(r"Recruiting Rank: (\d{4}) \((\d+)\)", text)
    return {
        "height_cm": int(height.group(1)) if height else None,
        "weight_kg": int(weight.group(1)) if weight else None,
        "birth_date": birth["data-birth"] if birth else None,
        "draft_year": int(draft.group(4)) if draft else None,
        "draft_round": int(draft.group(1)) if draft else None,
        "draft_pick": int(draft.group(3)) if draft else None,
        "rsci_rank": int(rsci.group(2)) if rsci else None,
        "cbb_path": cbb.group(1) if cbb else None,
    }


def draft_class(year: int, refresh: bool = False) -> pd.DataFrame:
    """Every pick in an NBA draft, with ``player_id`` and the drafting team."""
    df = parse_table(fetch(f"/draft/NBA_{year}.html", refresh=refresh), "stats")
    df = df[df["pick_overall"].notna()].copy()
    df.insert(0, "draft_year", year)
    return df.reset_index(drop=True)


def player_game_log(player_id: str, season: int, refresh: bool = False) -> pd.DataFrame:
    """A player's regular-season games for one season (end year).

    Games missed while on the roster are kept, with the reason ('Inactive',
    'Did Not Dress', 'Did Not Play', 'Not With Team') in ``status`` and no
    minutes. ``mp`` is converted from 'mm:ss' to minutes.
    """
    try:
        html = fetch(f"/players/{player_id[0]}/{player_id}/gamelog/{season}", refresh=refresh)
    except requests.HTTPError as e:  # a few listed player-seasons have no game log page
        if e.response.status_code == 404:
            return pd.DataFrame()
        raise
    try:
        df = parse_table(html, "player_game_log_reg")
    except ValueError:  # no regular-season games (e.g. playoffs only)
        return pd.DataFrame()
    df = df[df["team_game_num_season"].notna() & df["date"].notna()].copy()
    # The reason for a missed game fills the row from the first stat column on,
    # which is 'is_starter' unless the player never started.
    first = df["is_starter"] if "is_starter" in df else df["mp"]
    first = first.astype("string")
    df["status"] = first.where(first.str.contains("[A-Za-z]", na=False), None)
    df["started"] = first.eq("*").fillna(False).astype(int)
    mp = df["mp"].astype("string").where(df["status"].isna()).str.split(":", expand=True)
    df["mp"] = pd.to_numeric(mp[0], errors="coerce") + pd.to_numeric(mp[1], errors="coerce") / 60
    df.insert(0, "season", season)
    df.insert(1, "player_id", player_id)
    return df.reset_index(drop=True)


# --- Current season ---------------------------------------------------------

def schedule_months(season: int, max_age: float | None = None) -> list[str]:
    """Month names with a schedule page for a season (e.g. ['october', ...])."""
    html = fetch(f"/leagues/NBA_{season}_games.html", max_age=max_age)
    return list(dict.fromkeys(re.findall(rf"NBA_{season}_games-(\w+)\.html", html)))


def schedule_month(season: int, month: str, max_age: float | None = None) -> pd.DataFrame:
    """One month of the schedule with results so far.

    One row per game: ISO ``date``, ``away`` and ``home`` team ids, points
    (missing until played), ``arena`` and ``box_score`` (the box score path,
    once the game is final).
    """
    html = fetch(f"/leagues/NBA_{season}_games-{month}.html", max_age=max_age)
    table = BeautifulSoup(html, "lxml").find("table", id="schedule")
    records = []
    for row in table.tbody.find_all("tr"):
        cells = {c.get("data-stat"): c for c in row.find_all(["th", "td"])}
        if "home_team_name" not in cells:
            continue
        box = cells["box_score_text"].find("a")
        records.append({
            "date": pd.to_datetime(cells["date_game"].get_text(strip=True)).strftime("%Y-%m-%d"),
            "start_time": cells["game_start_time"].get_text(strip=True) or None,
            "away": TEAM_HREF.search(cells["visitor_team_name"].a["href"]).group(1),
            "home": TEAM_HREF.search(cells["home_team_name"].a["href"]).group(1),
            "away_pts": pd.to_numeric(cells["visitor_pts"].get_text(strip=True) or None),
            "home_pts": pd.to_numeric(cells["home_pts"].get_text(strip=True) or None),
            "arena": cells["arena_name"].get_text(strip=True) or None,
            "box_score": box["href"] if box else None,
        })
    df = pd.DataFrame.from_records(records)
    df.insert(0, "season", season)
    return df


def _minutes(mp: pd.Series) -> pd.Series:
    """'mm:ss' -> minutes."""
    parts = mp.astype("string").str.split(":", expand=True)
    return pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60


INACTIVE = re.compile(r"<strong>([A-Z]{3})</strong>((?:(?!<strong>).)*)", re.DOTALL)


def box_score(path: str, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Player rows and team totals for a finished game (e.g. '/boxscores/202610200DET.html').

    Player rows cover everyone listed: ``started`` marks the first five,
    ``status`` holds the reason for not playing ('Did Not Play', 'Did Not
    Dress', 'Not With Team', ...) or 'Inactive' for the inactive list. Team
    totals have one row per team with the box score counts.
    """
    html = fetch(path, refresh=refresh)
    soup = BeautifulSoup(html, "lxml")
    players, totals = [], []
    for team in dict.fromkeys(re.findall(r'id="box-([A-Z]{3})-game-basic"', html)):
        df = parse_table(html, f"box-{team}-game-basic")
        df = df[df["player_id"].notna()].reset_index(drop=True)
        df["team_id"] = team
        df["started"] = (df.index < 5).astype(int)
        df["status"] = df.get("reason")
        df["mp"] = _minutes(df["mp"])
        players.append(df)
        foot = soup.find("table", id=f"box-{team}-game-basic").tfoot
        row = {c["data-stat"]: c.get_text(strip=True) for c in foot.find_all(["th", "td"])
               if c.get("data-stat")}
        totals.append({"team_id": team, **{k: pd.to_numeric(v, errors="coerce")
                                           for k, v in row.items()}})
    block = html[html.find("Inactive:"):]
    block = block[:block.find("</div>")]
    for team, links in INACTIVE.findall(block):
        ids = PLAYER_HREF.findall(links)
        players.append(pd.DataFrame({"player_id": ids, "team_id": team, "started": 0,
                                     "status": "Inactive", "mp": float("nan")}))
    return pd.concat(players, ignore_index=True), pd.DataFrame(totals)


def team_roster(team_id: str, season: int, max_age: float | None = None) -> pd.DataFrame:
    """A team's current roster: ``player_id``, ``player``, ``pos``, experience."""
    df = parse_table(fetch(f"/teams/{team_id}/{season}.html", max_age=max_age), "roster")
    df = df[["player_id", "player", "pos", "years_experience"]].copy()
    df["player"] = df["player"].str.replace(r"\s*\(TW\)$", "", regex=True)  # two-way tag
    df.insert(0, "team_id", team_id)
    return df


INJURY_STATUS = re.compile(r"^(Out For Season|Out|Day To Day)")


def injury_list(max_age: float | None = None) -> pd.DataFrame:
    """Basketball-Reference's current injury list: player, team, status and note."""
    df = parse_table(fetch("/friv/injuries.fcgi", max_age=max_age), "injuries")
    df["status"] = df["note"].str.extract(INJURY_STATUS, expand=False)
    df["date_update"] = pd.to_datetime(df["date_update"]).dt.strftime("%Y-%m-%d")
    return df[["player_id", "player", "team_id", "date_update", "status", "note"]]
