"""The NBA's official injury report (PDF, published every 15 minutes on game days).

Each report lists, for the next day or two of games, every player a team has
declared Out, Doubtful, Questionable, Probable or Available, with a reason.
Reports are cached by their timestamp. The PDF is a table without ruling, so
words are placed in columns by their position under the header.
"""

from __future__ import annotations

import bisect
import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pdfplumber
import requests

from nba_analytics.config import CACHE_DIR

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"
EASTERN = ZoneInfo("America/New_York")
CACHE = CACHE_DIR / "injury_reports"
STEP = timedelta(minutes=15)

# Header words that start each column, left to right
COLUMNS = [("Game", "game_date"), ("Game", "game_time"), ("Matchup", "matchup"),
           ("Team", "team"), ("Player", "player"), ("Current", "status"), ("Reason", "reason")]
LINE_TOLERANCE = 2.0  # points; words this close vertically share a line
FOOTER = re.compile(r"^Page\s*\d+\s*of\s*\d+$")

# Official team names -> Basketball-Reference ids
TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}
ABBREVIATIONS = {"BKN": "BRK", "CHA": "CHO", "PHX": "PHO"}  # NBA -> Basketball-Reference


def stamp(ts: datetime) -> str:
    """Eastern time -> the report's file stamp, e.g. '2026-10-22_05_00PM'."""
    return ts.strftime("%Y-%m-%d_%I_%M%p")


def latest_stamp(now: datetime | None = None, hours_back: int = 24) -> str:
    """The stamp of the most recent published report at or before ``now``."""
    now = (now or datetime.now(EASTERN)).astimezone(EASTERN)
    ts = now.replace(minute=now.minute - now.minute % 15, second=0, microsecond=0)
    for _ in range(hours_back * 4):
        s = stamp(ts)
        if (CACHE / f"{s}.pdf").exists():
            return s
        resp = requests.head(URL.format(stamp=s), timeout=15)
        if resp.status_code == 200:
            return s
        ts -= STEP
    raise LookupError(f"no injury report in the {hours_back} hours before {now}")


def download(s: str) -> Path:
    """Cached path of the report with stamp ``s``."""
    path = CACHE / f"{s}.pdf"
    if not path.exists():
        resp = requests.get(URL.format(stamp=s), timeout=30)
        resp.raise_for_status()
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
    return path


def _lines(page, starts: list[float] | None) -> tuple[list[float], list]:
    """Text of each column on each line of a page, as (top, {column: text}).

    Only the first page has the column header; later pages reuse its
    ``starts`` (column left edges). Returns the starts and the lines.
    """
    words = page.extract_words(x_tolerance=1.5)
    header = next((w for w in words if w["text"] == "Matchup"), None)
    if header is not None:
        head = sorted((w for w in words if abs(w["top"] - header["top"]) < LINE_TOLERANCE),
                      key=lambda w: w["x0"])
        starts, i = [], 0
        for w in head:  # the first header word of each column, in order
            if i < len(COLUMNS) and w["text"] == COLUMNS[i][0]:
                starts.append(w["x0"] - 2)
                i += 1
    top_of_table = (header or next(w for w in words if w["text"] == "Report:"))["top"]
    lines: list[tuple[float, dict[str, list[str]]]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if w["top"] <= top_of_table + LINE_TOLERANCE:
            continue
        col = COLUMNS[max(bisect.bisect_right(starts, w["x0"]) - 1, 0)][1]
        if not lines or w["top"] - lines[-1][0] > LINE_TOLERANCE:
            lines.append((w["top"], {}))
        lines[-1][1].setdefault(col, []).append(w["text"])
    lines = [(top, {c: " ".join(t) for c, t in cols.items()}) for top, cols in lines]
    return starts, [(top, cols) for top, cols in lines
                    if not FOOTER.match(" ".join(cols.values()))]


def parse_report(path: Path) -> pd.DataFrame:
    """One row per listed player: game, team (Basketball-Reference id), name, status, reason.

    Teams that have not yet submitted appear with no player and status
    'Not Yet Submitted'. Names are as printed ('Last, First').
    """
    rows = []
    game = {"game_date": None, "game_time": None, "matchup": None, "team": None}
    starts = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            starts, lines = _lines(page, starts)
            listed = [i for i, (_, c) in enumerate(lines) if "status" in c or "player" in c]
            reasons: dict[int, list[str]] = {i: [] for i in listed}
            for i, (top, cols) in enumerate(lines):
                for k in game:
                    if k in cols:
                        game[k] = cols[k]
                if i in listed:
                    rows.append({**game, "player": cols.get("player"),
                                 "status": cols.get("status"), "line": (page.page_number, i)})
                    if "reason" in cols:
                        reasons[i].append(cols["reason"])
                elif "NOT YET SUBMITTED" in cols.get("reason", ""):
                    rows.append({**game, "player": None, "status": "Not Yet Submitted",
                                 "line": (page.page_number, i)})
                elif "reason" in cols and listed:  # a wrapped reason: nearest listed line
                    j = min(listed, key=lambda j: abs(lines[j][0] - top))
                    reasons[j].append(cols["reason"])
            for r in rows:
                if r["line"][0] == page.page_number and r["line"][1] in reasons:
                    r["reason"] = " ".join(reasons[r["line"][1]])
    df = pd.DataFrame(rows).drop(columns="line")
    df["game_date"] = pd.to_datetime(df["game_date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    teams = df["matchup"].str.split("@", expand=True).replace(ABBREVIATIONS)
    df["away"], df["home"] = teams[0], teams[1]
    df["team_id"] = df["team"].map(TEAMS)
    return df[["game_date", "game_time", "away", "home", "team_id", "player", "status",
               "reason"]]


def latest_report(now: datetime | None = None) -> tuple[str, pd.DataFrame]:
    """The most recent report's stamp and rows."""
    s = latest_stamp(now)
    return s, parse_report(download(s))
