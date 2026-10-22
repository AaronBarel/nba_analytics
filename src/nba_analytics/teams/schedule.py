"""Game venues, travel and rest.

Venues are the home team's arena except for neutral-site games: the 2020
restart bubble near Orlando, the NBA Cup semifinals in Las Vegas, and games in
Mexico City and Europe. Time zones come from ``zoneinfo`` so daylight saving
(and Phoenix not observing it) is handled on the game date.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# team_id -> (latitude, longitude, IANA time zone, altitude in meters)
ARENAS = {
    "ATL": (33.757, -84.396, "America/New_York", 320),
    "BOS": (42.366, -71.062, "America/New_York", 5),
    "BRK": (40.683, -73.976, "America/New_York", 10),
    "NJN": (40.770, -74.120, "America/New_York", 5),
    "CHA": (35.225, -80.839, "America/New_York", 230),
    "CHO": (35.225, -80.839, "America/New_York", 230),
    "CHH": (35.225, -80.839, "America/New_York", 230),
    "CHI": (41.881, -87.674, "America/Chicago", 180),
    "CLE": (41.497, -81.688, "America/New_York", 200),
    "DAL": (32.790, -96.810, "America/Chicago", 140),
    "DEN": (39.749, -105.008, "America/Denver", 1609),
    "DET": (42.341, -83.055, "America/Detroit", 190),
    "GSW": (37.768, -122.388, "America/Los_Angeles", 5),
    "HOU": (29.751, -95.362, "America/Chicago", 15),
    "IND": (39.764, -86.156, "America/Indiana/Indianapolis", 220),
    "LAC": (34.043, -118.267, "America/Los_Angeles", 90),
    "LAL": (34.043, -118.267, "America/Los_Angeles", 90),
    "MEM": (35.138, -90.051, "America/Chicago", 80),
    "VAN": (49.278, -123.109, "America/Vancouver", 5),
    "MIA": (25.781, -80.188, "America/New_York", 2),
    "MIL": (43.045, -87.917, "America/Chicago", 180),
    "MIN": (44.980, -93.276, "America/Chicago", 260),
    "NOH": (29.949, -90.082, "America/Chicago", 2),
    "NOP": (29.949, -90.082, "America/Chicago", 2),
    "NOK": (35.463, -97.515, "America/Chicago", 370),  # Hornets in Oklahoma City after Katrina
    "NYK": (40.751, -73.993, "America/New_York", 10),
    "OKC": (35.463, -97.515, "America/Chicago", 370),
    "SEA": (47.622, -122.354, "America/Los_Angeles", 40),
    "ORL": (28.539, -81.384, "America/New_York", 30),
    "PHI": (39.901, -75.172, "America/New_York", 5),
    "PHO": (33.446, -112.071, "America/Phoenix", 330),
    "POR": (45.532, -122.667, "America/Los_Angeles", 15),
    "SAC": (38.580, -121.500, "America/Los_Angeles", 8),
    "SAS": (29.427, -98.438, "America/Chicago", 200),
    "TOR": (43.643, -79.379, "America/Toronto", 80),
    "UTA": (40.768, -111.901, "America/Denver", 1288),
    "WAS": (38.898, -77.021, "America/New_York", 10),
}

CITIES = {
    "Orlando bubble": (28.337, -81.556, "America/New_York", 30),
    "Las Vegas": (36.103, -115.178, "America/Los_Angeles", 610),
    "Mexico City": (19.404, -99.096, "America/Mexico_City", 2240),
    "Paris": (48.839, 2.379, "Europe/Paris", 35),
    "Berlin": (52.508, 13.443, "Europe/Berlin", 35),
    "London": (51.503, 0.003, "Europe/London", 5),
}

# Neutral-site regular-season games Basketball-Reference marks 'N', by date
NEUTRAL_DATES = {
    "2024-11-02": "Mexico City", "2024-12-14": "Las Vegas", "2025-01-23": "Paris",
    "2025-01-25": "Paris", "2025-11-01": "Mexico City", "2025-12-13": "Las Vegas",
    "2026-01-15": "Berlin", "2026-01-18": "London",
}
BUBBLE_START = "2020-07-30"  # 2019-20 restart, all games at Walt Disney World
MAX_REST = 4  # rest days beyond this are treated the same
EARTH_RADIUS_KM = 6371.0


def venue(date: str, home_id: str, neutral: bool) -> str:
    """Key into ``ARENAS`` or ``CITIES`` for where a game was played."""
    if date >= BUBBLE_START and date < "2020-10-15":
        return "Orlando bubble"
    if neutral:
        return NEUTRAL_DATES.get(date, home_id)
    return home_id


def _place(key: str) -> tuple:
    return ARENAS.get(key) or CITIES[key]


def utc_offset(key: str, date: str) -> float:
    """Hours from UTC at a venue on a date (noon local, so DST changes resolve)."""
    tz = ZoneInfo(_place(key)[2])
    return datetime.fromisoformat(date).replace(hour=12, tzinfo=tz).utcoffset().total_seconds() / 3600


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def schedule_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Rest, travel and time-zone change for each team before each game.

    ``team_games`` has one row per team and game with ``season``, ``team_id``,
    ``date`` and ``venue``. A team's first game of the season counts as
    starting from its own arena with full rest.
    """
    df = team_games.sort_values(["season", "team_id", "date"]).copy()
    g = df.groupby(["season", "team_id"])
    date = pd.to_datetime(df["date"])
    prev_date = g["date"].shift(1).pipe(pd.to_datetime)
    df["rest_days"] = ((date - prev_date).dt.days - 1).clip(upper=MAX_REST).fillna(MAX_REST)
    df["back_to_back"] = (df["rest_days"] == 0).astype(int)
    # games in the four days before this one (e.g. three-in-four nights)
    day = date.dt.normalize()
    df["games_last_4d"] = 0
    for k in (1, 2, 3):
        prev = g["date"].shift(k).pipe(pd.to_datetime)
        df["games_last_4d"] += ((day - prev).dt.days <= 4).fillna(False).astype(int)

    prev_venue = g["venue"].shift(1).fillna(df["team_id"])
    here = np.array([_place(v)[:2] for v in df["venue"]], dtype=float)
    there = np.array([_place(v)[:2] for v in prev_venue], dtype=float)
    df["travel_km"] = haversine_km(there[:, 0], there[:, 1], here[:, 0], here[:, 1])
    offsets = {}
    for v, d in zip(pd.concat([df["venue"], prev_venue]), pd.concat([df["date"], df["date"]])):
        offsets.setdefault((v, d), utc_offset(v, d))
    now = np.array([offsets[(v, d)] for v, d in zip(df["venue"], df["date"])])
    before = np.array([offsets[(v, d)] for v, d in zip(prev_venue, df["date"])])
    df["tz_shift"] = now - before  # positive = travelled east
    df["altitude_m"] = [_place(v)[3] for v in df["venue"]]
    return df
