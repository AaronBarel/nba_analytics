"""What first-round picks are worth, and who owns each 2027 first-round pick.

A pick's value is the VORP its player put up in the four seasons after the
draft (zero for players who never played), averaged by draft slot over the
1995-2022 drafts and smoothed with an exponential curve. Stashed and busted
picks count, since a team's incentive is about the slot, not the player.

2027 ownership follows each trade's terms as listed by SalarySwish
(salaryswish.com/draft/2027), checked against Tankathon's resolved draft
order on 2026-10-23. Only first-round picks are modelled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import draft_class, player_season_table
from nba_analytics.players.usage_efficiency import season_rows
from nba_analytics.teams.lottery import TEAMS

WINDOW = 4  # seasons after the draft


def draft_outcomes(first: int = 1995, last: int = 2022, refresh: bool = False) -> pd.DataFrame:
    """Every first-round pick with its player's VORP over the four seasons after the draft."""
    cache = PROCESSED_DIR / f"draft_outcomes_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    adv = pd.concat([season_rows(player_season_table(s, "advanced"))[["season", "player_id", "vorp"]]
                     for s in range(first + 1, last + WINDOW + 1)])
    parts = []
    for year in range(first, last + 1):
        d = draft_class(year)[["pick_overall", "player_id", "player"]].dropna(subset=["player_id"])
        d = d[d["pick_overall"].astype(int) <= 30].assign(draft_year=year)
        vorp = adv[adv["season"].between(year + 1, year + WINDOW)].groupby("player_id")["vorp"].sum()
        parts.append(d.assign(pick=d["pick_overall"].astype(int),
                              vorp=d["player_id"].map(vorp).fillna(0.0)))
    out = pd.concat(parts, ignore_index=True)[["draft_year", "pick", "player_id", "player", "vorp"]]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def _curve(pick, a, b, c):
    return a * np.exp(-b * (np.asarray(pick) - 1)) + c


def fit_pick_value(outcomes: pd.DataFrame) -> np.ndarray:
    """Coefficients of value(pick) = a·exp(-b·(pick - 1)) + c, fit on every pick."""
    coef, _ = curve_fit(_curve, outcomes["pick"], outcomes["vorp"], p0=[6.0, 0.2, 0.5])
    return coef


def pick_value(pick: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """Expected four-season VORP of a first-round pick."""
    return _curve(pick, *coef)


# --- 2027 ownership ---------------------------------------------------------

# Plain transfers: original team -> owner, with the pick kept when it falls at or above
# ``protected`` (0 = unprotected).
TRANSFERS_2027 = [
    ("ATL", "SAS", 0),
    ("NYK", "BRK", 0),
    ("PHO", "HOU", 0),
    ("DAL", "CHO", 2),
    ("MIA", "CHO", 14),
    ("LAL", "MEM", 4),
    ("DEN", "OKC", 5),
    ("SAS", "SAC", 0),
    ("SAS", "OKC", 16),  # Sacramento sends the Spurs' pick on unless it is 1-16
]
TERMS_2027 = pd.DataFrame([
    ("ATL", "SAS", "Unprotected"),
    ("NYK", "BRK", "Unprotected"),
    ("PHO", "HOU", "Unprotected"),
    ("DAL", "CHO", "Top-2 protected"),
    ("MIA", "CHO", "Top-14 protected"),
    ("LAL", "MEM", "Top-4 protected"),
    ("DEN", "OKC", "Top-5 protected"),
    ("SAS", "SAC / OKC", "To SAC; SAC sends it to OKC unless 1-16"),
    ("BRK", "HOU / BRK", "HOU takes the better of the HOU and BRK picks"),
    ("CLE", "MEM / UTA / PHO", ("Of the CLE, MIN and UTA picks: MEM the best, UTA the middle, "
                              "PHO the worst")),
    ("MIN", "MEM / UTA / PHO", "As CLE"),
    ("UTA", "MEM / UTA / PHO", "As CLE"),
    ("MIL", "NOP / ATL", ("NOP the better of the MIL and NOP picks; ATL the worse unless it is "
                         "1-4")),
    ("LAC", "OKC / LAC", "OKC may swap its own or the DEN pick for the LAC pick"),
    ("TOR", "LAC / TOR", "LAC may swap the pick it ends with for the TOR pick"),
], columns=["original", "owner", "terms"])


def owners_2027(pick: np.ndarray, teams: list[str] = TEAMS) -> np.ndarray:
    """Owner (team index) of each original team's 2027 first, given its pick number (S, 30)."""
    i = {t: k for k, t in enumerate(teams)}
    S = len(pick)
    rows = np.arange(S)
    owner = np.tile(np.arange(len(teams)), (S, 1))
    for orig, to, protected in TRANSFERS_2027:
        moves = pick[:, i[orig]] > protected
        owner[moves, i[orig]] = i[to]

    hou_better = pick[:, i["HOU"]] < pick[:, i["BRK"]]
    owner[:, i["HOU"]] = np.where(hou_better, i["HOU"], i["BRK"])
    owner[:, i["BRK"]] = np.where(hou_better, i["BRK"], i["HOU"])

    pool = np.array([i["CLE"], i["MIN"], i["UTA"]])
    ranked = pool[np.argsort(pick[:, pool], axis=1)]
    for k, to in enumerate(["MEM", "UTA", "PHO"]):
        owner[rows, ranked[:, k]] = i[to]

    better, worse = (np.where(pick[:, i["MIL"]] < pick[:, i["NOP"]], i[a], i[b])
                     for a, b in [("MIL", "NOP"), ("NOP", "MIL")])
    owner[rows, better] = i["NOP"]
    owner[rows, worse] = np.where(pick[rows, worse] > 4, i["ATL"], i["NOP"])

    # OKC holds its own pick and, if conveyed, Denver's; it may give the worse for the
    # Clippers' pick. The Clippers then may swap what they hold for Toronto's pick.
    group = np.array([i["OKC"], i["DEN"], i["LAC"]])
    held = pick[:, group].astype(float)
    held[owner[:, i["DEN"]] != i["OKC"], 1] = -np.inf  # Denver kept its pick
    worst = group[np.argmax(held, axis=1)]
    for t in group:
        in_group = (t != i["DEN"]) | (owner[:, i["DEN"]] == i["OKC"])
        owner[in_group, t] = i["OKC"]
    owner[rows, worst] = i["LAC"]
    tor_better = pick[:, i["TOR"]] < pick[rows, worst]
    owner[tor_better, i["TOR"]] = i["LAC"]
    owner[rows[tor_better], worst[tor_better]] = i["TOR"]
    return owner


def holdings(pick: np.ndarray, owner: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Total value of the first-round picks each team owns (S, 30)."""
    out = np.zeros(pick.shape)
    for t in range(pick.shape[1]):
        out[:, t] = np.where(owner == t, value, 0.0).sum(axis=1)
    return out
