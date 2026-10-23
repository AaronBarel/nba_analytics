"""Roster talent, availability and turnover for the game model (T3).

A player's value is his projected Box Plus/Minus for the season: the
walk-forward season projection (P2) for veterans and the college/draft model
(R1) for rookies, both made before the season with no later data. Expected
minutes start at the projected minutes per game and move toward his actual
minutes per game this season as games accumulate.

For each game, the team's available players (those who played, or were
dressed but not used) get minutes in order of expected minutes until 240 are
used; any minutes left over go to a replacement-level player. Talent is then
the minutes-weighted sum of BPM divided by 48, which is on the scale of a
team net rating. The same calculation over the whole roster, injured players
included, gives the talent lost to absences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import draft_class, player_game_log, player_season_table
from nba_analytics.players import projections, rookies
from nba_analytics.players.defense import career_start
from nba_analytics.teams.net_rating import franchise

REPLACEMENT_BPM = -2.0
REPLACEMENT_MPG = 8.0  # expected minutes for a player with no projection
TEAM_MINUTES = 240
MINUTES_PRIOR_GAMES = 10  # weight of projected minutes against minutes played so far
AVAILABLE = {"Did Not Play"}  # dressed but unused; every other reason is an absence
BOX = ["pts", "trb", "orb", "drb", "ast", "stl", "blk", "tov", "fg", "fga", "fg3", "fg3a", "ft",
       "fta", "pf", "plus_minus"]
PROJECTED_RATES = ["pts_per_100", "trb_per_100", "ast_per_100", "stl_per_100", "blk_per_100",
                   "tov_per_100", "fg3a_per_100", "usg_pct"]


def gamelogs(first: int, last: int, refresh: bool = False) -> pd.DataFrame:
    """Every player's game rows (played or not) for the seasons, from the HTML cache."""
    cache = PROCESSED_DIR / f"player_gamelogs_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    parts = []
    for season in range(first, last + 1):
        for pid in player_season_table(season, "totals")["player_id"].unique():
            log = player_game_log(pid, season)
            if not log.empty:
                parts.append(log[["season", "player_id", "date", "team_id", "game_location",
                                  "opp_name_abbr", "status", "started", "mp", *BOX]])
    df = pd.concat(parts, ignore_index=True)
    df["available"] = df["status"].isna() | df["status"].isin(AVAILABLE)
    for c in ["mp", *BOX]:
        df[c] = df[c].astype(float)
    df["mp"] = df["mp"].fillna(0.0)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _rookie_predictions(rk: pd.DataFrame, rows: pd.DataFrame, features: list[str]
                        ) -> pd.DataFrame:
    """Rookie BPM and minutes for ``rows`` from R1 models that never saw their draft class."""
    out = rows[["player_id", "season"]].copy()
    for target, col in [("rookie_bpm", "bpm"), ("rookie_mpg", "mpg")]:
        train = rookies.training_rows(rk, target)
        out[col] = np.nan
        for year, part in rows.groupby("entry_year"):
            other = train[train["entry_year"] != year]
            fit = rookies.model(features).fit(other[features], other[target])
            out.loc[part.index, col] = fit.predict(part[features])
    return out


def player_values(first: int, last: int, refresh: bool = False) -> pd.DataFrame:
    """Pre-season projected BPM and minutes per game for each (player, season).

    Veterans also get projected per-100 rates and usage (rookies have none).

    Veterans: P2 walk-forward projections. Rookies with college stats: the
    full R1 model. Other drafted rookies (international or straight from high
    school): the draft-slot-only R1 model. Rookie models are leave-one-class-out.
    """
    cache = PROCESSED_DIR / f"player_values_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    seasons = projections.player_seasons()
    frame = projections.build_frame(seasons, last=seasons["season"].max())
    vets = _veteran_values(frame, projections.walk_forward(frame, seasons, first, last))

    rk = rookies.rookie_dataset()
    debut = career_start(last)
    college = rk[~rk["draft_class"]].assign(season=lambda d: d["debut"])
    picks = _draft_picks(range(first - 6, last), rk).assign(
        season=lambda d: d["player_id"].map(debut))
    out = _combine(vets, _rookie_values(rk, college, picks), first, last)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def season_values(season: int, refresh: bool = False) -> pd.DataFrame:
    """Pre-season values, as in ``player_values``, for a season not yet in the tables.

    Veterans who played in any of the last three seasons get the season
    projection fit on every earlier season (so players returning from a lost
    season are covered); last summer's draft class gets the rookie models.
    Cached to ``data/processed``.
    """
    cache = PROCESSED_DIR / f"season_values_{season}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    seasons = projections.player_seasons(last=season - 1)
    frame = projections.build_frame(seasons, last=season - 1, require_last=False)
    rows = frame["season"] == season
    preds = projections.fit_predict(frame, seasons, frame["season"] < season, rows)
    vets = _veteran_values(frame, preds)
    rk = rookies.rookie_dataset(last=season - 1)
    college = rk[rk["draft_class"]].assign(season=season)
    picks = _draft_picks(range(season - 1, season), rk).assign(season=season)
    out = _combine(vets, _rookie_values(rk, college, picks), season, season)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def _veteran_values(frame: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[preds.index, ["player_id", "season"]].assign(
        bpm=preds[("model", "bpm")].to_numpy(), mpg=preds[("model", "mp_per_g")].to_numpy(),
        **{c: preds[("model", c)].to_numpy() for c in PROJECTED_RATES},
        source="Season projection")


def _draft_picks(years: range, rk: pd.DataFrame) -> pd.DataFrame:
    """Drafted players with no college stats in ``rk`` (international, high school)."""
    picks = pd.concat([draft_class(y)[["player_id", "pick_overall"]].assign(entry_year=y)
                       for y in years], ignore_index=True)
    picks = picks.dropna(subset=["player_id"]).drop_duplicates("player_id")
    return picks[~picks["player_id"].isin(rk["player_id"])].assign(
        log_pick=lambda d: np.log(d["pick_overall"]), drafted=1).reset_index(drop=True)


def _rookie_values(rk: pd.DataFrame, college: pd.DataFrame, picks: pd.DataFrame
                   ) -> pd.DataFrame:
    college = _rookie_predictions(rk, college, rookies.FEATURE_SETS["All (adds draft slot, RSCI)"])
    picks = picks.dropna(subset=["season"]).reset_index(drop=True)
    drafted = _rookie_predictions(rk, picks, rookies.FEATURE_SETS["Draft slot"])
    return pd.concat([college.assign(source="Rookie (college)"),
                      drafted.assign(source="Rookie (draft slot)")], ignore_index=True)


def _combine(vets: pd.DataFrame, rook: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    rook = rook[rook["season"].between(first, last)]
    known = pd.MultiIndex.from_frame(vets[["player_id", "season"]])
    rook = rook[~pd.MultiIndex.from_frame(rook[["player_id", "season"]]).isin(known)]
    out = pd.concat([vets, rook], ignore_index=True)
    out["season"] = out["season"].astype(int)
    out["mpg"] = out["mpg"].clip(0, 40)
    return out


def _allocate(expected: np.ndarray, bpm: np.ndarray) -> float:
    """Team talent: fill 240 minutes in order of expected minutes, then replacement."""
    order = np.argsort(-expected)
    left, total = TEAM_MINUTES, 0.0
    for i in order:
        m = min(expected[i], left)
        total += m * bpm[i]
        left -= m
        if left <= 0:
            break
    total += max(left, 0) * REPLACEMENT_BPM
    return total / 48


def expected_minutes(logs: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Game logs with pre-season values and pre-game expected minutes for each row.

    Expected minutes blend the projection with minutes played for the team in
    earlier games this season (``MINUTES_PRIOR_GAMES`` games of the projection),
    so role changes show up as the season goes on. Rows for games not yet
    played have ``mp`` missing and count for nothing.
    """
    cols = ["player_id", "season", "bpm", "mpg", *[c for c in PROJECTED_RATES if c in values]]
    df = logs.merge(values[cols], on=["player_id", "season"], how="left")
    df["projected"] = df["bpm"].notna()
    df["bpm"] = df["bpm"].fillna(REPLACEMENT_BPM)
    df["mpg"] = df["mpg"].fillna(REPLACEMENT_MPG)
    df = df.sort_values(["season", "player_id", "date"]).reset_index(drop=True)
    keys = [df["season"], df["player_id"]]
    done = df["mp"].notna()
    mp = df["mp"].fillna(0.0)
    mp_before = mp.groupby(keys).cumsum() - mp
    games_before = done.groupby(keys).cumsum() - done  # roster games, played or not
    df["expected_mp"] = ((MINUTES_PRIOR_GAMES * df["mpg"] + mp_before)
                         / (MINUTES_PRIOR_GAMES + games_before))
    return df


def team_talent(logs: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Per team and game: talent of available players, of the full roster, and the gap."""
    df = expected_minutes(logs, values)
    records = []
    for (season, team, date), grp in df.groupby(["season", "team_id", "date"], sort=False):
        exp, bpm, avail = grp["expected_mp"].to_numpy(), grp["bpm"].to_numpy(), \
            grp["available"].to_numpy()
        full = _allocate(exp, bpm)
        here = _allocate(exp[avail], bpm[avail])
        top = np.argmax(exp * (bpm - REPLACEMENT_BPM))  # the most valuable player by role
        records.append((season, team, date, here, full, full - here, int(avail.sum()),
                        int(not avail[top])))
    return pd.DataFrame(records, columns=["season", "team_id", "date", "talent",
                                          "talent_full", "talent_missing", "n_available",
                                          "best_player_out"])


def returning_minutes(gl: pd.DataFrame, first: int) -> pd.DataFrame:
    """Per team and game: share of the franchise's minutes last season played by its roster.

    The roster is everyone listed for that game, so trades change it mid-season.
    """
    prev = []
    for season in range(first - 1, int(gl["season"].max())):
        t = player_season_table(season, "totals")
        t = t[t["team_id"].notna()]  # per-team rows, not the combined '2TM' rows
        prev.append(t[["player_id", "team_id", "mp"]].assign(season=season + 1))
    prev = pd.concat(prev, ignore_index=True)
    prev["franchise"] = franchise(prev["team_id"])
    prev = prev.groupby(["season", "franchise", "player_id"], as_index=False)["mp"].sum()
    totals = prev.groupby(["season", "franchise"])["mp"].sum().rename("team_mp")

    roster = gl[["season", "team_id", "date", "player_id"]].assign(
        franchise=lambda d: franchise(d["team_id"]))
    roster = roster.merge(prev, on=["season", "franchise", "player_id"], how="left")
    out = (roster.groupby(["season", "team_id", "franchise", "date"], as_index=False)["mp"]
           .sum().join(totals, on=["season", "franchise"]))
    out["returning_share"] = out["mp"] / out["team_mp"]
    return out[["season", "team_id", "date", "returning_share"]]
