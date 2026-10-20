"""Net rating persistence (T1) and early-season predictiveness (T2).

T1: how well does one season's net rating explain the next season's?
T2: after N games, how well do the record and net rating so far explain the
same measures over the rest of the season (games N+1 to the end)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import team_game_log, team_season_table

# Relocations, mapped to the current team id so consecutive seasons line up.
FRANCHISE = {"NJN": "BRK", "SEA": "OKC", "VAN": "MEM", "CHH": "NOP", "NOH": "NOP",
             "NOK": "NOP", "CHA": "CHO"}

# Lockout / pandemic seasons with fewer than 82 games
SHORTENED = {1999, 2012, 2020, 2021}


def franchise(team_id: pd.Series) -> pd.Series:
    return team_id.replace(FRANCHISE)


def r_squared(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """R² of a one-variable least-squares fit (the squared Pearson correlation)."""
    return float(np.corrcoef(x, y)[0, 1] ** 2)


# --- T1 ---------------------------------------------------------------------

def team_seasons(seasons: range) -> pd.DataFrame:
    df = pd.concat([team_season_table(s) for s in seasons], ignore_index=True)
    df["franchise"] = franchise(df["team_id"])
    df["win_pct"] = df["wins"] / (df["wins"] + df["losses"])
    df["shortened"] = df["season"].isin(SHORTENED)
    return df


def consecutive_seasons(df: pd.DataFrame, cols: tuple[str, ...] = ("net_rtg", "win_pct")
                        ) -> pd.DataFrame:
    """One row per franchise and season pair: ``<col>_prev`` and ``<col>_next``."""
    keep = ["franchise", "season", "team_id", "team", "shortened", *cols]
    prev = df[keep].rename(columns=lambda c: c if c == "franchise" else f"{c}_prev")
    nxt = df[keep].rename(columns=lambda c: c if c == "franchise" else f"{c}_next")
    prev["season_next"] = prev["season_prev"] + 1
    out = prev.merge(nxt, on=["franchise", "season_next"])
    out["any_shortened"] = out["shortened_prev"] | out["shortened_next"]
    return out


def r_squared_by_pair(pairs: pd.DataFrame, col: str = "net_rtg") -> pd.DataFrame:
    """R² across teams for each season-to-season transition."""
    rows = [{"season_next": s, "r2": r_squared(g[f"{col}_prev"], g[f"{col}_next"]),
             "any_shortened": g["any_shortened"].iloc[0]}
            for s, g in pairs.groupby("season_next")]
    return pd.DataFrame(rows)


# --- T2 ---------------------------------------------------------------------

def possessions(g: pd.DataFrame) -> pd.Series:
    """Basketball-Reference's possession estimate, averaged over both teams."""
    def one_side(fga, fta, orb, opp_drb, fg, tov):
        return fga + 0.4 * fta - 1.07 * (orb / (orb + opp_drb)) * (fga - fg) + tov

    team = one_side(g["fga"], g["fta"], g["orb"], g["opp_drb"], g["fg"], g["tov"])
    opp = one_side(g["opp_fga"], g["opp_fta"], g["opp_orb"], g["drb"], g["opp_fg"],
                   g["opp_tov"])
    return 0.5 * (team + opp)


def game_logs(seasons: range, cache: bool = True) -> pd.DataFrame:
    """Every team's regular-season games, with points, possessions and wins."""
    path = PROCESSED_DIR / f"team_game_logs_{seasons.start}_{seasons.stop - 1}.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)

    logs = pd.concat([team_game_log(t, s) for s in seasons
                      for t in team_season_table(s)["team_id"]], ignore_index=True)
    logs = logs.rename(columns={"team_game_num_season": "game_num", "team_game_score": "pts",
                                "opp_team_game_score": "opp_pts"})
    logs["poss"] = possessions(logs)
    logs["win"] = (logs["team_game_result"] == "W").astype(int)
    logs["franchise"] = franchise(logs["team_id"])
    cols = ["season", "team_id", "franchise", "game_num", "date", "game_location",
            "opp_name_abbr", "pts", "opp_pts", "poss", "win",
            "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "drb", "ast", "stl", "blk", "tov",
            "pf", "opp_fg", "opp_fga", "opp_fg3", "opp_fg3a", "opp_ft", "opp_fta", "opp_orb",
            "opp_drb", "opp_ast", "opp_stl", "opp_blk", "opp_tov", "opp_pf"]
    logs = logs[cols]
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        logs.to_parquet(path)
    return logs


def split_at(logs: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per team-season: win% and net rating over games 1..n and over games n+1..end."""
    first = logs["game_num"] <= n
    parts = {}
    for name, mask in [("first", first), ("rest", ~first)]:
        g = logs[mask].groupby(["season", "team_id"])
        parts[name] = pd.DataFrame({
            f"win_pct_{name}": g["win"].mean(),
            f"net_rtg_{name}": 100 * (g["pts"].sum() - g["opp_pts"].sum()) / g["poss"].sum(),
        })
    return parts["first"].join(parts["rest"], how="inner").reset_index()


PREDICTORS = ["win_pct", "net_rtg"]


def r_squared_by_games(logs: pd.DataFrame, max_n: int = 81, n_boot: int = 500,
                       seed: int = 0) -> pd.DataFrame:
    """R² of every first-N measure against every rest-of-season measure, N = 1..max_n.

    Bootstrap intervals resample team-seasons. Returns one row per
    (N, predictor, target) with ``r2``, ``lo`` and ``hi`` (2.5% / 97.5%).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(1, max_n + 1):
        split = split_at(logs, n)
        idx = rng.integers(0, len(split), size=(n_boot, len(split)))
        for pred in PREDICTORS:
            for target in PREDICTORS:
                x = split[f"{pred}_first"].to_numpy()
                y = split[f"{target}_rest"].to_numpy()
                boot = [r_squared(x[i], y[i]) for i in idx]
                rows.append({"n": n, "predictor": pred, "target": target,
                             "r2": r_squared(x, y), "lo": np.quantile(boot, 0.025),
                             "hi": np.quantile(boot, 0.975), "team_seasons": len(split)})
    return pd.DataFrame(rows)


def first_n_above(curves: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Smallest N at which R² exceeds ``threshold`` for each predictor/target pair."""
    above = curves[curves["r2"] > threshold]
    return (above.groupby(["predictor", "target"])["n"].min()
            .reindex(pd.MultiIndex.from_product([PREDICTORS, PREDICTORS],
                                                names=["predictor", "target"]))
            .rename("first_n").reset_index())
