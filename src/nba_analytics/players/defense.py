"""Defensive value (P3) and the hidden-defender screen.

The defensive index averages three families of z-scores, each computed
within position group (G, F, C) so guards are compared with guards:

- Box score: steals, blocks and defensive rebounds per 100 possessions, plus
  charges drawn per 36 minutes
- Advanced: DBPM, defensive win shares per 48 minutes, and (negated) DRtg
- On/off: (negated) change in team defensive rating with the player on the
  court vs. off it, minutes-weighted across teams for traded players

Hidden defenders are below-average offensive players for their position group
(OBPM and TS% relative to the group) who still play heavy minutes. Players in
their first four NBA seasons are flagged as likely rookie-scale development
minutes.
"""

from __future__ import annotations

import pandas as pd

from nba_analytics.config import POS_TO_GROUP
from nba_analytics.data.bbref import player_season_table, team_on_off, team_season_table
from nba_analytics.players.usage_efficiency import season_rows

HISTORY_START = 2001  # first season of per-game tables used for career length
MIN_GAMES = 20
MIN_MINUTES = 500
HEAVY_MINUTES_QUANTILE = 0.75
ROOKIE_SCALE_SEASONS = 4

FAMILIES = {
    "box_score": ["z_stl_per_100", "z_blk_per_100", "z_drb_per_100", "z_charges_per_36"],
    "advanced": ["z_dbpm", "z_dws_per_48", "z_neg_def_rtg"],
    "on_off": ["z_neg_drtg_on_off"],
}


def career_start(last_season: int) -> pd.Series:
    """First season each player appears in the per-game tables (from 2000-01)."""
    seen = pd.concat([player_season_table(s, "per_game")[["season", "player_id"]]
                      for s in range(HISTORY_START, last_season + 1)])
    return seen.groupby("player_id")["season"].min().rename("first_season")


def on_off_drtg(on_off: pd.DataFrame) -> pd.DataFrame:
    """Team defensive rating with the player on minus off, per player-season.

    Traded players get a minutes-weighted average of their per-team splits.
    """
    split = on_off.pivot_table(index=["season", "team_id", "player_id"], columns="split_id",
                               values=["opp_off_rtg", "mp"], aggfunc="first")
    per_team = pd.DataFrame({
        "on_court_mp": split[("mp", "On Court")],
        "drtg_on_off": split[("opp_off_rtg", "On Court")] - split[("opp_off_rtg", "Off Court")],
    }).dropna().reset_index()
    per_team["weighted"] = per_team["drtg_on_off"] * per_team["on_court_mp"]
    g = per_team.groupby(["season", "player_id"])
    return pd.DataFrame({"drtg_on_off": g["weighted"].sum() / g["on_court_mp"].sum()}
                        ).reset_index()


def _zscore_within(df: pd.DataFrame, col: str, by: str = "pos_group") -> pd.Series:
    g = df.groupby(by)[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def build_defense(per_poss: pd.DataFrame, advanced: pd.DataFrame, per_game: pd.DataFrame,
                  pbp: pd.DataFrame, on_off: pd.DataFrame, first_season: pd.Series,
                  min_games: int = MIN_GAMES, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    """One row per qualifying player-season with component scores and flags."""
    adv = season_rows(advanced)[["season", "player_id", "name_display", "age", "team", "pos",
                                 "games", "mp", "ts_pct", "usg_pct", "obpm", "dbpm", "dws"]]
    poss = season_rows(per_poss)[["player_id", "stl_per_poss", "blk_per_poss", "drb_per_poss",
                                  "def_rtg"]]
    pg = season_rows(per_game)[["player_id", "mp_per_g"]]
    charges = season_rows(pbp)[["player_id", "drawn_offensive"]]

    df = (adv.merge(poss, on="player_id", how="left")
             .merge(pg, on="player_id", how="left")
             .merge(charges, on="player_id", how="left")
             .merge(on_off_drtg(on_off).drop(columns="season"), on="player_id", how="left"))
    df = df[(df["games"] >= min_games) & (df["mp"] >= min_minutes)
            & df["pos"].isin(POS_TO_GROUP)].copy()
    df["pos_group"] = df["pos"].map(POS_TO_GROUP)

    df = df.rename(columns={"stl_per_poss": "stl_per_100", "blk_per_poss": "blk_per_100",
                            "drb_per_poss": "drb_per_100", "name_display": "player"})
    df["charges_per_36"] = df["drawn_offensive"].fillna(0) / df["mp"] * 36
    df["dws_per_48"] = df["dws"] / df["mp"] * 48
    df["neg_def_rtg"] = -df["def_rtg"]
    df["neg_drtg_on_off"] = -df["drtg_on_off"]
    df["ts_pct"] = df["ts_pct"] * 100
    df["ts_diff_group"] = df["ts_pct"] - df.groupby("pos_group")["ts_pct"].transform("mean")

    for col in ["stl_per_100", "blk_per_100", "drb_per_100", "charges_per_36", "dbpm",
                "dws_per_48", "neg_def_rtg", "neg_drtg_on_off", "obpm", "ts_diff_group"]:
        df[f"z_{col}"] = _zscore_within(df, col)
    for family, cols in FAMILIES.items():
        df[family] = df[cols].mean(axis=1)
    df["defense_index"] = df[list(FAMILIES)].mean(axis=1)
    df["defense_pctile"] = df.groupby("pos_group")["defense_index"].rank(pct=True) * 100
    df["offense_score"] = df[["z_obpm", "z_ts_diff_group"]].mean(axis=1)

    df["career_season"] = df["season"] - df["player_id"].map(first_season) + 1
    df["rookie_scale"] = df["career_season"] <= ROOKIE_SCALE_SEASONS
    heavy = df.groupby("pos_group")["mp_per_g"].transform(
        lambda s: s.quantile(HEAVY_MINUTES_QUANTILE))
    df["heavy_minutes"] = df["mp_per_g"] >= heavy
    df["hidden_defender"] = (df["offense_score"] < 0) & df["heavy_minutes"]
    return df.reset_index(drop=True)


def defense_frame(season: int, **kwargs) -> pd.DataFrame:
    """Defensive scores for every qualifying player in a season (end year)."""
    on_off = pd.concat([team_on_off(t, season) for t in team_season_table(season)["team_id"]],
                       ignore_index=True)
    return build_defense(player_season_table(season, "per_poss"),
                         player_season_table(season, "advanced"),
                         player_season_table(season, "per_game"),
                         player_season_table(season, "play-by-play"),
                         on_off, career_start(season), **kwargs)


DISPLAY_COLUMNS = {
    "player": "Player", "team": "Team", "pos": "Pos", "age": "Age",
    "career_season": "NBA season #", "mp_per_g": "MPG", "defense_index": "Defense index",
    "defense_pctile": "Group pctile", "box_score": "Box score", "advanced": "Advanced",
    "on_off": "On/off", "drtg_on_off": "DRtg on-off", "offense_score": "Offense score",
    "rookie_scale": "Rookie scale",
}


def top_defenders(df: pd.DataFrame, group: str, n: int = 15) -> pd.DataFrame:
    """Highest defensive index in a position group, formatted for display."""
    out = df[df["pos_group"] == group].nlargest(n, "defense_index")
    return out[list(DISPLAY_COLUMNS)].rename(columns=DISPLAY_COLUMNS).round(2)


def hidden_defenders(df: pd.DataFrame) -> pd.DataFrame:
    """Below-average offense, heavy minutes; sorted by minutes per game."""
    out = df[df["hidden_defender"]].sort_values(["pos_group", "mp_per_g"],
                                                ascending=[True, False])
    cols = ["pos_group", *DISPLAY_COLUMNS]
    return out[cols].rename(columns={"pos_group": "Group", **DISPLAY_COLUMNS}).round(2)
