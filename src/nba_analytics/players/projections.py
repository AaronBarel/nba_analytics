"""Next-season player projections (P2).

Baseline: a Marcel-style projection. Each rate stat is a weighted average of
the last three seasons (weights 5/4/3, times minutes), blended with a
stat-specific number of minutes (``REGRESS_MINUTES``) of the average player, plus an age adjustment from the
delta method (average year-over-year change at each age). The average is
for the player's position group (G, F, C), so a center's rebounding regresses
toward other centers rather than toward guards. Playing time
follows Marcel's 0.5 / 0.1 / constant rule for share of team games, and
minutes per game are a games-weighted 5/4/3 average.

Model: gradient boosted trees fit to the baseline's residuals, using three
seasons of per-100, advanced and per-game stats, age, career length, prior
awards and changes in role. Evaluation is walk-forward: each season is
projected by models fit only on earlier seasons.

Rookies are projected by the college model (``players.rookies``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from nba_analytics.config import POS_TO_GROUP, PROCESSED_DIR
from nba_analytics.data.bbref import player_season_table, team_season_table
from nba_analytics.players.defense import career_start
from nba_analytics.players.usage_efficiency import _is_combined, season_rows

HISTORY_START = 2005  # first season with per-100 and advanced history loaded
FIRST_TARGET = HISTORY_START + 3
LAGS = (1, 2, 3)
WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6}  # Marcel's 5/4/3
# Minutes of the position-group average blended into each rate stat: few for stable
# skills (assists, 3PA), more for noisy ones (TS%). Tuned on 2009-2013, before any test season.
REGRESS_MINUTES = {"pts_per_100": 250, "trb_per_100": 100, "ast_per_100": 100,
                   "stl_per_100": 500, "blk_per_100": 250, "tov_per_100": 500,
                   "fg3a_per_100": 100, "ts_pct": 1000, "usg_pct": 250, "bpm": 500}
AGE_RANGE = (20, 37)  # ages outside are clipped for the age curves
MIN_RATE_MINUTES = 250  # rate targets need this many minutes in the target season
EVAL_RATE_MINUTES = 500  # rate errors are scored on players with at least this many

PER_100 = ["pts", "trb", "orb", "drb", "ast", "stl", "blk", "tov", "fg3a", "fta", "pf"]
RATE_TARGETS = {
    "pts_per_100": "Points per 100", "trb_per_100": "Rebounds per 100",
    "ast_per_100": "Assists per 100", "stl_per_100": "Steals per 100",
    "blk_per_100": "Blocks per 100", "tov_per_100": "Turnovers per 100",
    "fg3a_per_100": "3PA per 100", "ts_pct": "TS%", "usg_pct": "Usage %", "bpm": "BPM",
}
TIME_TARGETS = {"games": "Games (82-game scale)", "mp_per_g": "Minutes per game"}
TARGETS = {**TIME_TARGETS, **RATE_TARGETS}

ADVANCED = ["ts_pct", "usg_pct", "bpm", "obpm", "dbpm", "ws_per_48", "ast_pct", "orb_pct",
            "drb_pct", "stl_pct", "blk_pct", "tov_pct", "fg3a_per_fga_pct", "fta_per_fga_pct"]
LAG_STATS = (["mp", "games_frac", "mp_per_g", "gs_frac", "n_teams", "ft_pct"]
             + [f"{s}_per_100" for s in PER_100] + ADVANCED + ["all_star", "all_nba", "all_def",
                                                              "mvp_votes", "dpoy_votes"])
AWARDS = {"AS": ("all_star", 1), "NBA1": ("all_nba", 3), "NBA2": ("all_nba", 2),
          "NBA3": ("all_nba", 1), "DEF1": ("all_def", 2), "DEF2": ("all_def", 1)}


def parse_awards(awards: pd.Series) -> pd.DataFrame:
    """'MVP-3,AS,NBA1,DEF2' -> All-Star flag, All-NBA and All-Defense levels, vote flags."""
    out = pd.DataFrame(0, index=awards.index,
                       columns=["all_star", "all_nba", "all_def", "mvp_votes", "dpoy_votes"])
    for idx, text in awards.dropna().items():
        for token in str(text).split(","):
            if token in AWARDS:
                col, value = AWARDS[token]
                out.at[idx, col] = value
            elif token.startswith("MVP-"):
                out.at[idx, "mvp_votes"] = 1
            elif token.startswith("DPOY-"):
                out.at[idx, "dpoy_votes"] = 1
    return out


def _season_frame(season: int) -> pd.DataFrame:
    adv = season_rows(player_season_table(season, "advanced"))
    poss = season_rows(player_season_table(season, "per_poss"))
    per_game = player_season_table(season, "per_game")
    pg = season_rows(per_game)
    n_teams = per_game[~_is_combined(per_game["team_name_abbr"])].groupby("player_id").size()
    team_games = team_season_table(season).set_index("team_id")
    team_games = team_games["wins"] + team_games["losses"]

    df = adv[["season", "player_id", "name_display", "age", "pos", "team", "games",
              "games_started", "mp", *ADVANCED]].rename(columns={"name_display": "player"})
    df = df.merge(poss[["player_id", "ft_pct", *[f"{s}_per_poss" for s in PER_100]]],
                  on="player_id", how="left")
    df = df.merge(pg[["player_id", "mp_per_g"]], on="player_id", how="left")
    df = df.rename(columns={f"{s}_per_poss": f"{s}_per_100" for s in PER_100})
    df["ts_pct"] = df["ts_pct"] * 100
    df["n_teams"] = df["player_id"].map(n_teams).fillna(1)
    df["team_games"] = df["team"].map(team_games).fillna(team_games.max())
    df["games_frac"] = (df["games"] / df["team_games"]).clip(upper=1)
    df["gs_frac"] = df["games_started"] / df["games"]
    return df.join(parse_awards(pg.set_index("player_id")["awards"]), on="player_id")


def player_seasons(first: int = HISTORY_START, last: int = 2026,
                   refresh: bool = False) -> pd.DataFrame:
    """One row per player-season with per-100, advanced, playing-time and award columns.

    Cached to ``data/processed`` since parsing every season table takes a minute or two.
    """
    cache = PROCESSED_DIR / f"player_seasons_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = pd.concat([_season_frame(s) for s in range(first, last + 1)], ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def build_frame(seasons: pd.DataFrame, last: int) -> pd.DataFrame:
    """One row per (player, target season) with lagged features and targets.

    Target seasons run from ``FIRST_TARGET`` to ``last + 1``. A player is
    included if he played in any of the three prior seasons; for the
    unplayed season ``last + 1`` he must have played in ``last``.
    """
    keys = seasons[["player_id", "season"]]
    base = pd.concat([keys.assign(season=keys["season"] + k) for k in LAGS])
    base = base.drop_duplicates()
    base = base[(base["season"] >= FIRST_TARGET) & (base["season"] <= last + 1)]
    active = set(seasons.loc[seasons["season"] == last, "player_id"])
    base = base[(base["season"] <= last) | base["player_id"].isin(active)]

    indexed = seasons.set_index(["player_id", "season"])
    frame = base.reset_index(drop=True)
    for k in LAGS:
        lag = indexed[LAG_STATS + ["age", "player", "pos"]].add_suffix(f"_{k}")
        lag.index = lag.index.set_levels(lag.index.levels[1] + k, level=1)
        frame = frame.join(lag, on=["player_id", "season"])
    frame = frame.copy()  # defragment after the wide joins

    frame["player"] = frame["player_1"].fillna(frame["player_2"]).fillna(frame["player_3"])
    group = frame["pos_1"].fillna(frame["pos_2"]).fillna(frame["pos_3"]).map(POS_TO_GROUP)
    frame["pos_group"] = group
    frame["age"] = (frame["age_1"] + 1).fillna(frame["age_2"] + 2).fillna(frame["age_3"] + 3)
    frame = frame.drop(columns=[f"{c}_{k}" for c in ["player", "age", "pos"] for k in LAGS])
    debut = career_start(last).reindex(frame["player_id"]).to_numpy()
    frame["career_season"] = frame["season"] - debut + 1
    for col in ["all_star", "all_nba", "all_def", "mvp_votes"]:
        frame[f"{col}_3yr"] = frame[[f"{col}_{k}" for k in LAGS]].sum(axis=1)
    frame["usg_change"] = frame["usg_pct_1"] - frame["usg_pct_2"]
    frame["mpg_change"] = frame["mp_per_g_1"] - frame["mp_per_g_2"]

    targets = indexed[["mp", "games_frac", "mp_per_g", "team", "pos", *RATE_TARGETS]]
    frame = frame.join(targets, on=["player_id", "season"])
    frame["games"] = frame["games_frac"] * 82
    return frame


def age_curves(seasons: pd.DataFrame, before: int) -> pd.DataFrame:
    """Delta-method aging: mean change from one season to the next by age.

    Uses only pairs whose second season is earlier than ``before``. Changes are
    weighted by the harmonic mean of the two seasons' minutes and smoothed over
    neighboring ages.
    """
    cols = ["games_frac", "mp_per_g", *RATE_TARGETS]
    prev = seasons.assign(season=seasons["season"] + 1).set_index(["player_id", "season"])
    pairs = seasons.join(prev[cols + ["mp"]], on=["player_id", "season"], rsuffix="_prev")
    pairs = pairs[(pairs["season"] < before) & pairs["mp_prev"].notna()]
    w = 2 / (1 / pairs["mp"].clip(lower=1) + 1 / pairs["mp_prev"].clip(lower=1))
    age = pairs["age"].clip(*AGE_RANGE)
    curves = {}
    for c in cols:
        delta = pairs[c] - pairs[f"{c}_prev"]
        ok = delta.notna()
        weight = w[ok] if c in RATE_TARGETS else pd.Series(1.0, index=w[ok].index)
        by_age = (delta[ok] * weight).groupby(age[ok]).sum() / weight.groupby(age[ok]).sum()
        curves[c] = by_age.rolling(3, center=True, min_periods=1).mean()
    return pd.DataFrame(curves).reindex(range(AGE_RANGE[0], AGE_RANGE[1] + 1)).ffill().bfill()


def marcel(frame: pd.DataFrame, seasons: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """Baseline projection for every target."""
    group = seasons["pos"].map(POS_TO_GROUP).rename("pos_group")
    means = seasons.groupby(["season", group]).apply(
        lambda d: pd.Series({c: np.average(d[c].fillna(0), weights=d["mp"])
                             for c in RATE_TARGETS}), include_groups=False)
    prior = means.reindex(pd.MultiIndex.from_arrays([frame["season"] - 1, frame["pos_group"]]))
    prior = prior.set_index(frame.index)
    adj = curves.reindex(frame["age"].clip(*AGE_RANGE).round()).set_index(frame.index)

    out = pd.DataFrame(index=frame.index)
    for c in RATE_TARGETS:
        num = sum(WEIGHTS[k] * frame[f"mp_{k}"].fillna(0) * frame[f"{c}_{k}"].fillna(0)
                  for k in LAGS)
        den = sum(WEIGHTS[k] * frame[f"mp_{k}"].fillna(0) * frame[f"{c}_{k}"].notna()
                  for k in LAGS)
        r = REGRESS_MINUTES[c]
        out[c] = (num + r * prior[c]) / (den + r) + adj[c]

    gf = [frame[f"games_frac_{k}"].fillna(0) for k in LAGS]
    out["games"] = ((0.5 * gf[0] + 0.1 * gf[1] + 0.25 + adj["games_frac"]).clip(0, 1)) * 82
    g = [WEIGHTS[k] * frame[f"games_frac_{k}"].fillna(0) for k in LAGS]
    mpg = sum(gk * frame[f"mp_per_g_{k}"].fillna(0) for gk, k in zip(g, LAGS))
    out["mp_per_g"] = (mpg / sum(g).replace(0, np.nan) + adj["mp_per_g"]).clip(0, 48)
    return out[list(TARGETS)]


def last_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline: the most recent season played."""
    out = pd.DataFrame(index=frame.index)
    for c in TARGETS:
        src = "games_frac" if c == "games" else c
        vals = frame[f"{src}_1"].fillna(frame[f"{src}_2"]).fillna(frame[f"{src}_3"])
        out[c] = vals * 82 if c == "games" else vals
    return out


def features(frame: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    lagged = [f"{c}_{k}" for c in LAG_STATS for k in LAGS]
    extra = ["age", "career_season", "usg_change", "mpg_change",
             *[f"{c}_3yr" for c in ["all_star", "all_nba", "all_def", "mvp_votes"]]]
    return pd.concat([frame[lagged + extra], base.add_prefix("marcel_")], axis=1)


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_depth=4,
                                         min_samples_leaf=40, l2_regularization=5.0,
                                         random_state=0)


def _train_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    ok = frame[target].notna()
    if target in RATE_TARGETS:
        ok &= frame["mp"] >= MIN_RATE_MINUTES
    return ok


def _weights(frame: pd.DataFrame, target: str) -> pd.Series:
    if target in RATE_TARGETS:
        return frame["mp"].clip(upper=2500)
    return pd.Series(1.0, index=frame.index)


def fit_predict(frame: pd.DataFrame, seasons: pd.DataFrame, train_seasons: pd.Series,
                predict_rows: pd.Series) -> pd.DataFrame:
    """Fit baselines and residual models on ``train_seasons`` rows; predict ``predict_rows``.

    Returns last-season, Marcel and model projections for every target.
    """
    first_predicted = int(frame.loc[predict_rows, "season"].min())
    curves = age_curves(seasons, before=first_predicted)
    base = marcel(frame, seasons, curves)
    X = features(frame, base)
    out = {"last": last_season(frame.loc[predict_rows]), "marcel": base.loc[predict_rows]}
    model_preds = {}
    for t in TARGETS:
        rows = train_seasons & _train_mask(frame, t)
        fit = model().fit(X[rows], frame.loc[rows, t] - base.loc[rows, t],
                          sample_weight=_weights(frame[rows], t))
        model_preds[t] = base.loc[predict_rows, t] + fit.predict(X[predict_rows])
    out["model"] = pd.DataFrame(model_preds)
    out["model"]["games"] = out["model"]["games"].clip(0, 82)
    out["model"]["mp_per_g"] = out["model"]["mp_per_g"].clip(0, 48)
    return pd.concat(out, axis=1)


def walk_forward(frame: pd.DataFrame, seasons: pd.DataFrame, first_test: int,
                 last_test: int) -> pd.DataFrame:
    """Project each test season from models fit on all earlier seasons."""
    parts = []
    for s in range(first_test, last_test + 1):
        parts.append(fit_predict(frame, seasons, frame["season"] < s, frame["season"] == s))
    return pd.concat(parts)


def evaluate(frame: pd.DataFrame, preds: pd.DataFrame, by_season: bool = False) -> pd.DataFrame:
    """MAE of each method; rate stats minutes-weighted among 500+ minute players."""
    records = []
    for t, label in TARGETS.items():
        rows = frame.loc[preds.index]
        ok = rows[t].notna()
        if t in RATE_TARGETS:
            ok &= rows["mp"] >= EVAL_RATE_MINUTES
        w = _weights(rows, t)[ok]
        groups = rows.loc[ok, "season"] if by_season else pd.Series(0, index=w.index)
        for method in ["last", "marcel", "model"]:
            err = (preds.loc[ok, (method, t)] - rows.loc[ok, t]).abs()
            mae = (err * w).groupby(groups).sum() / w.groupby(groups).sum()
            for g, v in mae.items():
                records.append({"target": label, "method": method, "season": g, "mae": v,
                                "n": int((groups == g).sum())})
    out = pd.DataFrame(records)
    return out if by_season else out.drop(columns="season")


def residual_bands(frame: pd.DataFrame, preds: pd.DataFrame, target: str,
                   q: tuple[float, float] = (0.1, 0.9)) -> pd.DataFrame:
    """Walk-forward residual quantiles of the model, by prior-season minutes bucket."""
    rows = frame.loc[preds.index]
    ok = rows[target].notna()
    if target in RATE_TARGETS:
        ok &= rows["mp"] >= EVAL_RATE_MINUTES
    resid = rows.loc[ok, target] - preds.loc[ok, ("model", target)]
    bucket = minutes_bucket(rows.loc[ok, "mp_1"])
    return resid.groupby(bucket).quantile(list(q)).unstack()


def minutes_bucket(mp: pd.Series) -> pd.Series:
    return pd.cut(mp.fillna(0), [-1, 1000, 2000, np.inf],
                  labels=["Under 1,000", "1,000-2,000", "2,000+"])


def project(frame: pd.DataFrame, seasons: pd.DataFrame, season: int,
            walk_forward_preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Model projections for ``season`` (not yet played), fit on every earlier season.

    ``{target}_1`` columns hold the most recent season played. With walk-forward
    predictions, ``{target}_lo`` / ``_hi`` give an 80% band from past errors of
    players with similar prior-season minutes.
    """
    preds = fit_predict(frame, seasons, frame["season"] < season, frame["season"] == season)
    rows = frame.loc[preds.index, ["player_id", "player", "pos_group", "age", "career_season",
                                   "mp_1"]]
    out = (rows.join(preds["model"]).join(last_season(frame.loc[preds.index]).add_suffix("_1"))
           .join(preds["marcel"].add_prefix("marcel_")))
    if walk_forward_preds is not None:
        bucket = minutes_bucket(out["mp_1"])
        for t in TARGETS:
            bands = residual_bands(frame, walk_forward_preds, t)
            out[f"{t}_lo"] = out[t] + bucket.map(bands[0.1]).astype(float)
            out[f"{t}_hi"] = out[t] + bucket.map(bands[0.9]).astype(float)
    return out.reset_index(drop=True)
