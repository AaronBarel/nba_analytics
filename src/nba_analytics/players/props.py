"""Per-game player projections (P4): points, rebounds, assists, steals, blocks and threes.

A stat's expected value is expected minutes times an expected per-minute rate,
each from its own boosted-tree model; a negative binomial around that mean
gives the probability of any line. Everything is known before tip-off, and
availability is taken as known (in live use, from the injury report), so the
models are fit and scored on games the player played.

Inputs per player and game:

* Pre-season projections (P2): minutes per game, per-100 rates, usage. Projected
  3-pointers made are projected attempts times the player's 3P% over the last
  three seasons, shrunk toward the league rate.
* Season to date and last five games: minutes, per-minute rates, starts.
* Teammates: expected minutes and usage of absent teammates.
* Opponent: what it allows per possession for each stat, relative to the
  league, season to date blended with last season.
* Game: expected pace, the team's Elo edge (blowout risk), home, rest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.ensemble import HistGradientBoostingRegressor

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import player_season_table
from nba_analytics.players.projections import player_seasons
from nba_analytics.teams import roster
from nba_analytics.teams.games import ELO_PARAMS, blended_counts, game_features, team_games
from nba_analytics.teams.net_rating import game_logs

STATS = {"pts": "Points", "trb": "Rebounds", "ast": "Assists", "stl": "Steals",
         "blk": "Blocks", "fg3": "3-pointers"}
RECENT_GAMES = 5
# Minutes of the pre-season rate blended into the season-to-date rate (the
# season projection's regression minutes, which were tuned on 2009-2013)
PRIOR_MINUTES = {"pts": 250, "trb": 100, "ast": 100, "stl": 500, "blk": 250, "fg3": 250}
# Opponent counts that measure what a defense allows for each stat
ALLOWED = {"pts": "opp_pts", "trb": "opp_trb", "ast": "opp_ast", "stl": "tov",
           "blk": "opp_blk", "fg3": "opp_fg3"}
LINES = {"pts": [10, 15, 20, 25, 30], "trb": [4, 6, 8, 10, 12], "ast": [2, 4, 6, 8, 10],
         "stl": [1, 2, 3], "blk": [1, 2, 3], "fg3": [1, 2, 3, 4, 5]}
# Season projections have 3PA but not 3PM: projected makes use the player's 3P% over
# the last FG3_SEASONS seasons with FG3_PRIOR_ATTEMPTS league-rate attempts added
# (best next-season log loss on 2012-2026); players with no history get FG3_PCT
FG3_PCT = 0.36
FG3_SEASONS = 3
FG3_PRIOR_ATTEMPTS = 300
FG3_PRIOR = (FG3_SEASONS, FG3_PRIOR_ATTEMPTS)  # stored with fitted models to spot stale caches


# --- Features ---------------------------------------------------------------

def shrunk_fg3_pct(totals: pd.DataFrame, seasons) -> pd.DataFrame:
    """Each player's pre-season 3P% for each of ``seasons``: makes and attempts over the
    previous FG3_SEASONS seasons plus FG3_PRIOR_ATTEMPTS at last season's league rate.

    ``totals`` has one row per player and season (``fg3``, ``fg3a``).
    """
    out = []
    for season in seasons:
        last = totals[totals["season"] == season - 1]
        league = last["fg3"].sum() / last["fg3a"].sum()
        past = totals[totals["season"].between(season - FG3_SEASONS, season - 1)]
        g = past.groupby("player_id")[["fg3", "fg3a"]].sum()
        pct = (g["fg3"] + FG3_PRIOR_ATTEMPTS * league) / (g["fg3a"] + FG3_PRIOR_ATTEMPTS)
        out.append(pct.rename("fg3_pct_prior").reset_index().assign(season=season))
    return pd.concat(out, ignore_index=True)[["season", "player_id", "fg3_pct_prior"]]


def fg3_pct_prior(seasons) -> pd.DataFrame:
    """``shrunk_fg3_pct`` from Basketball-Reference season totals (traded players' combined
    row, which comes first)."""
    seasons = sorted(set(seasons))
    years = range(seasons[0] - FG3_SEASONS, seasons[-1])
    totals = pd.concat([player_season_table(y, "totals").drop_duplicates("player_id")
                        [["season", "player_id", "fg3", "fg3a"]] for y in years],
                       ignore_index=True)
    for c in ["fg3", "fg3a"]:
        totals[c] = pd.to_numeric(totals[c], errors="coerce").fillna(0.0)
    return shrunk_fg3_pct(totals, seasons)


def _player_history(df: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date and last-five-game minutes, starts and per-minute rates.

    Every row gets the values from games before it, including rows for games
    the player missed or that haven't been played yet.
    """
    df = df.sort_values(["season", "player_id", "date"]).reset_index(drop=True)
    played = df["mp"] > 0
    g = df.groupby(["season", "player_id"])
    keys = [df["season"], df["player_id"]]
    out = pd.DataFrame(index=df.index)
    out["games_td"] = g["mp"].transform(lambda s: (s > 0).cumsum().shift(1, fill_value=0))
    mp = df["mp"].fillna(0.0)
    out["mp_td"] = mp.groupby(keys).cumsum() - mp
    out["mpg_td"] = out["mp_td"] / out["games_td"].replace(0, np.nan)
    out["start_share_td"] = (g["started"].cumsum() - df["started"]) / \
        out["games_td"].replace(0, np.nan)

    # last five games played: totals through each played game, carried to later rows
    p = df[played]
    pg = p.groupby(["season", "player_id"])
    through = pg[["mp", "started", *STATS]].transform(
        lambda s: s.rolling(RECENT_GAMES, min_periods=1).sum())
    through["n"] = (pg.cumcount() + 1).clip(upper=RECENT_GAMES)
    roll = through.reindex(df.index).groupby(keys).shift(1).groupby(keys).ffill()
    n_recent = roll["n"]
    out["mpg_recent"] = roll["mp"] / n_recent
    out["start_share_recent"] = roll["started"] / n_recent
    for s in STATS:
        out[f"{s}_rate_recent"] = roll[s] / roll["mp"]
    for s in STATS:
        out[f"{s}_td"] = g[s].transform(lambda x: x.fillna(0).cumsum()) - df[s].fillna(0)
        out[f"{s}_rate_td"] = out[f"{s}_td"] / out["mp_td"].replace(0, np.nan)
    return pd.concat([df, out], axis=1)


def _teammates(df: pd.DataFrame) -> pd.DataFrame:
    """Per team and game: expected minutes and usage-weighted minutes of absent players."""
    absent = ~df["available"]
    usg = df["usg_pct"].fillna(18.0)  # league-average usage for unprojected players
    t = df.assign(missing_mp=df["expected_mp"].where(absent, 0.0),
                  missing_usg_mp=(df["expected_mp"] * usg).where(absent, 0.0),
                  avail_mp=df["expected_mp"].where(~absent, 0.0))
    return (t.groupby(["season", "team_id", "date"], as_index=False)
            [["missing_mp", "missing_usg_mp", "avail_mp"]].sum())


def _opponent(team_logs: pd.DataFrame) -> pd.DataFrame:
    """What each team allows per possession for each stat, relative to the league average."""
    logs = team_logs.assign(opp_trb=team_logs["opp_orb"] + team_logs["opp_drb"])
    counts = sorted(set(ALLOWED.values()) | {"poss"})
    b = blended_counts(logs, counts)
    per_poss = b[counts].div(b["poss"], axis=0)
    league = per_poss.groupby(b["season"]).transform("mean")
    out = b[["season", "team_id", "date"]].copy()
    for s, c in ALLOWED.items():
        out[f"opp_allows_{s}"] = per_poss[c] / league[c]
    out["pace"] = b["poss"]
    return out


def player_game_rows(gl: pd.DataFrame, values: pd.DataFrame, team_logs: pd.DataFrame,
                     fg3_pct: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per player and game on a roster, with every pre-game feature and the outcomes.

    ``fg3_pct`` is ``fg3_pct_prior`` for the seasons in ``gl``, loaded when not given.
    """
    df = roster.expected_minutes(gl, values)
    df = _player_history(df)
    df = df.merge(_teammates(df), on=["season", "team_id", "date"], how="left")
    own = df["expected_mp"].where(~df["available"], 0.0)
    df["teammate_avail_mp"] = df["avail_mp"] - own  # other available players' expected minutes
    df["minutes_rank"] = df[df["available"]].groupby(["season", "team_id", "date"])[
        "expected_mp"].rank(ascending=False)

    games = game_features(team_logs, ELO_PARAMS)
    tg = team_games(games).merge(
        games[["game_id", "elo_diff", "home_court", "home_rest_days", "away_rest_days",
               "home_back_to_back", "away_back_to_back"]], on="game_id")
    home = tg["is_home"]
    tg["elo_edge"] = np.where(home, tg["elo_diff"], -tg["elo_diff"]) \
        + np.where(home, 1, -1) * ELO_PARAMS["home_adv"] * tg["home_court"]
    tg["rest_days"] = np.where(home, tg["home_rest_days"], tg["away_rest_days"])
    tg["back_to_back"] = np.where(home, tg["home_back_to_back"], tg["away_back_to_back"])
    tg["is_home"] = home.astype(int)
    df = df.merge(tg[["season", "date", "team_id", "opp_id", "is_home", "elo_edge",
                      "rest_days", "back_to_back"]], on=["season", "date", "team_id"],
                  how="left")
    opp = _opponent(team_logs)
    df = df.merge(opp.drop(columns="pace").rename(columns={"team_id": "opp_id"}),
                  on=["season", "date", "opp_id"], how="left")
    pace = opp[["season", "team_id", "date", "pace"]]
    df = df.merge(pace.rename(columns={"pace": "team_pace"}),
                  on=["season", "team_id", "date"], how="left")
    df = df.merge(pace.rename(columns={"team_id": "opp_id", "pace": "opp_pace"}),
                  on=["season", "opp_id", "date"], how="left")
    df["game_pace"] = (df["team_pace"] + df["opp_pace"]) / 2

    # pre-season per-minute rates, then blended with the season to date
    if fg3_pct is None:
        fg3_pct = fg3_pct_prior(df["season"].unique())
    df = df.merge(fg3_pct, on=["season", "player_id"], how="left")
    df["fg3_pct_prior"] = df["fg3_pct_prior"].fillna(FG3_PCT)
    for s in STATS:
        per_100 = df["fg3a_per_100"] * df["fg3_pct_prior"] if s == "fg3" else df[f"{s}_per_100"]
        proj = per_100 * df["team_pace"] / 100 / 48
        df[f"{s}_rate_proj"] = proj
        # players without a projection start from the median projected player
        prior = proj.fillna(proj.groupby(df["season"]).transform("median"))
        df[f"{s}_rate_blend"] = ((df[f"{s}_td"] + PRIOR_MINUTES[s] * prior)
                                 / (df["mp_td"] + PRIOR_MINUTES[s]))
    return df


MINUTES_FEATURES = ["expected_mp", "mpg", "mpg_td", "mpg_recent", "games_td",
                    "start_share_td", "start_share_recent", "missing_mp", "teammate_avail_mp",
                    "minutes_rank", "elo_edge", "is_home", "rest_days", "back_to_back",
                    "projected"]


def rate_features(stat: str) -> list[str]:
    own = [f"{stat}_rate_blend", f"{stat}_rate_td", f"{stat}_rate_recent", f"{stat}_rate_proj"]
    return own + ["usg_pct", "mp_td", "expected_mp", f"opp_allows_{stat}", "game_pace",
                  "missing_usg_mp", "missing_mp", "elo_edge", "is_home", "back_to_back"]


# Kept with the predictions for the situation charts
CONTEXT = ["missing_usg_mp", "opp_allows_pts", "opp_allows_trb", "opp_allows_ast",
           "back_to_back", "elo_edge", "is_home", "game_pace"]


def props_dataset(first: int, last: int, refresh: bool = False) -> pd.DataFrame:
    """Player-game rows for ``first``..``last``, cached to ``data/processed``."""
    cache = PROCESSED_DIR / f"player_games_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        # else built before a stat or the 3P% prior was added
        if all(f"{s}_rate_blend" in df for s in STATS) and "fg3_pct_prior" in df:
            return df
    df = player_game_rows(roster.gamelogs(first, last), roster.player_values(first, last),
                          game_logs(range(2001, last + 1)))
    names = player_seasons()[["player_id", "season", "player"]].drop_duplicates(
        ["player_id", "season"])
    df = df.merge(names, on=["player_id", "season"], how="left")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


# --- Models -----------------------------------------------------------------

def _trees(loss: str = "squared_error") -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss=loss, max_iter=300, learning_rate=0.05,
                                         max_depth=5, min_samples_leaf=200,
                                         l2_regularization=5.0, random_state=0)


def fit_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Negative binomial size ``r`` (Var = mu + mu^2 / r) by maximum likelihood."""
    mu = np.clip(mu, 1e-3, None)

    def nll(log_r):
        r = np.exp(log_r)
        return -np.sum(special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1)
                       + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))

    return float(np.exp(optimize.minimize_scalar(nll, bounds=(-3, 8), method="bounded").x))


MINUTE_BINS = [0, 16, 24, 30, 36, 48]  # dispersion is fit separately in each bin


def minute_bin(mp_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(mp_pred, MINUTE_BINS[1:-1]), 0, len(MINUTE_BINS) - 2)


def prob_over(mu: np.ndarray, r: float | np.ndarray, line: float) -> np.ndarray:
    """P(stat >= line) for a negative binomial with mean ``mu`` and size ``r``."""
    mu = np.clip(mu, 1e-6, None)
    return stats.nbinom.sf(np.ceil(line) - 1, r, r / (r + mu))


def fit(train: pd.DataFrame) -> dict:
    """Fit the minutes model, one rate model per stat, and negative binomial sizes.

    ``train`` holds played rows. Sizes are fit per stat and predicted-minutes
    bin: bench players' totals swing more with their minutes.
    """
    minutes = _trees().fit(train[MINUTES_FEATURES], train["mp"])
    tr_mp = minutes.predict(train[MINUTES_FEATURES]).clip(1, 48)
    models = {"minutes": minutes, "rates": {}, "sizes": {}, "fg3_prior": FG3_PRIOR}
    for s in STATS:
        f = rate_features(s)
        rows = train[s].notna()
        rate = _trees("poisson").fit(train.loc[rows, f], train.loc[rows, s] / train.loc[rows, "mp"],
                                     sample_weight=train.loc[rows, "mp"])
        mu_tr = tr_mp[rows.to_numpy()] * rate.predict(train.loc[rows, f])
        bins_tr, y_tr = minute_bin(tr_mp[rows.to_numpy()]), train.loc[rows, s].to_numpy()
        models["rates"][s] = rate
        models["sizes"][s] = np.array([fit_dispersion(y_tr[bins_tr == b], mu_tr[bins_tr == b])
                                       for b in range(len(MINUTE_BINS) - 1)])
    return models


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Minutes, rate, mean and negative binomial size for each stat, plus two baselines."""
    keys = ["player_id", "player", "season", "date", "team_id", "opp_id"]
    out = rows[[c for c in keys if c in rows] + ["mp", *STATS, *CONTEXT]].copy()
    out["mp_pred"] = models["minutes"].predict(rows[MINUTES_FEATURES]).clip(1, 48)
    for s in STATS:
        out[f"{s}_rate_pred"] = models["rates"][s].predict(rows[rate_features(s)])
        out[f"{s}_pred"] = out["mp_pred"] * out[f"{s}_rate_pred"]
        out[f"{s}_baseline"] = rows["expected_mp"] * rows[f"{s}_rate_blend"]
        out[f"{s}_avg"] = (rows[f"{s}_td"] / rows["games_td"].replace(0, np.nan)).fillna(
            out[f"{s}_baseline"])
        out[f"{s}_size"] = models["sizes"][s][minute_bin(out["mp_pred"].to_numpy())]
    return out


def fit_predict(df: pd.DataFrame, train: pd.Series, test: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Fit on ``train`` played rows and predict ``test`` played rows.

    Returns the predictions and the fitted sizes by stat.
    """
    played = df["mp"] > 0
    models = fit(df[train & played])
    return predict(models, df[test & played]), models["sizes"]


def walk_forward(df: pd.DataFrame, first_test: int, last_test: int,
                 first_train: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each test season from models fit on every earlier season."""
    first_train = first_train or int(df["season"].min())
    parts, disp = [], []
    for s in range(first_test, last_test + 1):
        train = df["season"].between(first_train, s - 1)
        out, d = fit_predict(df, train, df["season"] == s)
        parts.append(out)
        for stat, sizes in d.items():
            disp += [{"season": s, "stat": stat, "minutes": f"{lo}-{hi}", "size": r}
                     for lo, hi, r in zip(MINUTE_BINS[:-1], MINUTE_BINS[1:], sizes)]
    return pd.concat(parts, ignore_index=True), pd.DataFrame(disp)


# --- Evaluation -------------------------------------------------------------

METHODS = {"avg": "Season average so far", "baseline": "Blended rate x expected minutes",
           "pred": "Model"}


def evaluate(preds: pd.DataFrame, by_season: bool = False) -> pd.DataFrame:
    """MAE of each method for minutes and each stat."""
    groups = preds["season"] if by_season else pd.Series(0, index=preds.index)
    rows = []
    for s, label in {"mp": "Minutes", **STATS}.items():
        cols = {"pred": "mp_pred"} if s == "mp" else {m: f"{s}_{m}" for m in METHODS}
        for m, col in cols.items():
            err = (preds[col] - preds[s]).abs()
            for g, v in err.groupby(groups).mean().items():
                rows.append({"stat": label, "method": METHODS[m], "season": g, "mae": v})
    out = pd.DataFrame(rows)
    return out if by_season else out.drop(columns="season")


def over_calibration(preds: pd.DataFrame, stat: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual rate of reaching each line in ``LINES``, pooled, by bin."""
    parts = []
    for line in LINES[stat]:
        p = prob_over(preds[f"{stat}_pred"].to_numpy(), preds[f"{stat}_size"].to_numpy(), line)
        parts.append(pd.DataFrame({"p": p, "hit": (preds[stat] >= line).astype(int)}))
    both = pd.concat(parts, ignore_index=True)
    b = pd.cut(both["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (both.groupby(b, observed=True)
            .agg(predicted=("p", "mean"), actual=("hit", "mean"), n=("hit", "size"))
            .reset_index(drop=True))


def interval_coverage(preds: pd.DataFrame, level: float = 0.8) -> pd.DataFrame:
    """Share of games inside the central ``level`` negative binomial interval, by stat."""
    rows = []
    for s, label in STATS.items():
        r = preds[f"{s}_size"].to_numpy()
        mu = np.clip(preds[f"{s}_pred"].to_numpy(), 1e-6, None)
        lo = stats.nbinom.ppf((1 - level) / 2, r, r / (r + mu))
        hi = stats.nbinom.ppf(1 - (1 - level) / 2, r, r / (r + mu))
        y = preds[s].to_numpy()
        rows.append({"stat": label, "coverage": np.mean((y >= lo) & (y <= hi))})
    return pd.DataFrame(rows)
