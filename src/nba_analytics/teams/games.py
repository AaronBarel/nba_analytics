"""Game outcome model (T3): home win probability and point margin.

Every feature is known before tip-off. Team strength comes from an Elo rating
and from season-to-date box score rates blended with the previous season;
schedule features from ``schedule``; roster talent and availability from
``roster``. Models are fit walk-forward: each season is predicted by a model
trained only on earlier seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.teams import roster
from nba_analytics.teams.net_rating import franchise, game_logs
from nba_analytics.teams.schedule import schedule_features, venue

# --- Games ------------------------------------------------------------------


def game_table(logs: pd.DataFrame) -> pd.DataFrame:
    """One row per game from the home team's side.

    Neutral-site games (marked 'N') list the alphabetically first team as home
    and set ``neutral``; games in the 2020 bubble are neutral too.
    """
    home = logs["game_location"].isna()
    neutral = logs["game_location"].eq("N") & (logs["team_id"] < logs["opp_name_abbr"])
    g = logs[home | neutral].rename(columns={"team_id": "home", "opp_name_abbr": "away",
                                             "pts": "home_pts", "opp_pts": "away_pts"})
    g = g[["season", "date", "home", "away", "home_pts", "away_pts", "poss"]].copy()
    g["neutral"] = neutral[home | neutral].to_numpy()
    g["venue"] = [venue(d, h, n) for d, h, n in zip(g["date"], g["home"], g["neutral"])]
    g["neutral"] |= g["venue"].eq("Orlando bubble")
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["home_win"] = (g["margin"] > 0).astype(int)
    g = g.sort_values(["date", "home"]).reset_index(drop=True)
    g.insert(0, "game_id", g["date"].str.replace("-", "") + g["home"])
    return g


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team), carrying the game's venue."""
    sides = []
    for side, other in [("home", "away"), ("away", "home")]:
        s = games[["game_id", "season", "date", "venue", side, other]].rename(
            columns={side: "team_id", other: "opp_id"})
        s["is_home"] = side == "home"
        sides.append(s)
    return pd.concat(sides, ignore_index=True)


# --- Elo --------------------------------------------------------------------

ELO_MEAN = 1505
ELO_EXPANSION = 1300  # rating for a franchise first seen after the first season


def elo(games: pd.DataFrame, k: float = 20.0, home_adv: float = 100.0,
        carryover: float = 0.75) -> pd.DataFrame:
    """Pre-game Elo ratings with a margin-of-victory multiplier (FiveThirtyEight's NBA Elo).

    Between seasons each rating keeps ``carryover`` of its distance from the
    mean. Returns ``elo_home``, ``elo_away`` and ``elo_prob`` (home win).
    Scheduled games (no margin yet) get a prediction but move no ratings.
    """
    rating: dict[str, float] = {}
    first_season = games["season"].min()
    season = None
    home_f, away_f = franchise(games["home"]).to_numpy(), franchise(games["away"]).to_numpy()
    out = np.empty((len(games), 3))
    for i, (s, h, a, margin, neutral) in enumerate(zip(
            games["season"], home_f, away_f, games["margin"], games["neutral"])):
        if s != season:
            rating = {t: ELO_MEAN + carryover * (r - ELO_MEAN) for t, r in rating.items()}
            season = s
        start = ELO_MEAN if s == first_season else ELO_EXPANSION
        rh, ra = rating.setdefault(h, start), rating.setdefault(a, start)
        diff = rh - ra + (0 if neutral else home_adv)
        prob = 1 / (1 + 10 ** (-diff / 400))
        out[i] = rh, ra, prob
        if np.isnan(margin):
            continue
        win_diff = diff if margin > 0 else -diff  # rating edge of the winner
        mult = (abs(margin) + 3) ** 0.8 / (7.5 + 0.006 * win_diff)
        shift = k * mult * ((margin > 0) - prob)
        rating[h], rating[a] = rh + shift, ra - shift
    return pd.DataFrame(out, columns=["elo_home", "elo_away", "elo_prob"], index=games.index)


def tune_elo(games: pd.DataFrame, last: int, grid: dict[str, list[float]]) -> pd.DataFrame:
    """Log loss of Elo win probabilities on seasons up to ``last`` for each setting."""
    rows = []
    scored = games["season"] > games["season"].min()  # the first season is burn-in
    scored &= games["season"] <= last
    for k in grid["k"]:
        for h in grid["home_adv"]:
            for c in grid["carryover"]:
                p = elo(games, k, h, c)["elo_prob"]
                rows.append({"k": k, "home_adv": h, "carryover": c,
                             "log_loss": log_loss(games.loc[scored, "home_win"], p[scored])})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


# --- Team strength ----------------------------------------------------------

COUNTS = ["pts", "opp_pts", "poss", "fg", "fga", "fg3", "fta", "ft", "orb", "drb", "tov",
          "opp_fg", "opp_fga", "opp_fg3", "opp_fta", "opp_ft", "opp_orb", "opp_drb", "opp_tov"]
PRIOR_GAMES = 15  # weight of last season's rates, in games, early in a season
PRIOR_REGRESS = 1 / 3  # last season is regressed this far toward the league average
FORM_GAMES = 10
STRENGTH = ["net_rtg", "off_rtg", "def_rtg", "pace", "efg", "tov_pct", "orb_pct", "ft_rate",
            "opp_efg", "opp_tov_pct", "drb_pct", "opp_ft_rate", "form_net_rtg"]


def _rates(c: pd.DataFrame) -> pd.DataFrame:
    """Ratings and Four Factors from (per-game or summed) counting stats."""
    return pd.DataFrame({
        "off_rtg": 100 * c["pts"] / c["poss"],
        "def_rtg": 100 * c["opp_pts"] / c["poss"],
        "net_rtg": 100 * (c["pts"] - c["opp_pts"]) / c["poss"],
        "pace": c["poss"],
        "efg": (c["fg"] + 0.5 * c["fg3"]) / c["fga"],
        "tov_pct": c["tov"] / (c["fga"] + 0.44 * c["fta"] + c["tov"]),
        "orb_pct": c["orb"] / (c["orb"] + c["opp_drb"]),
        "ft_rate": c["ft"] / c["fga"],
        "opp_efg": (c["opp_fg"] + 0.5 * c["opp_fg3"]) / c["opp_fga"],
        "opp_tov_pct": c["opp_tov"] / (c["opp_fga"] + 0.44 * c["opp_fta"] + c["opp_tov"]),
        "drb_pct": c["drb"] / (c["drb"] + c["opp_orb"]),
        "opp_ft_rate": c["opp_ft"] / c["opp_fga"],
    }, index=c.index)


def blended_counts(logs: pd.DataFrame, counts: list[str]) -> pd.DataFrame:
    """Each team's per-game ``counts`` before each game, blended with last season.

    Season-to-date per-game counts are averaged with ``PRIOR_GAMES`` games at
    last season's per-game counts (regressed toward the league), so early
    games lean on last season and later games on this one. A franchise's first
    season uses the league average as its prior. Scheduled games (no result
    yet) are kept but count for nothing. Rows follow ``logs`` sorted by
    season, team and date, with those keys and ``games_played`` included.
    """
    df = logs.sort_values(["season", "team_id", "date"]).reset_index(drop=True)
    keys = [df["season"], df["team_id"]]
    played = df["pts"].notna()
    filled = df[counts].fillna(0)
    cum = filled.groupby(keys).cumsum() - filled  # before this game
    n = played.groupby(keys).cumsum() - played

    league = df.groupby("season")[counts].mean()
    prior = df.groupby(["season", "franchise"])[counts].mean().reset_index()
    prior["season"] += 1
    prior = prior.set_index(["season", "franchise"])
    key = pd.MultiIndex.from_frame(df[["season", "franchise"]])
    team_prior = prior.reindex(key).to_numpy()
    league_prior = league.shift(1).reindex(df["season"]).to_numpy()
    league_prior = np.where(np.isnan(league_prior), league.reindex(df["season"]).to_numpy(),
                            league_prior)
    team_prior = np.where(np.isnan(team_prior), league_prior, team_prior)
    prior_counts = (1 - PRIOR_REGRESS) * team_prior + PRIOR_REGRESS * league_prior

    blended = (cum.to_numpy() + PRIOR_GAMES * prior_counts) / (n.to_numpy()[:, None]
                                                               + PRIOR_GAMES)
    out = pd.DataFrame(blended, columns=counts, index=df.index)
    return pd.concat([df[["season", "team_id", "date"]], out.assign(games_played=n)], axis=1)


def strength_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Each team's ratings and Four Factors before each game (see ``blended_counts``).

    ``form_net_rtg`` is the net rating over the previous ``FORM_GAMES`` games.
    """
    blended = blended_counts(logs, COUNTS)
    out = _rates(blended[COUNTS])
    df = logs.sort_values(["season", "team_id", "date"]).reset_index(drop=True)
    keys = [df["season"], df["team_id"]]
    played = df[df["pts"].notna()]
    through = played.groupby(["season", "team_id"])[["pts", "opp_pts", "poss"]].transform(
        lambda s: s.rolling(FORM_GAMES, min_periods=1).sum()).reindex(df.index)
    roll = through.groupby(keys).shift(1).groupby(keys).ffill()  # before this game
    form = 100 * (roll["pts"] - roll["opp_pts"]) / roll["poss"]
    out["form_net_rtg"] = form.fillna(out["net_rtg"])
    out["games_played"] = blended["games_played"]
    return pd.concat([blended[["season", "team_id", "date"]], out], axis=1)


# --- Dataset ----------------------------------------------------------------

SCHEDULE = ["rest_days", "back_to_back", "games_last_4d", "travel_km", "tz_shift"]


def game_features(logs: pd.DataFrame, elo_params: dict | None = None) -> pd.DataFrame:
    """Games with pre-game features for both teams and home-minus-away differences."""
    games = game_table(logs)
    games = games.join(elo(games, **(elo_params or {})))
    tg = schedule_features(team_games(games))
    tg = tg.merge(strength_features(logs), on=["season", "team_id", "date"], how="left")
    per_team = ["games_played", *STRENGTH, *SCHEDULE]
    for side in ["home", "away"]:
        part = tg[tg["team_id"] == tg["game_id"].map(games.set_index("game_id")[side])]
        part = part.set_index("game_id")[per_team].add_prefix(f"{side}_")
        games = games.join(part, on="game_id")
    games["altitude_m"] = tg.drop_duplicates("game_id").set_index("game_id")["altitude_m"] \
        .reindex(games["game_id"]).to_numpy()
    games["elo_diff"] = games["elo_home"] - games["elo_away"]
    for c in [*STRENGTH, "rest_days", "travel_km"]:
        games[f"diff_{c}"] = games[f"home_{c}"] - games[f"away_{c}"]
    games["home_court"] = (~games["neutral"]).astype(int)
    return games


ELO_PARAMS = {"k": 20.0, "home_adv": 85.0, "carryover": 0.6}  # tuned on 2001-02 to 2012-13
ROSTER = ["talent", "talent_full", "talent_missing", "n_available", "best_player_out",
          "returning_share"]


def game_dataset(first: int, last: int, refresh: bool = False) -> pd.DataFrame:
    """Games ``first``..``last`` with team strength, schedule and roster features.

    Elo and team rates are built from 2000-01 on so early seasons are warmed up.
    Cached to ``data/processed``.
    """
    cache = PROCESSED_DIR / f"games_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    games = game_rows(game_logs(range(2001, last + 1)), roster.gamelogs(first, last),
                      roster.player_values(first, last), first)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    games.to_parquet(cache)
    return games


def game_rows(logs: pd.DataFrame, gl: pd.DataFrame, values: pd.DataFrame, first: int
              ) -> pd.DataFrame:
    """Games from season ``first`` on with every feature.

    ``logs`` are team game logs (from earlier seasons too, to warm up Elo and
    team rates), ``gl`` player game rows and ``values`` pre-season player
    values for the seasons in ``gl``. Scheduled games may be included.
    """
    games = game_features(logs, ELO_PARAMS)
    games = games[games["season"] >= first].reset_index(drop=True)
    per_team = roster.team_talent(gl, values).merge(
        roster.returning_minutes(gl, first), on=["season", "team_id", "date"], how="left")
    for side in ["home", "away"]:
        part = per_team.rename(columns={"team_id": side, **{c: f"{side}_{c}" for c in ROSTER}})
        games = games.merge(part, on=["season", "date", side], how="left")
    for c in ["talent", "talent_missing", "returning_share"]:
        games[f"diff_{c}"] = games[f"home_{c}"] - games[f"away_{c}"]
    return games


# Feature families for the model comparison; each set adds one family
FAMILIES = {
    "Home court": ["home_court"],
    "Elo": ["elo_diff"],
    "Team ratings": [f"diff_{c}" for c in STRENGTH if c != "form_net_rtg"]
                    + ["home_games_played"],
    "Recent form": ["diff_form_net_rtg"],
    "Schedule & travel": [f"{side}_{c}" for c in SCHEDULE for side in ["home", "away"]]
                         + ["altitude_m"],
    "Roster talent": ["diff_talent", "home_talent", "away_talent"],
    "Availability": ["diff_talent_missing", "home_talent_missing", "away_talent_missing",
                     "home_best_player_out", "away_best_player_out"],
    "Roster turnover": ["diff_returning_share", "home_returning_share",
                        "away_returning_share"],
}
ALL_FEATURES = [f for cols in FAMILIES.values() for f in cols]


# --- Models -----------------------------------------------------------------

def _logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _boosted_classifier():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3,
                                          min_samples_leaf=100, l2_regularization=5.0,
                                          random_state=0)


def _ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _boosted_regressor():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3,
                                         min_samples_leaf=100, l2_regularization=5.0,
                                         random_state=0)


MODELS = {"Logistic regression": ("win", _logistic),
          "Boosted trees": ("win", _boosted_classifier),
          "Ridge (margin)": ("margin", _ridge),
          "Boosted trees (margin)": ("margin", _boosted_regressor)}


def fit(train: pd.DataFrame, features: list[str], models: dict = MODELS) -> dict:
    """Fit each model; margin models also keep their training residual SD."""
    fitted = {}
    for name, (kind, make) in models.items():
        if kind == "win":
            fitted[name] = (kind, make().fit(train[features], train["home_win"]), None)
        else:
            m = make().fit(train[features], train["margin"])
            fitted[name] = (kind, m, np.std(train["margin"] - m.predict(train[features])))
    return fitted


def predict(fitted: dict, test: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Home win probability from each model, plus ``<name> pred`` margins.

    Margin models turn the predicted margin into a win probability with a
    normal distribution whose spread is the training residual SD.
    """
    out = test[["game_id", "season", "home_win", "margin"]].copy()
    for name, (kind, m, sd) in fitted.items():
        if kind == "win":
            out[name] = m.predict_proba(test[features])[:, 1]
        else:
            pred = m.predict(test[features])
            out[f"{name} pred"] = pred
            out[name] = norm.cdf(pred / sd)
    return out


def walk_forward(df: pd.DataFrame, features: list[str], first_test: int, last_test: int,
                 first_train: int | None = None, models: dict = MODELS) -> pd.DataFrame:
    """Out-of-sample home win probability (and margin) for each test season."""
    first_train = first_train or int(df["season"].min())
    parts = []
    for s in range(first_test, last_test + 1):
        train = df[(df["season"] >= first_train) & (df["season"] < s)]
        parts.append(predict(fit(train, features, models), df[df["season"] == s], features))
    return pd.concat(parts, ignore_index=True)


def scores(preds: pd.DataFrame, prob_cols: list[str], by_season: bool = False
           ) -> pd.DataFrame:
    """Log loss, Brier score and accuracy for each probability column."""
    groups = preds["season"] if by_season else pd.Series(0, index=preds.index)
    rows = []
    for col in prob_cols:
        for g, part in preds.groupby(groups):
            y, p = part["home_win"], part[col].clip(1e-6, 1 - 1e-6)
            rows.append({"model": col, "season": g, "n": len(part),
                         "log_loss": log_loss(y, p, labels=[0, 1]),
                         "brier": brier_score_loss(y, p),
                         "accuracy": ((p > 0.5) == y).mean()})
    out = pd.DataFrame(rows)
    return out if by_season else out.drop(columns="season")


def calibration(preds: pd.DataFrame, col: str, bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual home win rate by probability bin."""
    b = pd.cut(preds[col], np.linspace(0, 1, bins + 1), include_lowest=True)
    return (preds.groupby(b, observed=True)
            .agg(predicted=(col, "mean"), actual=("home_win", "mean"), n=("home_win", "size"))
            .reset_index(drop=True))


def compare_feature_sets(df: pd.DataFrame, first_test: int, last_test: int,
                         model: str = "Logistic regression") -> pd.DataFrame:
    """Walk-forward scores as each feature family is added in ``FAMILIES`` order."""
    rows, features = [], []
    for name, cols in FAMILIES.items():
        features = features + cols
        preds = walk_forward(df, features, first_test, last_test,
                             models={model: MODELS[model]})
        rows.append(scores(preds, [model]).assign(features=f"+ {name}" if rows else name))
    return pd.concat(rows, ignore_index=True)


def family_importance(df: pd.DataFrame, first_test: int, last_test: int,
                      model: str = "Logistic regression") -> pd.DataFrame:
    """Rise in walk-forward log loss when each family is removed from the full model."""
    one = {model: MODELS[model]}
    base = scores(walk_forward(df, ALL_FEATURES, first_test, last_test, models=one), [model])
    rows = []
    for name, cols in FAMILIES.items():
        rest = [f for f in ALL_FEATURES if f not in cols]
        s = scores(walk_forward(df, rest, first_test, last_test, models=one), [model])
        rows.append({"family": name,
                     "log_loss_rise": s["log_loss"].iloc[0] - base["log_loss"].iloc[0],
                     "accuracy_drop": base["accuracy"].iloc[0] - s["accuracy"].iloc[0]})
    return pd.DataFrame(rows).sort_values("log_loss_rise", ascending=False).reset_index(drop=True)


def margin_residuals(df: pd.DataFrame, first_test: int, last_test: int) -> pd.DataFrame:
    """Actual minus expected home margin, where the expectation uses Elo and team ratings only.

    Residuals are centered within each season.

    Shows how much schedule and roster situations move results beyond team strength.
    """
    base = FAMILIES["Home court"] + FAMILIES["Elo"] + FAMILIES["Team ratings"]
    preds = walk_forward(df, base, first_test, last_test,
                         models={"Ridge (margin)": MODELS["Ridge (margin)"]})
    out = df.set_index("game_id").loc[preds["game_id"]].reset_index()
    out["residual"] = out["margin"] - preds["Ridge (margin) pred"].to_numpy()
    # Home edge shrinks over time, so a model fit on earlier seasons runs high; center by season
    out["residual"] -= out.groupby("season")["residual"].transform("mean")
    return out
