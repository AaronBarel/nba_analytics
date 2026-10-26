"""Margin and total distributions: the probability of any home margin or combined score.

The margin model is T3's ridge on every game feature. The total starts from
the points the two teams' ratings imply (expected pace times the average of
each offense and the defense it faces), and a ridge on league-relative rates
and game situations predicts what that misses; predicting the gap rather
than the total keeps the model from extrapolating the league's rising
scoring from older seasons.

The total is normal around its prediction, with the residual SD of the last
``SD_SEASONS`` training seasons (margins and totals have spread out as
scoring rose). The margin is whole points and never zero (overtime), and
end-of-game fouling makes one-point finishes rarer and three-to-nine-point
finishes more common than a normal has them, so its probabilities come from
a normal's one-point bins reweighted by final margin (``fit_close_weights``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from nba_analytics.teams import games

SD_SEASONS = 2
MODEL_KEYS = ("margin", "total", "margin_sd", "margin_resid_sd", "total_sd", "close_weights")
RELATIVE = ["pace", "off_rtg", "def_rtg", "efg", "opp_efg", "ft_rate", "opp_ft_rate",
            "tov_pct", "opp_tov_pct", "orb_pct", "drb_pct"]
SITUATION = ["home_court", "home_games_played", "home_rest_days", "away_rest_days",
             "home_back_to_back", "away_back_to_back", "altitude_m", "home_talent_missing",
             "away_talent_missing"]
TOTAL_FEATURES = [f"{side}_{c}_rel" for c in RELATIVE for side in ["home", "away"]] + SITUATION


# --- Features ---------------------------------------------------------------

def league_relative(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """``<side>_<col>_rel``: each team's pre-game rate minus the league average that day.

    The league average is over every team's latest pre-game value in the
    season, so it only uses what is known before the games of that date.
    """
    out = pd.DataFrame(index=g.index)
    for c in cols:
        long = pd.concat([g[["season", "date", side]].set_axis(["season", "date", "team"], axis=1)
                          .assign(v=g[f"{side}_{c}"]) for side in ["home", "away"]])
        wide = long.pivot_table(index=["season", "date"], columns="team", values="v")
        league = wide.groupby(level="season").ffill().mean(axis=1)
        avg = league.reindex(pd.MultiIndex.from_frame(g[["season", "date"]])).to_numpy()
        for side in ["home", "away"]:
            out[f"{side}_{c}_rel"] = g[f"{side}_{c}"].to_numpy() - avg
    return out


def total_features(g: pd.DataFrame) -> pd.DataFrame:
    """Games with ``total``, the ratings' ``expected_total`` and the league-relative rates."""
    g = g.copy()
    g["total"] = g["home_pts"] + g["away_pts"]
    pace = (g["home_pace"] + g["away_pace"]) / 2
    g["expected_total"] = pace * (g["home_off_rtg"] + g["away_def_rtg"]
                                  + g["away_off_rtg"] + g["home_def_rtg"]) / 200
    return pd.concat([g, league_relative(g, RELATIVE)], axis=1)


# --- Model ------------------------------------------------------------------

def _recent_sd(resid: pd.Series, season: pd.Series) -> float:
    return float(resid[season > season.max() - SD_SEASONS].std())


CLOSE_MARGIN = 25  # final margins up to this many points get their own weight
SUPPORT = np.array([m for m in range(-80, 81) if m != 0])  # possible home margins


def margin_pmf(mu: np.ndarray, sd: float | np.ndarray, weights: np.ndarray | None = None
               ) -> np.ndarray:
    """P(home margin = m) for each game (rows) and each margin in ``SUPPORT`` (columns).

    The normal's mass in each one-point bin, times the weight for that
    absolute margin (``weights[k]`` for ``k <= CLOSE_MARGIN``, else 1), with the
    tie bin dropped and each row renormalized.
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))[:, None]
    sd = np.atleast_1d(np.asarray(sd, dtype=float))[:, None]
    p = stats.norm.cdf((SUPPORT + 0.5 - mu) / sd) - stats.norm.cdf((SUPPORT - 0.5 - mu) / sd)
    if weights is not None:
        k = np.abs(SUPPORT)
        p = p * np.where(k <= CLOSE_MARGIN, weights[np.minimum(k, CLOSE_MARGIN)], 1.0)
    return p / p.sum(axis=1, keepdims=True)


def fit_close_weights(margin: np.ndarray, mu: np.ndarray, sd: float, iters: int = 10
                      ) -> np.ndarray:
    """Weights per absolute margin so predicted and actual counts match on ``margin``.

    Close finishes aren't normal: teams trailing late foul, so one-point games
    are rarer and three-to-nine-point games more common than the normal has
    them. The fitted weights are smoothed with a three-margin moving average
    (margins past ``CLOSE_MARGIN`` count as 1). Index ``k`` is the weight for a
    final margin of ``k`` (0 is unused).
    """
    w = np.ones(CLOSE_MARGIN + 1)
    k = np.abs(SUPPORT)
    actual = np.bincount(np.abs(margin).astype(int), minlength=81)
    for _ in range(iters):
        expected = margin_pmf(mu, sd, w).sum(axis=0)
        for m in range(1, CLOSE_MARGIN + 1):
            w[m] *= actual[m] / expected[k == m].sum()
    padded = np.concatenate([w[1:2], w[1:], [1.0]])
    w[1:] = np.convolve(padded, np.ones(3) / 3, mode="valid")
    return w


def fit_margin_sd(margin: np.ndarray, mu: np.ndarray, weights: np.ndarray) -> float:
    """SD of the margin's underlying normal by maximum likelihood, given the weights."""
    idx = np.searchsorted(SUPPORT, margin)
    rows = np.arange(len(margin))

    def nll(sd):
        return -np.log(margin_pmf(mu, sd, weights)[rows, idx]).mean()

    return float(optimize.minimize_scalar(nll, bounds=(5, 25), method="bounded").x)


def fit(train: pd.DataFrame) -> dict:
    """Margin and total ridges, each distribution's SD and the close-game weights.

    SDs are fit on the last ``SD_SEASONS`` seasons, the weights on all of them.
    The margin's SD and weights depend on each other, so they are fit in turn
    twice, starting from the residual SD.
    """
    margin = games._ridge().fit(train[games.ALL_FEATURES], train["margin"])
    gap = train["total"] - train["expected_total"]
    total = games._ridge().fit(train[TOTAL_FEATURES], gap)
    y, mu = train["margin"].to_numpy(), margin.predict(train[games.ALL_FEATURES])
    recent = (train["season"] > train["season"].max() - SD_SEASONS).to_numpy()
    margin_sd = _recent_sd(train["margin"] - mu, train["season"])
    for _ in range(2):
        weights = fit_close_weights(y, mu, margin_sd)
        margin_sd = fit_margin_sd(y[recent], mu[recent], weights)
    return {
        "margin": margin, "total": total, "margin_sd": margin_sd,
        "margin_resid_sd": _recent_sd(train["margin"] - mu, train["season"]),
        "total_sd": _recent_sd(gap - total.predict(train[TOTAL_FEATURES]), train["season"]),
        "close_weights": fit_close_weights(y, mu, margin_sd),
    }


def predict(models: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """Predicted home margin and total, with each distribution's SD.

    ``margin_resid_sd`` is the plain residual SD, for a normal without the
    close-game weights.
    """
    keys = ["game_id", "season", "date", "home", "away", "margin", "total"]
    out = rows[[c for c in keys if c in rows]].copy()
    out["margin_pred"] = models["margin"].predict(rows[games.ALL_FEATURES])
    out["total_pred"] = rows["expected_total"] + models["total"].predict(rows[TOTAL_FEATURES])
    out["margin_sd"], out["total_sd"] = models["margin_sd"], models["total_sd"]
    out["margin_resid_sd"] = models["margin_resid_sd"]
    out["expected_total"] = rows["expected_total"]
    return out


def walk_forward(df: pd.DataFrame, first_test: int, last_test: int
                 ) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Out-of-sample margins and totals, each season from a model fit on earlier ones.

    Also returns each test season's close-game weights.
    """
    parts, weights = [], {}
    for s in range(first_test, last_test + 1):
        models = fit(df[df["season"] < s])
        parts.append(predict(models, df[df["season"] == s]))
        weights[s] = models["close_weights"]
    return pd.concat(parts, ignore_index=True), weights


def prob_total_over(mu: np.ndarray, sd: float | np.ndarray, line: float | np.ndarray
                    ) -> np.ndarray:
    """P(total > line) for a half-point line."""
    return stats.norm.sf((line - mu) / sd)


def prob_margin_over(mu: np.ndarray, sd: float | np.ndarray, line: float | np.ndarray,
                     weights: np.ndarray | None = None) -> np.ndarray:
    """P(home margin > line) for each game; ``line`` may differ by game.

    For the away team winning by more than ``x``, use ``1 - prob_margin_over(..., -x)``.
    """
    p = margin_pmf(mu, sd, weights)
    line = np.broadcast_to(np.asarray(line, dtype=float), (p.shape[0],))
    return (p * (SUPPORT[None, :] > line[:, None])).sum(axis=1)


# --- Evaluation -------------------------------------------------------------

def scores(preds: pd.DataFrame, by_season: bool = False) -> pd.DataFrame:
    """MAE, bias (actual minus predicted), RMSE and the distribution's SD."""
    groups = preds["season"] if by_season else pd.Series(0, index=preds.index)
    rows = []
    for kind in ["margin", "total"]:
        err = preds[kind] - preds[f"{kind}_pred"]
        for g, e in err.groupby(groups):
            rows.append({"target": kind, "season": g, "n": len(e), "mae": e.abs().mean(),
                         "bias": e.mean(), "rmse": np.sqrt((e ** 2).mean()),
                         "sd": preds.loc[e.index, f"{kind}_sd"].mean()})
    out = pd.DataFrame(rows)
    return out if by_season else out.drop(columns="season")


LINE_OFFSETS = [-15, -10, -5, -2, 0, 2, 5, 10, 15]


def line_calibration(preds: pd.DataFrame, kind: str, weights: dict[int, np.ndarray] | None = None,
                     offsets: list[int] = LINE_OFFSETS) -> pd.DataFrame:
    """Predicted vs. actual rate of going over half-point lines set around each prediction.

    The line for offset ``k`` is the prediction plus ``k``, rounded down to a
    half point. ``weights`` are the close-game weights by season (margin only).
    """
    rows = []
    for k in offsets:
        line = np.floor(preds[f"{kind}_pred"] + k) + 0.5
        if kind == "total":
            p = np.asarray(prob_total_over(preds["total_pred"], preds["total_sd"], line))
        else:
            p = np.empty(len(preds))
            for s, idx in preds.groupby("season").indices.items():
                part = preds.iloc[idx]
                p[idx] = prob_margin_over(part["margin_pred"].to_numpy(),
                                          part["margin_sd"].to_numpy(),
                                          line.iloc[idx].to_numpy(),
                                          weights.get(s) if weights else None)
        rows.append({"offset": k, "predicted": p.mean(), "actual": (preds[kind] > line).mean(),
                     "n": len(preds)})
    return pd.DataFrame(rows)


def margin_shape(preds: pd.DataFrame, weights: dict[int, np.ndarray], up_to: int = 25
                 ) -> pd.DataFrame:
    """Share of games by final margin (either team), actual and expected.

    Expected under the close-game weighted distribution and under a plain
    normal with the residual SD.
    """
    k = np.abs(SUPPORT)
    weighted, normal = np.zeros(81), np.zeros(81)
    for s, idx in preds.groupby("season").indices.items():
        part = preds.iloc[idx]
        mu = part["margin_pred"].to_numpy()
        weighted += np.bincount(k, margin_pmf(mu, part["margin_sd"].to_numpy(), weights[s])
                                .sum(axis=0), minlength=81)
        normal += np.bincount(k, margin_pmf(mu, part["margin_resid_sd"].to_numpy())
                              .sum(axis=0), minlength=81)
    actual = np.bincount(preds["margin"].abs().astype(int), minlength=81)
    m = np.arange(1, up_to + 1)
    n = len(preds)
    return pd.DataFrame({"margin": m, "actual": actual[m] / n, "weighted": weighted[m] / n,
                         "normal": normal[m] / n})


def total_pit(preds: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Share of totals in each decile of the predicted distribution (flat when calibrated)."""
    u = stats.norm.cdf((preds["total"] - preds["total_pred"]) / preds["total_sd"])
    b = pd.cut(u, np.linspace(0, 1, bins + 1), include_lowest=True)
    share = pd.Series(1, index=preds.index).groupby(b, observed=False).size() / len(u)
    return pd.DataFrame({"decile": np.arange(1, bins + 1), "share": share.to_numpy()})
