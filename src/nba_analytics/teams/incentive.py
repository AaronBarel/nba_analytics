"""Incentive to win (T4): what one more win is worth in draft value.

From a date in the season, the rest of the regular season is simulated many
times, then the play-in and the lottery that season's draft uses
(``lottery``). Each team's draft value is the expected value of the
first-round picks it owns (``picks``). The draft incentive is the change in
that value when the team wins its next game rather than loses it; a negative
number means losing helps its draft position. Both runs share every random
draw, so the difference comes only from that one game.

Games are simulated from frozen team ratings: Elo and full-roster talent
(``roster``) at the date, combined by a logistic fit on earlier seasons,
plus a random shock to each team's strength in each simulated season so
that standings far ahead are uncertain enough.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.teams import games, lottery, picks, roster
from nba_analytics.teams.lottery import EAST, TEAMS
from nba_analytics.teams.net_rating import franchise, game_logs

SIMS = 2000
STRENGTH_SD = 0.35  # per-season shock to a team's win logit (see ``calibration``)
TIEBREAK = 0.5  # random tiebreak, in wins
ELO_SCALE = np.log(10) / 400  # Elo points -> win logit
HOME_ELO = games.ELO_PARAMS["home_adv"]


def sim_model(df: pd.DataFrame) -> np.ndarray:
    """Logistic coefficients (intercept, Elo logit, talent gap) for simulated games."""
    X = np.column_stack([np.log(df["elo_prob"] / (1 - df["elo_prob"])),
                         df["home_talent_full"] - df["away_talent_full"]])
    ok = ~np.isnan(X).any(axis=1)
    fit = LogisticRegression(C=1e6).fit(X[ok], df.loc[ok, "home_win"])
    return np.r_[fit.intercept_, fit.coef_[0]]


@dataclass
class Snapshot:
    """Where every team stands on ``date``: wins so far, ratings, games left."""
    season: int
    date: str
    wins: np.ndarray  # (30,)
    losses: np.ndarray
    elo: np.ndarray
    talent: np.ndarray
    home: np.ndarray  # remaining games, as team indices (G,)
    away: np.ndarray
    neutral: np.ndarray
    dates: np.ndarray


def snapshot(df: pd.DataFrame, season: int, date: str) -> Snapshot:
    """Standings and ratings before the games on ``date``, from ``games.game_rows`` output.

    Games on or after ``date``, or with no result yet, are left to simulate.
    Ratings are each team's pre-game Elo for its next game and its latest
    full-roster talent.
    """
    df = df[df["season"] == season].sort_values("date")
    left = (df["date"] >= date) | df["margin"].isna()
    index = {t: i for i, t in enumerate(TEAMS)}
    home, away = (franchise(df[s]).map(index).to_numpy() for s in ["home", "away"])
    won = (df["margin"] > 0).to_numpy()
    wins = (np.bincount(home[~left & won], minlength=30)
            + np.bincount(away[~left & ~won], minlength=30))
    losses = (np.bincount(home[~left & ~won], minlength=30)
              + np.bincount(away[~left & won], minlength=30))
    long = pd.concat([pd.DataFrame({"team": side, "date": df["date"], "left": left,
                                    "elo": df[f"elo_{s}"], "talent": df[f"{s}_talent_full"]})
                      for side, s in [(home, "home"), (away, "away")]]).sort_values("date")
    long["talent"] = long.groupby("team")["talent"].ffill()
    nxt = long[long["left"]].groupby("team").first()
    last = long[~long["left"]].groupby("team").last()
    elo = nxt["elo"].reindex(range(30)).fillna(last["elo"].reindex(range(30)))
    talent = nxt["talent"].reindex(range(30)).fillna(last["talent"].reindex(range(30)))
    return Snapshot(season, date, wins.astype(float), losses.astype(float), elo.to_numpy(),
                    talent.fillna(0).to_numpy(),
                    home[left], away[left], df.loc[left, "neutral"].to_numpy(),
                    df.loc[left, "date"].to_numpy())


def _logit(snap: Snapshot, coef: np.ndarray, h, a, neutral) -> np.ndarray:
    elo = ELO_SCALE * (snap.elo[h] - snap.elo[a] + np.where(neutral, 0.0, HOME_ELO))
    return coef[0] + coef[1] * elo + coef[2] * (snap.talent[h] - snap.talent[a])


def own_pick_value(coef: np.ndarray) -> Callable:
    """Draft value when every team owns its pick (the historical replay)."""
    return lambda pick: picks.pick_value(pick, coef)


def holdings_2027(coef: np.ndarray) -> Callable:
    """Draft value of the 2027 firsts each team owns, after protections and swaps."""
    return lambda pick: picks.holdings(pick, picks.owners_2027(pick), picks.pick_value(pick, coef))


def simulate(snap: Snapshot, coef: np.ndarray, value: Callable, n: int = SIMS, seed: int = 0,
             force: tuple[int, bool] | None = None, sd: float = STRENGTH_SD,
             rules: lottery.Rules | None = None) -> dict:
    """Simulate the rest of the season, the play-in and the lottery ``n`` times.

    ``force`` = (game, home wins) fixes one remaining game's result. Runs with
    the same ``seed`` share every other random draw. ``rules`` overrides the
    season's lottery. Returns final wins, original pick numbers, playoff
    misses and each team's draft value (S, 30).
    """
    rng = np.random.default_rng(seed)
    G = len(snap.home)
    u_games, shock = rng.random((n, G)), rng.normal(0.0, sd, (n, 30))
    u_tie, u_play, u_draw = rng.random((n, 30)), rng.random((n, 6)), rng.random((n, 16))
    logit = _logit(snap, coef, snap.home, snap.away, snap.neutral)
    p = 1 / (1 + np.exp(-(logit + shock[:, snap.home] - shock[:, snap.away])))
    home_won = u_games < p
    if force is not None:
        home_won[:, force[0]] = force[1]
    teams = np.arange(30)
    wins = (snap.wins + home_won.astype(float) @ (snap.home[:, None] == teams)
            + (~home_won).astype(float) @ (snap.away[:, None] == teams))
    rows = np.arange(n)

    def game_prob(h, a):
        return 1 / (1 + np.exp(-(_logit(snap, coef, h, a, False) + shock[rows, h]
                                 - shock[rows, a])))

    east = np.isin(np.array(TEAMS), list(EAST))
    rules = rules or lottery.rules_for(snap.season)
    pick, missed = lottery.draft_order(wins + TIEBREAK * u_tie, east, rules, game_prob, u_play,
                                       u_draw)
    return {"wins": wins, "pick": pick, "missed": missed, "value": value(pick)}


def incentives(snap: Snapshot, coef: np.ndarray, value: Callable, n: int = SIMS, seed: int = 0,
               sd: float = STRENGTH_SD, rules: lottery.Rules | None = None) -> pd.DataFrame:
    """Per team: projected wins, playoff odds, expected pick and draft value, and what the
    next game's result changes in each.

    ``draft_incentive`` = E[draft value | win next game] - E[... | lose it].
    """
    base = simulate(snap, coef, value, n, seed, sd=sd, rules=rules)
    out = pd.DataFrame({
        "season": snap.season, "date": snap.date, "team_id": TEAMS, "wins": snap.wins,
        "losses": snap.losses,
        "proj_wins": base["wins"].mean(axis=0), "wins_sd": base["wins"].std(axis=0),
        "playoffs": 1 - base["missed"].mean(axis=0), "exp_pick": base["pick"].mean(axis=0),
        "draft_value": base["value"].mean(axis=0)})
    runs: dict[tuple[int, bool], dict] = {}
    out["next_game"] = None
    for col in ["draft_win", "draft_loss", "playoffs_win", "playoffs_loss"]:
        out[col] = np.nan
    for t in range(30):
        mine = np.flatnonzero((snap.home == t) | (snap.away == t))
        if not len(mine):
            continue
        g = mine[0]
        for won in (True, False):
            key = (g, bool(won == (snap.home[g] == t)))
            if key not in runs:
                runs[key] = simulate(snap, coef, value, n, seed, force=key, sd=sd, rules=rules)
            r = runs[key]
            tag = "win" if won else "loss"
            out.loc[t, f"draft_{tag}"] = r["value"][:, t].mean()
            out.loc[t, f"playoffs_{tag}"] = 1 - r["missed"][:, t].mean()
        out.loc[t, "next_game"] = snap.dates[g]
    out["draft_incentive"] = out["draft_win"] - out["draft_loss"]
    out["playoff_incentive"] = out["playoffs_win"] - out["playoffs_loss"]
    return out


LIVE_SIMS = 8000


def current(today: str, n: int = LIVE_SIMS) -> pd.DataFrame:
    """Incentives for the season in progress, with each team's actual 2027 pick holdings."""
    from nba_analytics import live  # live predictions pull in the per-game player models

    season = live.current_season(today)
    sched = live.schedule(season, today)
    totals, gl = live.finished(sched)
    logs = pd.concat([game_logs(range(2001, season)), live._team_rows(sched, totals)],
                     ignore_index=True)
    rows = games.game_rows(logs, gl, roster.season_values(season), season)
    coef = sim_model(games.game_dataset(live.FIRST_TRAIN, season - 1))
    value = holdings_2027(picks.fit_pick_value(picks.draft_outcomes()))
    return incentives(snapshot(rows, season, today), coef, value, n)


# --- Historical replay --------------------------------------------------------

def checkpoints(df: pd.DataFrame, season: int, start: str = "01-01", every: int = 7) -> list[str]:
    """Dates from ``start`` (month-day, in the new year) to the last game, ``every`` days apart."""
    dates = df.loc[df["season"] == season, "date"]
    first = max(pd.Timestamp(f"{season}-{start}"), pd.Timestamp(dates.min()))
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(first, dates.max(), freq=f"{every}D")]


EXCLUDED = {2020}  # the lottery used standings frozen when the season was suspended


THREE_TWO_ONE = lottery.Rules("3-2-1", 16, balls=True)  # without the repeat-pick limits


def replay(first: int, last: int, n: int = SIMS, rules: lottery.Rules | None = None,
           refresh: bool = False) -> pd.DataFrame:
    """Weekly incentives from January on for past seasons, assuming every team owns its pick.

    Each season's game model is fit on the seasons before it. ``rules``
    replays the seasons under another lottery (the season's own by default).
    Cached.
    """
    tag = "" if rules is None else "_" + rules.name.replace("-", "")
    cache = PROCESSED_DIR / f"incentive_replay_{first}_{last}{tag}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    df = games.game_dataset(first - 1, last)
    value = own_pick_value(picks.fit_pick_value(picks.draft_outcomes()))
    parts = []
    for season in range(first, last + 1):
        if season in EXCLUDED:
            continue
        coef = sim_model(df[df["season"] < season])
        for date in checkpoints(df, season):
            parts.append(incentives(snapshot(df, season, date), coef, value, n, rules=rules))
    out = pd.concat(parts, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def calibration(df: pd.DataFrame, seasons: list[int], sds: list[float], date_md: str = "01-15",
                n: int = 1000) -> pd.DataFrame:
    """Spread of actual final wins around the simulated mean, against the simulated SD.

    A ratio near 1 means the simulation is as uncertain as reality.
    """
    rows = []
    for sd in sds:
        z = []
        for season in seasons:
            year = season - 1 if int(date_md[:2]) >= 8 else season
            snap = snapshot(df, season, f"{year}-{date_md}")
            coef = sim_model(df[df["season"] < season])
            sim = simulate(snap, coef, lambda p: np.zeros(p.shape), n, sd=sd)
            final = snapshot(df, season, "9999-12-31").wins
            z.append((final - sim["wins"].mean(axis=0)) / sim["wins"].std(axis=0))
        z = np.concatenate(z)
        rows.append({"strength_sd": sd, "z_sd": z.std(), "n": len(z)})
    return pd.DataFrame(rows)


# --- As a game-model feature -------------------------------------------------

SIDES = ["draft_incentive", "playoff_incentive"]
FEATURES = [f"{side}_{c}" for c in SIDES for side in ["home", "away"]] + [f"diff_{c}"
                                                                          for c in SIDES]


def with_incentives(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Games with each team's incentives from the latest checkpoint on or before the game.

    Games before a season's first checkpoint get zero.
    """
    hist = hist.assign(date=pd.to_datetime(hist["date"])).sort_values("date")
    out = df.assign(_date=pd.to_datetime(df["date"]), _row=np.arange(len(df)))
    for side in ["home", "away"]:
        out["_team"] = franchise(out[side])
        m = pd.merge_asof(out.sort_values("_date"), hist[["season", "date", "team_id", *SIDES]],
                          left_on="_date", right_on="date", left_by=["season", "_team"],
                          right_by=["season", "team_id"], suffixes=("", "_h"))
        m = m.set_index("_row").reindex(out["_row"])
        for c in SIDES:
            out[f"{side}_{c}"] = m[c].fillna(0.0).to_numpy()
    for c in SIDES:
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out.drop(columns=["_date", "_row", "_team"])


def _late(df: pd.DataFrame) -> pd.Series:
    """Games from January 1 on (the replay's checkpoints start then)."""
    return df["date"].str[5:7].astype(int) < 8


def feature_test(df: pd.DataFrame, first_test: int, last_test: int) -> pd.DataFrame:
    """Walk-forward log loss on games from January 1 as incentive features join the model.

    ``df`` is ``with_incentives`` output. The logistic game model is refit
    each season with and without the availability family, which is where
    resting players already show up.
    """
    use = {"Logistic regression": games.MODELS["Logistic regression"]}
    draft = [f for f in FEATURES if "draft" in f]
    playoff = [f for f in FEATURES if "playoff" in f]
    no_avail = [f for f in games.ALL_FEATURES if f not in games.FAMILIES["Availability"]]
    late = df.loc[_late(df) & ~df["season"].isin(EXCLUDED), "game_id"]
    rows = []
    for base_name, base in [("All T3 features", games.ALL_FEATURES),
                            ("Without availability", no_avail)]:
        for added, extra in [("none", []), ("draft", draft), ("playoff", playoff),
                             ("both", FEATURES)]:
            p = games.walk_forward(df, base + extra, first_test, last_test, models=use)
            s = games.scores(p[p["game_id"].isin(late)], ["Logistic regression"]).iloc[0]
            rows.append({"base": base_name, "incentives added": added, "games": s["n"],
                         "log_loss": s["log_loss"], "brier": s["brier"],
                         "accuracy": s["accuracy"]})
    return pd.DataFrame(rows)


def residual_bins(df: pd.DataFrame, first_test: int, last_test: int, col: str,
                  bins: int = 8) -> pd.DataFrame:
    """Home result minus the T3 walk-forward probability, by quantile bin of ``col``."""
    use = {"Logistic regression": games.MODELS["Logistic regression"]}
    p = games.walk_forward(df, games.ALL_FEATURES, first_test, last_test, models=use)
    p = p.merge(df[["game_id", "date", col]], on="game_id")
    p = p[_late(p) & ~p["season"].isin(EXCLUDED)]
    p["res"] = p["home_win"] - p["Logistic regression"]
    p["bin"] = pd.qcut(p[col], bins, duplicates="drop")
    out = p.groupby("bin", observed=True).agg(mean=("res", "mean"), sem=("res", "sem"),
                                              count=("res", "size"), mid=(col, "median"))
    return out.reset_index()
