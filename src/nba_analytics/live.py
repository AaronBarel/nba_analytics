"""Predictions for upcoming games: win probability, margin and player stat lines.

The current season comes from Basketball-Reference's schedule and box scores
(one page per finished game, cached once final) and its team rosters.
Availability comes from the NBA's official injury report for the game, or
from Basketball-Reference's injury list for games the report doesn't cover
yet. Features are built exactly as for the historical models (T3 and P4),
with the upcoming games added as rows that have no result, and the models are
fit on every past season.
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data import bbref, injuries
from nba_analytics.players import props
from nba_analytics.teams import games, roster, totals
from nba_analytics.teams.net_rating import franchise, game_logs, possessions
from nba_analytics.teams.schedule import NEUTRAL_ARENAS, NEUTRAL_DATES

FIRST_TRAIN = 2014  # first season with player game logs
FRESH_HOURS = 3  # re-download rosters, injury lists and open schedule months after this
OUT = {"Out", "Doubtful"}  # statuses treated as not playing; Questionable counts as playing
PREDICTIONS_DIR = PROCESSED_DIR.parent / "predictions"
COUNTS = ["pts", "fg", "fga", "fg3", "fg3a", "ft", "fta", "orb", "drb", "ast", "stl", "blk",
          "tov", "pf"]


def current_season(today: str) -> int:
    """Season (end year) for an ISO date; a new season starts in August."""
    d = pd.Timestamp(today)
    return d.year + 1 if d.month >= 8 else d.year


def eastern_today() -> str:
    return datetime.now(injuries.EASTERN).strftime("%Y-%m-%d")


# --- Schedule and finished games -----------------------------------------------

def schedule(season: int, today: str) -> pd.DataFrame:
    """The season's games with results so far, refreshing months with open games."""
    parts = []
    for month in bbref.schedule_months(season, max_age=24 * 7):
        df = bbref.schedule_month(season, month)
        if (df["box_score"].isna() & (df["date"] <= today)).any():  # results may be out
            df = bbref.schedule_month(season, month, max_age=FRESH_HOURS / 3)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    city = df["arena"].map(NEUTRAL_ARENAS)
    df["neutral"] = city.notna() | (df["arena"].isna() & df["date"].isin(NEUTRAL_DATES))
    return df


def _team_rows(games_: pd.DataFrame, totals: dict) -> pd.DataFrame:
    """Two team-log rows per game, with box score counts when the game is final."""
    rows = []
    for g in games_.itertuples():
        for side, other, loc in [("home", "away", None), ("away", "home", "@")]:
            team, opp = getattr(g, side), getattr(g, other)
            row = {"season": g.season, "team_id": team, "date": g.date,
                   "game_location": "N" if g.neutral else loc, "opp_name_abbr": opp}
            if g.box_score in totals:
                t = totals[g.box_score].set_index("team_id")
                row.update({c: t.at[team, c] for c in COUNTS})
                row.update({f"opp_{c}": t.at[opp, c] for c in COUNTS})
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in [*COUNTS, *[f"opp_{c}" for c in COUNTS]]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["franchise"] = franchise(df["team_id"])
    df["poss"] = possessions(df)
    df["win"] = (df["pts"] > df["opp_pts"]).astype(int)
    df = df.sort_values(["team_id", "date"])
    df["game_num"] = df.groupby("team_id").cumcount() + 1
    return df.reset_index(drop=True)


def finished(sched: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Box score totals by game and player rows (as in ``roster.gamelogs``) for final games."""
    totals, players = {}, []
    for g in sched[sched["box_score"].notna()].itertuples():
        p, t = bbref.box_score(g.box_score)
        totals[g.box_score] = t
        home = p["team_id"] == g.home
        p = p.assign(season=g.season, date=g.date,
                     game_location=np.where(home, None, "@"),
                     opp_name_abbr=np.where(home, g.away, g.home))
        players.append(p)
    gl = pd.concat(players, ignore_index=True)
    gl["available"] = gl["status"].isna() | gl["status"].isin(roster.AVAILABLE)
    for c in ["mp", *roster.BOX]:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").astype(float)
    gl["mp"] = gl["mp"].fillna(0.0)
    cols = ["season", "player_id", "date", "team_id", "game_location", "opp_name_abbr",
            "status", "started", "mp", *roster.BOX, "available"]
    return totals, gl[cols]


# --- Availability --------------------------------------------------------------

def _key(name: str) -> str:
    """Accents, case, punctuation and spaces removed: 'Nikola Jokić' -> 'nikolajokic'."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", plain.lower())


def match_names(report: pd.DataFrame, rosters: pd.DataFrame) -> pd.Series:
    """Basketball-Reference ``player_id`` for each injury report row ('Last, First').

    A roster name matches if some split of it into first and last names gives
    the same key; failing that, a unique last name on the team.
    """
    keys: dict[tuple[str, str], str] = {}
    last_names: dict[tuple[str, str], list[str]] = {}
    for r in rosters.itertuples():
        words = r.player.split()
        for k in range(1, len(words)):
            keys[(r.team_id, _key("".join(words[k:]) + "".join(words[:k])))] = r.player_id
        last_names.setdefault((r.team_id, _key(words[-1])), []).append(r.player_id)
    out = []
    for r in report.itertuples():
        if not isinstance(r.player, str):
            out.append(None)
            continue
        last, _, first = r.player.partition(",")
        pid = keys.get((r.team_id, _key(last + first)))
        if pid is None:
            ids = last_names.get((r.team_id, _key(last.split()[0] if last.split() else last)), [])
            pid = ids[0] if len(ids) == 1 else None
        out.append(pid)
    return pd.Series(out, index=report.index, dtype=object)


def availability(slate: pd.DataFrame, rosters: pd.DataFrame, report: pd.DataFrame,
                 bbref_list: pd.DataFrame) -> pd.DataFrame:
    """Each rostered player's status for each game in ``slate``.

    Teams the official report covers for that game use it (unlisted players
    are available); other teams fall back to Basketball-Reference's injury
    list ('Day To Day' counts as Questionable). ``source`` says which.
    """
    report = report.assign(player_id=match_names(report, rosters))
    fallback = bbref_list.assign(status=bbref_list["status"].replace(
        {"Out For Season": "Out", "Day To Day": "Questionable"})).set_index("player_id")["status"]
    rows = []
    for g in slate.itertuples():
        for team in (g.home, g.away):
            listed = report[(report["game_date"] == g.date) & (report["team_id"] == team)]
            covered = len(listed) and not listed["status"].eq("Not Yet Submitted").all()
            status = listed.dropna(subset=["player_id"]).set_index("player_id")["status"]
            for p in rosters[rosters["team_id"] == team].itertuples():
                if covered:
                    s, src = status.get(p.player_id), "Official report"
                else:
                    s, src = fallback.get(p.player_id), "Basketball-Reference list"
                rows.append({"date": g.date, "team_id": team, "player_id": p.player_id,
                             "player": p.player, "injury_status": s or "Available",
                             "source": src})
    return pd.DataFrame(rows)


def unmatched(report: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Report rows whose name matched no rostered player (e.g. a signing not yet listed)."""
    ids = match_names(report, rosters)
    return report[report["player"].notna() & ids.isna()]


# --- Features and models -------------------------------------------------------

def build(target: str, today: str | None = None) -> dict:
    """Everything needed to predict the games on ``target`` (ISO date).

    Returns the schedule, team logs and player rows (finished games plus the
    games through ``target``), the pre-season player values, availability on
    ``target`` and the injury report stamp used.
    """
    today = today or eastern_today()
    season = current_season(target)
    sched = schedule(season, today)
    totals, gl = finished(sched)
    final = sched["box_score"].notna()
    ahead = sched[~final & (sched["date"] >= today) & (sched["date"] <= target)]
    slate = ahead[ahead["date"] == target]
    if slate.empty:
        raise LookupError(f"no unplayed games on {target}")

    teams = sorted(set(slate["home"]) | set(slate["away"]))
    rosters = pd.concat([bbref.team_roster(t, season, max_age=FRESH_HOURS) for t in teams],
                        ignore_index=True)
    stamp, report = injuries.latest_report()
    avail = availability(slate, rosters, report, bbref.injury_list(max_age=FRESH_HOURS))

    upcoming = avail.merge(slate[["date", "home", "away", "neutral"]], on="date")
    upcoming = upcoming[(upcoming["team_id"] == upcoming["home"])
                        | (upcoming["team_id"] == upcoming["away"])]
    home = upcoming["team_id"] == upcoming["home"]
    rows = pd.DataFrame({
        "season": season, "player_id": upcoming["player_id"], "date": upcoming["date"],
        "team_id": upcoming["team_id"], "game_location": np.where(home, None, "@"),
        "opp_name_abbr": np.where(home, upcoming["away"], upcoming["home"]),
        "status": upcoming["injury_status"], "started": 0, "mp": np.nan,
        **{c: np.nan for c in roster.BOX},
        "available": ~upcoming["injury_status"].isin(OUT)})

    logs = pd.concat([game_logs(range(2001, season)),
                      _team_rows(sched[final | sched.index.isin(ahead.index)], totals)],
                     ignore_index=True)
    return {"season": season, "target": target, "schedule": sched, "slate": slate,
            "logs": logs, "gl": pd.concat([gl, rows], ignore_index=True),
            "values": roster.season_values(season), "availability": avail,
            "report": report, "stamp": stamp, "rosters": rosters}


def _props_models(last: int):
    """Per-game models fit on every season through ``last``, cached."""
    path = PROCESSED_DIR / f"props_models_{FIRST_TRAIN}_{last}.pkl"
    if path.exists():
        models = pickle.loads(path.read_bytes())
        if set(models["rates"]) == set(props.STATS) \
                and models.get("fg3_prior") == props.FG3_PRIOR:
            return models
    df = props.props_dataset(FIRST_TRAIN, last)
    models = props.fit(df[df["mp"] > 0])
    path.write_bytes(pickle.dumps(models))
    return models


def totals_models(last: int) -> dict:
    """Margin and total distribution models fit on every season through ``last``, cached.

    Pass ``models["close_weights"]`` to ``totals.prob_margin_over``.
    """
    path = PROCESSED_DIR / f"totals_models_{FIRST_TRAIN}_{last}.pkl"
    if path.exists():
        models = pickle.loads(path.read_bytes())
        if set(models) == set(totals.MODEL_KEYS):
            return models
    models = totals.fit(totals.total_features(games.game_dataset(FIRST_TRAIN, last)))
    path.write_bytes(pickle.dumps(models))
    return models


def predict_games(data: dict) -> pd.DataFrame:
    """Home win probability (logistic), margin (ridge) and total for the slate.

    ``margin_sd``, ``total_pred`` and ``total_sd`` give the margin and total
    distributions (``totals.prob_margin_over`` and ``totals.prob_total_over``).
    """
    season = data["season"]
    rows = games.game_rows(data["logs"], data["gl"], data["values"], season)
    rows = totals.total_features(rows)  # league averages need the whole season so far
    rows = rows[rows["date"] == data["target"]]
    train = games.game_dataset(FIRST_TRAIN, season - 1)
    use = {k: games.MODELS[k] for k in ["Logistic regression", "Ridge (margin)"]}
    fitted = games.fit(train, games.ALL_FEATURES, use)
    pred = games.predict(fitted, rows, games.ALL_FEATURES)
    out = rows[["game_id", "date", "home", "away", "neutral", "elo_prob", "home_talent",
                "away_talent", "home_talent_missing", "away_talent_missing",
                "home_rest_days", "away_rest_days", "away_travel_km"]].copy()
    out["home_win_prob"] = pred["Logistic regression"].to_numpy()
    out["home_margin"] = pred["Ridge (margin) pred"].to_numpy()
    out["margin_win_prob"] = pred["Ridge (margin)"].to_numpy()
    dist = totals.predict(totals_models(season - 1), rows)
    for c in ["margin_sd", "total_pred", "total_sd"]:
        out[c] = dist[c].to_numpy()
    start = data["slate"].set_index(["home", "away"])["start_time"]
    out["start_time"] = [start.get((h, a), start.get((a, h))) for h, a in
                         zip(out["home"], out["away"])]
    return out.reset_index(drop=True)


def predict_players(data: dict, min_minutes: float = 10.0) -> pd.DataFrame:
    """Projected minutes and stat lines for available players on the slate.

    Each stat has its mean, an 80% interval and the negative binomial size,
    so the probability of any line is ``props.prob_over(mean, size, line)``.
    Players projected under ``min_minutes`` are left out.
    """
    season = data["season"]
    df = props.player_game_rows(data["gl"], data["values"], data["logs"])
    rows = df[(df["date"] == data["target"]) & df["available"]]
    out = props.predict(_props_models(season - 1), rows)
    names = data["rosters"].drop_duplicates("player_id").set_index("player_id")["player"]
    status = data["availability"].set_index(["date", "player_id"])["injury_status"]
    out["player"] = out["player_id"].map(names)
    out["injury_status"] = [status.get((d, p)) for d, p in zip(out["date"], out["player_id"])]
    for s in props.STATS:
        r, mu = out[f"{s}_size"], out[f"{s}_pred"].clip(lower=1e-6)
        out[f"{s}_lo"] = stats.nbinom.ppf(0.1, r, r / (r + mu))
        out[f"{s}_hi"] = stats.nbinom.ppf(0.9, r, r / (r + mu))
    keep = ["date", "player_id", "player", "team_id", "opp_id", "injury_status", "mp_pred",
            *[f"{s}_{k}" for s in props.STATS for k in ["pred", "lo", "hi", "size"]]]
    out = out[out["mp_pred"] >= min_minutes][keep]
    return out.sort_values(["team_id", "mp_pred"], ascending=[True, False]).reset_index(drop=True)


def save(data: dict, game_preds: pd.DataFrame, player_preds: pd.DataFrame) -> None:
    """Keep each run's predictions, stamped, to score them once the games are played."""
    run = datetime.now(injuries.EASTERN).strftime("%Y-%m-%dT%H%M")
    folder = PREDICTIONS_DIR / data["target"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in [("games", game_preds), ("players", player_preds)]:
        df.assign(run=run, injury_report=data["stamp"]).to_parquet(folder / f"{name}_{run}.parquet")
