"""G League exploration (R2): how G League production translates to the NBA.

G League tables have totals and advanced rates but no possessions or BPM, so
counting stats are compared per 36 minutes on both sides. A G League season
combines the regular season and the Showcase Cup.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import player_season_table
from nba_analytics.data.gleague import PARTS, gleague_table, nba_id
from nba_analytics.players.usage_efficiency import season_rows

COUNTING = ["pts", "trb", "ast", "stl", "blk", "tov", "fg3a", "fta"]
RATES = ["usg_pct", "ast_pct", "trb_pct", "stl_pct", "blk_pct", "tov_pct", "per", "ws_per_48"]
STATS = {**{f"{s}_per_36": f"{s.upper()} per 36" for s in COUNTING},
         "ts_pct": "TS%", "usg_pct": "Usage %", "ast_pct": "AST%", "trb_pct": "TRB%",
         "stl_pct": "STL%", "blk_pct": "BLK%", "tov_pct": "TOV%"}
MIN_GLEAGUE_MINUTES = 300
MIN_NBA_MINUTES = 250


def _ts(df: pd.DataFrame) -> pd.Series:
    """True shooting % (0-100) from totals."""
    return 100 * df["pts"] / (2 * (df["fga"] + 0.44 * df["fta"])).replace(0, np.nan)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Minutes-weighted mean over the parts with a value (NaN if none)."""
    ok = values.notna()
    return np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan


def _one_row_per_player(df: pd.DataFrame) -> pd.DataFrame:
    return season_rows(df.rename(columns={"team_id": "team_name_abbr"}))


def gleague_seasons(first: int, last: int) -> pd.DataFrame:
    """One row per G League player-season: totals, per-36, TS% and advanced rates."""
    rows = []
    for season in range(first, last + 1):
        tot = pd.concat([_one_row_per_player(gleague_table(season, "totals", p))
                         for p in PARTS if not gleague_table(season, "totals", p).empty])
        adv = pd.concat([_one_row_per_player(gleague_table(season, "advanced", p))
                         for p in PARTS if not gleague_table(season, "advanced", p).empty])
        totals = tot.groupby("player_id")[["g", "mp", "fga", *COUNTING]].sum()
        info = tot.groupby("player_id")[["player", "age"]].first()
        adv = adv[adv["mp"] > 0]
        rates = adv.groupby("player_id").apply(
            lambda d: pd.Series({c: _weighted_mean(d[c], d["mp"]) for c in RATES}),
            include_groups=False)
        df = info.join(totals).join(rates).reset_index()
        df.insert(0, "season", season)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(columns={"player_id": "gleague_id"})
    for s in COUNTING:
        df[f"{s}_per_36"] = df[s] / df["mp"].replace(0, np.nan) * 36
    df["ts_pct"] = _ts(df)
    return df


def nba_seasons(first: int, last: int) -> pd.DataFrame:
    """NBA player-seasons on the same scale: per 36, TS% and advanced rates."""
    rows = []
    for season in range(first, last + 1):
        pg = season_rows(player_season_table(season, "per_game"))
        adv = season_rows(player_season_table(season, "advanced"))
        df = pg[["season", "player_id", "name_display", "age", "pos", "games", "mp_per_g",
                 *[f"{s}_per_g" for s in COUNTING], "fga_per_g"]].merge(
            adv[["player_id", "mp", "ts_pct", "bpm", *[r for r in RATES if r != "per"],
                 "per"]], on="player_id", how="left")
        for s in COUNTING:
            df[f"{s}_per_36"] = df[f"{s}_per_g"] / df["mp_per_g"].replace(0, np.nan) * 36
        df["ts_pct"] = df["ts_pct"] * 100
        rows.append(df.rename(columns={"name_display": "player"}))
    return pd.concat(rows, ignore_index=True)


def _normalize(name: pd.Series) -> pd.Series:
    def norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
        return "".join(ch for ch in s.lower() if ch.isalnum())
    return name.map(norm)


LINK_WINDOW = 4  # seasons between a G League season and the NBA career for a name match


def link_players(gl: pd.DataFrame, nba: pd.DataFrame) -> pd.DataFrame:
    """Add the NBA ``player_id`` (NaN if none) and how it was found.

    1. 'id': the G League id minus its 'd' is an NBA id with the same name.
    2. 'name': otherwise, the one NBA player with that name.
    Either way the NBA career must come within ``LINK_WINDOW`` seasons of the
    G League season, so namesakes from different eras don't link.
    """
    names = nba.drop_duplicates("player_id").set_index("player_id")["player"]
    norm_nba = _normalize(names)
    career = nba.groupby("player_id")["season"].agg(["min", "max"])
    out = gl.copy()
    norm_gl = _normalize(out["player"])

    def in_window(pid, season) -> bool:
        return (pid in career.index and career.at[pid, "min"] - LINK_WINDOW <= season
                <= career.at[pid, "max"] + LINK_WINDOW)

    candidate = nba_id(out["gleague_id"])
    by_id = (norm_gl == candidate.map(norm_nba)) & pd.Series(
        [in_window(p, s) for p, s in zip(candidate, out["season"])], index=out.index)
    out["player_id"] = candidate.where(by_id)
    out["link"] = np.where(by_id, "id", None)

    ids_by_name = norm_nba.groupby(norm_nba).groups
    for idx in out.index[~by_id]:
        matches = [pid for pid in ids_by_name.get(norm_gl[idx], [])
                   if in_window(pid, out.at[idx, "season"])]
        if len(matches) == 1:
            out.at[idx, "player_id"], out.at[idx, "link"] = matches[0], "name"
    return out


def link_report(gl: pd.DataFrame) -> pd.DataFrame:
    """G League players by how (or whether) they link to an NBA player."""
    players = gl.drop_duplicates("gleague_id")
    counts = players["link"].fillna("none").value_counts()
    return pd.DataFrame({
        "G League players": [len(players)],
        "Linked by id and name": [counts.get("id", 0)],
        "Linked by name and career window": [counts.get("name", 0)],
        "No NBA match": [counts.get("none", 0)],
    })


def translation_pairs(gl: pd.DataFrame, nba: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """G League season t joined to the same player's NBA season t + lag."""
    g = gl[gl["mp"] >= MIN_GLEAGUE_MINUTES]
    n = nba[nba["mp"] >= MIN_NBA_MINUTES].assign(season=lambda d: d["season"] - lag)
    cols = ["player_id", "season", "mp", "age", *STATS]
    return g[cols + ["player"]].merge(n[cols + ["bpm"]], on=["player_id", "season"],
                                      suffixes=("_gl", "_nba"))


def _weighted_corr(x: pd.Series, y: pd.Series, w: pd.Series) -> float:
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2,
                                                                           weights=w))


def translation_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per stat: G League and NBA averages, the NBA/G League ratio, and the correlation.

    Weighted by the harmonic mean of the two seasons' minutes.
    """
    w_all = 2 / (1 / pairs["mp_gl"] + 1 / pairs["mp_nba"])
    records = []
    for stat, label in STATS.items():
        x, y = pairs[f"{stat}_gl"], pairs[f"{stat}_nba"]
        ok = x.notna() & y.notna()
        w = w_all[ok]
        gl_mean, nba_mean = np.average(x[ok], weights=w), np.average(y[ok], weights=w)
        records.append({"stat": label, "G League": gl_mean, "NBA": nba_mean,
                        "NBA / G League": nba_mean / gl_mean if gl_mean else np.nan,
                        "r": _weighted_corr(x[ok], y[ok], w), "n": int(ok.sum())})
    return pd.DataFrame(records)


def gleague_experience(gl: pd.DataFrame, nba: pd.DataFrame, first: int) -> pd.DataFrame:
    """Share of NBA players and minutes from players with earlier G League minutes."""
    first_gl = gl[gl["mp"] > 0].groupby("player_id")["season"].min()
    n = nba[nba["season"] >= first].copy()
    n["gleague_before"] = n["player_id"].map(first_gl) <= n["season"]
    rotation = n["mp"] >= 1000
    return pd.DataFrame({
        "players": n.groupby("season")["gleague_before"].mean(),
        "rotation players": n[rotation].groupby("season")["gleague_before"].mean(),
        "minutes": n.groupby("season").apply(
            lambda d: d.loc[d["gleague_before"], "mp"].sum() / d["mp"].sum(),
            include_groups=False),
    }).reset_index()


def lagged_gleague_features(gl: pd.DataFrame) -> pd.DataFrame:
    """G League features keyed by the following season, for the projection models.

    ``gl_mp_1`` is G League minutes in the prior season (0 when none).
    """
    cols = {"mp": "gl_mp_1", "pts_per_36": "gl_pts_per_36_1", "ts_pct": "gl_ts_pct_1",
            "usg_pct": "gl_usg_pct_1", "ast_pct": "gl_ast_pct_1", "trb_pct": "gl_trb_pct_1",
            "per": "gl_per_1", "ws_per_48": "gl_ws_per_48_1"}
    out = gl[["player_id", "season", *cols]].rename(columns=cols)
    out["season"] = out["season"] + 1
    return out


PRE_DEBUT = {"mp": "gl_pre_mp", "pts_per_36": "gl_pre_pts_per_36", "ts_pct": "gl_pre_ts_pct",
             "usg_pct": "gl_pre_usg_pct", "ast_pct": "gl_pre_ast_pct",
             "trb_pct": "gl_pre_trb_pct", "per": "gl_pre_per", "ws_per_48": "gl_pre_ws_per_48"}


def pre_debut_features(gl: pd.DataFrame, debut: pd.Series) -> pd.DataFrame:
    """Each player's last G League season before his NBA debut, for the rookie model.

    ``debut`` maps NBA player id -> first NBA season. ``gl_pre_mp`` is 0 for
    players with no G League minutes before debuting.
    """
    g = gl[gl["player_id"].notna()]
    g = g[g["season"] < g["player_id"].map(debut)]
    last = g.sort_values("season").groupby("player_id").last()
    out = last[list(PRE_DEBUT)].rename(columns=PRE_DEBUT).reindex(debut.index)
    out["gl_pre_mp"] = out["gl_pre_mp"].fillna(0)
    return out.rename_axis("player_id").reset_index()


def _compare(without: pd.Series, with_gl: pd.Series, n: pd.Series, group: str) -> pd.DataFrame:
    return pd.DataFrame({"group": group, "without": without, "with G League": with_gl,
                         "change %": (with_gl / without - 1) * 100, "n": n}).reset_index()


def projection_test(gl: pd.DataFrame, first_test: int = 2014,
                    last_test: int = 2026) -> pd.DataFrame:
    """Walk-forward MAE of the season-projection model with and without the prior
    season's G League stats, for all players and for those with G League minutes."""
    from nba_analytics.players import projections as pr

    seasons = pr.player_seasons(pr.HISTORY_START, last_test)
    frame = pr.build_frame(seasons, last=last_test)
    lag = lagged_gleague_features(gl[gl["player_id"].notna()])
    frame = frame.merge(lag, on=["player_id", "season"], how="left")
    frame["gl_mp_1"] = frame["gl_mp_1"].fillna(0)
    extra = [c for c in lag.columns if c.startswith("gl_")]
    base = pr.walk_forward(frame, seasons, first_test, last_test)
    with_gl = pr.walk_forward(frame, seasons, first_test, last_test, extra=extra)

    parts = []
    for group, idx in [("All players", base.index),
                       ("G League minutes last season",
                        base.index[frame.loc[base.index, "gl_mp_1"] > 0])]:
        a = pr.evaluate(frame, base.loc[idx]).query("method == 'model'").set_index("target")
        b = pr.evaluate(frame, with_gl.loc[idx]).query("method == 'model'").set_index("target")
        parts.append(_compare(a["mae"], b["mae"], b["n"], group))
    return pd.concat(parts, ignore_index=True)


def rookie_test(gl: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-class-out MAE of the rookie model with and without the last G League
    season before the NBA debut, for all players and for those with one."""
    from sklearn.metrics import mean_absolute_error

    from nba_analytics.players import rookies as r1

    df = r1.rookie_dataset()
    debut = df.loc[~df["draft_class"]].set_index("player_id")["debut"]
    df = df.merge(pre_debut_features(gl, debut), on="player_id", how="left")
    full = r1.FEATURE_SETS["All (adds draft slot, RSCI)"]
    records = []
    for t, label in r1.TARGETS.items():
        rows = r1.training_rows(df, t)
        a = r1.out_of_fold(df, t, full)
        b = r1.out_of_fold(df, t, full + list(PRE_DEBUT.values()))
        for group, mask in [("All players", rows.index == rows.index),
                            ("G League before debut", (rows["gl_pre_mp"] > 0).to_numpy())]:
            y = rows.loc[mask, t]
            ma, mb = mean_absolute_error(y, a[mask]), mean_absolute_error(y, b[mask])
            records.append({"group": group, "target": label, "without": ma,
                            "with G League": mb, "change %": (mb / ma - 1) * 100,
                            "n": int(mask.sum())})
    return pd.DataFrame(records)


def gleague_dataset(first: int = 2008, last: int = 2026,
                    refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Linked G League seasons and NBA seasons on the same scale, cached to ``data/processed``."""
    paths = [PROCESSED_DIR / f"{name}_{first}_{last}.parquet" for name in ("gleague", "nba_per36")]
    if all(p.exists() for p in paths) and not refresh:
        return pd.read_parquet(paths[0]), pd.read_parquet(paths[1])
    nba = nba_seasons(first, last)
    gl = link_players(gleague_seasons(first, last), nba)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gl.to_parquet(paths[0])
    nba.to_parquet(paths[1])
    return gl, nba
