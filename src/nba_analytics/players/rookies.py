"""College-to-NBA rookie projections (R1).

Features come from a player's final college season (per 100 possessions and
advanced), his college career, and his bio: age at the end of his last college
season, height, weight, position, conference, recruiting rank and draft slot.
College BPM already adjusts for strength of schedule; a power-conference flag
adds a coarser league-strength signal.

Targets are rookie minutes per game and BPM, and total minutes and
minutes-weighted BPM over the first four NBA seasons. The model averages
gradient boosted trees (draft pick constrained to be monotonic) and a ridge
regression, scored out of sample by leaving one draft class out at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_analytics.config import PROCESSED_DIR
from nba_analytics.data.bbref import draft_class, player_bio, player_page, player_season_table
from nba_analytics.data.cbb import college_table
from nba_analytics.players.defense import career_start
from nba_analytics.players.usage_efficiency import season_rows

ADVANCED_START = 2010  # college per-100 and advanced tables start in 2009-10
MAX_GAP = 2  # seasons allowed between the last college season and the NBA debut
WINDOW = 4  # seasons in the multi-year targets
UNDRAFTED_PICK = 61
UNRANKED_RSCI = 150
POWER_CONFS = {"ACC", "Big Ten", "Big 12", "SEC", "Pac-12", "Pac-10", "Big East"}
CLASS_YEAR = {"FR": 1, "SO": 2, "JR": 3, "SR": 4}

PER_100 = ["pts", "trb", "orb", "drb", "ast", "stl", "blk", "tov", "pf", "fg3a", "fta"]
ADVANCED = ["ts_pct", "usg_pct", "ast_pct", "orb_pct", "drb_pct", "stl_pct", "blk_pct",
            "tov_pct", "obpm", "dbpm", "bpm", "ws_per_40", "fg3a_per_fga_pct",
            "fta_per_fga_pct"]

# Feature families, used for drop-family importance
FAMILIES = {
    "Draft slot": ["log_pick", "drafted"],
    "Recruiting rank": ["rsci_rank", "rsci_missing"],
    "Age": ["age"],
    "Size": ["height_cm", "weight_kg"],
    "Position & conference": ["guard", "forward", "center", "power_conf"],
    "Role & experience": ["games", "mp", "mp_per_g", "start_pct", "class_year",
                          "college_seasons", "career_mp"],
    "Scoring & shooting": ["pts_per_100", "fg3a_per_100", "fta_per_100", "ts_pct", "usg_pct",
                           "fg3_pct", "ft_pct", "career_ft_pct", "career_fg3_pct",
                           "fg3a_per_fga_pct", "fta_per_fga_pct"],
    "Playmaking": ["ast_per_100", "tov_per_100", "ast_pct", "tov_pct"],
    "Rebounding": ["trb_per_100", "orb_per_100", "drb_per_100", "orb_pct", "drb_pct"],
    "Defensive box score": ["stl_per_100", "blk_per_100", "pf_per_100", "stl_pct", "blk_pct"],
    "Impact (BPM, WS)": ["obpm", "dbpm", "bpm", "ws_per_40", "career_bpm"],
}
SCOUTING = FAMILIES["Draft slot"] + FAMILIES["Recruiting rank"]
COLLEGE = [f for name, cols in FAMILIES.items() for f in cols if f not in SCOUTING]
FEATURE_SETS = {
    "Draft slot": FAMILIES["Draft slot"],
    "College stats + bio": COLLEGE,
    "All (adds draft slot, RSCI)": COLLEGE + SCOUTING,
}

TARGETS = {
    "rookie_mpg": "Rookie minutes per game",
    "rookie_bpm": "Rookie BPM (500+ minutes)",
    "mp_4yr": "Minutes, first four seasons",
    "bpm_4yr": "BPM, first four seasons (2,000+ minutes)",
}
MIN_ROOKIE_MP = 500
MIN_4YR_MP = 2000


def _top_row(df: pd.DataFrame, season: int) -> pd.Series:
    """The season's row with the most minutes (a mid-season transfer has two)."""
    rows = df[df["season"] == season]
    return rows.loc[rows["mp"].fillna(0).idxmax()]


def _with_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add missing columns as NaN; a stat with no entries (e.g. 3P% for a player
    who never attempted a three) has no column at all."""
    return df.reindex(columns=df.columns.union(cols, sort=False))


def _ratio(made: float, attempts: float) -> float:
    return made / attempts if attempts else np.nan


def college_features(cbb_path: str) -> dict | None:
    """Final-season and career features, or None without per-100 and advanced tables."""
    per_poss, adv = college_table(cbb_path, "per_poss"), college_table(cbb_path, "advanced")
    totals = college_table(cbb_path, "totals")
    if per_poss.empty or adv.empty or totals.empty:
        return None
    per_poss = _with_columns(per_poss, [f"{s}_per_poss" for s in PER_100]
                             + ["fg3_pct", "ft_pct"])
    adv = _with_columns(adv, ADVANCED + ["games_started"])
    totals = _with_columns(totals, ["ft", "fta", "fg3", "fg3a"])
    last = int(adv["season"].max())
    a, p = _top_row(adv, last), _top_row(per_poss, last)
    pos = str(a["pos"])
    career = totals.sum(numeric_only=True)
    has_bpm = adv["bpm"].notna() & (adv["mp"] > 0)
    return {
        "cbb_path": cbb_path, "last_college_season": last, "school": a["team_name_abbr"],
        "conf": a["conf_abbr"], "power_conf": int(a["conf_abbr"] in POWER_CONFS),
        "guard": int("G" in pos), "forward": int("F" in pos), "center": int("C" in pos),
        "class_year": CLASS_YEAR.get(a["class"]), "college_seasons": totals["season"].nunique(),
        "games": a["games"], "mp": a["mp"], "mp_per_g": a["mp"] / a["games"],
        "start_pct": a["games_started"] / a["games"],
        **{f"{s}_per_100": p[f"{s}_per_poss"] for s in PER_100},
        "fg3_pct": p["fg3_pct"], "ft_pct": p["ft_pct"],
        **{s: a[s] for s in ADVANCED},
        "career_mp": career["mp"],
        "career_ft_pct": _ratio(career["ft"], career["fta"]),
        "career_fg3_pct": _ratio(career["fg3"], career["fg3a"]),
        "career_bpm": np.average(adv.loc[has_bpm, "bpm"], weights=adv.loc[has_bpm, "mp"])
        if has_bpm.any() else np.nan,
    }


def nba_targets(first: int, last: int) -> pd.DataFrame:
    """Rookie-season and first-four-season outcomes for players debuting in the range."""
    debut = career_start(last)
    debut = debut[(debut >= first) & (debut <= last)]
    adv = pd.concat([season_rows(player_season_table(s, "advanced")) for s in range(first, last + 1)])
    pg = pd.concat([season_rows(player_season_table(s, "per_game")) for s in range(first, last + 1)])
    adv = adv[adv["player_id"].isin(debut.index)].copy()
    adv["career_season"] = adv["season"] - adv["player_id"].map(debut) + 1

    rookie = adv[adv["career_season"] == 1].set_index("player_id")
    rookie_pg = pg.set_index(["player_id", "season"])["mp_per_g"]
    window = adv[adv["career_season"] <= WINDOW].assign(bpm_mp=lambda d: d["bpm"] * d["mp"])
    w = window.groupby("player_id")[["mp", "bpm_mp"]].sum()

    out = pd.DataFrame({"debut": debut})
    out["player"] = rookie["name_display"]
    out["rookie_mpg"] = [rookie_pg.get((pid, s)) for pid, s in debut.items()]
    out["rookie_mp"] = rookie["mp"]
    out["rookie_bpm"] = rookie["bpm"].where(out["rookie_mp"] >= MIN_ROOKIE_MP)
    complete = out["debut"] + WINDOW - 1 <= last
    out["mp_4yr"] = w["mp"].reindex(out.index).fillna(0).where(complete)
    out["bpm_4yr"] = (w["bpm_mp"] / w["mp"]).reindex(out.index).where(out["mp_4yr"] >= MIN_4YR_MP)
    return out.rename_axis("player_id").reset_index()


def build_rookies(bios: pd.DataFrame, college: pd.DataFrame, targets: pd.DataFrame,
                  draft: pd.DataFrame) -> pd.DataFrame:
    """Join bios, college features and outcomes; add the upcoming draft class.

    ``draft`` is the most recent draft (no NBA stats yet). NBA players must have
    a final college season from 2009-10 on, at most two seasons before their debut.
    """
    df = bios.merge(college, on="cbb_path")
    df = df[df["last_college_season"] >= ADVANCED_START]
    nba = df.merge(targets, on="player_id")
    gap = nba["debut"] - nba["last_college_season"]
    nba = nba[(gap >= 1) & (gap <= MAX_GAP)]
    new = df.merge(draft[["player_id", "player", "pick_overall"]], on="player_id")
    new = new[new["last_college_season"] >= int(draft["draft_year"].iloc[0]) - 1]
    new["draft_pick"] = new["draft_pick"].fillna(new["pick_overall"])
    new["draft_year"] = new["draft_year"].fillna(draft["draft_year"].iloc[0])
    out = pd.concat([nba.assign(draft_class=False),
                     new.drop(columns="pick_overall").assign(draft_class=True)],
                    ignore_index=True)

    out["drafted"] = out["draft_pick"].notna().astype(int)
    out["draft_pick"] = out["draft_pick"].fillna(UNDRAFTED_PICK)
    out["log_pick"] = np.log(out["draft_pick"])  # the gap between picks 1 and 5 >> 41 and 45
    out["rsci_missing"] = out["rsci_rank"].isna().astype(int)
    out["rsci_rank"] = out["rsci_rank"].fillna(UNRANKED_RSCI)
    season_end = pd.to_datetime(out["last_college_season"].astype(str) + "-06-25")
    out["age"] = (season_end - pd.to_datetime(out["birth_date"])).dt.days / 365.25
    out["entry_year"] = out["last_college_season"]  # draft class, drafted or not
    return out


def rookie_dataset(first: int = 2010, last: int = 2026, refresh: bool = False) -> pd.DataFrame:
    """Modeling table: NBA players debuting first..last plus the draft held after ``last``.

    Parsing ~1,600 player pages takes a few minutes, so the result is cached
    to ``data/processed``.
    """
    cache = PROCESSED_DIR / f"rookies_{first}_{last}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)
    targets = nba_targets(first, last)
    draft = draft_class(last)
    ids = sorted(set(targets["player_id"]) | set(draft["player_id"].dropna()))
    bios = pd.DataFrame([{"player_id": pid, **player_bio(player_page(pid))} for pid in ids])
    paths = bios["cbb_path"].dropna().unique()
    college = pd.DataFrame([f for f in map(college_features, paths) if f is not None])
    df = build_rookies(bios, college, targets, draft)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def model(features: list[str]) -> VotingRegressor:
    """Average of shallow boosted trees and a ridge regression.

    Trees pick up interactions (e.g. age x production); ridge is steadier on
    the noisy BPM targets. A later pick can never raise the tree prediction.
    """
    trees = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=40,
        l2_regularization=5.0, random_state=0,
        monotonic_cst=[-1 if f == "log_pick" else 0 for f in features])
    ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    return VotingRegressor([("trees", trees), ("ridge", ridge)])


def training_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df[~df["draft_class"] & df[target].notna()]


def out_of_fold(df: pd.DataFrame, target: str, features: list[str]) -> pd.Series:
    """Predictions for each player from a model that never saw his draft class."""
    rows = training_rows(df, target)
    pred = cross_val_predict(model(features), rows[features], rows[target],
                             groups=rows["entry_year"], cv=LeaveOneGroupOut())
    return pd.Series(pred, index=rows.index, name=f"pred_{target}")


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-class-out R² and MAE for every target and feature set."""
    records = []
    for target, label in TARGETS.items():
        y = training_rows(df, target)[target]
        for name, features in FEATURE_SETS.items():
            pred = out_of_fold(df, target, features)
            records.append({"target": label, "features": name, "n": len(y),
                            "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred)})
    return pd.DataFrame(records)


def family_importance(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Drop in out-of-sample R² when each feature family is removed from the full model."""
    features = FEATURE_SETS["All (adds draft slot, RSCI)"]
    y = training_rows(df, target)[target]
    base = r2_score(y, out_of_fold(df, target, features))
    drops = {name: base - r2_score(y, out_of_fold(df, target, [f for f in features
                                                                if f not in cols]))
             for name, cols in FAMILIES.items()}
    return (pd.Series(drops, name="r2_drop").rename_axis("family").sort_values(ascending=False)
            .reset_index())


def project_draft_class(df: pd.DataFrame) -> pd.DataFrame:
    """Every target for the upcoming class, from the full model and the college-only model."""
    new = df[df["draft_class"]].copy()
    for target in TARGETS:
        rows = training_rows(df, target)
        for suffix, name in [("", "All (adds draft slot, RSCI)"),
                             ("_college", "College stats + bio")]:
            features = FEATURE_SETS[name]
            fit = model(features).fit(rows[features], rows[target])
            new[f"proj_{target}{suffix}"] = fit.predict(new[features])
    return new.sort_values("draft_pick").reset_index(drop=True)
