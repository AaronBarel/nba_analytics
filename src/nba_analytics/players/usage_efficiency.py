"""Usage rate vs. true shooting percentage by position.

``usage_ts_frame`` builds one row per qualifying player-season;
``plot_usage_ts`` draws a single position (or position group) scatter, and
``plot_usage_ts_interactive`` draws the same chart with Plotly, showing player
details on hover.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from adjustText import adjust_text
from matplotlib.lines import Line2D

from nba_analytics.config import POS_TO_GROUP, POSITION_GROUPS, season_label
from nba_analytics.data.bbref import player_season_table
from nba_analytics.viz.teams import TEAM_COLORS

MIN_GAMES = 20
MIN_MPG = 6.0


def _is_combined(team: pd.Series) -> pd.Series:
    """Multi-team season rows: 'TOT' in older pages, '2TM'/'3TM'/... in newer ones."""
    return team.str.fullmatch(r"TOT|\dTM")


def season_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a published player table to one row per player.

    Traded players keep their combined row, with ``team`` set to the last
    team they played for that season.
    """
    df = df.copy()
    combined = _is_combined(df["team_name_abbr"])
    last_team = df[~combined].groupby("player_id")["team_name_abbr"].last()
    has_combined = set(df.loc[combined, "player_id"])
    keep = combined | ~df["player_id"].isin(has_combined)
    out = df[keep].copy()
    out["team"] = out["player_id"].map(last_team)
    return out.reset_index(drop=True)


def build_usage_ts(advanced: pd.DataFrame, per_game: pd.DataFrame,
                   min_games: int = MIN_GAMES, min_mpg: float = MIN_MPG) -> pd.DataFrame:
    """Join, filter and add positional TS% differences (percentage points)."""
    adv = season_rows(advanced)
    pg = season_rows(per_game)[["player_id", "mp_per_g"]]
    df = adv.merge(pg, on="player_id", how="left")
    df = df[(df["games"] >= min_games) & (df["mp_per_g"] >= min_mpg)].copy()
    df = df[df["pos"].isin(POS_TO_GROUP)]

    df["pos_group"] = df["pos"].map(POS_TO_GROUP)
    df["ts_pct"] = df["ts_pct"] * 100  # fraction -> percent, to match usg_pct
    df["ts_diff_pos"] = df["ts_pct"] - df.groupby("pos")["ts_pct"].transform("mean")
    df["ts_diff_group"] = df["ts_pct"] - df.groupby("pos_group")["ts_pct"].transform("mean")

    cols = ["season", "player_id", "name_display", "age", "team", "pos", "pos_group",
            "games", "mp_per_g", "usg_pct", "ts_pct", "ts_diff_pos", "ts_diff_group"]
    return df[cols].rename(columns={"name_display": "player"}).reset_index(drop=True)


def usage_ts_frame(season: int, **kwargs) -> pd.DataFrame:
    """Qualifying player rows for a season (end year, e.g. 2026 = 2025-26)."""
    return build_usage_ts(player_season_table(season, "advanced"),
                          player_season_table(season, "per_game"), **kwargs)


def extreme_players(df: pd.DataFrame, y: str, n: int) -> pd.Index:
    """Index of the ``n`` points farthest from the centre in standardised units."""
    z = (df[["usg_pct", y]] - df[["usg_pct", y]].mean()) / df[["usg_pct", y]].std()
    return np.hypot(z["usg_pct"], z[y]).nlargest(n).index


def _select(df: pd.DataFrame, position: str, relative: bool) -> tuple[pd.DataFrame, str, bool]:
    """Rows for one position or group, the y column to plot, and whether it is a group.

    'C' is both a position and a group, so it is treated as a position.
    """
    is_group = position in POSITION_GROUPS and position != "C"
    sub = df[df["pos_group" if is_group else "pos"] == position]
    if relative:
        y = "ts_diff_group" if is_group else "ts_diff_pos"
    else:
        y = "ts_pct"
    return sub, y, is_group


def _titles(sub: pd.DataFrame, position: str, relative: bool, is_group: bool) -> tuple[str, str]:
    season = season_label(int(sub["season"].iloc[0]))
    kind = "group" if is_group else "position"
    ylabel = f"TS% minus {kind} average (pct. points)" if relative else "True shooting %"
    title = (f"{season} {position}: usage vs. "
             f"{'relative ' if relative else ''}true shooting (n={len(sub)})")
    return title, ylabel


def plot_usage_ts(df: pd.DataFrame, position: str, relative: bool = False,
                  n_labels: int = 12, ax: plt.Axes | None = None) -> plt.Axes:
    """Scatter of usage vs. TS% for one position ('PG'...'C') or group ('G', 'F').

    ``relative=True`` plots TS% minus the position (or group) average.
    """
    sub, y, is_group = _select(df, position, relative)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7.5))

    fill = sub["team"].map(lambda t: TEAM_COLORS.get(t, ("#888888", "#444444"))[0])
    edge = sub["team"].map(lambda t: TEAM_COLORS.get(t, ("#888888", "#444444"))[1])
    ax.scatter(sub["usg_pct"], sub[y], s=55, c=fill, edgecolors=edge, linewidths=1.4,
               alpha=0.9, zorder=3)

    ref = 0 if relative else sub[y].mean()
    ax.axhline(ref, color="#9a9a9a", lw=1, ls="--", zorder=1)
    ax.axvline(sub["usg_pct"].mean(), color="#9a9a9a", lw=1, ls="--", zorder=1)

    texts = [ax.text(sub.at[i, "usg_pct"], sub.at[i, y], sub.at[i, "player"], fontsize=8.5,
                     color="#222222", zorder=4)
             for i in extreme_players(sub, y, min(n_labels, len(sub)))]
    adjust_text(texts, ax=ax, arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6})

    teams = sorted(t for t in sub["team"].unique() if t in TEAM_COLORS)
    handles = [Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=TEAM_COLORS[t][0],
                      markeredgecolor=TEAM_COLORS[t][1], markeredgewidth=1.4, label=t)
               for t in teams]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), ncol=2,
              fontsize=8, frameon=False, title="Team", title_fontsize=9)

    title, ylabel = _titles(sub, position, relative, is_group)
    ax.set_xlabel("Usage %")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    ax.grid(color="#e6e6e6", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_usage_ts_interactive(df: pd.DataFrame, position: str, relative: bool = False,
                              n_labels: int = 12) -> go.Figure:
    """Plotly version of ``plot_usage_ts``: hover any dot for the player's details.

    One trace per team, so clicking a team in the legend hides or isolates it.
    """
    sub, y, is_group = _select(df, position, relative)
    title, ylabel = _titles(sub, position, relative, is_group)
    hover = ("<b>%{customdata[0]}</b> (%{customdata[1]}, %{customdata[2]})<br>"
             "Usage: %{x:.1f}%<br>TS: %{customdata[3]:.1f}%<br>"
             "TS vs. avg: %{customdata[4]:+.1f} pts<br>"
             "%{customdata[5]:.0f} games, %{customdata[6]:.1f} min/game<extra></extra>")
    diff = "ts_diff_group" if is_group else "ts_diff_pos"

    fig = go.Figure()
    for team, rows in sub.groupby("team", sort=True):
        fill, edge = TEAM_COLORS.get(team, ("#888888", "#444444"))
        fig.add_trace(go.Scatter(
            x=rows["usg_pct"], y=rows[y], mode="markers", name=team,
            marker={"size": 10, "color": fill, "line": {"color": edge, "width": 1.6},
                    "opacity": 0.9},
            customdata=rows[["player", "team", "pos", "ts_pct", diff, "games", "mp_per_g"]],
            hovertemplate=hover,
        ))

    for i in extreme_players(sub, y, min(n_labels, len(sub))):
        fig.add_annotation(x=sub.at[i, "usg_pct"], y=sub.at[i, y], text=sub.at[i, "player"],
                           showarrow=True, arrowhead=0, arrowwidth=0.6, arrowcolor="#777777",
                           ax=0, ay=-18, font={"size": 10, "color": "#222222"})

    ref = 0 if relative else sub[y].mean()
    fig.add_hline(y=ref, line={"color": "#9a9a9a", "width": 1, "dash": "dash"})
    fig.add_vline(x=sub["usg_pct"].mean(), line={"color": "#9a9a9a", "width": 1, "dash": "dash"})
    fig.update_layout(
        title={"text": title, "x": 0.01}, xaxis_title="Usage %", yaxis_title=ylabel,
        template="plotly_white", width=1000, height=860, legend_title_text="Team",
        legend={"font": {"size": 10}, "tracegroupgap": 0}, margin={"t": 60},
        hoverlabel={"bgcolor": "white"},
    )
    return fig
