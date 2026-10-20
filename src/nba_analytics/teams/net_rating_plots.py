"""Interactive charts for the net rating analyses (T1, T2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nba_analytics.config import season_label
from nba_analytics.teams.net_rating import r_squared
from nba_analytics.viz.teams import TEAM_COLORS

LABELS = {"net_rtg": "Net rating", "win_pct": "Win %"}
SERIES = {"win_pct": "#2a78d6", "net_rtg": "#eb6834"}  # validated categorical slots 1-2
GRAY = "#9a9a9a"


def _fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    return xs, intercept + slope * xs


def plot_consecutive(pairs: pd.DataFrame, col: str = "net_rtg") -> go.Figure:
    """Season N vs. season N+1, one dot per franchise and season pair.

    Pairs touching a shortened season are drawn as open circles.
    """
    label = LABELS[col]
    hover = ("<b>%{customdata[0]}</b><br>%{customdata[1]}: %{x:.1f}<br>"
             "%{customdata[2]}: %{y:.1f}<extra></extra>")
    if col == "win_pct":
        hover = hover.replace(":.1f}", ":.3f}")
    fig = go.Figure()
    for team, rows in pairs.groupby("franchise", sort=True):
        fill, edge = TEAM_COLORS.get(team, ("#888888", "#444444"))
        short = rows["any_shortened"]
        fig.add_trace(go.Scatter(
            x=rows[f"{col}_prev"], y=rows[f"{col}_next"], mode="markers", name=team,
            marker={"size": 8, "color": np.where(short, "rgba(0,0,0,0)", fill),
                    "line": {"color": np.where(short, fill, edge), "width": 1.4},
                    "opacity": 0.85},
            customdata=np.column_stack([
                rows["team_next"],
                rows["season_prev"].map(season_label),
                rows["season_next"].map(season_label)]),
            hovertemplate=hover,
        ))
    xs, ys = _fit_line(pairs[f"{col}_prev"], pairs[f"{col}_next"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit", hoverinfo="skip",
                             line={"color": "#222222", "width": 2}))
    r2_all = r_squared(pairs[f"{col}_prev"], pairs[f"{col}_next"])
    full = pairs[~pairs["any_shortened"]]
    r2_full = r_squared(full[f"{col}_prev"], full[f"{col}_next"])
    seasons = f"{season_label(pairs['season_prev'].min())} to {season_label(pairs['season_next'].max())}"
    fig.update_layout(
        title={"text": f"{label}: season N vs. season N+1, {seasons}<br>"
                       f"<sup>R² = {r2_all:.3f} (all pairs, n={len(pairs)}) · "
                       f"{r2_full:.3f} (excluding shortened seasons, n={len(full)}) · "
                       "open circles touch a shortened season</sup>", "x": 0.01},
        xaxis_title=f"{label}, season N", yaxis_title=f"{label}, season N+1",
        template="plotly_white", width=1000, height=860, legend_title_text="Team",
        legend={"font": {"size": 10}, "tracegroupgap": 0}, hoverlabel={"bgcolor": "white"},
    )
    return fig


def plot_r2_by_pair(by_pair: pd.DataFrame, col: str = "net_rtg") -> go.Figure:
    """R² across the league for each season-to-season transition."""
    labels = [f"{season_label(s - 1)} → {season_label(s)}" for s in by_pair["season_next"]]
    colors = np.where(by_pair["any_shortened"], GRAY, SERIES[col])
    fig = go.Figure(go.Bar(
        x=labels, y=by_pair["r2"], marker={"color": colors, "cornerradius": 4},
        hovertemplate="%{x}<br>R² = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=by_pair["r2"].mean(), line={"color": "#222222", "width": 1, "dash": "dash"},
                  annotation_text=f"mean {by_pair['r2'].mean():.2f}",
                  annotation_position="top right")
    fig.update_layout(
        title={"text": f"{LABELS[col]}: R² of season N vs. season N+1, by transition"
                       "<br><sup>Gray bars touch a shortened season</sup>", "x": 0.01},
        yaxis_title="R²", yaxis_range=[0, 1], template="plotly_white", width=1000,
        height=540, bargap=0.25, xaxis_tickangle=-45, margin={"b": 150},
    )
    return fig


def plot_r2_by_games(curves: pd.DataFrame, threshold: float = 0.5) -> go.Figure:
    """R² vs. games played, one panel per rest-of-season target."""
    targets = ["net_rtg", "win_pct"]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[f"Target: rest-of-season {LABELS[t].lower()}"
                                        for t in targets])
    for col, target in enumerate(targets, start=1):
        for pred in ["win_pct", "net_rtg"]:
            c = curves[(curves["predictor"] == pred) & (curves["target"] == target)]
            color = SERIES[pred]
            fig.add_trace(go.Scatter(
                x=np.concatenate([c["n"], c["n"][::-1]]),
                y=np.concatenate([c["hi"], c["lo"][::-1]]),
                fill="toself", fillcolor=_rgba(color, 0.15), line={"width": 0},
                hoverinfo="skip", showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=c["n"], y=c["r2"], mode="lines", line={"color": color, "width": 2},
                name=f"First-N {LABELS[pred].lower()}", legendgroup=pred, showlegend=col == 1,
                customdata=np.column_stack([c["lo"], c["hi"]]),
                hovertemplate=(f"First-N {LABELS[pred].lower()}<br>After %{{x}} games: "
                               "R² = %{y:.3f}<br>95% CI %{customdata[0]:.3f} to "
                               "%{customdata[1]:.3f}<extra></extra>")), row=1, col=col)
            crossed = c[c["r2"] > threshold]
            if len(crossed):
                first = crossed.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[first["n"]], y=[first["r2"]], mode="markers+text",
                    marker={"size": 10, "color": color, "line": {"color": "white", "width": 2}},
                    text=[f"{int(first['n'])} games"],
                    textposition="top left" if pred == "net_rtg" else "bottom right",
                    textfont={"color": "#222222", "size": 11}, hoverinfo="skip",
                    showlegend=False), row=1, col=col)
        fig.add_hline(y=threshold, line={"color": GRAY, "width": 1, "dash": "dash"},
                      row=1, col=col)
        fig.update_xaxes(title_text="Games played (N)", row=1, col=col)
    n_ts = int(curves["team_seasons"].iloc[0])
    fig.update_yaxes(title_text="R² vs. rest of season (games N+1 to end)", range=[0, 1],
                     row=1, col=1)
    fig.update_layout(
        title={"text": "How many games until early results predict the rest of the season?"
                       f"<br><sup>{n_ts} team-seasons; bands are 95% bootstrap intervals; "
                       f"dots mark the first N with R² > {threshold}</sup>", "x": 0.01},
        template="plotly_white", width=1100, height=560, hovermode="closest",
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"
