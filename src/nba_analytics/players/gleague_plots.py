"""Interactive charts for the G League exploration (R2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nba_analytics.config import season_label
from nba_analytics.players.gleague import STATS

# Validated categorical slots, in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#9a9a9a"


def plot_experience(exp: pd.DataFrame) -> go.Figure:
    """Share of NBA players, rotation players and minutes with earlier G League minutes."""
    fig = go.Figure()
    for (col, name), color in zip([("players", "All players"),
                                   ("rotation players", "Rotation players (1,000+ min)"),
                                   ("minutes", "Share of NBA minutes")], SERIES):
        fig.add_trace(go.Scatter(
            x=exp["season"].map(season_label), y=exp[col] * 100, mode="lines+markers",
            name=name, line={"color": color, "width": 2}, marker={"size": 7, "color": color},
            hovertemplate=f"<b>{name}</b> %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "NBA players with earlier G League minutes<br><sup>G League "
                       "history counted from 2007-08, so early seasons undercount</sup>",
               "x": 0.01},
        xaxis={"type": "category"},  # '2011-12' would otherwise parse as a date
        yaxis={"title": "Percent", "range": [0, 100]}, template="plotly_white", width=1000,
        height=480, hoverlabel={"bgcolor": "white"}, hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    return fig


def plot_translation_r(table: pd.DataFrame) -> go.Figure:
    """Correlation between a G League season and the next NBA season, by stat."""
    rows = table.sort_values("r")
    fig = go.Figure(go.Bar(
        y=rows["stat"], x=rows["r"], orientation="h",
        marker={"color": SERIES[0], "cornerradius": 4},
        customdata=rows[["G League", "NBA", "NBA / G League", "n"]],
        text=rows["r"].map("{:.2f}".format), textposition="outside",
        hovertemplate="<b>%{y}</b><br>r = %{x:.3f} (n=%{customdata[3]})<br>"
                      "G League %{customdata[0]:.1f} → NBA %{customdata[1]:.1f} "
                      "(x%{customdata[2]:.2f})<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "How well each G League stat carries over to the next NBA season<br>"
                       "<sup>Weighted correlation; 300+ G League minutes and 250+ NBA minutes"
                       "</sup>", "x": 0.01},
        xaxis={"title": "Correlation (r)", "range": [0, 1]}, template="plotly_white",
        width=1000, height=560, hoverlabel={"bgcolor": "white"}, showlegend=False,
    )
    return fig


def plot_translation_scatter(pairs: pd.DataFrame, stats: list[str]) -> go.Figure:
    """G League season vs. next NBA season for several stats (small multiples)."""
    cols = 3
    nrows = int(np.ceil(len(stats) / cols))
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=[STATS[s] for s in stats],
                        horizontal_spacing=0.07, vertical_spacing=0.14)
    custom = np.column_stack([pairs["player"], pairs["season"].map(season_label),
                              (pairs["season"] + 1).map(season_label), pairs["mp_gl"],
                              pairs["mp_nba"]])
    for i, stat in enumerate(stats):
        r, c = i // cols + 1, i % cols + 1
        x, y = pairs[f"{stat}_gl"], pairs[f"{stat}_nba"]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", showlegend=False, customdata=custom,
            marker={"size": 6, "color": SERIES[0], "opacity": 0.6,
                    "line": {"color": "white", "width": 0.5}},
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>{STATS[stat]}: G League "
                          "%{customdata[1]} %{x:.1f} (%{customdata[3]:,} min)<br>"
                          "NBA %{customdata[2]} %{y:.1f} (%{customdata[4]:,} min)<extra></extra>",
        ), row=r, col=c)
        lo, hi = np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 hoverinfo="skip",
                                 line={"color": MUTED, "width": 1, "dash": "dash"}),
                      row=r, col=c)
    fig.update_xaxes(title_text="G League", title_font={"size": 10})
    fig.update_yaxes(title_text="Next NBA season", title_font={"size": 10})
    fig.update_layout(
        title={"text": f"G League season vs. next NBA season (n={len(pairs)})<br><sup>Dashed "
                       "line = same value in both leagues</sup>", "x": 0.01},
        template="plotly_white", width=1000, height=330 * nrows + 120,
        hoverlabel={"bgcolor": "white"},
    )
    return fig


def plot_feature_test(test: pd.DataFrame, title: str) -> go.Figure:
    """Percent change in MAE from adding G League features, by target and group."""
    targets = list(dict.fromkeys(test["target"]))[::-1]
    fig = go.Figure()
    for group, color in zip(dict.fromkeys(test["group"]), SERIES):
        rows = test[test["group"] == group].set_index("target").loc[targets].reset_index()
        fig.add_trace(go.Bar(
            y=rows["target"], x=rows["change %"], name=group, orientation="h",
            marker={"color": color, "cornerradius": 4, "line": {"color": "white", "width": 2}},
            customdata=rows[["without", "with G League", "n"]],
            hovertemplate=f"<b>{group}</b><br>%{{y}}<br>MAE %{{customdata[0]:.3f}} → "
                          "%{customdata[1]:.3f} (%{x:+.1f}%, n=%{customdata[2]})<extra></extra>",
        ))
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    fig.update_layout(
        title={"text": f"{title}<br><sup>Change in out-of-sample MAE from adding G League "
                       "features; negative = more accurate</sup>", "x": 0.01},
        barmode="group", bargap=0.2, xaxis_title="Change in MAE (%)", template="plotly_white",
        width=1000, height=max(420, 52 * len(targets) + 180), hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.12, "x": 0, "traceorder": "normal"},
    )
    return fig
