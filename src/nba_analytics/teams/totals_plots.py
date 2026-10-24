"""Interactive charts for the margin and total distributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from nba_analytics.config import season_label
from nba_analytics.teams.games_plots import LAYOUT, MUTED, SERIES
from nba_analytics.teams.totals import SUPPORT, margin_pmf


def plot_scoring_by_season(df: pd.DataFrame) -> go.Figure:
    """Average combined score and the spread of home margins, by season."""
    g = df.groupby("season").agg(total=("total", "mean"), margin_sd=("margin", "std"),
                                 n=("total", "size"))
    labels = g.index.map(season_label)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["Average combined score (points)",
                                        "SD of the home margin (points)"])
    for col, y in enumerate([g["total"], g["margin_sd"]], start=1):
        fig.add_trace(go.Scatter(
            x=labels, y=y, mode="lines+markers", line={"color": SERIES[0], "width": 2},
            marker={"size": 8, "color": SERIES[0], "line": {"color": "white", "width": 2}},
            showlegend=False, customdata=g["n"],
            hovertemplate="<b>%{x}</b>: %{y:.1f} (%{customdata:,} games)<extra></extra>",
        ), row=1, col=col)
    fig.update_xaxes(type="category", tickangle=-45, tickfont={"size": 9})
    fig.update_layout(title={"text": "Scoring and margins by season", "x": 0.01}, height=420,
                      **LAYOUT)
    return fig


def plot_margin_shape(shape: pd.DataFrame) -> go.Figure:
    """Share of games by final margin: actual against both margin distributions."""
    fig = go.Figure(go.Bar(
        x=shape["margin"], y=100 * shape["actual"], name="Actual",
        marker={"color": "#c9d6e8", "cornerradius": 4},
        hovertemplate="Won by %{x}: %{y:.2f}% of games<extra>Actual</extra>"))
    for col, name, color in [("normal", "Normal", SERIES[1]),
                             ("weighted", "Normal with close-game weights", SERIES[0])]:
        fig.add_trace(go.Scatter(
            x=shape["margin"], y=100 * shape[col], name=name, mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color, "line": {"color": "white", "width": 2}},
            hovertemplate=f"Won by %{{x}}: %{{y:.2f}}%<extra>{name}</extra>"))
    fig.update_layout(
        title={"text": "How games end: final margin of the winner<br><sup>Share of games, "
                       "actual and expected from the walk-forward predictions</sup>",
               "x": 0.01},
        xaxis={"title": "Final margin (points)", "dtick": 1}, yaxis_title="Games (%)",
        height=460, bargap=0.15, legend={"orientation": "h", "y": -0.18, "x": 0}, **LAYOUT)
    return fig


def plot_line_calibration(cal: dict[str, pd.DataFrame]) -> go.Figure:
    """Predicted minus actual rate of going over lines set around each prediction.

    ``cal`` maps a series name to ``totals.line_calibration`` output; names
    starting with 'Margin' go in the left panel, the rest in the right.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=["Home margin", "Total"])
    for (name, rows), color in zip(cal.items(), [SERIES[1], SERIES[0], SERIES[2]]):
        col = 1 if name.startswith("Margin") else 2
        gap = 100 * (rows["predicted"] - rows["actual"])
        fig.add_trace(go.Scatter(
            x=rows["offset"], y=gap, name=name, mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color, "line": {"color": "white", "width": 2}},
            customdata=np.column_stack([100 * rows["predicted"], 100 * rows["actual"]]),
            hovertemplate=f"<b>{name}</b><br>Line at prediction %{{x:+d}}<br>"
                          "Predicted %{customdata[0]:.1f}%, actual %{customdata[1]:.1f}%<br>"
                          "Gap %{y:+.2f} points<extra></extra>"), row=1, col=col)
        fig.add_hline(y=0, line={"color": MUTED, "width": 1}, row=1, col=col)
    fig.update_xaxes(title_text="Line minus prediction (points)", dtick=5)
    fig.update_yaxes(title_text="Predicted minus actual (percentage points)", row=1, col=1)
    fig.update_layout(
        title={"text": "Calibration of over probabilities<br><sup>For lines set from 15 below "
                       "to 15 above each game's prediction: predicted minus actual share of "
                       "games over the line</sup>", "x": 0.01},
        height=440, legend={"orientation": "h", "y": -0.2, "x": 0}, **LAYOUT)
    return fig


def plot_total_pit(pit: pd.DataFrame) -> go.Figure:
    """Share of totals in each decile of the predicted distribution."""
    fig = go.Figure(go.Bar(
        x=pit["decile"], y=100 * pit["share"], marker={"color": SERIES[0], "cornerradius": 4},
        hovertemplate="Decile %{x}: %{y:.1f}% of games<extra></extra>"))
    fig.add_hline(y=10, line={"color": MUTED, "width": 1, "dash": "dash"},
                  annotation_text="Calibrated (10%)", annotation_position="top left",
                  annotation_font={"size": 10, "color": MUTED})
    fig.update_layout(
        title={"text": "Where actual totals fall in the predicted distribution<br><sup>Share "
                       "of games in each tenth of the predicted distribution</sup>", "x": 0.01},
        xaxis={"title": "Decile of the predicted distribution", "dtick": 1},
        yaxis={"title": "Games (%)", "range": [0, 13]}, height=400, bargap=0.15, **LAYOUT)
    return fig


def plot_game(row: pd.Series, weights: np.ndarray) -> go.Figure:
    """One game's home margin and total distributions."""
    p = margin_pmf(row["margin_pred"], row["margin_sd"], weights)[0]
    keep = np.abs(SUPPORT - row["margin_pred"]) <= 35
    totals_x = np.arange(int(row["total_pred"] - 55), int(row["total_pred"] + 56))
    pt = stats.norm.pdf(totals_x, row["total_pred"], row["total_sd"])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=[f"{row['home']} margin", "Total"])
    fig.add_trace(go.Bar(x=SUPPORT[keep], y=100 * p[keep], showlegend=False,
                         marker={"color": np.where(SUPPORT[keep] > 0, SERIES[0], SERIES[1]),
                                 "cornerradius": 2},
                         hovertemplate=f"{row['home']} by %{{x}}: %{{y:.1f}}%<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=totals_x, y=100 * pt, showlegend=False,
                         marker={"color": SERIES[0], "cornerradius": 2},
                         hovertemplate="Total %{x}: %{y:.2f}%<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_layout(
        title={"text": f"{row['away']} at {row['home']}: predicted {row['home']} by "
                       f"{row['margin_pred']:+.1f}, total {row['total_pred']:.1f}", "x": 0.01},
        height=400, bargap=0.05, **LAYOUT)
    return fig
