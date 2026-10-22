"""Interactive charts for the game outcome model (T3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nba_analytics.config import season_label

# Validated categorical slots, in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#9a9a9a"
LAYOUT = {"template": "plotly_white", "hoverlabel": {"bgcolor": "white"}, "width": 1000}


def plot_home_court(games: pd.DataFrame) -> go.Figure:
    """Home win rate and average home margin by season (non-neutral games)."""
    g = games[~games["neutral"]].groupby("season").agg(
        win=("home_win", "mean"), margin=("margin", "mean"), n=("home_win", "size"))
    labels = g.index.map(season_label)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Home win rate (%)",
                                                        "Average home margin (points)"],
                        horizontal_spacing=0.08)
    for col, (y, fmt) in enumerate([(g["win"] * 100, ".1f"), (g["margin"], ".2f")], start=1):
        fig.add_trace(go.Scatter(
            x=labels, y=y, mode="lines+markers", line={"color": SERIES[0], "width": 2},
            marker={"size": 8, "color": SERIES[0], "line": {"color": "white", "width": 2}},
            showlegend=False, customdata=g["n"],
            hovertemplate=f"<b>%{{x}}</b>: %{{y:{fmt}}} (%{{customdata:,}} games)<extra></extra>",
        ), row=1, col=col)
    fig.update_xaxes(type="category", tickangle=-45, tickfont={"size": 9})
    fig.update_yaxes(range=[50, 65], row=1, col=1)
    fig.update_yaxes(range=[0, 4.5], row=1, col=2)
    fig.update_layout(
        title={"text": "Home-court advantage by season<br><sup>Neutral-site and 2020 bubble "
                       "games excluded; 2020-21 was played mostly without fans</sup>",
               "x": 0.01}, height=440, **LAYOUT)
    return fig


def plot_feature_sets(comp: pd.DataFrame, baselines: dict[str, float]) -> go.Figure:
    """Walk-forward log loss as feature families are added, with reference lines."""
    fig = go.Figure(go.Scatter(
        y=comp["features"][::-1], x=comp["log_loss"][::-1], mode="markers+text",
        marker={"size": 11, "color": SERIES[0], "line": {"color": "white", "width": 2}},
        text=comp["log_loss"][::-1].map("{:.4f}".format), textposition="middle right",
        textfont={"size": 10},
        customdata=np.column_stack([comp["accuracy"][::-1] * 100, comp["brier"][::-1]]),
        hovertemplate="<b>%{y}</b><br>Log loss %{x:.4f}<br>Brier %{customdata[1]:.4f}<br>"
                      "Accuracy %{customdata[0]:.1f}%<extra></extra>",
    ))
    for (name, value), color in zip(baselines.items(), [MUTED, SERIES[1]]):
        fig.add_vline(x=value, line={"color": color, "width": 1.5, "dash": "dash"},
                      annotation_text=f"{name} {value:.4f}", annotation_position="top",
                      annotation_font={"size": 10, "color": color})
    lo = min(comp["log_loss"].min(), *baselines.values())
    hi = max(comp["log_loss"].max(), *baselines.values())
    fig.update_layout(
        title={"text": "Out-of-sample log loss as each feature family is added<br><sup>"
                       "Logistic regression, walk-forward by season; lower is better</sup>",
               "x": 0.01},
        xaxis={"title": "Log loss", "range": [lo - 0.004, hi + 0.008]}, height=480,
        margin={"t": 110}, **LAYOUT)
    return fig


def plot_importance(imp: pd.DataFrame) -> go.Figure:
    """Rise in log loss when each feature family is dropped from the full model."""
    rows = imp.sort_values("log_loss_rise")
    fig = go.Figure(go.Bar(
        y=rows["family"], x=rows["log_loss_rise"] * 1000, orientation="h",
        marker={"color": SERIES[0], "cornerradius": 4},
        customdata=rows["accuracy_drop"] * 100,
        hovertemplate="<b>%{y}</b><br>Log loss +%{x:.2f} (x 1,000)<br>"
                      "Accuracy %{customdata:+.2f} pts lower<extra></extra>",
    ))
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    fig.update_layout(
        title={"text": "What each feature family adds<br><sup>Rise in walk-forward log loss "
                       "(x 1,000) when the family is removed from the full model</sup>",
               "x": 0.01},
        xaxis_title="Log loss increase (x 1,000)", height=440, **LAYOUT)
    return fig


def plot_scores_by_season(by_season: pd.DataFrame, models: list[str]) -> go.Figure:
    """Walk-forward log loss by season for several models."""
    fig = go.Figure()
    for name, color in zip(models, SERIES):
        rows = by_season[by_season["model"] == name]
        fig.add_trace(go.Scatter(
            x=rows["season"].map(season_label), y=rows["log_loss"], name=name,
            mode="lines+markers", line={"color": color, "width": 2},
            marker={"size": 8, "color": color, "line": {"color": "white", "width": 2}},
            customdata=rows["accuracy"] * 100,
            hovertemplate=f"<b>{name}</b> %{{x}}: log loss %{{y:.4f}}, "
                          "accuracy %{customdata:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "Out-of-sample log loss by season", "x": 0.01},
        xaxis={"type": "category"}, yaxis_title="Log loss", height=460, hovermode="x unified",
        legend={"orientation": "h", "y": -0.15, "x": 0}, **LAYOUT)
    return fig


def plot_calibration(cal: dict[str, pd.DataFrame]) -> go.Figure:
    """Predicted vs. actual home win rate by probability bin, one series per model."""
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                               line={"color": MUTED, "width": 1, "dash": "dash"},
                               name="Perfect calibration"))
    for (name, rows), color in zip(cal.items(), SERIES):
        fig.add_trace(go.Scatter(
            x=rows["predicted"], y=rows["actual"], name=name, mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 9, "color": color, "line": {"color": "white", "width": 2}},
            customdata=rows["n"],
            hovertemplate=f"<b>{name}</b><br>Predicted %{{x:.1%}}<br>Actual %{{y:.1%}}<br>"
                          "%{customdata:,} games<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "Calibration: predicted vs. actual home win rate<br><sup>Walk-forward "
                       "predictions in 10 probability bins</sup>", "x": 0.01},
        xaxis={"title": "Predicted home win probability", "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": "Actual home win rate", "range": [0, 1], "tickformat": ".0%"},
        width=640, height=600, template="plotly_white", hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.15, "x": 0})
    return fig


def plot_situations(res: pd.DataFrame, groups: dict[str, pd.Series]) -> go.Figure:
    """Average margin residual (actual minus expected, home side) for game situations.

    ``groups`` maps a panel title to a Series of situation labels aligned with ``res``.
    """
    sizes = [pd.Series(labels).nunique() for labels in groups.values()]
    fig = make_subplots(rows=len(groups), cols=1, subplot_titles=list(groups),
                        vertical_spacing=0.08, row_heights=sizes)
    for i, (title, labels) in enumerate(groups.items(), start=1):
        g = res.groupby(labels, observed=True, sort=True)["residual"].agg(
            ["mean", "sem", "size"])
        fig.add_trace(go.Bar(
            y=g.index.astype(str), x=g["mean"], orientation="h", showlegend=False, width=0.6,
            marker={"color": [SERIES[0] if v >= 0 else SERIES[1] for v in g["mean"]],
                    "cornerradius": 4},
            error_x={"type": "data", "array": 1.96 * g["sem"], "color": MUTED, "thickness": 1},
            customdata=g["size"],
            hovertemplate="<b>%{y}</b><br>%{x:+.2f} points vs. expected (%{customdata:,} "
                          "games)<extra></extra>",
        ), row=i, col=1)
        fig.add_vline(x=0, line={"color": MUTED, "width": 1}, row=i, col=1)
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_xaxes(title_text="Home margin vs. expected (points)", title_font={"size": 10})
    fig.update_layout(
        title={"text": "How game situations move the result beyond team strength<br><sup>"
                       "Actual minus expected home margin (expected from Elo and team ratings), "
                       "centered within each season; bars show 95% intervals</sup>", "x": 0.01},
        height=30 * sum(sizes) + 120 * len(groups) + 80, **LAYOUT)
    return fig


def plot_talent_missing(res: pd.DataFrame, bins: list[float]) -> go.Figure:
    """Margin residual against the difference in talent lost to absences."""
    gap = res["away_talent_missing"] - res["home_talent_missing"]
    b = pd.cut(gap, bins)
    g = res.groupby(b, observed=True).agg(gap=("residual", lambda s: gap[s.index].mean()),
                                          residual=("residual", "mean"),
                                          sem=("residual", "sem"), n=("residual", "size"))
    fig = go.Figure(go.Scatter(
        x=g["gap"], y=g["residual"], mode="lines+markers", line={"color": SERIES[0], "width": 2},
        marker={"size": 9, "color": SERIES[0], "line": {"color": "white", "width": 2}},
        error_y={"type": "data", "array": 1.96 * g["sem"], "color": MUTED, "thickness": 1},
        customdata=g["n"],
        hovertemplate="Away minus home talent missing %{x:+.1f}<br>Home margin %{y:+.2f} vs. "
                      "expected (%{customdata:,} games)<extra></extra>",
    ))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_layout(
        title={"text": "Margin vs. expected by talent lost to absences<br><sup>Talent lost "
                       "to absences, away minus home (net-rating points), against the margin "
                       "residual</sup>", "x": 0.01},
        xaxis_title="Away minus home talent missing (points per 100 possessions)",
        yaxis_title="Home margin vs. expected (points)", height=480, **LAYOUT)
    return fig
