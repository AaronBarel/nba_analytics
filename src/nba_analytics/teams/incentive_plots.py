"""Interactive charts for the incentive-to-win factor (T4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nba_analytics.teams.games_plots import LAYOUT, MUTED, SERIES

MARKER = {"size": 8, "line": {"color": "white", "width": 2}}


def _line(x, y, name, color, hover, **kw) -> go.Scatter:
    return go.Scatter(x=x, y=y, name=name, mode="lines+markers", line={"color": color, "width": 2},
                      marker={**MARKER, "color": color}, hovertemplate=hover, **kw)


def plot_lottery_odds(odds: dict[str, np.ndarray]) -> go.Figure:
    """Chance of the first pick and expected pick by lottery rank, for each system."""
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=["Chance of the 1st pick (%)", "Expected pick"])
    for (name, o), color in zip(odds.items(), SERIES):
        rank = np.arange(1, len(o) + 1)
        expected = (o * np.arange(1, o.shape[1] + 1)).sum(axis=1)
        fig.add_trace(_line(rank, 100 * o[:, 0], name, color, f"<b>{name}</b> rank %{{x}}: "
                            "%{y:.1f}%<extra></extra>", legendgroup=name), row=1, col=1)
        fig.add_trace(_line(rank, expected, name, color, f"<b>{name}</b> rank %{{x}}: pick "
                            "%{y:.2f}<extra></extra>", legendgroup=name, showlegend=False),
                      row=1, col=2)
    fig.update_xaxes(title_text="Lottery rank (1 = worst record)", dtick=1)
    fig.update_yaxes(rangemode="tozero", row=1, col=1)
    fig.update_yaxes(range=[0, 16], row=1, col=2)
    fig.update_layout(
        title={"text": "Lottery odds by rank<br><sup>3-2-1 ranks: 1-3 relegated, 4-10 other "
                       "non-play-in teams, 11-14 the 9 and 10 seeds, 15-16 the 7-vs-8 losers"
                       "</sup>", "x": 0.01},
        height=440, legend={"orientation": "h", "y": -0.2}, **LAYOUT)
    return fig


def plot_pick_value(outcomes: pd.DataFrame, curve: np.ndarray) -> go.Figure:
    """Average four-season VORP by draft slot, with a 95% interval, and the fitted curve."""
    g = outcomes.groupby("pick")["vorp"].agg(["mean", "sem", "count"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=g.index, y=g["mean"], mode="markers", name="Average",
        marker={**MARKER, "size": 9, "color": SERIES[0]},
        error_y={"type": "data", "array": 1.96 * g["sem"], "color": MUTED, "thickness": 1},
        customdata=g["count"],
        hovertemplate="<b>Pick %{x}</b>: %{y:.2f} VORP (%{customdata} players)<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.arange(1, 31), y=curve, mode="lines", name="Fitted curve",
                             line={"color": SERIES[1], "width": 2},
                             hovertemplate="<b>Pick %{x}</b>: %{y:.2f} fitted<extra></extra>"))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_xaxes(title_text="Pick", dtick=1)
    fig.update_yaxes(title_text="VORP, first four seasons")
    fig.update_layout(
        title={"text": "Value of a first-round pick by slot<br><sup>1995-2022 drafts; VORP summed "
                       "over the four seasons after the draft, zero for players who never "
                       "played</sup>", "x": 0.01},
        height=460, legend={"orientation": "h", "y": -0.2}, **LAYOUT)
    return fig


def plot_value_by_rank(values: dict[str, np.ndarray]) -> go.Figure:
    """Expected value of a team's own pick by lottery rank, for each system."""
    fig = go.Figure()
    for (name, v), color in zip(values.items(), SERIES):
        fig.add_trace(_line(np.arange(1, len(v) + 1), v, name, color,
                            f"<b>{name}</b> rank %{{x}}: %{{y:.2f}} VORP<extra></extra>"))
    fig.update_xaxes(title_text="Lottery rank (1 = worst record)", dtick=1)
    fig.update_yaxes(title_text="Expected VORP of the pick", rangemode="tozero")
    fig.update_layout(title={"text": "Expected pick value by lottery rank", "x": 0.01},
                      height=440, legend={"orientation": "h", "y": -0.2}, **LAYOUT)
    return fig


def incentive_by_rank(replay: pd.DataFrame) -> pd.DataFrame:
    """Average draft incentive by projected final league rank (1 = fewest projected wins)."""
    r = replay.assign(rank=replay.groupby(["season", "date"])["proj_wins"].rank(method="first"))
    return r.groupby("rank")["draft_incentive"].agg(["mean", "sem", "count"])


def plot_incentive_by_rank(curves: dict[str, pd.DataFrame]) -> go.Figure:
    """Average draft incentive by projected rank, one line per lottery system."""
    fig = go.Figure()
    for (name, g), color in zip(curves.items(), SERIES):
        fig.add_trace(_line(
            g.index, g["mean"], name, color,
            f"<b>{name}</b> rank %{{x}}: %{{y:.3f}} VORP (%{{customdata:,}} team-weeks)"
            "<extra></extra>", customdata=g["count"],
            error_y={"type": "data", "array": 1.96 * g["sem"], "color": color, "thickness": 1}))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_xaxes(title_text="Projected final record rank (1 = fewest wins)", dtick=1)
    fig.update_yaxes(title_text="Draft value of winning minus losing (VORP)")
    fig.update_layout(
        title={"text": "Draft incentive by projected record<br><sup>Weekly from January 1, "
                       "every team owning its pick; 95% intervals</sup>", "x": 0.01},
        height=480, legend={"orientation": "h", "y": -0.2}, **LAYOUT)
    return fig


def plot_residuals(binned: pd.DataFrame, label: str) -> go.Figure:
    """Average home result minus predicted win probability by incentive bin."""
    fig = go.Figure(go.Scatter(
        x=binned["mid"], y=100 * binned["mean"], mode="markers",
        marker={**MARKER, "size": 10, "color": SERIES[0]},
        error_y={"type": "data", "array": 196 * binned["sem"], "color": MUTED, "thickness": 1},
        customdata=np.column_stack([binned["bin"].astype(str), binned["count"]]),
        hovertemplate="<b>%{customdata[0]}</b><br>%{y:.1f} points of win probability "
                      "(%{customdata[1]} games)<extra></extra>"))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Home wins minus predicted (% points)")
    fig.update_layout(
        title={"text": "Game model residuals by incentive gap<br><sup>Walk-forward logistic "
                       "model, games from January 1, 2015-16 to 2025-26; 95% intervals</sup>",
               "x": 0.01}, height=440, **LAYOUT)
    return fig


def plot_current(cur: pd.DataFrame) -> go.Figure:
    """Each team's draft and playoff incentive now, sorted by the draft incentive."""
    c = cur.sort_values("draft_incentive", ascending=False)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=["Draft value (VORP)", "Playoff chance (% points)"])
    hover = np.column_stack([c["wins"].astype(int), c["losses"].astype(int), c["proj_wins"],
                             100 * c["playoffs"], c["exp_pick"], c["draft_value"]])
    tip = ("<b>%{y}</b> %{customdata[0]}-%{customdata[1]}, projected %{customdata[2]:.1f} wins"
           "<br>Playoffs %{customdata[3]:.0f}%, own pick %{customdata[4]:.1f}, "
           "holdings %{customdata[5]:.2f} VORP<br>")
    for col, (x, fmt) in enumerate([(c["draft_incentive"], ".3f"),
                                    (100 * c["playoff_incentive"], ".1f")], start=1):
        fig.add_trace(go.Scatter(
            x=x, y=c["team_id"], mode="markers", showlegend=False, customdata=hover,
            marker={**MARKER, "size": 10, "color": SERIES[col - 1]},
            hovertemplate=tip + f"Win minus loss: %{{x:{fmt}}}<extra></extra>"), row=1, col=col)
        fig.add_vline(x=0, line={"color": MUTED, "width": 1}, row=1, col=col)
    fig.update_yaxes(autorange="reversed", tickfont={"size": 10})
    date = cur["date"].iloc[0]
    fig.update_layout(
        title={"text": f"What winning the next game is worth, {date}<br><sup>2027 lottery "
                       "(3-2-1) with each team's actual first-round holdings; winning minus "
                       "losing</sup>", "x": 0.01}, height=760, **LAYOUT)
    return fig


def plot_holdings(cur: pd.DataFrame) -> go.Figure:
    """Expected value of the 2027 first-round picks each team owns."""
    c = cur.sort_values("draft_value")
    fig = go.Figure(go.Bar(
        x=c["draft_value"], y=c["team_id"], orientation="h", marker={"color": SERIES[0]},
        hovertemplate="<b>%{y}</b>: %{x:.2f} VORP<extra></extra>"))
    fig.update_xaxes(title_text="Expected VORP of 2027 firsts owned", rangemode="tozero")
    fig.update_yaxes(tickfont={"size": 10})
    fig.update_layout(title={"text": f"2027 first-round holdings, {cur['date'].iloc[0]}",
                             "x": 0.01}, height=720, bargap=0.25, **LAYOUT)
    return fig
