"""Interactive charts for the per-game player projections (P4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from nba_analytics.players.props import STATS

# Validated categorical slots, in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#9a9a9a"
LAYOUT = {"template": "plotly_white", "hoverlabel": {"bgcolor": "white"}, "width": 1000}


def plot_mae(ev: pd.DataFrame) -> go.Figure:
    """MAE by stat and method, as percent change from the season-average baseline."""
    base = ev[ev["method"] == "Season average so far"].set_index("stat")["mae"]
    rows = ev[(ev["method"] != "Season average so far") & ev["stat"].isin(base.index)].copy()
    rows["change"] = 100 * (rows["mae"] / rows["stat"].map(base) - 1)
    order = [s for s in STATS.values() if s in base.index][::-1]
    fig = go.Figure()
    methods = sorted(dict.fromkeys(rows["method"]), key=lambda m: m != "Model")
    for method, color in zip(methods, SERIES):
        r = rows[rows["method"] == method].set_index("stat").loc[order].reset_index()
        fig.add_trace(go.Bar(
            y=r["stat"], x=r["change"], name=method, orientation="h",
            marker={"color": color, "cornerradius": 4, "line": {"color": "white", "width": 2}},
            customdata=np.column_stack([r["mae"], r["stat"].map(base)]),
            hovertemplate=f"<b>{method}</b><br>%{{y}}: MAE %{{customdata[0]:.3f}} vs. "
                          "%{customdata[1]:.3f} (%{x:+.1f}%)<extra></extra>",
        ))
    fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    fig.update_layout(
        title={"text": "Per-game error against the season average so far<br><sup>Change in "
                       "walk-forward MAE; negative = more accurate</sup>", "x": 0.01},
        barmode="group", xaxis_title="Change in MAE (%)", height=460,
        legend={"orientation": "h", "y": -0.15, "x": 0, "traceorder": "normal"}, **LAYOUT)
    return fig


def plot_over_calibration(cal: dict[str, pd.DataFrame]) -> go.Figure:
    """Predicted vs. actual probability of reaching a line, one panel per stat."""
    names = list(cal)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=[STATS[s] for s in names],
                        horizontal_spacing=0.04, shared_yaxes=True)
    for i, s in enumerate(names, start=1):
        rows = cal[s]
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", hoverinfo="skip",
                                 showlegend=False,
                                 line={"color": MUTED, "width": 1, "dash": "dash"}),
                      row=1, col=i)
        fig.add_trace(go.Scatter(
            x=rows["predicted"], y=rows["actual"], mode="lines+markers", showlegend=False,
            line={"color": SERIES[0], "width": 2},
            marker={"size": 7, "color": SERIES[0], "line": {"color": "white", "width": 1.5}},
            customdata=rows["n"],
            hovertemplate=f"<b>{STATS[s]}</b><br>Predicted %{{x:.1%}}<br>Actual %{{y:.1%}}<br>"
                          "%{customdata:,} player-games x lines<extra></extra>",
        ), row=1, col=i)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Predicted",
                     title_font={"size": 10})
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    fig.update_layout(
        title={"text": "Calibration of over probabilities<br><sup>Negative binomial "
                       "probability of reaching each line, pooled over the lines listed above, "
                       "in 10 bins</sup>", "x": 0.01}, height=360, **LAYOUT)
    return fig


def plot_distribution(row: pd.Series, stat: str, lines: list[float]) -> go.Figure:
    """The projected distribution for one player-game, with the actual result marked."""
    mu, r = row[f"{stat}_pred"], row[f"{stat}_size"]
    k = np.arange(0, int(stats.nbinom.ppf(0.995, r, r / (r + mu))) + 1)
    pmf = stats.nbinom.pmf(k, r, r / (r + mu))
    actual = int(row[stat])
    fig = go.Figure(go.Bar(
        x=k, y=pmf * 100,
        marker={"color": [SERIES[1] if v == actual else SERIES[0] for v in k],
                "cornerradius": 2},
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>", showlegend=False,
    ))
    for line in lines:
        p = stats.nbinom.sf(np.ceil(line) - 1, r, r / (r + mu))
        fig.add_vline(x=line - 0.5, line={"color": MUTED, "width": 1, "dash": "dot"},
                      annotation_text=f"{line}+: {p:.0%}", annotation_position="top",
                      annotation_font={"size": 10})
    fig.update_layout(
        title={"text": f"{row['player']} {STATS[stat].lower()} vs. {row['opp_id']}, "
                       f"{row['date']}<br><sup>Projected mean {mu:.1f} on {row['mp_pred']:.0f} "
                       f"projected minutes; actual {actual} (orange)</sup>", "x": 0.01},
        xaxis_title=STATS[stat], yaxis_title="Probability (%)", height=420, bargap=0.1,
        margin={"t": 110}, **LAYOUT)
    return fig


def plot_residual_by(preds: pd.DataFrame, panels: dict[str, tuple[str, pd.Series]]
                     ) -> go.Figure:
    """Actual minus season average so far, grouped by a situation, for a few stats.

    ``panels`` maps a panel title to (stat, labels aligned with ``preds``).
    """
    fig = make_subplots(rows=len(panels), cols=1, subplot_titles=list(panels),
                        vertical_spacing=0.14)
    for i, (title, (stat, labels)) in enumerate(panels.items(), start=1):
        resid = preds[stat] - preds[f"{stat}_avg"]
        g = resid.groupby(labels, observed=True).agg(["mean", "sem", "size"])
        fig.add_trace(go.Bar(
            x=g.index.astype(str), y=g["mean"], showlegend=False,
            marker={"color": [SERIES[0] if v >= 0 else SERIES[1] for v in g["mean"]],
                    "cornerradius": 4},
            error_y={"type": "data", "array": 1.96 * g["sem"], "color": MUTED, "thickness": 1},
            customdata=g["size"],
            hovertemplate=f"<b>%{{x}}</b><br>{STATS[stat]} %{{y:+.2f}} vs. season average "
                          "(%{customdata:,} player-games)<extra></extra>",
        ), row=i, col=1)
        fig.add_hline(y=0, line={"color": MUTED, "width": 1}, row=i, col=1)
        fig.update_yaxes(title_text=f"{STATS[stat]} vs. average", title_font={"size": 10},
                         row=i, col=1)
    fig.update_layout(
        title={"text": "Game situations and player output<br><sup>Actual minus the player's "
                       "season average so far; bars show 95% intervals</sup>", "x": 0.01},
        height=300 * len(panels) + 80, **LAYOUT)
    return fig
