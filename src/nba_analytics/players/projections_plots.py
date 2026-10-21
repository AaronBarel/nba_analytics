"""Interactive charts for next-season projections (P2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nba_analytics.config import season_label
from nba_analytics.players.projections import AGE_RANGE, TARGETS

# Validated categorical slots, in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#9a9a9a"
METHODS = {"last": "Last season", "marcel": "Marcel baseline", "model": "Model"}


def plot_mae_reduction(scores: pd.DataFrame) -> go.Figure:
    """Percent reduction in MAE vs. repeating last season, for Marcel and the model."""
    t = scores.pivot(index="target", columns="method", values="mae").loc[list(TARGETS.values())]
    t = t.iloc[::-1]
    fig = go.Figure()
    for method, color in [("marcel", SERIES[0]), ("model", SERIES[1])]:
        pct = (1 - t[method] / t["last"]) * 100
        fig.add_trace(go.Bar(
            y=t.index, x=pct, name=METHODS[method], orientation="h",
            marker={"color": color, "cornerradius": 4, "line": {"color": "white", "width": 2}},
            customdata=np.column_stack([t[method], t["last"]]),
            text=pct.map("{:.0f}%".format), textposition="outside",
            hovertemplate=f"<b>{METHODS[method]}</b><br>%{{y}}<br>MAE %{{customdata[0]:.3f}} vs. "
                          "%{customdata[1]:.3f} last season (%{x:.1f}% lower)<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "Error reduction vs. repeating last season<br><sup>Walk-forward "
                       "2013-14 to 2025-26; rate stats minutes-weighted, players with 500+ "
                       "minutes</sup>", "x": 0.01},
        barmode="group", bargap=0.2, xaxis_title="MAE reduction (%)", template="plotly_white",
        width=1000, height=720, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.1, "x": 0, "traceorder": "normal"},
    )
    return fig


def plot_mae_by_season(by_season: pd.DataFrame, target: str) -> go.Figure:
    """MAE of each method by test season for one target."""
    label = TARGETS[target]
    rows = by_season[by_season["target"] == label]
    fig = go.Figure()
    for (method, name), color in zip(METHODS.items(), [MUTED, SERIES[0], SERIES[1]]):
        r = rows[rows["method"] == method].sort_values("season")
        fig.add_trace(go.Scatter(
            x=r["season"].map(season_label), y=r["mae"], mode="lines+markers", name=name,
            line={"color": color, "width": 2}, marker={"size": 8, "color": color},
            customdata=r["n"],
            hovertemplate=f"<b>{name}</b> %{{x}}<br>MAE %{{y:.3f}} (n=%{{customdata}})"
                          "<extra></extra>",
        ))
    fig.update_layout(
        title={"text": f"{label}: walk-forward MAE by season", "x": 0.01},
        yaxis_title="MAE", template="plotly_white", width=1000, height=440,
        hoverlabel={"bgcolor": "white"}, hovermode="x unified",
        legend={"orientation": "h", "y": -0.18, "x": 0},
    )
    return fig


def plot_age_curves(curves: pd.DataFrame, stats: dict[str, str], peak_age: int = 27) -> go.Figure:
    """Cumulative delta-method aging curves, set to zero at ``peak_age``."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(stats.values()),
                        horizontal_spacing=0.1, vertical_spacing=0.16)
    ages = curves.index
    for i, (col, label) in enumerate(stats.items()):
        # curves[age] is the change from age-1 to age, so the level at age a is the cumsum
        level = curves[col].cumsum()
        level = level - level.loc[peak_age]
        fig.add_trace(go.Scatter(
            x=ages, y=level, mode="lines+markers", showlegend=False,
            line={"color": SERIES[0], "width": 2}, marker={"size": 6, "color": SERIES[0]},
            hovertemplate=f"<b>{label}</b><br>Age %{{x}}: %{{y:+.2f}} vs. age {peak_age}"
                          "<extra></extra>",
        ), row=i // 2 + 1, col=i % 2 + 1)
        fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dash"},
                      row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(range=[AGE_RANGE[0] - 0.5, AGE_RANGE[1] + 0.5])
    fig.update_layout(
        title={"text": f"Aging curves (delta method), relative to age {peak_age}<br><sup>"
                       "Average change between consecutive seasons, minutes-weighted, "
                       "2005-06 to 2025-26</sup>", "x": 0.01},
        template="plotly_white", width=1000, height=680, hoverlabel={"bgcolor": "white"},
    )
    return fig


def plot_projection_bands(proj: pd.DataFrame, target: str, n: int = 30) -> go.Figure:
    """Top ``n`` projections with 80% bands from walk-forward residuals."""
    rows = proj.nlargest(n, target).iloc[::-1]
    label = TARGETS[target]
    last = f"{target}_1"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rows[f"{target}_hi"], y=rows["player"], mode="markers", showlegend=False,
        hoverinfo="skip", marker={"size": 0.1, "color": "rgba(0,0,0,0)"},
        error_x={"type": "data", "symmetric": False, "color": "#b7d3f6", "thickness": 6,
                 "width": 0, "arrayminus": rows[f"{target}_hi"] - rows[f"{target}_lo"],
                 "array": np.zeros(len(rows))},
    ))
    if last in rows:
        fig.add_trace(go.Scatter(
            x=rows[last], y=rows["player"], mode="markers", name="2025-26 actual",
            marker={"size": 9, "color": "white", "line": {"color": SERIES[1], "width": 2}},
            hovertemplate="<b>%{y}</b><br>2025-26: %{x:.1f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=rows[target], y=rows["player"], mode="markers", name="2026-27 projection",
        marker={"size": 10, "color": SERIES[0], "line": {"color": "white", "width": 2}},
        customdata=np.column_stack([rows[f"{target}_lo"], rows[f"{target}_hi"], rows["age"]]),
        hovertemplate="<b>%{y}</b> (age %{customdata[2]:.0f})<br>Projection %{x:.1f}<br>"
                      "80% band %{customdata[0]:.1f} to %{customdata[1]:.1f}<extra></extra>",
    ))
    fig.update_layout(
        title={"text": f"2026-27 {label}: top {n} projections<br><sup>Bar = 80% band from "
                       "walk-forward errors for players with similar prior minutes</sup>",
               "x": 0.01},
        xaxis_title=label, template="plotly_white", width=1000,
        height=26 * len(rows) + 200, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.06, "x": 0},
    )
    return fig


AGE_GROUPS = [("24 and under", 0, 24), ("25-30", 25, 30), ("31 and over", 31, 99)]


def plot_projected_change(proj: pd.DataFrame, target: str = "bpm", min_mpg: float = 20,
                          min_prior_minutes: float = 1000, n_labels: int = 8) -> go.Figure:
    """2025-26 value vs. 2026-27 projection, colored by age next season."""
    rows = proj[(proj["mp_per_g"] >= min_mpg) & (proj["mp_1"] >= min_prior_minutes)].copy()
    rows["change"] = rows[target] - rows[f"{target}_1"]
    label = TARGETS[target]
    fig = go.Figure()
    for (name, lo, hi), color in zip(AGE_GROUPS, SERIES):
        r = rows[rows["age"].between(lo, hi)]
        fig.add_trace(go.Scatter(
            x=r[f"{target}_1"], y=r[target], mode="markers", name=name,
            marker={"size": 8, "color": color, "opacity": 0.85,
                    "line": {"color": "white", "width": 1}},
            customdata=np.column_stack([r["player"], r["age"], r["mp_per_g"], r["change"]]),
            hovertemplate="<b>%{customdata[0]}</b> (age %{customdata[1]:.0f} next season)<br>"
                          "2025-26 %{x:.1f} → 2026-27 %{y:.1f} (%{customdata[3]:+.1f})<br>"
                          "Projected %{customdata[2]:.1f} MPG<extra></extra>",
        ))
    lo, hi = rows[[target, f"{target}_1"]].min().min(), rows[[target, f"{target}_1"]].max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             hoverinfo="skip", line={"color": MUTED, "width": 1, "dash": "dash"}))
    for _, r in rows.loc[rows["change"].abs().nlargest(n_labels).index].iterrows():
        # Risers sit above the no-change line, decliners below: label away from it
        fig.add_annotation(x=r[f"{target}_1"], y=r[target], text=r["player"], showarrow=True,
                           arrowhead=0, arrowwidth=0.6, arrowcolor="#777777", ax=0,
                           ay=-18 if r["change"] > 0 else 18,
                           font={"size": 10, "color": "#222222"})
    fig.update_layout(
        title={"text": f"{label}: 2025-26 actual vs. 2026-27 projection (n={len(rows)})<br>"
                       f"<sup>{min_prior_minutes:,.0f}+ minutes in 2025-26 and projected "
                       f"{min_mpg:.0f}+ MPG; dashed line = no change; labels = largest "
                       "projected changes</sup>", "x": 0.01},
        xaxis_title=f"2025-26 {label}", yaxis_title=f"2026-27 projected {label}",
        template="plotly_white", width=1000, height=700, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.12, "x": 0},
    )
    return fig
