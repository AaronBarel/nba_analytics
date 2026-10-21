"""Interactive charts for the college-to-NBA rookie model (R1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nba_analytics.players.rookies import FEATURE_SETS, TARGETS

# Validated categorical slots, in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRID = "#9a9a9a"
DRAFT_TIERS = [("Lottery (1-14)", 1, 14), ("Picks 15-30", 15, 30), ("Second round", 31, 60),
               ("Undrafted", 61, 61)]


def plot_cv_scores(scores: pd.DataFrame) -> go.Figure:
    """Leave-one-class-out R² by target, one bar per feature set."""
    fig = go.Figure()
    order = list(TARGETS.values())[::-1]
    for (name, color) in zip(FEATURE_SETS, SERIES):
        rows = scores[scores["features"] == name].set_index("target").loc[order].reset_index()
        fig.add_trace(go.Bar(
            y=rows["target"], x=rows["r2"], name=name, orientation="h",
            marker={"color": color, "cornerradius": 4, "line": {"color": "white", "width": 2}},
            customdata=rows[["mae", "n"]], text=rows["r2"].map("{:.2f}".format),
            textposition="outside",
            hovertemplate=f"<b>{name}</b><br>%{{y}}<br>R² %{{x:.3f}} · MAE %{{customdata[0]:.2f}}"
                          " · n=%{customdata[1]}<extra></extra>",
        ))
    fig.update_layout(
        title={"text": "Out-of-sample R² by target and feature set<br><sup>Each draft class "
                       "is predicted by a model trained on the other classes</sup>", "x": 0.01},
        barmode="group", bargap=0.25, xaxis={"title": "R²", "range": [0, 1]},
        template="plotly_white", width=1000, height=560, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.15, "x": 0, "traceorder": "normal"},
    )
    return fig


def _tier(pick: pd.Series) -> pd.Series:
    return pd.Series(np.select([pick.between(lo, hi) for _, lo, hi in DRAFT_TIERS],
                               [name for name, _, _ in DRAFT_TIERS], default=""),
                     index=pick.index)


def plot_predicted_vs_actual(df: pd.DataFrame, pred: pd.Series, target: str,
                             n_labels: int = 10) -> go.Figure:
    """Out-of-fold prediction vs. outcome, colored by draft tier."""
    sub = df.loc[pred.index].assign(pred=pred, tier=lambda d: _tier(d["draft_pick"]))
    sub["resid"] = sub[target] - sub["pred"]
    fmt = ",.0f" if target == "mp_4yr" else ".1f"
    hover = ("<b>%{customdata[0]}</b> (%{customdata[1]}, %{customdata[2]} draft class)<br>"
             "Pick %{customdata[3]}<br>"
             f"Predicted %{{x:{fmt}}} · Actual %{{y:{fmt}}}<extra></extra>")
    fig = go.Figure()
    for (name, _, _), color in zip(DRAFT_TIERS, SERIES):
        rows = sub[sub["tier"] == name]
        pick = rows["draft_pick"].where(rows["drafted"] == 1).map(
            lambda p: f"{p:.0f}" if pd.notna(p) else "undrafted")
        fig.add_trace(go.Scatter(
            x=rows["pred"], y=rows[target], mode="markers", name=name,
            marker={"size": 8, "color": color, "opacity": 0.8,
                    "line": {"color": "white", "width": 1}},
            customdata=np.column_stack([rows["player"], rows["school"], rows["entry_year"], pick]),
            hovertemplate=hover,
        ))
    lo, hi = min(sub["pred"].min(), sub[target].min()), max(sub["pred"].max(), sub[target].max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                             hoverinfo="skip", line={"color": GRID, "width": 1, "dash": "dash"}))
    for _, r in sub.loc[sub["resid"].abs().nlargest(n_labels).index].iterrows():
        fig.add_annotation(x=r["pred"], y=r[target], text=r["player"], showarrow=True,
                           arrowhead=0, arrowwidth=0.6, arrowcolor="#777777", ax=0, ay=-16,
                           font={"size": 10, "color": "#222222"})
    fig.update_layout(
        title={"text": f"{TARGETS[target]}: predicted vs. actual (n={len(sub)})<br><sup>"
                       "Full model, out of sample by draft class; dashed line = perfect "
                       "prediction; labels = largest misses</sup>", "x": 0.01},
        xaxis_title="Predicted", yaxis_title="Actual", template="plotly_white",
        width=1000, height=700, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.12, "x": 0},
    )
    return fig


def plot_importance(imp: pd.DataFrame, target: str) -> go.Figure:
    """Drop-family importance as horizontal bars."""
    rows = imp.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=rows["family"], x=rows["r2_drop"], orientation="h",
        marker={"color": SERIES[0], "cornerradius": 4},
        hovertemplate="<b>%{y}</b><br>R² drop %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0, line={"color": GRID, "width": 1})
    fig.update_layout(
        title={"text": f"Feature family importance: {TARGETS[target].lower()}<br><sup>Drop in "
                       "out-of-sample R² when the family is removed from the full model</sup>",
               "x": 0.01},
        xaxis_title="R² drop", template="plotly_white", width=1000, height=520,
        hoverlabel={"bgcolor": "white"}, showlegend=False,
    )
    return fig


def plot_draft_class(proj: pd.DataFrame, target: str, n: int = 30) -> go.Figure:
    """Full-model vs. college-only projection for the first ``n`` picks."""
    rows = proj.nsmallest(n, "draft_pick").iloc[::-1]
    full, college = rows[f"proj_{target}"], rows[f"proj_{target}_college"]
    label = rows["draft_pick"].map("{:.0f}".format) + ". " + rows["player"]
    fmt = ",.0f" if target == "mp_4yr" else ".1f"
    fig = go.Figure()
    for i, (lbl, a, b) in enumerate(zip(label, full, college)):
        fig.add_trace(go.Scatter(x=[a, b], y=[lbl, lbl], mode="lines", showlegend=False,
                                 hoverinfo="skip", line={"color": "#c9c9c4", "width": 2}))
    custom = np.column_stack([rows["school"], rows["age"], full, college])
    hover = ("<b>%{y}</b> (%{customdata[0]}, age %{customdata[1]:.1f})<br>"
             f"Full model %{{customdata[2]:{fmt}}} · College only %{{customdata[3]:{fmt}}}"
             "<extra></extra>")
    for name, x, color in [("Full model", full, SERIES[0]),
                           ("College stats + bio only", college, SERIES[1])]:
        fig.add_trace(go.Scatter(x=x, y=label, mode="markers", name=name, customdata=custom,
                                 hovertemplate=hover,
                                 marker={"size": 10, "color": color,
                                         "line": {"color": "white", "width": 2}}))
    year = int(proj["draft_year"].iloc[0])
    fig.update_layout(
        title={"text": f"{year} draft, first {n} picks with college stats: projected "
                       f"{TARGETS[target].lower()}<br><sup>Gap = how much the draft slot and "
                       "recruiting rank move the projection</sup>", "x": 0.01},
        xaxis_title=f"Projected {TARGETS[target].lower()}", template="plotly_white",
        width=1000, height=max(500, 26 * len(rows) + 160), hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.08, "x": 0},
    )
    return fig
