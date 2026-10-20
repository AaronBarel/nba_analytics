"""Interactive charts for defensive value (P3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nba_analytics.config import season_label

# Validated categorical slots 1-2 plus a recessive gray for everyone else
CATEGORIES = [
    ("Other qualifiers", "#b8b8b4"),
    ("Hidden defender", "#2a78d6"),
    ("Hidden defender, rookie scale", "#eb6834"),
]
GROUP_NAMES = {"G": "Guards", "F": "Forwards", "C": "Centers"}


def _category(df: pd.DataFrame) -> pd.Series:
    return np.select(
        [df["hidden_defender"] & df["rookie_scale"], df["hidden_defender"]],
        [CATEGORIES[2][0], CATEGORIES[1][0]], default=CATEGORIES[0][0])


def plot_offense_defense(df: pd.DataFrame, group: str, n_labels: int = 10) -> go.Figure:
    """Offense score vs. defensive index for one position group; size = minutes per game."""
    sub = df[df["pos_group"] == group].assign(category=lambda d: _category(d))
    hover = ("<b>%{customdata[0]}</b> (%{customdata[1]}, %{customdata[2]})<br>"
             "%{customdata[3]:.1f} MPG · NBA season #%{customdata[4]}<br>"
             "Defense index %{y:+.2f} (group pctile %{customdata[5]:.0f})<br>"
             "Box score %{customdata[6]:+.2f} · Advanced %{customdata[7]:+.2f} · "
             "On/off %{customdata[8]:+.2f}<br>"
             "DRtg on-off %{customdata[9]:+.1f}<br>"
             "Offense score %{x:+.2f}<extra></extra>")
    cols = ["player", "team", "pos", "mp_per_g", "career_season", "defense_pctile",
            "box_score", "advanced", "on_off", "drtg_on_off"]

    fig = go.Figure()
    for name, color in CATEGORIES:
        rows = sub[sub["category"] == name]
        fig.add_trace(go.Scatter(
            x=rows["offense_score"], y=rows["defense_index"], mode="markers", name=name,
            marker={"size": 4 + rows["mp_per_g"] / 2.5, "color": color, "opacity": 0.85,
                    "line": {"color": "white", "width": 1}},
            customdata=rows[cols], hovertemplate=hover,
        ))

    # Label the hidden defenders with the best defensive index, plus the top overall
    labeled = pd.concat([sub[sub["hidden_defender"]].nlargest(n_labels, "defense_index"),
                         sub.nlargest(3, "defense_index")]).drop_duplicates("player_id")
    for _, r in labeled.iterrows():
        fig.add_annotation(x=r["offense_score"], y=r["defense_index"], text=r["player"],
                           showarrow=True, arrowhead=0, arrowwidth=0.6, arrowcolor="#777777",
                           ax=0, ay=-16, font={"size": 10, "color": "#222222"})

    fig.add_hline(y=0, line={"color": "#9a9a9a", "width": 1, "dash": "dash"})
    fig.add_vline(x=0, line={"color": "#9a9a9a", "width": 1, "dash": "dash"})
    season = season_label(int(sub["season"].iloc[0]))
    fig.update_layout(
        title={"text": f"{season} {GROUP_NAMES[group]}: offense vs. defensive index "
                       f"(n={len(sub)})<br><sup>Scores are z-scores within the position group; "
                       "marker size = minutes per game; hidden defenders = below-average "
                       "offense and top-quartile minutes</sup>", "x": 0.01},
        xaxis_title="Offense score (OBPM and relative TS%)",
        yaxis_title="Defensive index (box score, advanced, on/off)",
        template="plotly_white", width=1000, height=700, hoverlabel={"bgcolor": "white"},
        legend={"orientation": "h", "y": -0.12, "x": 0},
    )
    return fig
