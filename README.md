# NBA Analytics

Player projections, defensive value estimates, and team/game outcome models for the NBA.

## Setup

```bash
uv sync
uv run pytest
uv run jupyter lab notebooks/
```

Data is scraped from Basketball-Reference (rate-limited to about one request every 3.5
seconds) and cached under `data/cache/` and `data/processed/`, which are not committed.
The team notebooks need a one-time backfill of team game logs (~45 minutes):

```bash
uv run python scripts/backfill_team_data.py 2001 2026
```

Charts are interactive (hover for details) in JupyterLab or VS Code. GitHub shows static images.

## Layout

```
src/nba_analytics/
  data/       scrapers and loaders
  players/    player-level analyses and models
  teams/      team-level analyses and models
  viz/        shared plotting helpers (team colors)
notebooks/    reports and visualizations
scripts/      data backfills
tests/
```

## Reports

| Notebook | Contents |
|---|---|
| [01_usage_vs_true_shooting](notebooks/01_usage_vs_true_shooting.ipynb) | Usage % vs. TS% by position and position group, 2022-23 to 2025-26 |
| [02_usage_vs_relative_true_shooting](notebooks/02_usage_vs_relative_true_shooting.ipynb) | Same, with TS% relative to the position / group average |
| [03_net_rating_persistence](notebooks/03_net_rating_persistence.ipynb) | R² of net rating and win % from one season to the next, 2000-01 to 2025-26 |
| [04_early_season_predictiveness](notebooks/04_early_season_predictiveness.ipynb) | R² of first-N-games record and net rating vs. the rest of the season |

## Roadmap

- Players: season projection model, defensive value estimate, per-game stat projections
- Rookies: college → NBA translation model; G League exploration
- Teams: game outcome model, draft-incentive factor under the 2027 lottery rules
