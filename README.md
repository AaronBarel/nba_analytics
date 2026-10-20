# NBA Analytics

Player projections, defensive value estimates, and team/game outcome models for the NBA.

## Setup

```bash
uv sync
uv run pytest
uv run jupyter lab notebooks/
```

Data is scraped from Basketball-Reference (rate-limited to about one request every 3.5
seconds) and cached under `data/cache/`, which is not committed.

## Layout

```
src/nba_analytics/
  data/       scrapers and loaders
  players/    player-level analyses and models
  viz/        shared plotting helpers (team colors)
notebooks/    reports and visualizations
tests/
```

## Reports

| Notebook | Contents |
|---|---|
| [01_usage_vs_true_shooting](notebooks/01_usage_vs_true_shooting.ipynb) | Usage % vs. TS% by position and position group, 2022-23 to 2025-26 |
| [02_usage_vs_relative_true_shooting](notebooks/02_usage_vs_relative_true_shooting.ipynb) | Same, with TS% relative to the position / group average |

## Roadmap

- Players: season projection model, defensive value estimate, per-game stat projections
- Teams: net rating year-over-year persistence, early-season predictiveness, game outcome
  model, draft-incentive factor under the 2027 lottery rules
