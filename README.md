# NBA Analytics

Player projections, defensive value estimates, and team/game outcome models for the NBA.

## Setup

```bash
uv sync
uv run pytest
uv run jupyter lab notebooks/
```

Data is scraped from Basketball-Reference and Sports-Reference college basketball (rate-limited to about one request every 3.5
seconds) and cached under `data/cache/` and `data/processed/`, which are not committed.
The team notebooks need a one-time backfill of team game logs (~45 minutes):

```bash
uv run python scripts/backfill_team_data.py 2001 2026
uv run python scripts/backfill_player_data.py 2023 2026   # ~10 minutes, for notebook 05
uv run python scripts/backfill_college_data.py 2010 2026  # ~3 hours, for notebook 06
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
| [05_defensive_value](notebooks/05_defensive_value.ipynb) | Defensive index (box score, advanced, on/off) and hidden-defender screen, 2022-23 to 2025-26 |
| [06_rookie_projections](notebooks/06_rookie_projections.ipynb) | College stats → NBA rookie and first-four-season minutes and BPM; 2026 draft class projections |

## Roadmap

- Players: season projection model, per-game stat projections
- Rookies: G League exploration
- Teams: game outcome model, draft-incentive factor under the 2027 lottery rules
