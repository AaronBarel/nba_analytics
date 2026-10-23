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
uv run python scripts/backfill_player_data.py 2005 2026 --season-tables  # ~3 minutes, notebook 07
uv run python scripts/backfill_college_data.py 2010 2026  # ~3 hours, for notebook 06
uv run python scripts/backfill_gleague_data.py 2008 2026  # ~3 minutes, for notebook 08
uv run python scripts/backfill_player_gamelogs.py 2014 2026  # ~7 hours, for notebooks 09 and 10
```

Charts are interactive (hover for details) in JupyterLab or VS Code. GitHub shows static images.

### Predicting upcoming games

```bash
uv run python scripts/predict_games.py              # next date with unplayed games
uv run python scripts/predict_games.py 2026-10-23   # a given date
```

This pulls the season's box scores, the playing teams' rosters and the NBA's latest official
injury report, then prints home win probabilities, projected margins and player stat lines,
saving them under `data/predictions/<date>/`. The models are the ones in notebooks 09 and 10,
fit on every past season; players listed Out or Doubtful are treated as not playing.

## Layout

```
src/nba_analytics/
  data/       scrapers and loaders
  players/    player-level analyses and models
  teams/      team-level analyses and models
  viz/        shared plotting helpers (team colors)
  live.py     predictions for upcoming games
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
| [07_season_projections](notebooks/07_season_projections.ipynb) | Next-season games, minutes, per-100 stats, TS%, usage and BPM: Marcel baseline vs. residual model, walk-forward 2013-14 to 2025-26; aging curves; 2026-27 projections |
| [08_gleague_exploration](notebooks/08_gleague_exploration.ipynb) | G League → NBA linking, the G League pathway, how each stat translates to the next NBA season, and whether G League stats improve the projection models |
| [09_game_outcomes](notebooks/09_game_outcomes.ipynb) | Home win probability and margin from Elo, team ratings, schedule and travel, roster talent, availability and turnover; walk-forward 2015-16 to 2025-26 |
| [10_player_game_projections](notebooks/10_player_game_projections.ipynb) | Per-game points, rebounds, assists, steals and blocks as projected minutes x per-minute rate with a negative binomial spread; accuracy against season averages and calibration of over probabilities, walk-forward 2015-16 to 2025-26 |

## Roadmap

- Teams: draft-incentive factor under the 2027 lottery rules
- Season-to-date scoring of the live predictions
