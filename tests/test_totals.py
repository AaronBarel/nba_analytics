import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nba_analytics.teams.totals import league_relative, prob_margin_over, prob_total_over


def test_prob_total_over_half_point_line():
    assert prob_total_over(np.array([220.0]), 19.0, 220.5)[0] == pytest.approx(
        stats.norm.sf(0.5 / 19.0))
    assert prob_total_over(np.array([220.0]), 19.0, 200.5)[0] > 0.8


def test_prob_margin_over_excludes_ties():
    mu, sd = np.array([3.0]), 13.0
    no_tie = 1 - (stats.norm.cdf(0.5 / sd - mu / sd) - stats.norm.cdf(-0.5 / sd - mu / sd))
    assert prob_margin_over(mu, sd, 0.5)[0] == pytest.approx(stats.norm.sf((0.5 - 3) / sd)
                                                            / no_tie)
    # the home team wins or loses, never ties
    assert prob_margin_over(mu, sd, 0.5)[0] + prob_margin_over(-mu, sd, 0.5)[0] \
        == pytest.approx(1)


def test_league_relative_uses_each_teams_latest_value():
    g = pd.DataFrame({"season": 2025, "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                      "home": ["A", "C", "A"], "away": ["B", "D", "C"],
                      "home_pace": [100.0, 96.0, 102.0], "away_pace": [98.0, 94.0, 97.0]})
    rel = league_relative(g, ["pace"])
    assert rel.loc[0, "home_pace_rel"] == pytest.approx(100 - 97)  # league mean 97 on day 1
    # day 2: A 102, C 97, B and D carried from day 1 (98, 94)
    assert rel.loc[2, "away_pace_rel"] == pytest.approx(97 - (102 + 97 + 98 + 94) / 4)


def test_league_relative_starts_fresh_each_season():
    g = pd.DataFrame({"season": [2024, 2025], "date": ["2024-04-01", "2024-10-25"],
                      "home": ["A", "C"], "away": ["B", "D"],
                      "home_pace": [90.0, 100.0], "away_pace": [90.0, 98.0]})
    assert league_relative(g, ["pace"]).loc[1, "home_pace_rel"] == pytest.approx(1.0)


def test_prob_margin_over_takes_a_line_per_game():
    p = prob_margin_over(np.array([5.0, 5.0]), 12.0, np.array([0.5, 10.5]))
    assert p[0] > 0.6 and p[1] < 0.35


def test_close_weights_recover_a_hole_at_close_margins():
    from nba_analytics.teams.totals import SUPPORT, fit_close_weights, margin_pmf

    planted = np.ones(26)
    planted[1:4] = 0.5
    rng = np.random.default_rng(0)
    mu = rng.normal(2, 6, 20000)
    probs = margin_pmf(mu, 13.0, planted)
    y = np.array([rng.choice(SUPPORT, p=row) for row in probs])
    w = fit_close_weights(y, mu, 13.0)
    assert w[2] / w[10] == pytest.approx(0.5, abs=0.1)
    # home over 3.5 and the mirrored game under it cover every outcome
    over = prob_margin_over(mu[:5], 13.0, 3.5, w)
    under = prob_margin_over(-mu[:5], 13.0, -3.5, w)
    assert np.allclose(over + under, 1)
