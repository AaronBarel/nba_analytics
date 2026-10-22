import numpy as np
import pytest

from nba_analytics.players.props import fit_dispersion, prob_over


def test_fit_dispersion_recovers_negative_binomial_size():
    rng = np.random.default_rng(0)
    mu, r = rng.uniform(5, 25, 20000), 6.0
    y = rng.negative_binomial(r, r / (r + mu))
    assert fit_dispersion(y, mu) == pytest.approx(r, rel=0.1)


def test_prob_over_matches_simulation_and_is_monotone():
    rng = np.random.default_rng(1)
    mu, r = 18.0, 5.0
    y = rng.negative_binomial(r, r / (r + mu), 200000)
    assert prob_over(np.array([mu]), r, 20)[0] == pytest.approx((y >= 20).mean(), abs=0.01)
    assert prob_over(np.array([mu]), r, 19.5)[0] == pytest.approx((y >= 20).mean(), abs=0.01)
    p = prob_over(np.full(3, mu), r, 10), prob_over(np.full(3, mu), r, 25)
    assert (p[0] > p[1]).all()


def test_player_history_uses_only_earlier_games():
    import pandas as pd

    from nba_analytics.players.props import _player_history

    df = pd.DataFrame({
        "season": 2025, "player_id": "a", "date": ["2025-01-01", "2025-01-03", "2025-01-05"],
        "mp": [30.0, 0.0, 20.0], "started": [1, 0, 0],
        "pts": [30.0, None, 10.0], "trb": 5.0, "ast": 2.0, "stl": 1.0, "blk": 0.0,
    })
    h = _player_history(df)
    assert pd.isna(h.loc[0, "pts_rate_td"]) and h.loc[0, "games_td"] == 0
    assert h.loc[2, "games_td"] == 1 and h.loc[2, "mpg_td"] == 30  # the DNP game is skipped
    assert h.loc[2, "pts_rate_td"] == 1.0 and h.loc[2, "pts_rate_recent"] == 1.0
    assert h.loc[2, "start_share_td"] == 1.0
