import numpy as np
import pandas as pd
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
        "pts": [30.0, None, 10.0], "trb": 5.0, "ast": 2.0, "stl": 1.0, "blk": 0.0, "fg3": 2.0,
    })
    h = _player_history(df)
    assert pd.isna(h.loc[0, "pts_rate_td"]) and h.loc[0, "games_td"] == 0
    assert h.loc[2, "games_td"] == 1 and h.loc[2, "mpg_td"] == 30  # the DNP game is skipped
    assert h.loc[2, "pts_rate_td"] == 1.0 and h.loc[2, "pts_rate_recent"] == 1.0
    assert h.loc[2, "start_share_td"] == 1.0


def test_shrunk_fg3_pct_pools_the_last_three_seasons_toward_the_league():
    from nba_analytics.players.props import FG3_PRIOR_ATTEMPTS, shrunk_fg3_pct

    totals = pd.DataFrame({
        "season": [2021, 2022, 2023, 2024, 2025, 2022, 2023, 2024],
        "player_id": ["a"] * 5 + ["b"] * 3,
        "fg3": [999, 30, 40, 50, 0, 100, 100, 100],   # 2021 is too old, 2025 is the season itself
        "fg3a": [999, 100, 100, 200, 50, 250, 250, 250]})
    out = shrunk_fg3_pct(totals, [2025]).set_index("player_id")["fg3_pct_prior"]
    league_2024 = (50 + 100) / (200 + 250)
    k = FG3_PRIOR_ATTEMPTS
    assert out["a"] == pytest.approx((30 + 40 + 50 + k * league_2024) / (400 + k))
    assert out["b"] == pytest.approx((300 + k * league_2024) / (750 + k))


def test_shrunk_fg3_pct_without_history_is_the_league_rate():
    from nba_analytics.players.props import shrunk_fg3_pct

    totals = pd.DataFrame({"season": [2024, 2025], "player_id": ["a", "new"],
                           "fg3": [36.0, 5.0], "fg3a": [100.0, 10.0]})
    out = shrunk_fg3_pct(totals, [2025]).set_index("player_id")["fg3_pct_prior"]
    assert "new" not in out.index  # callers fill players without history with FG3_PCT
    assert out["a"] == pytest.approx(0.36)


def test_blend_threes_separates_attempts_from_accuracy():
    from nba_analytics.players.props import FG3A_PRIOR_MINUTES, blend_threes

    rows = pd.DataFrame({"fg3_td": [0.0, 5.0], "fg3a_td": [0.0, 8.0], "mp_td": [0.0, 100.0],
                         "fg3a_rate_proj": 0.06, "fg3_pct_prior": 0.32,
                         "fg3_pct_weight": 600.0})
    out = blend_threes(rows)
    # no games yet: the pre-season attempt rate at the pre-season 3P%
    assert out.loc[0, "fg3_rate_blend"] == pytest.approx(0.06 * 0.32)
    att = (8 + FG3A_PRIOR_MINUTES * 0.06) / (100 + FG3A_PRIOR_MINUTES)
    pct = (5 + 600 * 0.32) / (8 + 600)
    assert out.loc[1, "fg3a_rate_blend"] == pytest.approx(att)
    assert out.loc[1, "fg3_pct_blend"] == pytest.approx(pct)
    assert out.loc[1, "fg3_rate_blend"] == pytest.approx(att * pct)
    # 5 of 8 barely moves the percentage
    assert out.loc[1, "fg3_pct_blend"] < 0.33


def test_shrunk_fg3_pct_weight_counts_past_attempts():
    from nba_analytics.players.props import FG3_PRIOR_ATTEMPTS, shrunk_fg3_pct

    totals = pd.DataFrame({"season": [2023, 2024], "player_id": "a", "fg3": [30.0, 40.0],
                           "fg3a": [100.0, 120.0]})
    out = shrunk_fg3_pct(totals, [2025]).iloc[0]
    assert out["fg3_pct_weight"] == pytest.approx(220 + FG3_PRIOR_ATTEMPTS)
