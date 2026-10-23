import numpy as np
import pandas as pd

from nba_analytics.teams import incentive, lottery, picks
from nba_analytics.teams.lottery import TEAMS

# Tankathon's resolved 2027 order on 2026-10-23: (original team, owner) by pick
TANKATHON = [("SAC", "SAC"), ("BRK", "HOU"), ("MIL", "NOP"), ("CHI", "CHI"), ("NOP", "ATL"),
             ("DAL", "CHO"), ("UTA", "MEM"), ("PHO", "HOU"), ("ORL", "ORL"), ("MIA", "MIA"),
             ("GSW", "GSW"), ("CLE", "UTA"), ("DEN", "OKC"), ("BOS", "BOS"), ("NYK", "BRK"),
             ("SAS", "SAC"), ("MEM", "MEM"), ("LAC", "OKC"), ("WAS", "WAS"), ("CHO", "CHO"),
             ("POR", "POR"), ("ATL", "SAS"), ("IND", "IND"), ("LAL", "MEM"), ("HOU", "BRK"),
             ("TOR", "LAC"), ("MIN", "PHO"), ("DET", "DET"), ("PHI", "PHI"), ("OKC", "TOR")]


def test_2027_owners_match_tankathon():
    pick = np.zeros((1, 30), dtype=int)
    for k, (orig, _) in enumerate(TANKATHON, start=1):
        pick[0, TEAMS.index(orig)] = k
    owner = picks.owners_2027(pick)[0]
    assert [(o, TEAMS[owner[TEAMS.index(o)]]) for o, _ in TANKATHON] == TANKATHON


def test_protected_pick_stays_and_okc_keeps_two_best():
    pick = np.zeros((1, 30), dtype=int)
    order = ["DEN", "LAL", "SAS"] + [t for t in TEAMS if t not in {"DEN", "LAL", "SAS"}]
    for k, t in enumerate(order, start=1):
        pick[0, TEAMS.index(t)] = k
    owner = {TEAMS[k]: TEAMS[o] for k, o in enumerate(picks.owners_2027(pick)[0])}
    assert owner["DEN"] == "DEN" and owner["LAL"] == "LAL" and owner["SAS"] == "SAC"
    # Denver kept its pick, so OKC keeps the better of its own and the Clippers'
    assert sorted([owner["OKC"], owner["LAC"]]).count("OKC") == 1


def test_321_draw_matches_published_odds():
    S = 200_000
    balls = np.array([2, 2, 2] + [3] * 7 + [2] * 4 + [1, 1], float)
    floor = np.zeros((S, 16), bool)
    floor[:, :3] = True
    pick = lottery.draw(np.tile(balls, (S, 1)), 16, np.random.default_rng(0).random((S, 16)),
                        floor=floor)
    first = (pick == 1).mean(axis=0)
    assert np.allclose(first[[0, 3, 10, 14]], [2 / 37, 3 / 37, 2 / 37, 1 / 37], atol=0.003)
    assert (pick[:, :3] <= 12).all()
    assert abs((pick[:, 0] == 12).mean() - 0.251) < 0.005  # Tankathon's base odds table


def test_banned_team_never_takes_barred_picks():
    S = 20_000
    w = np.ones((S, 16))
    banned = {k: np.arange(16) == 0 for k in range(1, 6)}
    pick = lottery.draw(w, 16, np.random.default_rng(1).random((S, 16)), banned=banned)
    assert (pick[:, 0] > 5).all()


def test_old_lottery_drops_at_most_four_spots():
    S = 5_000
    rng = np.random.default_rng(2)
    score = np.tile(np.arange(30, dtype=float), (S, 1)) * 2 + rng.random((S, 30))
    east = np.isin(np.array(TEAMS), list(lottery.EAST))
    order, missed = lottery.draft_order(score, east, lottery.RULES_2019, lambda h, a: 0.5,
                                        rng.random((S, 6)), rng.random((S, 16)))
    worst = np.argmin(score[0])
    assert order[:, worst].max() == 5 and order[:, worst].min() == 1
    assert missed.sum(axis=1).tolist() == [14] * S


def _season(n_rounds=4, played=2):
    """Synthetic season: every team plays every round; the first ``played`` rounds are final."""
    rng = np.random.default_rng(3)
    rows = []
    for r in range(n_rounds):
        order = rng.permutation(TEAMS)
        for k in range(15):
            h, a = order[2 * k], order[2 * k + 1]
            rows.append({"season": 2026, "date": f"2026-01-{r + 1:02d}", "home": h, "away": a,
                         "neutral": False, "margin": rng.choice([-5, 5]) if r < played
                         else np.nan, "elo_home": 1500.0, "elo_away": 1500.0,
                         "home_talent_full": 0.0, "away_talent_full": 0.0})
    return pd.DataFrame(rows)


def test_snapshot_counts_results_and_leaves_open_games():
    df = _season()
    snap = incentive.snapshot(df, 2026, "2026-01-03")
    assert snap.wins.sum() == 30 and (snap.wins + snap.losses == 2).all()
    assert len(snap.home) == 30


def test_forcing_a_game_moves_only_that_result():
    snap = incentive.snapshot(_season(), 2026, "2026-01-03")
    value = lambda p: np.zeros(p.shape)
    coef = np.array([0.0, 1.0, 0.0])
    won = incentive.simulate(snap, coef, value, 500, force=(0, True))
    lost = incentive.simulate(snap, coef, value, 500, force=(0, False))
    diff = won["wins"] - lost["wins"]
    assert (diff[:, snap.home[0]] == 1).all() and (diff[:, snap.away[0]] == -1).all()
    assert (np.delete(diff, [snap.home[0], snap.away[0]], axis=1) == 0).all()


def test_team_without_its_pick_has_no_draft_incentive():
    snap = incentive.snapshot(_season(), 2026, "2026-01-03")
    coef = np.array([0.0, 1.0, 0.0])
    t = TEAMS.index("NYK")

    def value(pick):  # every team owns its pick except NYK, whose pick is worth nothing to it
        v = picks.pick_value(pick, np.array([6.0, 0.2, 0.9]))
        v[:, t] = 0.0
        return v

    out = incentive.incentives(snap, coef, value, 500).set_index("team_id")
    assert out.loc["NYK", "draft_incentive"] == 0.0
    assert out["draft_incentive"].abs().max() > 0
