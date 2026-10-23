"""Draft lottery rules: conference seeding, the play-in and the draft order.

Three systems cover the seasons modelled here:

- 2015-2018: the 14 non-playoff teams; three picks drawn, with odds of 25%
  down to 0.5% by record.
- 2019-2026: the 14 non-playoff teams (play-in losers from 2021); four picks
  drawn, with the three worst records sharing 14% each.
- 2027 on, the "3-2-1" lottery: 16 teams, every one of the 16 picks drawn by
  lottery balls. Teams outside the play-in get 3 balls, the 9 and 10 seeds 2,
  the 7-vs-8 losers 1. The three worst records ("draft relegated") lose a
  ball but can fall no lower than 12th. A team's original pick can't be 1st
  in consecutive drafts or top five in three consecutive drafts.

Picks after the lottery go by record, worst first. Every function works on
many simulated seasons at once: arrays are (simulations, 30 teams), and ties
in record are broken by a random draw built into ``score``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

EAST = {"ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DET", "IND", "MIA", "MIL", "NYK", "ORL",
        "PHI", "TOR", "WAS"}
TEAMS = sorted(EAST | {"DAL", "DEN", "GSW", "HOU", "LAC", "LAL", "MEM", "MIN", "NOP", "OKC",
                       "PHO", "POR", "SAC", "SAS", "UTA"})


@dataclass(frozen=True)
class Rules:
    name: str
    drawn: int  # picks decided by the drawing
    odds: tuple[float, ...] = ()  # chance of the top pick by lottery rank (older systems)
    play_in: bool = True
    balls: bool = False  # the 3-2-1 system
    floor: int = 12  # relegated teams fall no lower than this (3-2-1)
    no_first: frozenset[str] = field(default_factory=frozenset)  # original picks barred from 1st
    no_top5: frozenset[str] = field(default_factory=frozenset)  # ... barred from the top five


RULES_2015 = Rules("2015-2018", 3, (25.0, 19.9, 15.6, 11.9, 8.8, 6.3, 4.3, 2.8, 1.7, 1.1, 0.8,
                                     0.7, 0.6, 0.5), play_in=False)
RULES_2019 = Rules("2019-2026", 4, (14.0, 14.0, 14.0, 12.5, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 2.0,
                                     1.5, 1.0, 0.5))
# 2026 draft: Washington picked 1st with its own pick; Utah's own pick was top five in 2025
# and 2026.
RULES_2027 = Rules("3-2-1", 16, balls=True, no_first=frozenset({"WAS"}),
                   no_top5=frozenset({"UTA"}))
RELEGATED = 3


def rules_for(season: int) -> Rules:
    """The lottery that a season's records feed (the draft after the season)."""
    if season >= 2027:
        return RULES_2027
    if season >= 2019:
        return RULES_2019 if season >= 2021 else Rules(RULES_2019.name, 4, RULES_2019.odds,
                                                      play_in=False)
    return RULES_2015


def conference_seeds(score: np.ndarray, east: np.ndarray) -> np.ndarray:
    """Seed (1 = best) of each team within its conference, from ``score`` (higher is better)."""
    seeds = np.zeros(score.shape, dtype=int)
    for conf in (east, ~east):
        idx = np.flatnonzero(conf)
        order = np.argsort(-score[:, idx], axis=1)
        rows = np.arange(len(score))[:, None]
        seeds[rows, idx[order]] = np.arange(1, len(idx) + 1)
    return seeds


def _team_at(seeds: np.ndarray, conf: np.ndarray, seed: int) -> np.ndarray:
    """Index of the team holding ``seed`` in the conference, per simulation."""
    return np.argmax((seeds == seed) & conf, axis=1)


def play_in(seeds: np.ndarray, east: np.ndarray, game_prob: Callable, u: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Play the play-in: returns (losers of 7 vs 8, teams that miss the playoffs).

    ``game_prob(home, away)`` is the home team's win probability for arrays of
    team indices; the better seed is at home. ``u`` holds three uniforms per
    conference per simulation (S, 6).
    """
    rows = np.arange(len(seeds))
    lost_7v8 = np.zeros(seeds.shape, dtype=bool)
    out = seeds > 10
    for c, conf in enumerate((east, ~east)):
        s7, s8, s9, s10 = (_team_at(seeds, conf, s) for s in (7, 8, 9, 10))
        win78 = u[:, 3 * c] < game_prob(s7, s8)
        loser78 = np.where(win78, s8, s7)
        win910 = np.where(u[:, 3 * c + 1] < game_prob(s9, s10), s9, s10)
        loser910 = np.where(win910 == s9, s10, s9)
        last = np.where(u[:, 3 * c + 2] < game_prob(loser78, win910), win910, loser78)
        lost_7v8[rows, loser78] = True
        out[rows, loser910] = True
        out[rows, last] = True
    return lost_7v8, out


def draw(weights: np.ndarray, n: int, u: np.ndarray, banned: dict[int, np.ndarray] | None = None,
         floor: np.ndarray | None = None, floor_pick: int = 12) -> np.ndarray:
    """Draw ``n`` picks in turn, each with chance proportional to ``weights``.

    ``banned[k]`` marks teams that can't take pick ``k`` (1-based). ``floor``
    marks teams that must be drawn by ``floor_pick``: once the picks left up to
    it only just cover the floor teams still waiting, only they are drawn.
    Returns the pick number of each drawn team (0 for the rest).
    """
    S, T = weights.shape
    rows = np.arange(S)
    pick = np.zeros((S, T), dtype=int)
    for k in range(1, n + 1):
        w = np.where(pick > 0, 0.0, weights)
        if banned and k in banned:
            w = np.where(banned[k], 0.0, w)
        if floor is not None and k <= floor_pick:
            waiting = floor & (pick == 0)
            must = waiting.sum(axis=1) >= floor_pick - k + 1
            w = np.where(must[:, None] & ~waiting, 0.0, w)
        cum = w.cumsum(axis=1)
        idx = (cum <= (u[:, k - 1] * cum[:, -1])[:, None]).sum(axis=1)
        pick[rows, idx] = k
    return pick


def _fill(pick: np.ndarray, lottery: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Give the undrawn teams the remaining picks: lottery teams first, each group by record."""
    key = np.where(pick > 0, pick - 1e6, np.where(lottery, score, score + 1e6))
    order = np.argsort(key, axis=1)
    out = np.empty_like(pick)
    rows = np.arange(len(pick))[:, None]
    out[rows, order] = np.arange(1, pick.shape[1] + 1)
    return out


def lottery_balls(seeds: np.ndarray, lost_7v8: np.ndarray, score: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
    """3-2-1 balls for each team and the draft-relegated teams (three worst records)."""
    balls = np.select([seeds > 10, seeds >= 9, lost_7v8], [3.0, 2.0, 1.0], 0.0)
    worst = np.argsort(score, axis=1)[:, :RELEGATED]
    relegated = np.zeros(score.shape, dtype=bool)
    relegated[np.arange(len(score))[:, None], worst] = True
    return balls - (relegated & (balls == 3)), relegated


def draft_order(score: np.ndarray, east: np.ndarray, rules: Rules, game_prob: Callable,
                u_play: np.ndarray, u_draw: np.ndarray, teams: list[str] = TEAMS
                ) -> tuple[np.ndarray, np.ndarray]:
    """Pick number (1-30) of each team's original first-round pick, and who missed the playoffs.

    ``score`` is final wins plus a tiebreak below one win. ``u_play`` (S, 6)
    and ``u_draw`` (S, 16) are the uniforms for the play-in and the drawing,
    passed in so that two runs can share them.
    """
    seeds = conference_seeds(score, east)
    if rules.play_in:
        lost_7v8, out = play_in(seeds, east, game_prob, u_play)
    else:
        lost_7v8, out = np.zeros(seeds.shape, dtype=bool), seeds > 8
    if rules.balls:
        weights, relegated = lottery_balls(seeds, lost_7v8, score)
        index = {t: i for i, t in enumerate(teams)}
        banned = {}
        for k in range(1, 6):
            barred = rules.no_top5 | (rules.no_first if k == 1 else frozenset())
            banned[k] = np.isin(np.arange(len(teams)), [index[t] for t in barred])
        pick = draw(weights, rules.drawn, u_draw, banned, relegated, rules.floor)
        return _fill(pick, weights > 0, score), out
    rank = np.argsort(np.argsort(np.where(out, score, np.inf), axis=1), axis=1)  # 0 = worst
    odds = np.asarray(rules.odds)
    weights = np.where(out, odds[np.minimum(rank, len(odds) - 1)], 0.0)
    pick = draw(weights, rules.drawn, u_draw)
    return _fill(pick, out, score), out


def base_odds(rules: Rules, n: int = 200_000, seed: int = 0) -> np.ndarray:
    """Chance of each pick (columns) by pre-lottery record rank (rows, worst first).

    For the 3-2-1 system the ranks are the three relegated teams, the seven
    other non-play-in teams, the four 9-10 seeds and the two 7-8 losers.
    """
    rng = np.random.default_rng(seed)
    if rules.balls:
        balls = np.array([2] * RELEGATED + [3] * 7 + [2] * 4 + [1] * 2, float)
        floor = np.zeros((n, len(balls)), dtype=bool)
        floor[:, :RELEGATED] = True
        pick = draw(np.tile(balls, (n, 1)), len(balls), rng.random((n, len(balls))),
                    floor=floor, floor_pick=rules.floor)
    else:
        weights = np.tile(np.asarray(rules.odds), (n, 1))
        pick = draw(weights, rules.drawn, rng.random((n, rules.drawn)))
        pick = _fill(pick, np.ones(pick.shape, dtype=bool), np.tile(np.arange(len(rules.odds),
                                                                              dtype=float), (n, 1)))
    return np.stack([(pick == k).mean(axis=0) for k in range(1, pick.shape[1] + 1)], axis=1)
