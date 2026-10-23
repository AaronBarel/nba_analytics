"""Each team's incentive to win under the 2027 lottery, as of today (or a given date).

    uv run python scripts/incentives.py
    uv run python scripts/incentives.py 2027-02-15

Simulates the rest of the season, the play-in and the 3-2-1 lottery with
every team's actual 2027 first-round holdings (about a minute), prints the
table and saves it to ``data/predictions/incentives/<date>.parquet``.
"""

import sys
import warnings

from nba_analytics import live
from nba_analytics.teams import incentive

warnings.filterwarnings("ignore", category=FutureWarning)


def main(today: str | None) -> None:
    today = today or live.eastern_today()
    out = incentive.current(today)
    folder = live.PREDICTIONS_DIR / "incentives"
    folder.mkdir(parents=True, exist_ok=True)
    out.to_parquet(folder / f"{today}.parquet")

    print(f"Incentives before the games on {today} ({incentive.LIVE_SIMS} simulations)\n")
    table = out.sort_values("draft_incentive").assign(
        record=out["wins"].astype(int).astype(str) + "-" + out["losses"].astype(int).astype(str),
        proj_wins=out["proj_wins"].round(1), playoffs=(100 * out["playoffs"]).round(1),
        exp_pick=out["exp_pick"].round(1), draft_value=out["draft_value"].round(2),
        draft_incentive=out["draft_incentive"].round(3),
        playoff_incentive=(100 * out["playoff_incentive"]).round(1))
    print(table[["team_id", "record", "proj_wins", "playoffs", "exp_pick", "draft_value",
                 "draft_incentive", "playoff_incentive", "next_game"]].to_string(index=False))
    print("\nplayoffs and playoff_incentive in %; exp_pick is the team's "
          "own pick;\ndraft_value is the expected four-season VORP of the 2027 firsts it owns; "
          "the incentives\nare what winning the next game instead of losing it changes.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
