import pandas as pd
import pytest

from nba_analytics.players.rookies import UNDRAFTED_PICK, UNRANKED_RSCI, build_rookies


def _bio(pid, path, pick=None, rsci=None, birth="2000-06-25", draft_year=None):
    return {"player_id": pid, "cbb_path": path, "draft_pick": pick, "rsci_rank": rsci,
            "birth_date": birth, "draft_year": draft_year, "height_cm": 200, "weight_kg": 95}


def test_build_rookies_filters_and_fills():
    bios = pd.DataFrame([
        _bio("a", "/a", pick=3, rsci=5, draft_year=2019),
        _bio("b", "/b"),                              # undrafted, unranked
        _bio("c", "/c", pick=40, draft_year=2015),    # college long before debut
        _bio("d", "/d", pick=20, draft_year=2009),    # college before per-100 tables
        _bio("new", "/new", birth="2006-06-25"),      # upcoming class, bio not yet updated
        _bio("old", "/old", pick=55),                 # in the draft but left college in 2023
    ])
    college = pd.DataFrame({"cbb_path": ["/a", "/b", "/c", "/d", "/new", "/old"],
                            "last_college_season": [2019, 2020, 2015, 2009, 2026, 2023]})
    targets = pd.DataFrame({"player_id": ["a", "b", "c", "d"], "debut": [2020, 2022, 2019, 2010],
                            "player": ["A", "B", "C", "D"], "mp_4yr": [5000, 300, 1000, 900]})
    draft = pd.DataFrame({"draft_year": [2026, 2026], "player_id": ["new", "old"],
                          "player": ["New", "Old"], "pick_overall": [7, 55]})

    df = build_rookies(bios, college, targets, draft).set_index("player_id")
    assert set(df.index) == {"a", "b", "new"}
    assert not df.at["a", "draft_class"] and df.at["new", "draft_class"]
    assert df.at["b", "draft_pick"] == UNDRAFTED_PICK and df.at["b", "drafted"] == 0
    assert df.at["b", "rsci_rank"] == UNRANKED_RSCI and df.at["b", "rsci_missing"] == 1
    assert df.at["new", "draft_pick"] == 7 and df.at["new", "player"] == "New"
    assert df.at["a", "age"] == pytest.approx(19, abs=0.01)
    assert df.at["new", "age"] == pytest.approx(20, abs=0.01)
    assert df.at["a", "entry_year"] == 2019
