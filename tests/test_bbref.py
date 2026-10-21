import pytest
import requests

from nba_analytics.data import bbref


class _Resp:
    def __init__(self, status, text=""):
        self.status_code, self.text, self.encoding = status, text, None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture
def no_waiting(monkeypatch):
    monkeypatch.setattr(bbref, "MIN_REQUEST_INTERVAL", 0)
    monkeypatch.setattr(bbref, "RETRY_BACKOFF", (0, 0, 0, 0))


def _fake_get(responses):
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    return get, calls


def test_fetch_retries_network_errors_then_caches(tmp_path, monkeypatch, no_waiting):
    get, calls = _fake_get([requests.ConnectionError(), _Resp(503), _Resp(200, "<html>ok</html>")])
    monkeypatch.setattr(bbref.requests, "get", get)
    assert bbref.fetch("/x.html", cache_dir=tmp_path) == "<html>ok</html>"
    assert bbref.fetch("/x.html", cache_dir=tmp_path) == "<html>ok</html>"  # from cache
    assert len(calls) == 3


def test_fetch_does_not_retry_rate_limit(tmp_path, monkeypatch, no_waiting):
    get, calls = _fake_get([_Resp(429)])
    monkeypatch.setattr(bbref.requests, "get", get)
    with pytest.raises(requests.HTTPError):
        bbref.fetch("/y.html", cache_dir=tmp_path)
    assert len(calls) == 1


def test_player_bio_parses_meta_block():
    html = """<div id="meta"><p>6-9, 205lb (206cm, 92kg)</p>
      <p>Born: <span id="necro-birth" data-birth="2006-12-21">December 21, 2006</span></p>
      <p>College: <a href="https://www.sports-reference.com/cbb/players/cooper-flagg-1.html">Duke</a></p>
      <p>Recruiting Rank: 2024 (1)</p>
      <p>Draft: Dallas Mavericks, 1st round (1st pick, 1st overall), 2025 NBA Draft</p></div>"""
    assert bbref.player_bio(html) == {
        "height_cm": 206, "weight_kg": 92, "birth_date": "2006-12-21", "draft_year": 2025,
        "draft_round": 1, "draft_pick": 1, "rsci_rank": 1,
        "cbb_path": "/cbb/players/cooper-flagg-1.html",
    }


def test_player_bio_handles_undrafted_no_college():
    bio = bbref.player_bio('<div id="meta"><p>6-4, 190lb (193cm, 86kg)</p></div>')
    assert bio["height_cm"] == 193
    assert bio["draft_pick"] is None and bio["cbb_path"] is None


def test_parse_table_reads_player_id_from_link():
    html = """<table id="stats"><tbody><tr>
      <th data-stat="ranker">1</th>
      <td data-stat="player"><a href='/players/d/dybanaj01.html'>AJ Dybantsa</a></td>
      <td data-stat="pick_overall">1</td></tr></tbody></table>"""
    df = bbref.parse_table(html, "stats")
    assert df.loc[0, "player_id"] == "dybanaj01"
    assert df.loc[0, "pick_overall"] == 1
