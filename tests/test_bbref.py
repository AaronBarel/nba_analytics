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
