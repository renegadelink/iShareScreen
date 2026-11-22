"""The browser bridge only serves viewers that present this run's token."""
from __future__ import annotations

import inspect
import re
from pathlib import Path

from isharescreen.frontend.wt import server
from isharescreen.proxy.session import SessionConfig

PAGE = (Path(server.__file__).parent / "static" / "index.html").read_text()


def _bridge():
    return server.WebTransportBridge(SessionConfig(host="mac", username="u", password="p"))


def test_token_is_random_and_checked():
    a, b = _bridge(), _bridge()
    assert len(a.token) >= 32 and a.token != b.token
    assert a.check_token(a.token)
    for bad in ("", None, b.token, a.token[:-1], 123):
        assert not a.check_token(bad)


def test_connect_is_refused_without_the_token_and_not_logged():
    src = inspect.getsource(server._BridgeProtocol._h3_event)
    assert "check_token(token)" in src and "403" in src
    # the HEADERS log line prints base_path, not the path with its query
    assert re.search(r'event\.stream_id, method, base_path, protocol', src)


def test_printed_url_carries_the_token_in_the_fragment():
    src = inspect.getsource(server.WebTransportBridge.run_async)
    assert '/#token={self.token}' in src


def test_stats_and_log_need_the_token():
    src = inspect.getsource(server.WebTransportBridge._start_http_bootstrap)
    assert src.count("if not token_ok(req):") == 2


def test_page_sends_the_token_and_keeps_it_out_of_logs():
    assert 'new URLSearchParams(location.hash.slice(1)).get("token")' in PAGE
    assert "history.replaceState" in PAGE                 # gone from the address bar
    assert '?token=${encodeURIComponent(ISS_TOKEN)}' in PAGE
    assert PAGE.count('"x-iss-token": ISS_TOKEN') == 2   # /stats and /log
    assert "WT connect: ${wtBase}" in PAGE                # not the URL with the token
