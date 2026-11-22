"""--alt-session never falls back to the console, and the browser bridge
doesn't keep retrying a refused login or a session the Mac doesn't offer."""
from __future__ import annotations

import pytest

from isharescreen.proxy.protocol import negotiation as neg


class _Sock:
    def __init__(self):
        self.sent = b""
        self.closed = False

    def sendall(self, data):
        self.sent += data

    def close(self):
        self.closed = True


class _Reached(Exception):
    pass


def _negotiate_without_prompt(monkeypatch, **kw):
    sock = _Sock()
    monkeypatch.setattr(neg, "_phase_auth", lambda *a, **k: (sock, b"\x00" * 16))
    monkeypatch.setattr(neg, "_phase_client_init", lambda s: (1920, 1080))
    monkeypatch.setattr(neg, "_peek_ss_prompt", lambda s, *a, **k: False)

    def handshake(*a, **k):
        raise _Reached()
    monkeypatch.setattr(neg, "_phase_handshake_plaintext", handshake)
    neg.connect_and_negotiate("mac", 5900, "u", "p", audio_offer=b"a",
                              video_offer=b"v", **kw)


def test_alt_session_without_a_session_choice_is_refused(monkeypatch):
    sock = _Sock()
    monkeypatch.setattr(neg, "_phase_auth", lambda *a, **k: (sock, b"\x00" * 16))
    monkeypatch.setattr(neg, "_phase_client_init", lambda s: (1920, 1080))
    monkeypatch.setattr(neg, "_peek_ss_prompt", lambda s, *a, **k: False)
    with pytest.raises(neg.SessionSelectError, match="did not offer a separate session"):
        neg.connect_and_negotiate("mac", 5900, "u", "p", audio_offer=b"a",
                                  video_offer=b"v", alt_session=True)
    assert sock.sent == b"" and sock.closed       # nothing shared, nothing sent


def test_console_sharing_without_a_session_choice_still_proceeds(monkeypatch):
    with pytest.raises(_Reached):
        _negotiate_without_prompt(monkeypatch, share_console=True)
    with pytest.raises(_Reached):
        _negotiate_without_prompt(monkeypatch)


@pytest.mark.parametrize("make_error", [
    lambda: __import__("isharescreen.proxy.protocol.auth", fromlist=["AuthError"]).AuthError("bad password"),
    lambda: neg.SessionSelectError("no separate session"),
])
def test_bridge_does_not_retry_what_cannot_succeed(monkeypatch, make_error):
    from isharescreen.frontend.wt import server
    from isharescreen.proxy.session import SessionConfig
    bridge = server.WebTransportBridge(SessionConfig(host="mac", username="u", password="p"))
    bridge.viewers.append(object())

    class FailingSession:
        def __init__(self, cfg):
            pass

        def connect(self):
            raise make_error()
    timers = []
    monkeypatch.setattr(server, "Session", FailingSession)
    monkeypatch.setattr(server.threading, "Timer",
                        lambda *a, **k: timers.append(a) or pytest.fail("retry scheduled"))
    bridge._start_session_thread()
    assert timers == [] and bridge._session is None and not bridge._session_starting


def test_bridge_still_retries_a_transient_failure(monkeypatch):
    from isharescreen.frontend.wt import server
    from isharescreen.proxy.session import SessionConfig
    bridge = server.WebTransportBridge(SessionConfig(host="mac", username="u", password="p"))
    bridge.viewers.append(object())

    class FlakySession:
        def __init__(self, cfg):
            pass

        def connect(self):
            raise RuntimeError("burst starved")

    class Timer:
        started = []

        def __init__(self, delay, fn):
            self.delay = delay

        def start(self):
            Timer.started.append(self.delay)
    monkeypatch.setattr(server, "Session", FlakySession)
    monkeypatch.setattr(server.threading, "Timer", Timer)
    bridge._start_session_thread()
    assert Timer.started == [1.5]
