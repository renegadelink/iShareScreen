"""Non-Mac VNC servers: VNC password and no-auth security, RFB 3.7, refusals.

The fake server restates RFC 6143 independently of the client, so these tests
check the viewer against the specification as the fake reads it. They are not a
substitute for trying a real third-party server (TigerVNC, x11vnc, ...).
"""
import pytest

from desktop_helpers import pump, verify
from fake_vnc import FakeVNC, _des_response
from isharescreen.frontend.desktop import session as S
from isharescreen.proxy import standard as P
from isharescreen.proxy.address import ProbeError, probe_server


def connect(server, user='', password='', **kwargs):
    return P.StandardClient('127.0.0.1', user, password, server.port, **kwargs)


# ── protocol ─────────────────────────────────────────────────────────

def test_vnc_password_authentication_reaches_pixels(server):
    fake = server(security='vnc', users={'': 'sekret'}, colour=(1, 2, 3))
    client = connect(fake, password='sekret')
    try:
        assert client.read_update() == 'frame' and client.pixels[0, 0].tolist() == [1, 2, 3, 255]
    finally:
        client.close()
    assert fake.authenticated == 1


def test_vnc_password_rejection_carries_the_servers_reason(server):
    fake = server(security='vnc', users={'': 'sekret'}, reason='Too many authentication failures')
    with pytest.raises(P.AuthRejected) as caught:
        connect(fake, password='nope')
    assert 'password' in str(caught.value) and 'Too many authentication failures' in str(caught.value)
    assert 'nope' not in str(caught.value) and fake.rejected == 1


@pytest.mark.parametrize('password', ['a', 'sekret', 'exactly8', 'longer-than-eight', 'pässwörd'])
def test_des_response_matches_the_independent_reference(password):
    challenge = bytes(range(16))
    assert P._vnc_password_response(challenge, password) == _des_response(challenge, password)


def test_no_authentication_server_needs_no_credentials(server):
    client = connect(server(security='none'))
    try:
        assert client.read_update() == 'frame'
    finally:
        client.close()


def test_rfb_3_7_none_has_no_security_result_and_3_7_vnc_still_authenticates(server):
    fake = server(security='none', banner=b'RFB 003.007\n')
    client = connect(fake)
    try:
        assert client.read_update() == 'frame'
    finally:
        client.close()
    old = server(security='vnc', banner=b'RFB 003.007\n', users={'': 'sekret'})
    client = connect(old, password='sekret')
    client.close()
    with pytest.raises(P.AuthRejected):
        connect(old, password='wrong')


def test_unsupported_security_and_old_versions_are_explained(server):
    with pytest.raises(P.UnsupportedSecurity) as caught:
        connect(server(security='unsupported'))
    assert 'TLS' in str(caught.value)
    with pytest.raises(P.UnsupportedSecurity) as old:
        connect(server(banner=b'RFB 003.003\n'))
    assert '3.3' in str(old.value)


def test_server_refusal_reason_is_reported(server):
    with pytest.raises(P.ServerRefused) as caught:
        connect(server(refuse='Too many connections'))
    assert 'Too many connections' in str(caught.value)
    assert S.classify_error(caught.value)[0] == 'refused'


def test_apple_authentication_is_always_preferred():
    assert P.choose_security([1, 2, 30, 33]) == (30, 'account')
    assert P.choose_security([1, 2]) == (2, 'password')
    assert P.choose_security([18, 1]) == (1, 'none')
    assert P.choose_security([16, 18, 19]) is None


def test_probe_reports_what_each_server_asks_for(server):
    expected = {'ard': 'account', 'vnc': 'password', 'none': 'none'}
    for security, auth in expected.items():
        result = probe_server('127.0.0.1', server(security=security).port, timeout=3)
        assert result.auth == auth
    ard = probe_server('127.0.0.1', server().port, timeout=3)
    assert 30 in ard.security and ard.version == '003.889'


def test_probe_rejects_servers_it_cannot_talk_to(server):
    with pytest.raises(ProbeError) as unsupported:
        probe_server('127.0.0.1', server(security='unsupported').port, timeout=3)
    assert unsupported.value.kind == 'unsupported' and 'TLS' in unsupported.value.message
    with pytest.raises(ProbeError) as old:
        probe_server('127.0.0.1', server(banner=b'RFB 003.003\n').port, timeout=3)
    assert old.value.kind == 'unsupported'
    with pytest.raises(ProbeError) as refused:
        probe_server('127.0.0.1', server(refuse='Server is full').port, timeout=3)
    assert refused.value.kind == 'refused' and 'Server is full' in refused.value.message


# ── wizard adapts to the server ──────────────────────────────────────

def test_wizard_asks_only_for_a_password_when_the_server_wants_only_that(qapp, app, server):
    fake = server(security='vnc', users={'': 'sekret'})
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    assert not wizard.account_box.isVisibleTo(wizard) and wizard.password_box.isVisibleTo(wizard)
    assert wizard.password_label.text() == 'VNC Password:' and 'requires a password' in wizard.message.text()
    assert 'user name' not in wizard.message.text()
    wizard.password.setText('wrong')
    wizard.connect_session()
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS and wizard.cred_status.text())
    assert 'rejected' in wizard.cred_status.text() and not app.sessions and app.store.items == []
    wizard.password.setText('sekret')
    wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    saved = app.store.items[0]
    assert saved.user == '' and 'sekret' not in (app.store.path.read_text())
    assert app.sessions[0].auth == 'password'


def test_password_is_still_required_when_the_server_wants_one(qapp, app, server):
    wizard = app.open_wizard()
    verify(qapp, wizard, server(security='vnc', users={'': 'sekret'}))
    wizard.connect_session()
    assert wizard.stack.currentIndex() == wizard.CREDENTIALS and 'Enter the password' in wizard.cred_status.text()


def test_wizard_connects_without_credentials_when_the_server_wants_none(qapp, app, server):
    fake = server(security='none')
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    assert not wizard.account_box.isVisibleTo(wizard) and not wizard.password_box.isVisibleTo(wizard)
    assert wizard.auth_note.isVisibleTo(wizard)
    wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    window = app.sessions[0]
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')

    class NoPrompt:
        def __init__(self, *args): raise AssertionError('no password prompt expected')
    S.PasswordDialog, original = NoPrompt, S.PasswordDialog
    try:
        window.reconnect()
    finally:
        S.PasswordDialog = original
    assert pump(qapp, lambda: window.state == 'Connected')


def test_wizard_refuses_to_continue_past_an_unsupported_server(qapp, app, server):
    wizard = app.open_wizard()
    wizard.address.setText(f"127.0.0.1:{server(security='unsupported').port}")
    wizard.check_address()
    assert pump(qapp, lambda: not wizard.is_checking)
    assert wizard.stack.currentIndex() == wizard.ADDRESS and 'TLS' in wizard.address_status.text()
