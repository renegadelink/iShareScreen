"""High Performance (experimental): the Qt controller, gating and settings, against a fake engine.

Nothing here talks to a real Mac. In live tests the real engine did not deliver video
from either test Mac, which is why the option is off until switched on.
"""
import json
import os
import threading
import time

import numpy as np
import pytest
from PySide6.QtGui import QImage

from desktop_helpers import USER, PASSWORD, pump, verify
from isharescreen.frontend.desktop import hp as H
from isharescreen.frontend.desktop import wizard as W
from isharescreen.frontend.desktop.library import LibraryStore, valid_resolution
from isharescreen.frontend.desktop.settings import Settings
from isharescreen.proxy.media.tiles import TileFrame
from isharescreen.proxy.protocol.auth import AuthError


@pytest.fixture(autouse=True)
def keep_environment_clean():
    """The controller sets engine switches in os.environ (the engine reads them there);
    no other test may see them."""
    saved = {key: os.environ.get(key) for key in ('ISS_VIDEO_CODEC', 'ISS_CLIP')}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ── a stand-in for the engine ────────────────────────────────────────

class FakeInput:
    def __init__(self): self.calls = []
    def pointer_event(self, buttons, x, y): self.calls.append(('pointer', buttons, x, y))
    def scroll_event(self, x, y, dx, dy): self.calls.append(('scroll', x, y, dx, dy))
    def key_event(self, down, keysym): self.calls.append(('key', down, keysym))
    def cut_text(self, text): self.calls.append(('cut_text', text))


def make_tile(width, height, y, u, v, subsampled=False, pad=0):
    cw, ch = ((width + 1) // 2, (height + 1) // 2) if subsampled else (width, height)
    y_plane = np.full((height, width + pad), y, np.uint8)
    u_plane = np.full((ch, cw + pad), u, np.uint8)
    v_plane = np.full((ch, cw + pad), v, np.uint8)
    return TileFrame(y_plane.tobytes(), u_plane.tobytes(), v_plane.tobytes(), width, height,
                     width + pad, cw + pad, cw, ch)


class FakeSession:
    """Mimics the parts of Session the controller uses."""
    def __init__(self, config, behaviour='ok', canvas=(32, 16), tile_count=2):
        self.config, self.behaviour = config, behaviour
        self.canvas, self.tile_count = canvas, tile_count
        self.input = FakeInput()
        self.connected = False
        self.closed = threading.Event()
        self.close_calls = 0
        self.pending = []                      # colours to hand out as the next picture
        self.clipboard_callback = None
        self.fir_requests = 0
        self.lock = threading.Lock()

    def set_clipboard_text_callback(self, callback): self.clipboard_callback = callback

    def connect(self):
        if self.behaviour == 'hang':
            self.closed.wait(30)
            raise ConnectionError('closed while connecting')
        if self.behaviour == 'no_video':
            raise RuntimeError('HP stream never started: 0 HEVC video packets arrived across 3 fresh handshakes.')
        if self.behaviour == 'auth':
            raise AuthError('bad credentials')
        if self.behaviour == 'decoder':
            raise RuntimeError('no usable decoder for this stream')
        self.connected = True

    @property
    def is_connected(self): return self.connected and not self.closed.is_set()
    @property
    def canvas_dims(self): return self.canvas
    @property
    def num_tiles(self): return self.tile_count

    def push(self, y=100, u=128, v=128):
        with self.lock:
            self.pending.append((y, u, v))

    def wait_for_fresh_tile(self, timeout=0.03):
        end = time.time() + timeout
        while time.time() < end:
            if self.pending or self.closed.is_set():
                return bool(self.pending)
            time.sleep(0.005)
        return False

    def get_frame(self, index):
        with self.lock:
            if not self.pending:
                return None
            y, u, v = self.pending[0]
            if index == self.tile_count - 1:
                self.pending.pop(0)
            return make_tile(self.canvas[0], self.canvas[1] // self.tile_count, y, u, v)

    def request_fir(self, tile): self.fir_requests += 1

    def close(self):
        self.close_calls += 1
        self.closed.set()
        self.connected = False


def controller(qapp, behaviour='ok', host='127.0.0.1', port=5900, **kwargs):
    made = []

    def factory(config):
        made.append(FakeSession(config, behaviour, **kwargs))
        return made[0]
    c = H.HPController(host, port, USER, PASSWORD, resolution=kwargs.pop('resolution', 'auto'), session_factory=factory)
    c.fake = lambda: made[0] if made else None
    return c


def run(qapp, c, until, timeout=6):
    events = {'frames': [], 'established': [], 'failed': [], 'clipboard': [], 'phase': []}
    c.frame.connect(events['frames'].append); c.established.connect(events['established'].append)
    c.failed.connect(lambda kind, text: events['failed'].append((kind, text)))
    c.clipboard.connect(events['clipboard'].append); c.phase.connect(events['phase'].append)
    c.start()
    assert pump(qapp, lambda: until(events), timeout), events
    return events


# ── pictures ─────────────────────────────────────────────────────────

def reference_rgb(y, u, v):
    """BT.709, full range, in floating point."""
    y, u, v = (np.asarray(a, np.float32) for a in (y, u - 128, v - 128))
    return np.stack([y + 1.5748 * v, y - 0.1873 * u - 0.4681 * v, y + 1.8556 * u], axis=-1).clip(0, 255)


@pytest.mark.parametrize('subsampled', [False, True])
def test_conversion_matches_bt709_full_range(subsampled):
    rng = np.random.default_rng(7)
    width, height = 48, 24
    ys = rng.integers(0, 256, (height, width), dtype=np.uint8)
    cw, ch = (width // 2, height // 2) if subsampled else (width, height)
    us, vs = (rng.integers(0, 256, (ch, cw), dtype=np.uint8) for _ in range(2))
    tile = TileFrame(ys.tobytes(), us.tobytes(), vs.tobytes(), width, height, width, cw, cw, ch)
    rgba = H.compose_rgba({0: tile}, width, height)
    up = (lambda a: np.repeat(np.repeat(a, 2, 0), 2, 1)) if subsampled else (lambda a: a)
    expected = reference_rgb(ys, up(us).astype(np.int32), up(vs).astype(np.int32))
    assert np.abs(rgba[..., :3].astype(np.float32) - expected).max() <= 2.5
    assert (rgba[..., 3] == 255).all()


def test_known_colours():
    white = H.compose_rgba({0: make_tile(4, 4, 255, 128, 128)}, 4, 4)
    black = H.compose_rgba({0: make_tile(4, 4, 0, 128, 128)}, 4, 4)
    red = H.compose_rgba({0: make_tile(4, 4, 54, 99, 255)}, 4, 4)          # BT.709 red
    assert white[0, 0].tolist() == [255, 255, 255, 255] and black[0, 0].tolist() == [0, 0, 0, 255]
    assert red[0, 0, 0] > 240 and red[0, 0, 1] < 15 and red[0, 0, 2] < 15


def test_tiles_stack_into_rows_and_padding_is_clipped():
    tiles = {0: make_tile(16, 6, 50, 128, 128, pad=8), 1: make_tile(16, 6, 200, 128, 128, pad=8)}   # strided rows
    rgba = H.compose_rgba(tiles, 16, 10)                                    # canvas shorter than 2 * 6 rows
    assert rgba.shape == (10, 16, 4)
    assert rgba[0, 0, 0] == 50 and rgba[5, 15, 0] == 50                     # tile 0: rows 0..5
    assert rgba[6, 0, 0] == 200 and rgba[9, 15, 0] == 200                   # tile 1 starts at row 6, clipped at 10


def test_subsampled_tiles_stack_with_their_chroma():
    tiles = {0: make_tile(16, 8, 128, 90, 200, subsampled=True), 1: make_tile(16, 8, 128, 200, 60, subsampled=True)}
    rgba = H.compose_rgba(tiles, 16, 16)
    assert rgba[0, 0, 0] > 150 and rgba[0, 0, 2] < 120                       # strong V: reddish
    assert rgba[12, 5, 2] > 150 and rgba[12, 5, 0] < 120                     # strong U: bluish, lower strip


def test_nv12_passthrough_is_rejected():
    tile = TileFrame(b'\0' * 16, b'\0' * 8, None, 4, 4, 4, 4, 2, 2)
    with pytest.raises(ValueError):
        H.compose_rgba({0: tile}, 4, 4)


# ── gating, settings, parsing ────────────────────────────────────────

class Probe:
    def __init__(self, auth='account', security=(30, 33, 36)): self.auth, self.security = auth, security


def test_capability_reasons():
    ok, note = H.hp_capability(Probe(), True)
    assert ok and 'Experimental' in note and 'unresponsive' in note
    off, why = H.hp_capability(Probe(), False)
    assert not off and 'coming in a later version' in why
    assert not H.hp_capability(Probe(auth='password'), True)[0]
    assert 'account' in H.hp_capability(Probe(auth='none'), True)[1]
    assert 'login' in H.hp_capability(Probe(security=(30, 35)), True)[1]


def test_resolution_parsing_and_validation():
    assert H.parse_resolution('auto') == (1920, 1080) and H.parse_resolution('') == (1920, 1080)
    assert H.parse_resolution('1280x720') == (1280, 720)
    assert H.parse_resolution('9999x9999') == (1920, 1080) and H.parse_resolution('junk') == (1920, 1080)
    assert valid_resolution('1600x900') == '1600x900' and valid_resolution('10x10') == 'auto'
    assert valid_resolution(None) == 'auto' and valid_resolution(5) == 'auto' and valid_resolution('4000x3000') == 'auto'


def test_settings_persist_ignore_junk_and_survive_damage(tmp_path):
    path = tmp_path / 'sub' / 'settings.json'
    settings = Settings(path)
    assert settings.experimental_hp is False and not path.exists()
    assert settings.set('experimental_hp', True) and Settings(path).experimental_hp is True
    path.write_text(json.dumps({'experimental_hp': 'yes', 'password': 'x', 'other': True}))
    assert Settings(path).experimental_hp is False                           # only real booleans count
    path.write_text('{broken')
    assert Settings(path).experimental_hp is False
    with pytest.raises(KeyError):
        settings.set('unknown', True)


def test_settings_write_failure_keeps_the_old_value(tmp_path, monkeypatch):
    settings = Settings(tmp_path / 'settings.json')
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError(13, 'denied')))
    assert settings.set('experimental_hp', True) is False and settings.experimental_hp is False
    assert not [p for p in tmp_path.iterdir() if p.suffix == '.tmp']


# ── the controller ───────────────────────────────────────────────────

def test_config_uses_the_least_disruptive_settings(qapp, monkeypatch):
    monkeypatch.delenv('ISS_VIDEO_CODEC', raising=False)
    c = controller(qapp, resolution='1280x720')
    ev = run(qapp, c, lambda e: c.fake() is not None and c.fake().connected)
    config = c.fake().config
    assert (config.curtain, config.share_console, config.alt_session, config.audio) == (False, False, False, False)
    assert (config.advertise.width, config.advertise.height, config.advertise.hidpi_scale) == (1280, 720, 1.0)
    assert config.host == '127.0.0.1' and config.username == USER
    assert os.environ['ISS_VIDEO_CODEC'] == 'avc' and os.environ['ISS_CLIP'] == '0'
    with pytest.raises(RuntimeError, match='Another user'):                  # never asks someone else to share
        config.on_session_choice('bob')
    c.stop(); c.wait(3000)
    assert c._password == ''                                                 # the secret does not linger


def test_pictures_flow_and_input_maps_to_the_engine(qapp):
    c = controller(qapp)
    ev = run(qapp, c, lambda e: c.fake() is not None and c.fake().connected)
    fake = c.fake()
    assert 'Starting High Performance…' in ev['phase']
    fake.push(60); fake.push(90)
    assert pump(qapp, lambda: len(ev['frames']) >= 1)
    assert len(ev['established']) == 1 and ev['established'][0].width() == 32 and ev['established'][0].height() == 16
    assert c.last_image.pixelColor(5, 3).red() in range(58, 63) and c.is_established
    c.send('pointer', 7, 9, 1)                                                # left button
    c.send('pointer', 7, 9, 2 | 8)                                            # right button + wheel up
    c.send('pointer', 7, 9, 16)                                               # wheel down
    c.send('key', 0x61, True); c.send('key', 0xff0d, False)
    c.send('send_clipboard', 'héllo 日本'); c.send('request', True)
    assert fake.input.calls == [('pointer', 1, 7, 9), ('scroll', 7, 9, 0, -H.WHEEL_TICKS), ('pointer', 2, 7, 9),
                                ('scroll', 7, 9, 0, H.WHEEL_TICKS), ('pointer', 0, 7, 9),
                                ('key', True, 0x61), ('key', False, 0xff0d), ('cut_text', 'héllo 日本')]
    assert fake.fir_requests == 1                                             # Refresh asks for a fresh keyframe
    fake.clipboard_callback('from the mac')
    assert pump(qapp, lambda: ev['clipboard'] == ['from the mac'])
    c.stop()
    assert pump(qapp, lambda: fake.closed.is_set()) and c.wait(3000)
    assert ev['failed'] == []                                                 # a user stop is not a failure
    c.send('key', 0x62, True)                                                 # and later input is ignored
    assert ('key', True, 0x62) not in fake.input.calls


def test_converting_is_throttled_but_the_last_picture_is_not_lost(qapp):
    c = controller(qapp)
    ev = run(qapp, c, lambda e: c.fake() is not None and c.fake().connected)
    fake = c.fake()
    for shade in range(40):
        fake.push(shade * 5)
    assert pump(qapp, lambda: c.last_image.pixelColor(2, 2).red() in range(193, 198))   # the newest one arrived
    assert len(ev['frames']) < 40                                              # older ones were skipped, not all converted
    c.stop(); c.wait(3000)


@pytest.mark.parametrize('behaviour,kind,words', [
    ('no_video', 'hp_no_video', ['sent no video', 'Standard mode works', 'unresponsive']),
    ('auth', 'auth', ['rejected the user name or password']),
    ('decoder', 'hp_decoder', ['video decoder']),
])
def test_failures_are_explained(qapp, behaviour, kind, words):
    c = controller(qapp, behaviour)
    ev = run(qapp, c, lambda e: e['failed'])
    assert ev['failed'][0][0] == kind and all(w in ev['failed'][0][1] for w in words)
    assert c.fake().closed.is_set() and not ev['established']
    assert pump(qapp, lambda: c.isFinished())


def test_a_connection_lost_mid_session_is_reported(qapp):
    c = controller(qapp)
    ev = run(qapp, c, lambda e: c.fake() is not None and c.fake().connected)
    c.fake().push(); assert pump(qapp, lambda: ev['established'])
    c.fake().connected = False
    assert pump(qapp, lambda: ev['failed'])
    assert ev['failed'][0][0] == 'network' and 'closed' in ev['failed'][0][1]


def test_an_attempt_that_never_produces_a_picture_is_abandoned(qapp, monkeypatch):
    monkeypatch.setattr(H, 'START_TIMEOUT_S', 0.4)
    c = controller(qapp, 'hang')
    started = time.monotonic()
    ev = run(qapp, c, lambda e: e['failed'])
    assert ev['failed'][0][0] == 'hp_timeout' and 'Standard mode works' in ev['failed'][0][1]
    assert time.monotonic() - started < 4 and c.fake().closed.is_set()


def test_a_connected_session_without_pictures_also_times_out(qapp, monkeypatch):
    monkeypatch.setattr(H, 'START_TIMEOUT_S', 0.4)
    c = controller(qapp)                                   # connects, but the Mac never sends a picture
    ev = run(qapp, c, lambda e: e['failed'])
    assert ev['failed'][0][0] == 'hp_timeout' and not ev['established']


def test_the_deadline_stops_applying_once_a_picture_arrives(qapp, monkeypatch):
    monkeypatch.setattr(H, 'START_TIMEOUT_S', 0.5)
    c = controller(qapp)
    ev = run(qapp, c, lambda e: c.fake() is not None and c.fake().connected)
    c.fake().push()
    assert pump(qapp, lambda: ev['established'])
    time.sleep(0.9)
    qapp.processEvents()
    assert ev['failed'] == [] and not c.fake().closed.is_set()
    c.stop(); c.wait(3000)


def test_only_one_attempt_per_mac_at_a_time(qapp):
    first = controller(qapp, 'hang', host='10.9.9.9', port=5900)
    first.start()
    assert pump(qapp, lambda: first.fake() is not None)
    second = controller(qapp, host='10.9.9.9', port=5900)
    ev = run(qapp, second, lambda e: e['failed'])
    assert ev['failed'][0][0] == 'hp_busy' and second.fake() is None          # never reached the engine
    other = controller(qapp, host='10.9.9.8', port=5900)                       # a different Mac is unaffected
    run(qapp, other, lambda e: other.fake() is not None and other.fake().connected)
    other.stop(); other.wait(3000)
    first.stop()
    assert pump(qapp, lambda: first.isFinished(), 8)
    third = controller(qapp, host='10.9.9.9', port=5900)                       # and the slot is free again
    run(qapp, third, lambda e: third.fake() is not None and third.fake().connected)
    third.stop(); third.wait(3000)


def test_stop_never_blocks_the_caller_on_a_slow_close(qapp):
    class SlowClose(FakeSession):
        def close(self):
            time.sleep(1.5)
            super().close()
    c = H.HPController('127.0.0.1', 5900, USER, PASSWORD, session_factory=lambda config: SlowClose(config))
    ev = run(qapp, c, lambda e: c.session is not None)
    started = time.monotonic()
    c.stop()
    assert time.monotonic() - started < 0.5                                     # the close runs in the background
    c.wait(6000)


def test_clone_keeps_the_kind_and_the_resolution(qapp):
    c = H.HPController('h', 5901, 'u', 'old', auth='account', resolution='1280x720')
    fresh = c.clone('new')
    assert isinstance(fresh, H.HPController) and fresh is not c
    assert (fresh.host, fresh.port, fresh.user, fresh.resolution, fresh._password) == ('h', 5901, 'u', '1280x720', 'new')
    assert fresh.mode_label == 'High Performance' and c.__class__.__mro__[1].mode_label == 'Standard VNC'


# ── the whole flow, with High Performance chosen ─────────────────────

def hp_wizard(qapp, app, server, behaviour='ok'):
    """Experimental HP on; the wizard builds HPControllers whose engine is the fake."""
    app.settings.set('experimental_hp', True)
    sessions = []

    def make(host, port, user, secret, auth='account', resolution='auto'):
        def factory(config):
            sessions.append(FakeSession(config, behaviour))
            threading.Timer(0.2, lambda: sessions[-1].push(120) if sessions else None).start()
            return sessions[-1]
        return H.HPController(host, port, user, secret, auth, resolution, factory)
    W.HPController = make
    wizard = app.open_wizard()
    verify(qapp, wizard, server())
    return wizard, sessions


@pytest.fixture(autouse=True)
def restore_wizard_controller():
    original = W.HPController
    yield
    W.HPController = original


def test_high_performance_is_offered_only_when_switched_on(qapp, app, server):
    off = app.open_wizard(); verify(qapp, off, server())
    assert not off.mode.model().item(1).isEnabled() and 'coming in a later version' in off.mode.itemData(1, W.Qt.ItemDataRole.ToolTipRole)
    app.settings.set('experimental_hp', True)
    on = app.open_wizard(); verify(qapp, on, server())
    assert on.mode.model().item(1).isEnabled() and 'Experimental' in on.mode.itemData(1, W.Qt.ItemDataRole.ToolTipRole)
    assert not on.resolution.isVisibleTo(on) and not on.hp_note.isVisibleTo(on)       # only once it is chosen
    on.mode.setCurrentIndex(1)
    assert on.resolution.isVisibleTo(on) and on.hp_note.isVisibleTo(on) and 'unresponsive' in on.hp_note.text()
    assert [on.resolution.itemData(i) for i in range(on.resolution.count())][0] == 'auto'
    on.mode.setCurrentIndex(0)
    assert not on.resolution.isVisibleTo(on)


def test_password_only_and_passwordless_servers_never_offer_it(qapp, app, server):
    app.settings.set('experimental_hp', True)
    for security in ('vnc', 'none'):
        wizard = app.open_wizard(); verify(qapp, wizard, server(security=security, users={'': 'x'}))
        assert not wizard.mode.model().item(1).isEnabled()


def test_connecting_with_high_performance_opens_a_session_and_saves_the_choice(qapp, app, server):
    wizard, sessions = hp_wizard(qapp, app, server)
    wizard.mode.setCurrentIndex(1)
    wizard.resolution.setCurrentIndex(wizard.resolution.findData('1280x720'))
    wizard.username.setText(USER); wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live, 8)
    window = app.sessions[0]
    assert isinstance(window.controller, H.HPController) and window.template.mode_label == 'High Performance'
    assert 'High Performance' in window.info_label.text()
    saved = app.store.items[0]
    assert (saved.mode, saved.resolution) == ('high_performance', '1280x720')
    assert 'High Performance' in app.library.list.item(0).data(0x0100 + 2)
    assert (sessions[0].config.advertise.width, sessions[0].config.advertise.height) == (1280, 720)
    window.close()


def test_a_saved_high_performance_entry_reopens_as_high_performance(qapp, app, server):
    wizard, sessions = hp_wizard(qapp, app, server)
    wizard.mode.setCurrentIndex(1); wizard.resolution.setCurrentIndex(wizard.resolution.findData('1600x900'))
    wizard.username.setText(USER); wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live, 8)
    app.sessions[0].close()
    saved = app.store.items[0]
    app.open_wizard(prefill=saved)
    reopened = app.wizards[-1]
    assert pump(qapp, lambda: reopened.stack.currentIndex() == reopened.CREDENTIALS)
    assert reopened.mode.currentData() == 'high_performance' and reopened.resolution.currentData() == '1600x900'


def test_a_failed_high_performance_attempt_returns_to_sign_in_and_saves_nothing(qapp, app, server):
    wizard, sessions = hp_wizard(qapp, app, server, behaviour='no_video')
    wizard.mode.setCurrentIndex(1)
    wizard.username.setText(USER); wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS and wizard.cred_status.text(), 8)
    assert 'Standard mode works' in wizard.cred_status.text() and wizard.cred_status.text().startswith('✕')
    assert app.store.items == [] and not app.sessions and wizard.mode.currentData() == 'high_performance'
    wizard.mode.setCurrentIndex(0)                                            # fall back to Standard in the same dialog
    wizard.password.setText(PASSWORD)
    assert wizard.connect_button.isEnabled()


def test_reconnect_makes_another_high_performance_controller(qapp, app, server):
    wizard, sessions = hp_wizard(qapp, app, server)
    wizard.mode.setCurrentIndex(1)
    wizard.username.setText(USER); wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live, 8)
    window = app.sessions[0]
    sessions[0].connected = False                                             # the connection drops
    assert pump(qapp, lambda: window.state == 'Disconnected', 8)
    window.saved_secret = lambda: PASSWORD
    window.reconnect()
    assert isinstance(window.controller, H.HPController) and window.controller is not window.template
    assert pump(qapp, lambda: window.state == 'Connected', 8) and len(sessions) == 2
    window.close()


# ── the switch ───────────────────────────────────────────────────────

def test_the_menu_switch_asks_first_and_remembers(qapp, app):
    seen = []
    app.library.hp_toggled.connect(seen.append)
    assert app.library.hp_action.isCheckable() and not app.library.hp_action.isChecked()
    app._confirm_experimental = lambda: False
    app.library.hp_action.trigger()
    assert seen == [True] and app.settings.experimental_hp is False and not app.library.hp_action.isChecked()   # declined
    app._confirm_experimental = lambda: True
    app.library.hp_action.trigger()
    assert app.settings.experimental_hp is True and app.library.hp_action.isChecked()
    assert Settings(app.settings.path).experimental_hp is True               # remembered
    app.library.hp_action.trigger()                                           # turning it off needs no confirmation
    assert app.settings.experimental_hp is False and not app.library.hp_action.isChecked()


def test_the_switch_reports_a_setting_that_could_not_be_saved(qapp, app, monkeypatch):
    app._confirm_experimental = lambda: True
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError(13, 'denied')))
    app.library.hp_action.trigger()
    assert app.settings.experimental_hp is False and not app.library.hp_action.isChecked()
    assert any('could not be saved' in note for note in app.notices)


def test_the_switch_state_is_shown_on_start(qapp, tmp_path):
    from isharescreen.frontend.desktop.credentials import MemoryCredentialStore
    from isharescreen.frontend.desktop.standard_app import AppController
    settings = Settings(tmp_path / 's.json'); settings.set('experimental_hp', True)
    app = AppController(LibraryStore(tmp_path / 'l.json'), MemoryCredentialStore(), settings)
    try:
        assert app.library.hp_action.isChecked()
    finally:
        app.library.close = lambda: None


def test_the_sheet_offers_high_performance_and_resolution_together(qapp, app):
    from isharescreen.frontend.desktop.edit_dialog import EditSystemDialog
    saved = app.store.record_success('a.local', 5900, 'ana', mode='standard')
    dialog = EditSystemDialog(None, saved, hp_enabled=True)
    assert dialog.mode.model().item(1).isEnabled() and not dialog.resolution.isEnabled()
    dialog.mode.setCurrentIndex(1)
    assert dialog.resolution.isEnabled()
    dialog.resolution.setCurrentIndex(dialog.resolution.findData('1366x768'))
    dialog.save.click()
    assert dialog.values()['mode'] == 'high_performance' and dialog.values()['resolution'] == '1366x768'
    app._ask_edit = lambda system: dialog.values() | {'connect': False}
    app.edit_system(saved.id)
    item = app.store.get(saved.id)
    assert (item.mode, item.resolution) == ('high_performance', '1366x768')
