"""Library -> wizard -> session windows, against deterministic local RFB servers."""
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from desktop_helpers import USER, PASSWORD, pump, verify, sign_in, open_session
from isharescreen.frontend.desktop import session as S
from isharescreen.frontend.desktop import wizard as W
from isharescreen.frontend.desktop.library import LibraryError

def test_address_stage_blocks_unreachable_and_wrong_service(qapp, app, server, tmp_path):
    import socket
    closed = socket.socket(); closed.bind(('127.0.0.1', 0)); dead_port = closed.getsockname()[1]; closed.close()
    web = server(mode='http')
    wizard = app.open_wizard()
    for port in (dead_port, web.port):
        wizard.address.setText(f'127.0.0.1:{port}')
        wizard.check_address()
        assert wizard.is_checking and not wizard.continue_button.isEnabled()
        assert pump(qapp, lambda: not wizard.is_checking)
        assert wizard.stack.currentIndex() == wizard.ADDRESS           # never reaches credentials
        assert wizard.address_status.text().startswith('✕') and wizard.continue_button.text() == 'Try Again'
    wizard.address.setText('not a host')
    wizard.check_address()
    assert not wizard.is_checking and 'spaces' in wizard.address_status.text()
    assert wizard.address.text() == 'not a host'                       # input preserved for correction
    assert not (tmp_path / 'library.json').exists()


def test_verified_address_shows_credentials_and_hp_is_unavailable(qapp, app, server):
    wizard = app.open_wizard()
    fake = server(name='Verified')
    verify(qapp, wizard, fake)
    assert wizard.found_label.text().startswith('✓') and 'RFB 003.889' in wizard.found_label.text()
    assert f'127.0.0.1:{fake.port}' in wizard.message.text() and 'requires a user name and password' in wizard.message.text()
    assert wizard.mode.currentData() == 'standard' and wizard.mode.model().item(0).isEnabled()
    assert not wizard.mode.model().item(1).isEnabled() and wizard.mode.itemData(1, W.Qt.ItemDataRole.ToolTipRole)
    assert wizard.size().width() == W.CONTENT_WIDTH + 44                 # a compact sheet, like macOS
    wizard.username.setText(USER)
    wizard.go(wizard.ADDRESS)                                           # going back keeps everything
    assert wizard.address.text() == f'127.0.0.1:{fake.port}' and wizard.username.text() == USER


def test_password_field_is_masked_and_toggle_is_labelled(qapp, app):
    wizard = app.open_wizard()
    assert wizard.password.echoMode() == W.QLineEdit.EchoMode.Password
    assert wizard.show_password.text() == '' and not wizard.show_password.icon().isNull()   # a symbol, not a word
    assert wizard.show_password.accessibleName() == 'Show password'
    wizard.show_password.setChecked(True)
    assert wizard.password.echoMode() == W.QLineEdit.EchoMode.Normal
    assert wizard.show_password.accessibleName() == 'Hide password' and wizard.show_password.toolTip() == 'Hide password'
    wizard.show_password.setChecked(False)
    assert wizard.password.echoMode() == W.QLineEdit.EchoMode.Password


def test_wrong_password_is_recoverable_and_saves_nothing(qapp, app, server, tmp_path):
    fake = server()
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    sign_in(wizard, password='wrong-secret')
    assert wizard.stack.currentIndex() == wizard.CONNECTING
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS)
    assert 'rejected' in wizard.cred_status.text() and wizard.cred_status.text().startswith('✕')
    assert 'wrong-secret' not in wizard.cred_status.text()
    assert wizard.password.text() == '' and wizard.username.text() == USER and wizard.address.text() == f'127.0.0.1:{fake.port}'
    assert fake.rejected == 1 and not app.sessions and app.store.items == []
    assert not (tmp_path / 'library.json').exists()
    wizard.password.setText(PASSWORD)                                   # retry in the same wizard
    wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    assert len(app.store.items) == 1


def test_success_opens_session_window_saves_entry_and_never_the_password(qapp, app, server, tmp_path):
    fake = server(colour=(10, 200, 30), size=(4, 2))
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    wizard.name.setText('Studio')
    wizard.view.setCurrentIndex(1)
    sign_in(wizard)
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    window = app.sessions[0]
    assert pump(qapp, lambda: not app.wizards)                                # no blank wizard left behind
    assert window.state == 'Connected' and 'Studio' in window.windowTitle() and 'Connected' in window.windowTitle()
    assert window.screen_widget.image.size().width() == 4
    assert window.screen_widget.image.pixelColor(0, 0).green() == 200
    assert window.screen_widget.view == 'actual'                              # saved/selected view applied
    saved = app.store.items[0]
    assert (saved.host, saved.port, saved.user, saved.name, saved.view) == ('127.0.0.1', fake.port, USER, 'Studio', 'actual')
    assert saved.last_connected and window.system_id == saved.id
    assert PASSWORD not in (tmp_path / 'library.json').read_text()
    assert app.library.list.count() == 1
    assert app.library.list.item(0).data(Qt.ItemDataRole.UserRole + 1) == 1  # shown as connected


def test_black_remote_desktop_is_a_valid_success(qapp, app, server):
    window = open_session(qapp, app, server(colour=(0, 0, 0)))
    assert window.is_live and window.screen_widget.image.pixelColor(0, 0).red() == 0


def test_two_sessions_are_independent(qapp, app, server):
    one, two = server(colour=(200, 0, 0)), server(colour=(0, 0, 200))
    first = open_session(qapp, app, one)
    second = open_session(qapp, app, two)
    assert first is not second and first.controller is not second.controller
    assert len(app.store.items) == 2 and len(app.sessions) == 2
    two.push_frame(colour=(0, 100, 0))
    assert pump(qapp, lambda: second.screen_widget.image.pixelColor(0, 0).green() == 100)
    assert first.screen_widget.image.pixelColor(0, 0).red() == 200            # untouched
    first.close()
    assert pump(qapp, lambda: one.disconnects == 2)                           # its probe and its session
    assert second.is_live and two.disconnects == 1 and app.sessions == [second]   # only two's probe has ended
    two.push_frame(colour=(7, 8, 9))
    assert pump(qapp, lambda: second.screen_widget.image.pixelColor(0, 0).red() == 7)
    counts = [app.library.list.item(i).data(Qt.ItemDataRole.UserRole + 1) for i in range(app.library.list.count())]
    assert sorted(counts) == [0, 1]                                           # only the open one is "connected"


def test_double_submit_makes_one_connection(qapp, app, server):
    fake = server()
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    wizard.username.setText(USER); wizard.password.setText(PASSWORD)
    wizard.connect_session(); wizard.connect_session(); wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    assert fake.connections == 2 and fake.authenticated == 1                  # 1 probe + 1 session


def test_cancel_stops_network_work_and_cleans_up(qapp, app, server, tmp_path):
    stalled = server(mode='stall')
    wizard = app.open_wizard()
    verify(qapp, wizard, stalled)                                             # stall mode still sends a banner
    wizard.username.setText(USER); wizard.password.setText(PASSWORD)
    wizard.connect_session()
    assert pump(qapp, lambda: stalled.connections == 2)
    live = list(S._LIVE)
    assert wizard.stack.currentIndex() == wizard.CONNECTING and len(live) == 1
    started = time.monotonic()
    wizard.cancel_connection()
    assert wizard.stack.currentIndex() == wizard.CREDENTIALS and wizard.connect_button.isEnabled()
    assert pump(qapp, lambda: not S._LIVE and not live[0].isRunning())        # worker really ended
    assert time.monotonic() - started < 3
    assert pump(qapp, lambda: stalled.disconnects == 2)                       # server saw the socket close
    assert not app.sessions and app.store.items == [] and wizard.address.text() == f'127.0.0.1:{stalled.port}'


def test_closing_wizard_mid_connect_abandons_the_attempt(qapp, app, server):
    stalled = server(mode='stall')
    wizard = app.open_wizard()
    verify(qapp, wizard, stalled)
    wizard.username.setText(USER); wizard.password.setText(PASSWORD)
    wizard.connect_session()
    assert pump(qapp, lambda: stalled.connections == 2)
    wizard.close()
    assert pump(qapp, lambda: not S._LIVE and stalled.disconnects == 2)
    assert not app.sessions and not app.wizards


def test_quitting_in_the_middle_of_an_address_check_exits_cleanly(tmp_path):
    """Regression: the check ran in a QThread, and Qt aborts the process if one is still alive at exit."""
    import subprocess, sys, textwrap
    from pathlib import Path
    tests_dir = str(Path(__file__).parent)
    code = textwrap.dedent(f'''
        import os, sys, time
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        sys.path.insert(0, {tests_dir!r})
        from fake_vnc import FakeVNC
        from PySide6.QtWidgets import QApplication
        from isharescreen.frontend.desktop.credentials import MemoryCredentialStore
        from isharescreen.frontend.desktop.library import LibraryStore
        from isharescreen.frontend.desktop.settings import Settings
        from isharescreen.frontend.desktop.standard_app import AppController
        from isharescreen.frontend.desktop.theme import apply_theme
        quiet = FakeVNC(mode='silent')                       # accepts, then says nothing: the check keeps waiting
        app = QApplication([]); apply_theme(app)
        controller = AppController(LibraryStore({str(tmp_path / 'l.json')!r}), MemoryCredentialStore(),
                                   Settings({str(tmp_path / 's.json')!r}))
        wizard = controller.open_wizard()
        wizard.address.setText(f'127.0.0.1:{{quiet.port}}'); wizard.check_address()
        for _ in range(15):
            app.processEvents(); time.sleep(0.02)
        assert wizard.is_checking
        controller.quit_now()
        print('quit while checking')
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.returncode, result.stderr[-800:])
    assert 'quit while checking' in result.stdout


def test_escape_cancels_a_running_check(qapp, app, server):
    quiet = server(mode='silent')
    wizard = app.open_wizard()
    wizard.address.setText(f'127.0.0.1:{quiet.port}')
    wizard.check_address()
    assert wizard.is_checking
    wizard.cancel_check()
    assert not wizard.is_checking and wizard.continue_button.isEnabled()
    assert wizard.stack.currentIndex() == wizard.ADDRESS
    assert pump(qapp, lambda: not W._PROBES)                                  # probe thread finished, nothing leaked


def test_saved_entry_reopens_prefilled_without_password_and_verifies_first(qapp, app, server):
    fake = server()
    first = open_session(qapp, app, fake)
    saved_id = first.system_id
    first.close()
    app.store.rename(saved_id, 'Desk')
    app.open_system(saved_id)
    wizard = app.wizards[-1]
    assert wizard.address.text() == f'127.0.0.1:{fake.port}' and wizard.username.text() == USER   # port included
    assert wizard.password.text() == '' and wizard.name.text() == 'Desk'      # name is kept, the password never is
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS)   # auto-verified
    wizard.password.setText(PASSWORD)
    wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[-1].is_live)
    assert len(app.store.items) == 1 and app.store.items[0].id == saved_id    # no duplicate entry
    assert 'Desk' in app.sessions[-1].windowTitle()


def test_library_save_failure_does_not_disturb_the_live_session(qapp, app, server, monkeypatch):
    def broken():
        raise LibraryError('Could not save the library (disk full).')
    monkeypatch.setattr(app.store, 'save', broken)
    window = open_session(qapp, app, server())
    assert window.is_live and window.system_id is None and app.store.items == []
    assert app.notices and 'not saved to the library' in app.notices[0] and 'disk full' in app.notices[0]


def test_remote_disconnect_shows_banner_and_reconnect_is_offered(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    assert not window.banner.isHidden() and 'Connection lost' in window.banner_text.text()
    assert window.reconnect_action.isVisible() and 'Disconnected' in window.windowTitle()
    assert app.sessions == [window] and PASSWORD not in window.banner_text.text()
    assert pump(qapp, lambda: not S._LIVE)


def _answer_with(monkeypatch, password):
    class Dialog:
        def __init__(self, parent, user, address):
            assert user == USER and '127.0.0.1' in address       # asks for the secret again, never remembers it
        def exec(self): return S.QDialog.DialogCode.Accepted
        def password(self): return password
    monkeypatch.setattr(S, 'PasswordDialog', Dialog)


def test_reconnect_prompts_for_the_password_and_resumes_in_the_same_window(qapp, app, server, monkeypatch):
    fake = server()
    window = open_session(qapp, app, fake)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    _answer_with(monkeypatch, 'wrong-secret')
    window.reconnect()
    assert pump(qapp, lambda: 'rejected' in window.banner_text.text() and window.state == 'Disconnected')
    assert 'wrong-secret' not in window.banner_text.text() and not window.banner.isHidden()
    _answer_with(monkeypatch, PASSWORD)
    window.reconnect()
    assert pump(qapp, lambda: window.state == 'Connected')
    assert window.banner.isHidden() and not window.reconnect_action.isVisible()
    assert app.sessions == [window] and fake.authenticated == 2 and len(app.store.items) == 1


def test_input_reaches_only_its_own_server_and_is_released_on_blur(qapp, app, server):
    one, two = server(), server()
    first, second = open_session(qapp, app, one), open_session(qapp, app, two)
    screen = first.screen_widget
    screen.resize(400, 200)
    from PySide6.QtGui import QKeyEvent, QMouseEvent
    from PySide6.QtCore import QEvent, QPointF
    screen.grab()                                                                # paints once to compute the target rectangle
    screen.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, 'a'))
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(300, 100), screen.mapToGlobal(QPointF(300, 100)),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    screen.mousePressEvent(press)
    assert pump(qapp, lambda: ('key', ord('a'), True) in one.events and any(e[0] == 'pointer' and e[3] == 1 for e in one.events))
    screen.release_input()
    assert pump(qapp, lambda: ('key', ord('a'), False) in one.events and one.events[-1][0] == 'pointer' and one.events[-1][3] == 0)
    assert two.events == []


def test_quit_closes_every_session_and_wizard(qapp, app, server):
    one, two = server(), server()
    open_session(qapp, app, one); open_session(qapp, app, two)
    app.open_wizard()
    app._confirm_quit = lambda count: count == 2
    app.request_quit()
    assert not app.sessions and not app.wizards and not S._LIVE
    assert pump(qapp, lambda: one.disconnects == 2 and two.disconnects == 2)


def test_quit_prompt_can_be_declined(qapp, app, server):
    window = open_session(qapp, app, server())
    app._confirm_quit = lambda count: False
    app.request_quit()
    assert app.sessions == [window] and window.is_live


def test_library_actions_rename_and_remove_keep_sessions_open(qapp, app, server):
    window = open_session(qapp, app, server())
    system_id = window.system_id
    app._ask_name = lambda system: 'Renamed'
    app.rename_system(system_id)
    assert app.store.get(system_id).name == 'Renamed' and app.library.list.item(0).text() == 'Renamed'
    app._confirm_remove = lambda system: False
    app.remove_system(system_id)
    assert app.store.items
    app._confirm_remove = lambda system: True
    app.remove_system(system_id)
    assert app.store.items == [] and app.library.stack.currentIndex() == 0   # back to the empty state
    assert window.is_live and app.sessions == [window]                       # removal never disconnects
    assert window.system_id == system_id                                     # (and the window keeps working)


def test_library_search_and_empty_state(qapp, app):
    assert app.library.stack.currentIndex() == 0 and not app.library.search_button.isEnabled()   # nothing to search
    app.store.record_success('alpha.local', 5900, 'a', name='Alpha')
    app.store.record_success('beta.local', 5900, 'b', name='Beta')
    app.refresh_library()
    assert app.library.stack.currentIndex() == 1 and app.library.search_button.isEnabled()
    assert not app.library.search.isVisibleTo(app.library)                    # hidden until the ⌕ button is used
    app.library.toggle_search()
    assert app.library.search.isVisibleTo(app.library)
    app.library.search.setText('bet')
    assert app.library.list.count() == 1 and app.library.list.item(0).text() == 'Beta'
    app.library.search.setText('zzz')
    assert app.library.list.count() == 0 and 'zzz' in app.library.no_match.text()


def test_screen_maps_pointer_in_fit_and_actual_modes(qapp):
    sent = []

    class Sink:
        def send(self, *args): sent.append(args)
    screen = S.Screen(Sink())
    screen.set_image(QImage(100, 50, QImage.Format.Format_RGBA8888))
    screen.resize(400, 400)                       # letterboxed: image is 400x200 centred vertically
    screen.grab()
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF

    def click(x, y):
        screen.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                                           Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    click(200, 200)
    assert sent[-1] == ('pointer', 50, 25, 1)
    click(0, 0)                                   # outside the image: clamped to the corner
    assert sent[-1] == ('pointer', 0, 0, 1)
    screen.set_view('actual')
    screen.grab()
    assert screen.size().width() == 100 and screen.size().height() == 50
    click(10, 20)
    assert sent[-1] == ('pointer', 10, 20, 1)
    screen.release_input()
    assert sent[-1] == ('pointer', 10, 20, 0)
    screen.set_view('fit')
    assert screen.maximumWidth() > 1000


def test_classify_error_never_leaks_secrets_and_distinguishes_causes():
    import socket
    from isharescreen.proxy import standard as P
    cases = [(P.AuthRejected('x'), 'auth'), (P.UnsupportedSecurity('x'), 'unsupported'),
             (P.UnsupportedEncoding('7'), 'encoding'), (P.DesktopUnavailable('x'), 'unavailable'),
             (S.RenderError('x'), 'render'), (socket.timeout(), 'timeout'), (ConnectionRefusedError(), 'refused'),
             (ConnectionResetError(), 'disconnect'), (ValueError('bad'), 'protocol'), (OSError(1, 'boom'), 'network')]
    for error, kind in cases:
        assert S.classify_error(error)[0] == kind
