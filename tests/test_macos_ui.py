"""The macOS-style UI: symbols, theme, the session toolbar, library, dialogs."""
import re
import time

import pytest
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QIcon, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFrame, QToolButton

from desktop_helpers import USER, pump, open_session, verify
from isharescreen.frontend.desktop import icons, theme
from isharescreen.frontend.desktop import library_window as LW
from isharescreen.frontend.desktop.edit_dialog import EditSystemDialog, resolve_address
from isharescreen.frontend.desktop.library import describe_last_connected


# ── symbols and theme ────────────────────────────────────────────────

def opaque(pixmap):
    image = pixmap.toImage()
    return sum(1 for x in range(0, image.width(), 2) for y in range(0, image.height(), 2) if image.pixelColor(x, y).alpha() > 0)


def test_every_symbol_draws_in_every_state(qapp):
    assert len(icons.glyph_names()) >= 25
    for name in icons.glyph_names():
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Disabled, QIcon.Mode.Selected):
            for state in (QIcon.State.Off, QIcon.State.On):
                pixmap = icons.glyph(name).pixmap(QSize(32, 32), mode, state)
                assert not pixmap.isNull() and opaque(pixmap) > 6, (name, mode, state)
    with pytest.raises(KeyError):
        icons.glyph('no-such-symbol')


def test_symbols_follow_the_palette_and_selection(qapp):
    normal = icons.glyph('launchpad').pixmap(QSize(32, 32), QIcon.Mode.Normal, QIcon.State.Off).toImage()
    selected = icons.glyph('launchpad').pixmap(QSize(32, 32), QIcon.Mode.Selected, QIcon.State.Off).toImage()
    on = icons.glyph('launchpad').pixmap(QSize(32, 32), QIcon.Mode.Normal, QIcon.State.On).toImage()
    probe = QPoint(7, 7)            # well inside the first tile of the grid
    palette = QApplication.palette()
    assert normal.pixelColor(probe).rgb() != selected.pixelColor(probe).rgb()
    assert selected.pixelColor(probe).rgb() == on.pixelColor(probe).rgb() == palette.highlightedText().color().rgb()


def test_theme_switches_between_dark_and_light(qapp):
    try:
        theme.apply_theme(qapp, theme.DARK)
        assert theme.tokens()['name'] == 'dark' and qapp.palette().window().color().name() == theme.DARK['window']
        assert theme.DARK['accent'] in qapp.styleSheet()
        dark_icon = icons.glyph('plus').pixmap(QSize(24, 24)).toImage()
        theme.apply_theme(qapp, theme.LIGHT)
        assert theme.tokens()['name'] == 'light' and qapp.palette().window().color().name() == theme.LIGHT['window']
        light_icon = icons.glyph('plus').pixmap(QSize(24, 24)).toImage()
        assert dark_icon.pixelColor(12, 12).rgb() != light_icon.pixelColor(12, 12).rgb()    # same icon, new colours
    finally:
        theme.apply_theme(qapp)


def test_theme_can_be_forced_and_every_token_is_used_consistently(monkeypatch):
    monkeypatch.setenv('ISS_THEME', 'dark')
    assert theme.wanted_theme() is theme.DARK
    monkeypatch.setenv('ISS_THEME', 'light')
    assert theme.wanted_theme() is theme.LIGHT
    assert theme.DARK.keys() == theme.LIGHT.keys()
    for tokens in (theme.DARK, theme.LIGHT):
        style = theme.build_style(tokens)
        for colour in ('accent', 'window', 'group', 'text', 'screen'):
            assert tokens[colour] in style                                      # every key colour reaches the style sheet
        assert '{t[' not in style and "{'" not in style                         # no unformatted placeholders left over


# ── session toolbar ──────────────────────────────────────────────────

def names(frame):
    return [b.accessibleName() for b in frame.findChildren(QToolButton)]


def test_toolbar_is_symbols_only_in_the_macos_groups(qapp, app, server):
    window = open_session(qapp, app, server())
    buttons = window.toolbar.findChildren(QToolButton)
    assert len(buttons) == 11
    for button in buttons:
        # The action keeps its name (for accessibility and the tooltip), but only the symbol is drawn.
        assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly, button.accessibleName()
        assert not button.icon().isNull() and button.toolTip() and button.accessibleName()
    segments = window.toolbar.findChildren(QFrame, 'segment')
    assert [names(s) for s in segments] == [['Control', 'Share clipboard', 'Type clipboard'],
                                            ['Launchpad', 'Mission Control', 'Show Desktop'],
                                            ['Zoom out', 'Actual size', 'Zoom in', 'Fit to window'],
                                            ['More']]


def test_more_menu_holds_the_rare_actions(qapp, app, server):
    window = open_session(qapp, app, server())
    entries = [a.text() for a in window.more_menu.actions() if not a.isSeparator()]
    assert entries == ['Refresh Screen', 'Full Screen', 'Reconnect…', 'Ctrl = Command', 'Swap Alt/Win', 'Disconnect']
    assert not window.reconnect_action.isVisible()


def test_toggles_show_their_state_and_start_in_the_right_place(qapp, app, server):
    window = open_session(qapp, app, server())
    assert window.control_action.isChecked() and window.clip_action.isChecked()
    assert window.fit_action.isChecked() and not window.actual.isChecked()
    assert not window.ctrl_action.isChecked() and not window.swap_action.isChecked()


# ── view only ────────────────────────────────────────────────────────

def keys(fake):
    return [(e[1], e[2]) for e in fake.events if e[0] == 'key']


def test_control_off_means_nothing_reaches_the_mac(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.control_action.setChecked(False); window.set_control(False)
    assert 'View only' in window.statusBar().currentMessage() and not window.screen_widget.input_enabled
    screen = window.screen_widget
    from PySide6.QtGui import QKeyEvent, QMouseEvent
    from PySide6.QtCore import QEvent, QPointF
    screen.resize(400, 200); screen.grab()
    screen.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, 'a'))
    screen.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 100), screen.mapToGlobal(QPointF(100, 100)),
                                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    window.mission_control(); window.show_desktop(); window.launchpad(); window.type_text('typed')
    QApplication.clipboard().setText('private'); window.push_clipboard()
    time.sleep(0.4); qapp.processEvents()
    assert fake.events == []                                              # nothing at all was sent
    assert 'Control is off' in window.statusBar().currentMessage()
    window.control_action.setChecked(True); window.set_control(True)
    window.mission_control()
    assert pump(qapp, lambda: (0xff52, True) in keys(fake))               # and it works again


# ── zoom ─────────────────────────────────────────────────────────────

def test_zoom_steps_up_and_down_from_fit_and_stays_in_range(qapp, app, server):
    window = open_session(qapp, app, server(size=(1600, 1200)))
    window.resize(500, 400)
    assert pump(qapp, lambda: window.fit_zoom() < 0.4)
    fit = window.fit_zoom()
    window.zoom_in()
    assert window.screen_widget.zoom == next(s for s in window.ZOOM_STEPS if s > fit * 1.02)
    window.zoom_in(); window.zoom_in()
    assert window.screen_widget.zoom > fit
    window.set_zoom(0.5); window.zoom_out()
    assert window.screen_widget.zoom == 0.33
    window.zoom_out(); window.zoom_out(); window.zoom_out()
    assert window.screen_widget.zoom == window.ZOOM_STEPS[0]               # never below the smallest step
    window.set_zoom(window.ZOOM_STEPS[-1]); window.zoom_in()
    assert window.screen_widget.zoom == window.ZOOM_STEPS[-1]              # never past the largest
    assert 'largest zoom' in window.statusBar().currentMessage()
    window.set_zoom(1.5)
    assert 'Zoom 150%' in window.statusBar().currentMessage()


def test_actual_and_fit_buttons_track_the_zoom(qapp, app, server):
    window = open_session(qapp, app, server(size=(1600, 1200)))
    window.set_zoom(1.0)
    assert window.actual.isChecked() and not window.fit_action.isChecked() and window.screen_widget.view == 'actual'
    assert window.screen_widget.size() == QSize(1600, 1200)
    window.set_zoom(1.5)
    assert not window.actual.isChecked() and window.screen_widget.view == 'zoom'
    assert window.screen_widget.size() == QSize(2400, 1800) and window.pan_timer.isActive()
    window.fit_action.trigger()
    assert window.screen_widget.zoom is None and window.fit_action.isChecked() and not window.pan_timer.isActive()


def test_action_feedback_survives_incoming_frames_and_connection_info_has_its_own_place(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.set_zoom(1.5)
    before = window.info_label.text()
    for shade in (10, 60, 120):
        fake.push_frame(colour=(shade, 0, 0))
    assert pump(qapp, lambda: window.controller.frames >= 4)
    assert window.statusBar().currentMessage() == 'Zoom 150%'                  # not overwritten by the new frames
    assert window.info_label.text() != before and f'127.0.0.1:{fake.port}' in window.info_label.text()
    assert 'Standard VNC' in window.info_label.text() and 'updates' in window.info_label.text()


def test_zoom_in_from_a_fit_that_is_already_huge_does_not_shrink(qapp, app, server):
    window = open_session(qapp, app, server(size=(20, 10)))
    window.resize(1000, 700)
    assert pump(qapp, lambda: window.fit_zoom() > 4)
    window.zoom_in()
    assert window.screen_widget.zoom in (None, window.ZOOM_STEPS[-1]) or window.screen_widget.zoom >= window.fit_zoom()


def test_zoomed_pointer_mapping_follows_the_scale(qapp):
    sent = []

    class Sink:
        def send(self, *args): sent.append(args)
    from isharescreen.frontend.desktop import session as S
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtCore import QEvent, QPointF
    screen = S.Screen(Sink())
    screen.set_image(QImage(100, 50, QImage.Format.Format_RGBA8888))
    screen.set_zoom(2.0); screen.grab()
    assert screen.size() == QSize(200, 100) and screen.view == 'zoom'
    screen.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 60), QPointF(100, 60),
                                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert sent[-1] == ('pointer', 50, 30, 1)                              # widget pixels / 2 = remote pixels


# ── Mission Control, Show Desktop, Launchpad ─────────────────────────

def test_mission_control_and_show_desktop_send_the_measured_keystrokes(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.ctrl_action.setChecked(True); window.set_ctrl_as_command(True)       # these options must not matter
    window.swap_action.setChecked(True); window.set_swap_alt_win(True)
    window.mission_control()
    assert pump(qapp, lambda: len(keys(fake)) == 4)
    assert keys(fake) == [(0xffe3, True), (0xff52, True), (0xff52, False), (0xffe3, False)]    # Control+Up
    fake.events.clear()
    window.show_desktop()
    assert pump(qapp, lambda: len(keys(fake)) == 2)
    assert keys(fake) == [(0xffc8, True), (0xffc8, False)]                                      # F11


def test_shortcut_buttons_need_a_live_connection(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    window.mission_control()
    assert 'Not connected' in window.statusBar().currentMessage()


def test_launchpad_types_only_after_it_sees_spotlight_open(qapp, app, server):
    fake = server(colour=(10, 20, 30))
    window = open_session(qapp, app, fake)
    window.launchpad()
    assert pump(qapp, lambda: len(keys(fake)) >= 4)
    assert keys(fake)[:4] == [(0xffeb, True), (0x20, True), (0x20, False), (0xffeb, False)]    # Command+Space
    time.sleep(0.5)
    fake.push_frame(colour=(200, 90, 40))                                                      # Spotlight "appears"
    assert pump(qapp, lambda: keys(fake)[-1:] == [(0xff0d, False)], 6)
    typed = [k for k, down in keys(fake)[4:] if down and k != 0xff0d]
    assert bytes(typed).decode() == 'Launchpad'
    assert keys(fake)[-2:] == [(0xff0d, True), (0xff0d, False)]                                # then Return
    assert 'Opened Launchpad' in window.statusBar().currentMessage()


def test_launchpad_refuses_when_spotlight_never_opens(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.launchpad()
    assert pump(qapp, lambda: 'did not open' in window.statusBar().currentMessage(), 6)
    assert keys(fake) == [(0xffeb, True), (0x20, True), (0x20, False), (0xffeb, False)]        # nothing was typed
    window.launchpad()                                                                         # and it can be retried
    assert pump(qapp, lambda: len(keys(fake)) == 8)


def test_launchpad_refuses_when_the_screen_is_too_busy_to_tell(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    counter = [0]

    def churn():
        counter[0] += 1
        fake.push_frame(colour=(counter[0] % 250, (counter[0] * 7) % 250, 3))
    from PySide6.QtCore import QTimer
    timer = QTimer(); timer.timeout.connect(churn); timer.start(15)
    try:
        window.launchpad()
        assert pump(qapp, lambda: 'changing too much' in window.statusBar().currentMessage(), 8)
    finally:
        timer.stop()
    assert all(k in (0xffeb, 0x20) for k, _ in keys(fake))                                     # no letters, no Return


# ── library ──────────────────────────────────────────────────────────

def fill(app):
    app.store.record_success('b.local', 5900, 'u', name='Bravo', when=100.0)
    first = app.store.record_success('a.local', 5900, 'u', name='Alpha', when=200.0)
    app.store.record_success('c.local', 5900, '', name='Charlie', when=300.0)
    app.refresh_library()
    return first


def titles(app):
    return [app.library.list.item(i).text() for i in range(app.library.list.count())]


def test_library_header_is_symbols_only(qapp, app):
    from PySide6.QtWidgets import QLineEdit
    buttons = [b for b in app.library.findChildren(QToolButton) if not isinstance(b.parent(), QLineEdit)]   # not Qt's own ⓧ
    assert len(buttons) >= 7
    for button in buttons:
        assert button.text() == '' and (not button.icon().isNull()) and button.toolTip(), button.accessibleName()


def test_library_views_sorting_and_sidebar(qapp, app):
    first = fill(app)
    window = app.library
    assert titles(app) == ['Charlie', 'Alpha', 'Bravo']                         # last connected first
    window.set_sort('name')
    assert titles(app) == ['Alpha', 'Bravo', 'Charlie'] and window.sort_name.isChecked()
    window.set_view('grid')
    assert window.list.viewMode() == window.list.ViewMode.IconMode and window.delegate.grid and window.grid_button.isChecked()
    assert window.list.item(0).sizeHint() == QSize(150, 132)
    window.set_view('list')
    assert window.list.viewMode() == window.list.ViewMode.ListMode and not window.delegate.grid and window.list_button.isChecked()
    window.side_list.setCurrentRow(1)                                           # "Connected"
    assert titles(app) == [] and 'No sessions are open' in window.no_match.text() and window.title.text() == 'Connected'
    window.set_active({first.id: 1})
    assert titles(app) == ['Alpha']
    window.side_list.setCurrentRow(0)
    assert len(titles(app)) == 3 and window.title.text() == 'All Connections'
    window.sidebar_button.setChecked(False)
    assert not window.sidebar.isVisibleTo(window)


def test_info_button_opens_the_sheet_and_double_click_connects(qapp, app):
    first = fill(app)
    window = app.library
    app._ask_edit = lambda system: None                                         # the real sheet is modal
    app.open_system = lambda system_id: None                                    # and connecting is tested elsewhere
    window.show(); window.resize(900, 520); qapp.processEvents()
    opened, edits = [], []
    window.open_system.connect(opened.append); window.edit_system.connect(edits.append)
    row = next(i for i in range(window.list.count()) if window.list.item(i).data(LW.ID_ROLE) == first.id)
    rect = window.list.visualItemRect(window.list.item(row))
    info = window.delegate.info_rect(rect)
    QTest.mouseClick(window.list.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, info.center())
    assert edits == [first.id] and opened == []                                 # the ⓘ edits, it does not connect
    window.list.itemActivated.emit(window.list.item(row))                       # double-click / Enter
    assert opened == [first.id]


def test_search_button_shows_filters_and_hides(qapp, app):
    fill(app)
    window = app.library
    window.show(); qapp.processEvents()
    assert not window.search.isVisible()
    window.search_button.click()
    assert window.search.isVisible()
    window.search.setText('brav')
    assert titles(app) == ['Bravo']
    window.search_button.click()                                                # has text: refocus, stay open
    assert window.search.isVisible()
    window.search.clear()
    window.search_button.click()                                                # empty: hide
    assert not window.search.isVisible() and len(titles(app)) == 3


def test_rows_show_name_address_and_the_macos_date_format(qapp, app):
    fill(app)
    item = app.library.list.item(0)
    assert item.data(LW.SUBTITLE_ROLE) == 'c.local'                             # no account: just the address
    assert re.fullmatch(r'[A-Z][a-z]{2} \d{1,2}, \d{4} at \d{1,2}:\d{2} [AP]M', item.data(LW.WHEN_ROLE))
    assert describe_last_connected(None) == 'Not connected yet'
    assert describe_last_connected(300.0).startswith('Last connected ')


# ── connect dialogs ──────────────────────────────────────────────────

def test_connect_to_dialog_is_one_field_and_takes_host_and_port(qapp, app, server):
    fake = server()
    wizard = app.open_wizard()
    assert wizard.address.placeholderText() == 'Hostname or IP address'
    assert wizard.continue_button.text() == 'Connect' and wizard.continue_button.isDefault()
    assert not wizard.stop_check.isVisibleTo(wizard)
    narrow = wizard.size()
    verify(qapp, wizard, fake)                                                  # typed as host:port
    assert wizard.stack.currentIndex() == wizard.CREDENTIALS
    assert wizard.size().width() == narrow.width() and wizard.size().height() > narrow.height()


def test_sign_in_dialog_wording_follows_the_server(qapp, app, server):
    account = app.open_wizard(); verify(qapp, account, server())
    assert 'user name and password' in account.message.text() and account.instruction.isVisibleTo(account)
    password = app.open_wizard(); verify(qapp, password, server(security='vnc', users={'': 'x'}))
    assert 'requires a password' in password.message.text() and 'user name' not in password.message.text()
    nothing = app.open_wizard(); verify(qapp, nothing, server(security='none'))
    assert nothing.message.text().startswith('Connect to “') and not nothing.instruction.isVisibleTo(nothing)
    assert nothing.auth_note.isVisibleTo(nothing)


def test_more_options_and_the_high_performance_entry(qapp, app, server):
    wizard = app.open_wizard()
    verify(qapp, wizard, server())
    assert not wizard.more_box.isVisibleTo(wizard)
    before = wizard.size().height()
    wizard.options.setChecked(True)
    assert wizard.more_box.isVisibleTo(wizard) and wizard.size().height() > before and wizard.options.text() == 'More ▾'
    assert not wizard.mode.model().item(1).isEnabled()                          # no capability: unavailable
    capable = app.open_wizard(hp_available=lambda result: (True, ''))
    verify(qapp, capable, server())
    assert capable.mode.model().item(1).isEnabled()
    capable.mode.setCurrentIndex(1)
    assert capable.mode.currentData() == 'high_performance'


# ── the ⓘ sheet ──────────────────────────────────────────────────────

def test_resolve_address_combines_the_address_and_port_fields():
    assert resolve_address('mac.local', '') == ('mac.local', 5900)
    assert resolve_address('mac.local', '5901') == ('mac.local', 5901)
    assert resolve_address('mac.local:5902', '') == ('mac.local', 5902)
    assert resolve_address('mac.local:5902', '5902') == ('mac.local', 5902)
    with pytest.raises(ValueError, match='different ports'):
        resolve_address('mac.local:5902', '5903')
    with pytest.raises(ValueError):
        resolve_address('mac.local', 'abc')
    with pytest.raises(ValueError):
        resolve_address('bad host', '')


def test_sheet_buttons_connect_and_done_and_validation(qapp, app):
    saved = app.store.record_success('a.local', 5901, 'ana', name='A')
    dialog = EditSystemDialog(None, saved)
    assert (dialog.address.text(), dialog.port.text(), dialog.user.text()) == ('a.local', '5901', 'ana')
    assert [dialog.save.text(), dialog.connect_button.text()] == ['Done', 'Connect']
    dialog.address.setText('bad host')
    dialog.connect_button.click()
    assert dialog.result() != dialog.DialogCode.Accepted and not dialog.connect_after and dialog.error.text().startswith('✕')
    dialog.address.setText('b.local')
    dialog.connect_button.click()
    assert dialog.values()['connect'] is True and dialog.values()['host'] == 'b.local' and dialog.values()['port'] == 5901
    plain = EditSystemDialog(None, saved)
    plain.save.click()
    assert plain.values()['connect'] is False


def test_sheet_keeps_a_saved_high_performance_choice_but_only_offers_it_when_available(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', mode='high_performance')
    dialog = EditSystemDialog(None, saved)
    assert dialog.mode.currentData() == 'high_performance' and not dialog.mode.model().item(1).isEnabled()
    enabled = EditSystemDialog(None, saved, hp_enabled=True)
    assert enabled.mode.model().item(1).isEnabled()


def test_connect_from_the_sheet_saves_then_opens_the_connection(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', name='A', password_saved=True)
    app.credentials.set(saved.id, 'pw', 'ana')
    app._ask_edit = lambda system: dict(host='a.local', port=5900, user='ana', name='Renamed', view='fit',
                                        mode='standard', forget_password=False, connect=True)
    app.edit_system(saved.id)
    assert app.store.get(saved.id).name == 'Renamed'
    assert app.wizards and app.wizards[-1]._preset_password == 'pw'              # connecting with the saved password


def test_changing_only_the_mode_keeps_the_saved_password(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', password_saved=True)
    app.credentials.set(saved.id, 'pw', 'ana')
    app._ask_edit = lambda system: dict(host='a.local', port=5900, user='ana', name='', view='fit',
                                        mode='high_performance', forget_password=False)
    app.edit_system(saved.id)
    item = app.store.get(saved.id)
    assert item.mode == 'high_performance' and item.password_saved and app.credentials.get(saved.id) == 'pw'
