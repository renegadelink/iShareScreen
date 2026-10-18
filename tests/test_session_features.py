"""Clipboard sharing and actual-size edge scrolling in a session window."""
import time

import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from desktop_helpers import pump, open_session
from isharescreen.frontend.desktop import session as S
from isharescreen.proxy.standard import StandardClient


@pytest.fixture(autouse=True)
def clean_clipboard(qapp):
    QApplication.clipboard().clear()
    yield
    QApplication.clipboard().clear()


def test_text_already_on_the_clipboard_is_not_sent_when_a_session_starts(qapp, app, server):
    QApplication.clipboard().setText('copied before connecting')
    fake = server()
    window = open_session(qapp, app, fake)
    window.isActiveWindow = lambda: True
    window.push_clipboard(); window.changeEvent(QEvent(QEvent.Type.ActivationChange))
    time.sleep(0.3); qapp.processEvents()
    assert clipboard_events(fake) == []
    set_local('copied afterwards')
    assert pump(qapp, lambda: clipboard_events(fake) == ['copied afterwards'])


def clipboard_events(fake):
    return [event[1] for event in fake.events if event[0] == 'clipboard']


def set_local(text):
    QApplication.clipboard().setText(text)


def test_local_clipboard_reaches_the_mac_once(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    set_local('hello from windows')
    window.push_clipboard()                          # what window activation does
    assert pump(qapp, lambda: clipboard_events(fake) == ['hello from windows'])
    window.push_clipboard(); qapp.processEvents()    # unchanged text is not sent again
    time.sleep(0.2)
    assert clipboard_events(fake) == ['hello from windows']


def test_copy_while_the_session_is_active_is_sent_without_switching(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.isActiveWindow = lambda: True
    set_local('copied while active')
    assert pump(qapp, lambda: 'copied while active' in clipboard_events(fake))


def test_copy_in_another_window_waits_until_this_session_is_used(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.isActiveWindow = lambda: False
    set_local('private note')
    qapp.processEvents(); time.sleep(0.3); qapp.processEvents()
    assert clipboard_events(fake) == []              # not sent while another app has focus


def test_mac_clipboard_arrives_locally_without_echoing_back(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.isActiveWindow = lambda: True
    fake.push_clipboard('from the mac')
    assert pump(qapp, lambda: QApplication.clipboard().text() == 'from the mac')
    time.sleep(0.4); qapp.processEvents()
    assert clipboard_events(fake) == []              # no ping-pong


def test_sharing_toggle_blocks_both_directions(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.isActiveWindow = lambda: True
    window.clip_action.setChecked(False)
    window.set_clipboard_sync(False)
    set_local('while off'); window.push_clipboard()
    fake.push_clipboard('mac while off')
    time.sleep(0.4); qapp.processEvents()
    assert clipboard_events(fake) == [] and QApplication.clipboard().text() == 'while off'
    window.clip_action.setChecked(True)
    window.set_clipboard_sync(True)                  # turning it on sends the current text
    assert pump(qapp, lambda: clipboard_events(fake) == ['while off'])


def test_non_latin1_text_is_replaced_and_the_user_is_told(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    set_local('héllo ✓')
    window.push_clipboard()
    assert pump(qapp, lambda: clipboard_events(fake) == ['héllo ?'])
    assert 'Latin-1' in window.statusBar().currentMessage()


def test_clipboard_encoding_limits():
    client = StandardClient.__new__(StandardClient)
    assert client.clipboard_text('plain') == (b'plain', True)
    assert client.clipboard_text('é') == (b'\xe9', True)
    data, lossless = client.clipboard_text('日本')
    assert data == b'??' and not lossless
    data, _ = client.clipboard_text('x' * 2_000_000)
    assert len(data) == StandardClient.MAX_CLIPBOARD


class Keys:
    def __init__(self): self.sent = []
    def send(self, method, *args):
        if method == 'key': self.sent.append(args)


def press(screen, key, text='', modifiers=Qt.KeyboardModifier.NoModifier):
    screen.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, modifiers, text))


def release(screen, key, modifiers=Qt.KeyboardModifier.NoModifier):
    screen.keyReleaseEvent(QKeyEvent(QEvent.Type.KeyRelease, key, modifiers, ''))


def test_modifier_mapping_matches_what_the_mac_measured(qapp):
    """Measured live: Meta_L is Option; Alt_L, Super_L and Hyper_L are Command."""
    sink = Keys()
    screen = S.Screen(sink)
    for key, keysym in ((Qt.Key.Key_Alt, 0xffe7), (Qt.Key.Key_Meta, 0xffeb), (Qt.Key.Key_Control, 0xffe3),
                        (Qt.Key.Key_Shift, 0xffe1)):
        sink.sent.clear()
        press(screen, key); release(screen, key)
        assert sink.sent == [(keysym, True), (keysym, False)]
    assert 0xffe9 not in (v for v in S.SPECIAL.values())          # Alt_L would be Command


def test_alt_chord_sends_the_plain_letter_even_when_windows_gives_no_text(qapp):
    sink = Keys()
    screen = S.Screen(sink)
    press(screen, Qt.Key.Key_Alt)
    press(screen, Qt.Key.Key_A, '', Qt.KeyboardModifier.AltModifier)
    release(screen, Qt.Key.Key_A); release(screen, Qt.Key.Key_Alt)
    assert sink.sent == [(0xffe7, True), (0x61, True), (0x61, False), (0xffe7, False)]


def test_ctrl_as_command_swaps_only_the_ctrl_key_and_releases_what_was_pressed(qapp):
    sink = Keys()
    screen = S.Screen(sink)
    screen.ctrl_as_command = True
    press(screen, Qt.Key.Key_Control)
    press(screen, Qt.Key.Key_C, '\x03', Qt.KeyboardModifier.ControlModifier)
    screen.ctrl_as_command = False                                # toggled while held
    release(screen, Qt.Key.Key_C); release(screen, Qt.Key.Key_Control)
    assert sink.sent == [(0xffeb, True), (0x63, True), (0x63, False), (0xffeb, False)]   # released as pressed
    sink.sent.clear()
    press(screen, Qt.Key.Key_Control); release(screen, Qt.Key.Key_Control)
    assert sink.sent == [(0xffe3, True), (0xffe3, False)]         # off: plain Control


def test_swap_alt_and_windows_keys(qapp):
    """Swapped: Alt acts as Command (Super_L) and the Windows key as Option (Meta_L)."""
    sink = Keys()
    screen = S.Screen(sink)
    screen.swap_alt_win = True
    for key, keysym in ((Qt.Key.Key_Alt, 0xffeb), (Qt.Key.Key_Meta, 0xffe7)):
        sink.sent.clear()
        press(screen, key); release(screen, key)
        assert sink.sent == [(keysym, True), (keysym, False)]
    sink.sent.clear()
    for key in (Qt.Key.Key_Control, Qt.Key.Key_Shift):                # nothing else changes
        press(screen, key); release(screen, key)
    assert sink.sent == [(0xffe3, True), (0xffe3, False), (0xffe1, True), (0xffe1, False)]
    screen.swap_alt_win = False
    sink.sent.clear()
    press(screen, Qt.Key.Key_Alt); release(screen, Qt.Key.Key_Alt)
    press(screen, Qt.Key.Key_Meta); release(screen, Qt.Key.Key_Meta)
    assert sink.sent == [(0xffe7, True), (0xffe7, False), (0xffeb, True), (0xffeb, False)]   # back to the default


def test_swap_chords_send_the_letter_with_the_swapped_modifier(qapp):
    sink = Keys()
    screen = S.Screen(sink)
    screen.swap_alt_win = True
    press(screen, Qt.Key.Key_Alt)
    press(screen, Qt.Key.Key_Left)
    release(screen, Qt.Key.Key_Left); release(screen, Qt.Key.Key_Alt)
    assert sink.sent == [(0xffeb, True), (0xff51, True), (0xff51, False), (0xffeb, False)]   # Alt+Left = Command+Left
    sink.sent.clear()
    press(screen, Qt.Key.Key_Meta)
    press(screen, Qt.Key.Key_A, '', Qt.KeyboardModifier.MetaModifier)
    release(screen, Qt.Key.Key_A); release(screen, Qt.Key.Key_Meta)
    assert sink.sent == [(0xffe7, True), (0x61, True), (0x61, False), (0xffe7, False)]       # Win+A = Option+A


def test_swap_and_ctrl_as_command_combine(qapp):
    sink = Keys()
    screen = S.Screen(sink)
    screen.swap_alt_win = True
    screen.ctrl_as_command = True
    for key in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
        press(screen, key)
    assert [s for s, d in sink.sent] == [0xffeb, 0xffeb, 0xffe7]      # Ctrl and Alt both Command, Windows is Option


def test_swap_toggle_releases_held_keys_as_they_were_pressed(qapp, app, server):
    window = open_session(qapp, app, server())
    sent = []
    window.controller.send = lambda method, *args: sent.append((method, *args))
    press(window.screen_widget, Qt.Key.Key_Alt)                       # held as Option (Meta_L)
    window.swap_action.setChecked(True); window.set_swap_alt_win(True)
    assert sent == [('key', 0xffe7, True), ('key', 0xffe7, False)]    # released with the code it went down with
    assert window.screen_widget.swap_alt_win is True and 'Command' in window.statusBar().currentMessage()
    window.swap_action.setChecked(False); window.set_swap_alt_win(False)
    assert window.screen_widget.swap_alt_win is False


def test_keysym_for_char():
    f = S.keysym_for_char
    assert (f('a'), f('Z'), f(' '), f('~'), f('é'), f(' ')) == (0x61, 0x5a, 0x20, 0x7e, 0xe9, 0xa0)
    assert (f('\n'), f('\r'), f('\t')) == (0xff0d, 0xff0d, 0xff09)
    assert (f('日'), f('✓'), f('☃')) == (0x01000000 | 0x65e5, 0x01000000 | 0x2713, 0x01000000 | 0x2603)
    assert (f('\x00'), f('\x1b'), f('\x7f'), f('\x85')) == (None, None, None, None)
    assert (f('😀'), f('\U0001d49c'), f('￿') is not None) == (None, None, True)   # astral: the server drops or breaks them


def test_astral_characters_are_skipped_and_reported(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.type_text('a😀b😀😀')
    assert pump(qapp, lambda: not window.type_timer.isActive())
    assert pump(qapp, lambda: len(key_events(fake)) == 4)
    assert key_events(fake) == [(0x61, True), (0x61, False), (0x62, True), (0x62, False)]
    message = window.statusBar().currentMessage()
    assert 'Typed 2 characters' in message and '3 emoji' in message and 'skipped' in message
    window.type_text('😀')
    assert 'no text that can be typed' in window.statusBar().currentMessage() and not window.type_timer.isActive()


def key_events(fake):
    return [(e[1], e[2]) for e in fake.events if e[0] == 'key']


def test_type_clipboard_types_every_character_in_order(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    QApplication.clipboard().setText('Hi é日\r\nok\x07')            # CRLF becomes one Return; BEL is dropped
    window.type_clipboard()
    expected = []
    for keysym in (0x48, 0x69, 0x20, 0xe9, 0x01000000 | 0x65e5, 0xff0d, 0x6f, 0x6b):
        expected += [(keysym, True), (keysym, False)]
    assert pump(qapp, lambda: len(key_events(fake)) >= len(expected))
    assert key_events(fake) == expected
    assert 'Typed 8 characters' in window.statusBar().currentMessage()


def test_type_text_is_paced_and_respects_its_limit(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.TYPE_LIMIT = 50
    window.type_text('x' * 200)
    assert window.type_timer.isActive()                           # sent in the background, not in one blocking burst
    assert pump(qapp, lambda: not window.type_timer.isActive())
    assert pump(qapp, lambda: len(key_events(fake)) == 100)
    assert 'Typed 50 characters (only the first 50 of 200)' in window.statusBar().currentMessage()


def test_typing_stops_when_the_connection_drops(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.type_text('y' * 20000)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    assert pump(qapp, lambda: not window.type_timer.isActive()) and not window._typing


def test_type_clipboard_reports_empty_and_disconnected_states(qapp, app, server):
    fake = server()
    window = open_session(qapp, app, fake)
    window.type_text('\x07\x00')
    assert 'Nothing to type' in window.statusBar().currentMessage() and not window.type_timer.isActive()
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    window.type_text('abc')
    assert 'Not connected' in window.statusBar().currentMessage() and not window.type_timer.isActive()


def test_ctrl_command_toggle_is_wired_to_the_screen(qapp, app, server):
    window = open_session(qapp, app, server())
    assert window.screen_widget.ctrl_as_command is False
    window.ctrl_action.setChecked(True); window.set_ctrl_as_command(True)
    assert window.screen_widget.ctrl_as_command is True


def big_session(qapp, app, server):
    fake = server(size=(1600, 1200))
    window = open_session(qapp, app, fake)
    window.resize(500, 400)
    window.isActiveWindow = lambda: True
    window.actual.setChecked(True)
    window.set_actual_size(True)
    assert pump(qapp, lambda: window.scroll.horizontalScrollBar().maximum() > 0
                and window.scroll.verticalScrollBar().maximum() > 0)
    return fake, window


def global_at(window, x, y):
    return window.scroll.viewport().mapToGlobal(QPoint(x, y))


def test_actual_size_scrolls_toward_the_edge_the_cursor_pushes(qapp, app, server):
    fake, window = big_session(qapp, app, server)
    viewport = window.scroll.viewport()
    h, v = window.scroll.horizontalScrollBar(), window.scroll.verticalScrollBar()
    centre = global_at(window, viewport.width() // 2, viewport.height() // 2)
    window.pan_step(centre)
    assert (h.value(), v.value()) == (0, 0)                                   # the middle never scrolls
    window.pan_step(global_at(window, viewport.width() - 2, viewport.height() // 2))
    assert h.value() > 0 and v.value() == 0                                   # right edge scrolls right
    window.pan_step(global_at(window, viewport.width() // 2, viewport.height() - 2))
    assert v.value() > 0
    right, down = h.value(), v.value()
    window.pan_step(global_at(window, 1, viewport.height() // 2))
    window.pan_step(global_at(window, viewport.width() // 2, 1))
    assert h.value() < right and v.value() < down                             # and back again


def test_faster_nearer_the_edge_and_stops_at_the_end(qapp, app, server):
    fake, window = big_session(qapp, app, server)
    viewport = window.scroll.viewport()
    h = window.scroll.horizontalScrollBar()
    window.pan_step(global_at(window, viewport.width() - window.PAN_MARGIN + 5, 100)); slow = h.value()
    h.setValue(0)
    window.pan_step(global_at(window, viewport.width() - 1, 100)); fast = h.value()
    assert 0 < slow < fast <= window.PAN_SPEED
    for _ in range(400):
        window.pan_step(global_at(window, viewport.width() - 1, 100))
    assert h.value() == h.maximum()


def test_panning_keeps_the_remote_pointer_under_the_cursor(qapp, app, server):
    fake, window = big_session(qapp, app, server)
    viewport = window.scroll.viewport()
    before = len([e for e in fake.events if e[0] == 'pointer'])
    window.pan_step(global_at(window, viewport.width() - 2, viewport.height() // 2))
    assert pump(qapp, lambda: len([e for e in fake.events if e[0] == 'pointer']) > before)
    kind, x, y, buttons = [e for e in fake.events if e[0] == 'pointer'][-1]
    assert x > viewport.width() - 40 and 0 <= y < 1200 and buttons == 0       # the image moved, the pointer followed


def test_no_panning_in_fit_mode_outside_the_window_or_when_inactive(qapp, app, server):
    fake, window = big_session(qapp, app, server)
    viewport = window.scroll.viewport()
    h = window.scroll.horizontalScrollBar()
    edge = global_at(window, viewport.width() - 2, 100)
    window.isActiveWindow = lambda: False
    window.pan_step(edge)
    assert h.value() == 0                                                     # another window has focus
    window.isActiveWindow = lambda: True
    window.pan_step(global_at(window, viewport.width() + 50, 100))
    assert h.value() == 0                                                     # cursor outside this window
    window.actual.setChecked(False)
    window.set_actual_size(False)
    window.pan_step(edge)
    assert h.value() == 0 and not window.pan_timer.isActive()                 # fit mode never pans
