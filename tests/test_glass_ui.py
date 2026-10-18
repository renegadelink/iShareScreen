"""The Liquid Glass edition: same behaviour as the macOS-style one, a different look.
Every test here switches the theme to glass and back, so the other test files keep
exercising the default look."""
import time

import pytest
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QPushButton, QToolButton

from desktop_helpers import pump, open_session
from isharescreen.frontend.desktop import theme, widgets
from isharescreen.frontend.desktop.edit_dialog import EditSystemDialog
from isharescreen.frontend.desktop.library import SavedSystem


@pytest.fixture
def glass(qapp):
    theme.apply_theme(qapp, theme.GLASS)
    yield
    theme.apply_theme(qapp, theme.LIGHT)


def settle(qapp, seconds=0.4):
    end = time.time() + seconds
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)


def seed(app):
    app.store.record_success('studio-mac.local', 5900, 'alex', name='Studio Mac', when=time.time() - 3600)
    app.store.record_success('192.0.2.42', 5900, 'sam', name='Build Mini', when=time.time() - 86400 * 2)
    app.refresh_library()


def test_default_look_uses_plain_widgets(qapp):
    assert not theme.is_glass()
    assert type(widgets.push_button('x')) is QPushButton
    assert type(widgets.tool_button()) is QToolButton


def test_style_selection(monkeypatch):
    monkeypatch.delenv('ISS_STYLE', raising=False)
    assert theme.style_name() == 'macos' and theme.wanted_theme()['style'] == 'macos'
    monkeypatch.setenv('ISS_STYLE', 'glass')
    assert theme.style_name() == 'glass' and theme.wanted_theme() is theme.GLASS


def test_cli_style_option():
    """In a child process: main() changes the process-wide theme and imports the HP engine."""
    import os
    import subprocess
    import sys
    code = ('import os, sys\n'
            'from isharescreen.frontend.desktop.standard_app import main\n'
            "code = main(['--style', 'glass', '--smoke-test'])\n"
            "print(os.environ['ISS_STYLE'])\n"
            'sys.exit(code)\n')
    env = {key: value for key, value in os.environ.items() if key != 'ISS_STYLE'}
    result = subprocess.run([sys.executable, '-c', code], env={**env, 'QT_QPA_PLATFORM': 'offscreen'},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == 'glass'


def test_glass_widgets_are_used(qapp, glass):
    assert theme.is_glass()
    assert isinstance(widgets.push_button('x'), widgets.GlassButton)
    assert isinstance(widgets.tool_button(), widgets.GlassToolButton)
    assert theme.label('Connect To:', 'Hostname') == 'Hostname'


def test_library_renders_with_the_field_and_panels(qapp, glass, app):
    seed(app)
    window = app.library
    window.resize(980, 600); window.show(); settle(qapp)
    image = window.grab().toImage()
    assert not image.isNull()
    corner = QColor(image.pixel(4, image.height() - 4))
    assert corner.green() >= corner.red() and corner.green() > 200     # the pale green field, not grey or black
    assert window.list.count() == 2
    window.set_view('grid'); settle(qapp, 0.1)
    window.set_view('list')
    window.hide()


def test_reduced_effects_paint_opaque_panels(qapp, glass, app, monkeypatch):
    monkeypatch.setenv('ISS_EFFECTS', 'reduced')
    assert theme.effects_reduced()
    seed(app)
    window = app.library
    window.resize(900, 560); window.show(); settle(qapp, 0.2)
    assert not window.grab().toImage().isNull()
    window.hide()


def test_wizard_flow_in_glass(qapp, glass, app, server):
    fake = server()
    session = open_session(qapp, app, fake)
    assert session.is_live
    buttons = session.toolbar.findChildren(widgets.GlassToolButton)
    assert len(buttons) >= 11                                   # every symbol of the toolbar
    assert all(not button.icon().isNull() for button in buttons)
    session.close()


def test_toggle_buttons_follow_their_actions(qapp, glass, app, server):
    session = open_session(qapp, app, server())
    assert session.control_action.isChecked()
    session.control_action.trigger()
    assert not session.control_action.isChecked()
    session.control_action.trigger()
    assert session.control_action.isChecked()
    session.close()


def test_edit_dialog_in_glass_keeps_its_behaviour(qapp, glass):
    system = SavedSystem(id='1', host='studio-mac.local', port=5900, user='alex', name='Studio Mac')
    dialog = EditSystemDialog(None, system)
    dialog.show(); settle(qapp)
    assert dialog.save.isDefault()
    dialog.name.setText('Renamed')
    dialog.port.setText('5901')
    dialog.accept()
    values = dialog.values()
    assert values['name'] == 'Renamed' and values['port'] == 5901 and values['host'] == 'studio-mac.local'
    dialog.close()


def test_edit_dialog_validation_in_glass(qapp, glass):
    system = SavedSystem(id='1', host='studio-mac.local', port=5900, user='alex', name='Studio Mac')
    dialog = EditSystemDialog(None, system)
    dialog.address.setText('studio-mac.local:5901'); dialog.port.setText('5902')
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted and 'different ports' in dialog.error.text()
    dialog.close()


def test_reveal_is_opacity_only_and_short(qapp, glass):
    from PySide6.QtWidgets import QWidget
    items = [QWidget() for _ in range(10)]
    for item in items:
        item.show()
    animations = widgets.reveal(items)
    assert len(animations) == 10
    assert all(animation.propertyName() == b'opacity' for animation in animations)
    assert all(animation.duration() <= widgets.REVEAL_MS for animation in animations)
    assert widgets.MAX_STAGGERED == 6 and widgets.MAX_STAGGER_DELAY_MS <= 150
    settle(qapp, 0.4)
    assert all(item.graphicsEffect() is None for item in items)     # the effect is dropped once the fade ends
    for item in items:
        item.close()


