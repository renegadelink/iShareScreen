"""macOS Screen Sharing look, in native Qt styles: grouped rounded sections, a
blue accent, symbol-only toolbars, and light/dark that follow the system setting
(override with ISS_THEME=light|dark). Remote pixels are never given a styled
background."""
from __future__ import annotations
import os
import sys

from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Qt
from PySide6.QtGui import QColor, QPalette

DARK = dict(name='dark', window='#2b2829', sidebar='#252223', bar='#353132', group='#3b3738', group_hover='#464142',
            border='#4f4a4b', text='#ececec', muted='#a09a9b', accent='#0a84ff', accent_hover='#3a9bff',
            on_accent='#ffffff', stripe='#322e2f', field='#262324', error='#ff7b72', ok='#32d74b',
            selected='#0a84ff', selected_text='#ffffff', screen='#0c0c0c', banner='#4a3b1c', banner_text='#f3d9a0')
LIGHT = dict(name='light', window='#ececec', sidebar='#e3e1e2', bar='#f6f5f5', group='#ffffff', group_hover='#f0eff0',
             border='#d3d1d2', text='#1d1d1f', muted='#6e6e73', accent='#007aff', accent_hover='#2b8dff',
             on_accent='#ffffff', stripe='#f5f4f5', field='#ffffff', error='#c8342b', ok='#1c8c3a',
             selected='#007aff', selected_text='#ffffff', screen='#101010', banner='#fdf3e2', banner_text='#5d4515')

_current = LIGHT


def tokens() -> dict:
    """The colours of the active theme, for widgets that paint themselves."""
    return _current


# Back-compatible names used by older modules.
INK, MUTED, OK, FOCUS = LIGHT['text'], LIGHT['muted'], LIGHT['ok'], LIGHT['accent']
SCREEN_BG = LIGHT['screen']


def build_style(t: dict) -> str:
    return f'''
QWidget {{ font-family: 'Segoe UI'; font-size: 13px; color: {t['text']}; }}
QMainWindow, QDialog, #root {{ background: {t['window']}; }}
#header {{ background: {t['bar']}; border-bottom: 1px solid {t['border']}; }}
#sidebar {{ background: {t['sidebar']}; border-right: 1px solid {t['border']}; }}
#sidebarItem {{ border-radius: 7px; padding: 6px 8px; font-weight: 600; background: {t['group']}; }}
#title {{ font-size: 15px; font-weight: 700; }}
#heading {{ font-size: 13px; font-weight: 700; }}
#hint {{ color: {t['muted']}; font-size: 12px; }}
#fieldLabel {{ color: {t['text']}; }}
#section {{ background: {t['group']}; border-radius: 10px; }}
#sectionDivider {{ background: {t['border']}; max-height: 1px; min-height: 1px; }}
#emptyTitle {{ font-size: 24px; font-weight: 700; }}
#emptyBody {{ color: {t['muted']}; font-size: 14px; }}
QLabel[kind="error"] {{ color: {t['error']}; font-weight: 600; }}
QLabel[kind="ok"] {{ color: {t['ok']}; font-weight: 600; }}
QLineEdit {{ background: {t['field']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 5px 8px;
             min-height: 18px; selection-background-color: {t['accent']}; selection-color: {t['on_accent']}; }}
QLineEdit:focus {{ border: 2px solid {t['accent']}; padding: 4px 7px; }}
QLineEdit:disabled {{ color: {t['muted']}; }}
QLineEdit#bare {{ background: transparent; border: none; border-radius: 0; padding: 2px 0; text-align: right; }}
QLineEdit#bare:focus {{ border: none; padding: 2px 0; }}
QComboBox {{ background: {t['group']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 4px 10px; min-height: 18px; }}
QComboBox:focus {{ border: 2px solid {t['accent']}; padding: 3px 9px; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox#bare {{ background: transparent; border: none; padding: 2px 0; }}
QComboBox QAbstractItemView {{ background: {t['group']}; selection-background-color: {t['accent']};
                               selection-color: {t['on_accent']}; border: 1px solid {t['border']}; outline: 0; }}
QPushButton {{ background: {t['group']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 4px 16px; min-height: 18px; }}
QPushButton:hover {{ background: {t['group_hover']}; }}
QPushButton:pressed {{ background: {t['border']}; }}
QPushButton:focus {{ border: 2px solid {t['accent']}; padding: 3px 15px; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']}; border: 1px solid {t['accent']}; font-weight: 600; }}
QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
QPushButton#primary:disabled {{ background: {t['group']}; color: {t['muted']}; border-color: {t['border']}; }}
QPushButton#link {{ background: transparent; border: none; color: {t['accent']}; padding: 4px 2px; font-weight: 600; }}
QPushButton#link:hover {{ text-decoration: underline; background: transparent; }}
QPushButton#round {{ border-radius: 11px; padding: 0; min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; }}
QPushButton:disabled {{ color: {t['muted']}; }}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1px solid {t['border']}; background: {t['field']}; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['accent']}; }}
QCheckBox:disabled {{ color: {t['muted']}; }}
QProgressBar {{ background: {t['border']}; border: none; border-radius: 3px; max-height: 6px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
QListView#library {{ background: transparent; border: none; outline: none; }}
QStatusBar {{ background: {t['bar']}; color: {t['muted']}; border-top: 1px solid {t['border']}; }}
QStatusBar::item {{ border: none; }}
#segment {{ background: {t['group']}; border-radius: 8px; }}
QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 4px 7px; min-width: 20px; min-height: 20px; }}
QToolButton:hover {{ background: {t['group_hover']}; }}
QToolButton:checked {{ background: {t['accent']}; }}
QToolButton:disabled {{ background: transparent; }}
QToolButton:focus {{ border: 2px solid {t['accent']}; padding: 2px 5px; }}
QToolButton::menu-indicator {{ image: none; }}
#banner {{ background: {t['banner']}; border-bottom: 1px solid {t['border']}; }}
#banner QLabel {{ color: {t['banner_text']}; font-weight: 600; }}
QMenu {{ background: {t['group']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 4px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {t['accent']}; color: {t['on_accent']}; }}
QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 6px; }}
QToolTip {{ background: {t['group']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 3px 6px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea#screenScroll, QScrollArea#screenScroll > QWidget > QWidget {{ background: {t['screen']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: {t['muted']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 4px; min-width: 32px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
'''


STYLE = build_style(LIGHT)      # kept for callers (and tests) that apply a style directly


def _palette(t: dict) -> QPalette:
    palette = QPalette()
    pairs = {QPalette.ColorRole.Window: 'window', QPalette.ColorRole.WindowText: 'text', QPalette.ColorRole.Base: 'field',
             QPalette.ColorRole.AlternateBase: 'stripe', QPalette.ColorRole.Text: 'text', QPalette.ColorRole.Button: 'group',
             QPalette.ColorRole.ButtonText: 'text', QPalette.ColorRole.Highlight: 'accent',
             QPalette.ColorRole.HighlightedText: 'on_accent', QPalette.ColorRole.ToolTipBase: 'group',
             QPalette.ColorRole.ToolTipText: 'text', QPalette.ColorRole.PlaceholderText: 'muted'}
    for role, key in pairs.items():
        palette.setColor(role, QColor(t[key]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(t['muted']))
    return palette


def wanted_theme() -> dict:
    forced = os.environ.get('ISS_THEME', '').lower()
    if forced in ('light', 'dark'):
        return DARK if forced == 'dark' else LIGHT
    try:
        from PySide6.QtGui import QGuiApplication
        return DARK if QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark else LIGHT
    except Exception:
        return LIGHT


def apply_theme(app, theme: dict | None = None) -> dict:
    """Install the palette and style sheet, and follow later system changes."""
    global _current
    _current = theme or wanted_theme()
    app.setStyle('Fusion')
    app.setPalette(_palette(_current))
    app.setStyleSheet(build_style(_current))
    if not getattr(app, '_iss_theme_hooked', False) and not os.environ.get('ISS_THEME'):
        try:
            app.styleHints().colorSchemeChanged.connect(lambda *_: apply_theme(app))
            app._iss_theme_hooked = True
        except Exception:
            pass
    return _current


def animations_enabled() -> bool:
    """False when the user asked for reduced motion (ISS_REDUCE_MOTION=1, or
    Windows' “Show animations in Windows” switched off)."""
    if os.environ.get('ISS_REDUCE_MOTION') == '1':
        return False
    if sys.platform == 'win32':
        try:
            import ctypes
            enabled = ctypes.c_int(1)
            if ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(enabled), 0):  # SPI_GETCLIENTAREAANIMATION
                return bool(enabled.value)
        except Exception:
            pass
    return True


def fade_in(window, ms: int = 160) -> None:
    """Short opacity fade on first show. Skipped for reduced motion; it never
    moves a control, so hit targets stay put."""
    if not animations_enabled():
        return
    animation = QPropertyAnimation(window, b'windowOpacity', window)
    animation.setDuration(ms)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
