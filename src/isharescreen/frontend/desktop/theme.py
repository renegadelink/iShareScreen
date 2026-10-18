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

DARK['style'] = LIGHT['style'] = 'macos'

# Liquid Glass (https://github.com/pointlessduffin-21/liquid-glass-design): frosted white surfaces over a soft
# green field. Light only, as the kit specifies. Colour tuples are RGBA for painting; the *_soft/_line/hairline
# strings are for style sheets.
GLASS = dict(
    name='glass', style='glass',
    window='#eef3ee', sidebar='#eef3ee', bar='#f4f8f4', group='#f7faf7', group_hover='#ffffff', border='#d3dcd5',
    text='#10241a', muted='#4a5c54', accent='#127942', accent_hover='#1a6f3a', on_accent='#ffffff', stripe='#f1f6f1',
    field='#ffffff', error='#a8302a', ok='#0f7a36', selected='#127942', selected_text='#ffffff', screen='#101510',
    banner='#fdf3e2', banner_text='#5d4515',
    ink2='#3f5248', brand_deep='#0f4f34', brand_green='#1a6f3a', brand_mid='#5fae52', brand_lime='#a8cc4a',
    well='#f7faf7', hairline='rgba(16,36,26,23)', hairline2='rgba(16,36,26,41)',
    accent_soft='rgba(18,121,66,26)', accent_line='rgba(18,121,66,71)',
    glass_top=(255, 255, 255, 189), glass_bottom=(255, 255, 255, 138),
    strong_top=(255, 255, 255, 230), strong_bottom=(255, 255, 255, 194), flat=(255, 255, 255, 158),
    edge=(255, 255, 255, 199), field_a=(168, 204, 74, 77), field_b=(95, 174, 82, 56), field_c=(18, 121, 66, 41),
    shadow=(15, 79, 52), warn_ink='#8a5d06', icon_on='#127942', icon_selected='#127942',
)

_current = LIGHT


def tokens() -> dict:
    """The colours of the active theme, for widgets that paint themselves."""
    return _current


def is_glass() -> bool:
    return _current.get('style') == 'glass'


def label(macos: str, glass: str) -> str:
    """Text for the active look: macOS capitalises titles, Liquid Glass uses sentence case."""
    return glass if is_glass() else macos


def effects_reduced() -> bool:
    """True when frosted transparency should become opaque panels (ISS_EFFECTS=reduced, or
    Windows' “Transparency effects” switched off)."""
    if os.environ.get('ISS_EFFECTS') == 'reduced':
        return True
    if sys.platform == 'win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize') as key:
                return winreg.QueryValueEx(key, 'EnableTransparency')[0] == 0
        except OSError:
            pass
    return False


def motion_ms(normal: int) -> int:
    """Duration for an opacity/colour change. Reduced motion keeps these short fades (at most 160 ms)
    and drops all movement; this app never animates position or scale."""
    return normal if animations_enabled() else min(normal, 120)


# Back-compatible names used by older modules.
INK, MUTED, OK, FOCUS = LIGHT['text'], LIGHT['muted'], LIGHT['ok'], LIGHT['accent']
SCREEN_BG = LIGHT['screen']


def build_style(t: dict) -> str:
    return _glass_style(t) if t.get('style') == 'glass' else _macos_style(t)


def _check_icon() -> str:
    """A white tick for checked boxes (a style sheet can only point at a file)."""
    import tempfile
    folder = os.path.join(tempfile.gettempdir(), 'iShareScreen-glass')
    path = os.path.join(folder, 'check.svg')
    try:
        os.makedirs(folder, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="M4 8.4l2.7 2.7L12 5.6" '
                         'fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    except OSError:
        return 'none'
    return path.replace('\\', '/')


def _glass_style(t: dict) -> str:
    """Style sheet for the Liquid Glass look. Panels, pills and the field are painted by
    `widgets.py` (a style sheet cannot layer gradients or animate); this covers the controls."""
    return f'''
QWidget {{ font-family: 'Segoe UI Variable Text', 'Segoe UI'; font-size: 14px; color: {t['text']}; }}
QMainWindow, QDialog {{ background: {t['window']}; }}
#title {{ font-size: 15px; font-weight: 650; }}
#heading {{ font-size: 15px; font-weight: 650; }}
#hint {{ color: {t['muted']}; font-size: 13px; }}
QLabel#eyebrow {{ font-family: 'Cascadia Mono', Consolas, monospace; font-size: 10px; font-weight: 600; color: {t['muted']}; }}
QLabel#pageTitle {{ font-size: 26px; font-weight: 700; color: {t['text']}; }}
QLabel#pageSub {{ font-size: 15px; color: {t['muted']}; }}
#emptyTitle {{ font-size: 24px; font-weight: 700; }}
#emptyBody {{ color: {t['muted']}; font-size: 15px; }}
#fieldLabel {{ font-weight: 600; color: {t['ink2']}; }}
#sectionDivider {{ background: {t['hairline']}; max-height: 1px; min-height: 1px; }}
QLabel[kind="error"] {{ color: {t['error']}; font-weight: 600; }}
QLabel[kind="ok"] {{ color: {t['ok']}; font-weight: 600; }}
QLineEdit {{ background: #ffffff; border: 1px solid {t['hairline2']}; border-radius: 10px; padding: 8px 12px;
             min-height: 26px; selection-background-color: {t['accent_line']}; selection-color: {t['text']}; }}
QLineEdit:focus {{ border: 2px solid {t['accent']}; padding: 7px 11px; }}
QLineEdit:disabled {{ background: rgba(16,36,26,10); color: {t['muted']}; }}
QLineEdit#bare {{ background: transparent; border: none; border-radius: 0; padding: 2px 0; min-height: 22px; }}
QLineEdit#bare:focus {{ border: none; padding: 2px 0; }}
QComboBox {{ background: #ffffff; border: 1px solid {t['hairline2']}; border-radius: 10px; padding: 6px 12px; min-height: 26px; }}
QComboBox:focus {{ border: 2px solid {t['accent']}; padding: 5px 11px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox#bare {{ background: transparent; border: none; padding: 2px 0; min-height: 22px; }}
QComboBox QAbstractItemView {{ background: {t['well']}; selection-background-color: {t['accent_soft']};
                               selection-color: {t['brand_deep']}; border: 1px solid {t['hairline2']}; outline: 0; }}
QPushButton {{ background: {t['well']}; border: 1px solid {t['hairline2']}; border-radius: 17px; padding: 6px 16px; min-height: 22px; font-weight: 600; }}
QPushButton:hover {{ background: #ffffff; }}
QPushButton:disabled {{ color: {t['muted']}; }}
QCheckBox {{ spacing: 9px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 5px; border: 1px solid {t['hairline2']}; background: #ffffff; }}
QCheckBox::indicator:checked {{ background: {t['accent']}; border-color: {t['brand_deep']}; image: url({_check_icon()}); }}
QCheckBox:disabled {{ color: {t['muted']}; }}
QProgressBar {{ background: rgba(16,36,26,23); border: none; border-radius: 4px; max-height: 8px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {t['brand_lime']}, stop:1 {t['accent']}); border-radius: 4px; }}
QListView#library {{ background: transparent; border: none; outline: none; }}
QStatusBar {{ background: rgba(255,255,255,158); color: {t['muted']}; border-top: 1px solid {t['hairline']}; }}
QStatusBar::item {{ border: none; }}
QToolButton {{ background: transparent; border: none; border-radius: 12px; padding: 4px 8px; min-width: 22px; min-height: 22px; }}
QToolButton::menu-indicator {{ image: none; }}
#banner {{ background: {t['banner']}; border-bottom: 1px solid {t['hairline2']}; }}
#banner QLabel {{ color: {t['warn_ink']}; font-weight: 600; }}
QMenu {{ background: {t['well']}; border: 1px solid {t['hairline2']}; border-radius: 12px; padding: 5px; }}
QMenu::item {{ padding: 8px 22px 8px 12px; border-radius: 8px; }}
QMenu::item:selected {{ background: {t['accent_soft']}; color: {t['brand_deep']}; }}
QMenu::separator {{ height: 1px; background: {t['hairline']}; margin: 4px 6px; }}
QToolTip {{ background: {t['well']}; color: {t['text']}; border: 1px solid {t['hairline2']}; padding: 4px 8px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea#screenScroll, QScrollArea#screenScroll > QWidget > QWidget {{ background: {t['screen']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: rgba(16,36,26,46); border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:vertical:hover {{ background: rgba(16,36,26,77); }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: rgba(16,36,26,46); border-radius: 4px; min-width: 32px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
'''


def _macos_style(t: dict) -> str:
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


def style_name() -> str:
    """'glass' for the Liquid Glass look, else 'macos' (ISS_STYLE, or the --style option)."""
    return 'glass' if os.environ.get('ISS_STYLE', '').lower() == 'glass' else 'macos'


def wanted_theme() -> dict:
    if style_name() == 'glass':
        return GLASS                      # the kit is light only
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
