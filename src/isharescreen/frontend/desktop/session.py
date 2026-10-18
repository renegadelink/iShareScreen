"""One remote-desktop session: its controller (network thread), screen widget
and window. Every session owns its own socket, thread and lifecycle."""
from __future__ import annotations
import collections
import math
import socket

from PySide6.QtCore import Qt, QThread, Signal, QRectF, QSize, QTimer, QEvent
from PySide6.QtGui import QImage, QPainter, QColor, QKeySequence, QShortcut, QAction, QCursor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QScrollArea, QToolButton, QMenu, QDialog, QDialogButtonBox, QCheckBox)

from ...proxy.address import format_address
from ...proxy.standard import (StandardClient, AuthRejected, UnsupportedSecurity, UnsupportedEncoding,
                               DesktopUnavailable, ServerRefused)
from .icons import glyph
from .theme import tokens, fade_in, is_glass
from .widgets import push_button, surface, tool_button

QWIDGETSIZE_MAX = 16777215


class RenderError(RuntimeError):
    """A received frame could not be turned into a displayable image."""


def classify_error(error: BaseException) -> tuple[str, str]:
    """Map a failure to `(kind, message)`. Messages never contain credentials."""
    if isinstance(error, AuthRejected):
        return 'auth', 'The Mac rejected the username or password. Check them and try again.'
    if isinstance(error, UnsupportedSecurity):
        return 'unsupported', str(error)
    if isinstance(error, UnsupportedEncoding):
        return 'encoding', f'The computer sent screen data this viewer cannot decode ({error}).'
    if isinstance(error, DesktopUnavailable):
        return 'unavailable', str(error)
    if isinstance(error, ServerRefused):
        return 'refused', str(error)
    if isinstance(error, RenderError):
        return 'render', f'The screen arrived but could not be displayed: {error}'
    if isinstance(error, socket.gaierror):
        return 'dns', 'The computer’s name could not be found.'
    if isinstance(error, TimeoutError):
        return 'timeout', 'The computer stopped responding (timed out).'
    if isinstance(error, ConnectionRefusedError):
        return 'refused', 'The computer refused the connection. Is Screen Sharing turned on?'
    if isinstance(error, ConnectionError):
        return 'disconnect', 'The connection was closed by the computer or the network.'
    if isinstance(error, OSError):
        return 'network', f'Network error: {error.strerror or error}'
    if isinstance(error, ValueError):
        return 'protocol', f'The computer sent data this viewer could not understand: {error}'
    return 'error', str(error) or type(error).__name__


_LIVE: set = set()   # controllers whose thread has not finished yet


def shutdown_all(wait_ms: int = 2000) -> None:
    """Stop every running session and wait for their threads to end."""
    for controller in list(_LIVE):
        controller.stop()
    for controller in list(_LIVE):
        if controller.wait(wait_ms):
            _LIVE.discard(controller)


def close_socket(sock) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class SessionController(QThread):
    """Connects, then receives framebuffer updates until stopped.

    Signals (all delivered on the GUI thread):
      phase(text)        a real stage of connection setup
      established(image) the first decoded framebuffer update, once
      frame(image)       every decoded update, including the first
      failed(kind, text) the session ended abnormally (not after stop())
      clipboard(text)    remote clipboard text
    QThread.finished always fires last.
    """
    phase = Signal(str)
    established = Signal(QImage)
    frame = Signal(QImage)
    failed = Signal(str, str)
    clipboard = Signal(str)
    _raw_frame = Signal(QImage)
    mode_label = 'Standard VNC'

    def clone(self, secret):
        """A fresh, unstarted controller to the same computer (used by Reconnect)."""
        return SessionController(self.host, self.port, self.user, secret, auth=self.auth)

    def __init__(self, host, port, user, password, auth='account'):
        super().__init__()
        self.host, self.port, self.user, self.auth = host, port, user, auth
        self._password = password
        self.client = None
        self.sock = None
        self.stopping = False
        self.is_established = False
        self.last_image = QImage()
        self.frames = 0
        self._raw_frame.connect(self._on_raw_frame)

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self, *args):
        _LIVE.add(self)   # keeps the thread object alive until it has finished
        self.finished.connect(self._cleanup)
        super().start(*args)

    def _cleanup(self):
        self.wait(2000)   # `finished` fires just before the OS thread exits
        _LIVE.discard(self)

    def stop(self):
        """Thread-safe and idempotent: unblocks any pending network call."""
        self.stopping = True
        close_socket(self.sock)
        client = self.client
        if client:
            client.close()

    def _on_socket(self, sock):
        self.sock = sock
        if self.stopping:
            close_socket(sock)

    def _on_phase(self, name):
        text = {'connecting': f'Connecting to {format_address(self.host, self.port)}…',
                'authenticating': 'Authenticating…',
                'requesting': 'Requesting the desktop…'}.get(name, name)
        if not self.stopping:
            self.phase.emit(text)

    def run(self):
        try:
            client = StandardClient(self.host, self.user, self._password, self.port,
                                    on_socket=self._on_socket, on_phase=self._on_phase)
            self._password = ''
            self.client = client
            if self.stopping:
                return
            self.phase.emit('Waiting for the first screen update…')
            while not self.stopping:
                kind = client.read_update()
                if kind == 'frame':
                    pixels = client.pixels
                    image = QImage(pixels.data, client.width, client.height, client.width*4,
                                   QImage.Format.Format_RGBA8888).copy()
                    if image.isNull():
                        raise RenderError('empty image')
                    self._raw_frame.emit(image)
                elif kind == 'clipboard':
                    self.clipboard.emit(client.clipboard)
        except Exception as error:
            if not self.stopping:
                self.failed.emit(*classify_error(error))
        finally:
            self._password = ''
            if self.client:
                self.client.close()
                self.client = None
            close_socket(self.sock)
            self.sock = None

    def _on_raw_frame(self, image):
        if self.stopping:
            return
        self.last_image = image
        self.frames += 1
        if not self.is_established:
            self.is_established = True
            self.established.emit(image)
        self.frame.emit(image)

    def send(self, method, *args):
        client = self.client
        if client and not self.stopping:
            try:
                getattr(client, method)(*args)
            except OSError:
                pass


SPECIAL = {
    Qt.Key.Key_Escape:0xff1b, Qt.Key.Key_Tab:0xff09, Qt.Key.Key_Backtab:0xff09,
    Qt.Key.Key_Backspace:0xff08, Qt.Key.Key_Return:0xff0d, Qt.Key.Key_Enter:0xff0d,
    Qt.Key.Key_Insert:0xff63, Qt.Key.Key_Delete:0xffff, Qt.Key.Key_Home:0xff50,
    Qt.Key.Key_End:0xff57, Qt.Key.Key_Left:0xff51, Qt.Key.Key_Up:0xff52,
    Qt.Key.Key_Right:0xff53, Qt.Key.Key_Down:0xff54, Qt.Key.Key_PageUp:0xff55,
    Qt.Key.Key_PageDown:0xff56, Qt.Key.Key_Shift:0xffe1, Qt.Key.Key_Control:0xffe3,
    # Measured against a live Intel Mac running Apple's Screen Sharing server:
    # Meta_L (0xffe7) acts as Option, while Alt_L, Super_L and Hyper_L all act as
    # Command. So the Alt key is sent as Meta_L to reach Option, and the Windows
    # key stays Super_L, which is Command.
    Qt.Key.Key_Alt:0xffe7, Qt.Key.Key_Meta:0xffeb, Qt.Key.Key_CapsLock:0xffe5,
}
SPECIAL.update({int(Qt.Key.Key_F1)+i:0xffbe+i for i in range(12)})
KEYSYM_COMMAND = 0xffeb     # Super_L
KEYSYM_OPTION = 0xffe7      # Meta_L


def keysym_for_char(char):
    """X11 keysym that types `char`, or None if it cannot be typed.

    Latin-1 characters use their own code; other Basic Multilingual Plane
    characters use the Unicode keysym range (0x01000000 | code point), which
    Apple's server types correctly. Control characters, and characters above
    U+FFFF (emoji and similar), return None: measured against a live Mac, the
    server drops them as a single keysym and types broken halves for a UTF-16
    surrogate pair, so skipping them is the only safe choice."""
    if char in '\r\n':
        return 0xff0d
    if char == '\t':
        return 0xff09
    code = ord(char)
    if code < 32 or 127 <= code < 160 or code > 0xFFFF:
        return None
    return code if code <= 255 else 0x01000000 | code


class Screen(QWidget):
    """Paints the remote framebuffer and forwards input to `sink.send(...)`.

    `zoom` is None for fit (scaled to the widget, aspect preserved), otherwise a
    scale in device pixels per remote pixel, so 1.0 is actual size. The widget
    then takes the scaled image size and a surrounding scroll area scrolls it.
    `view` names the state: 'fit', 'actual' (zoom 1.0) or 'zoom'.
    `input_enabled` False makes the screen view-only.
    """
    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self.image = QImage()
        self.target = QRectF()
        self.held = {}
        self.view = 'fit'
        self.zoom = None
        self.input_enabled = True
        self.ctrl_as_command = False   # send Ctrl as Command, so Ctrl+C/V/Z behave as on a PC
        self.swap_alt_win = False      # Alt -> Command and Windows key -> Option (default: Alt -> Option, Windows -> Command)
        self.last_pointer = (0, 0, 0)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(320,240)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def set_image(self, image):
        resized = image.size() != self.image.size()
        self.image = image
        if resized:
            self._fit_widget()
        self.update()

    def set_zoom(self, zoom):
        """None fits the window; a number is device pixels per remote pixel."""
        self.zoom = zoom
        self.view = 'fit' if zoom is None else 'actual' if abs(zoom-1.0) < 1e-9 else 'zoom'
        self._fit_widget()
        self.update()

    def set_view(self, view):
        self.set_zoom(1.0 if view == 'actual' else None)

    def _scaled_size(self):
        scale = (self.zoom or 1.0) / (self.devicePixelRatioF() or 1.0)
        return QSize(math.ceil(self.image.width()*scale), math.ceil(self.image.height()*scale))

    def _fit_widget(self):
        if self.zoom is not None and not self.image.isNull():
            self.setFixedSize(self._scaled_size())
        else:
            self.setMinimumSize(320,240)
            self.setMaximumSize(QWIDGETSIZE_MAX,QWIDGETSIZE_MAX)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tokens()['screen']))
        if self.image.isNull():
            painter.setPen(QColor('#c9c4c5'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'Waiting for the computer’s screen…')
            return
        if self.zoom is not None:
            size = self._scaled_size()
            self.target = QRectF(0, 0, size.width(), size.height())
        else:
            size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.target = QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,
                                 size.width(),size.height())
        if self.view != 'actual':
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.target,self.image)

    def remote_point(self, position):
        """Map a widget position to remote pixel coordinates (clamped), or None."""
        if self.image.isNull() or self.target.isEmpty():
            return None
        x = min(self.image.width()-1,max(0,int((position.x()-self.target.x())*self.image.width()/self.target.width())))
        y = min(self.image.height()-1,max(0,int((position.y()-self.target.y())*self.image.height()/self.target.height())))
        return x,y

    def sync_pointer(self, position):
        """Re-send the pointer for a stationary cursor after the view scrolled
        under it, keeping the current button state."""
        point = self.remote_point(position)
        if point and self.input_enabled:
            mask = self.last_pointer[2]
            self.sink.send('pointer',point[0],point[1],mask)
            self.last_pointer = (point[0],point[1],mask)

    def pointer(self, event):
        point = self.remote_point(event.position())
        if point is None or not self.input_enabled:
            return
        x,y = point
        buttons = event.buttons()
        # Apple swaps the right/middle button bits relative to generic RFB.
        mask = (1 if buttons & Qt.MouseButton.LeftButton else 0) | (2 if buttons & Qt.MouseButton.RightButton else 0) | (4 if buttons & Qt.MouseButton.MiddleButton else 0)
        self.sink.send('pointer',x,y,mask)
        self.last_pointer = (x,y,mask)

    def mouseMoveEvent(self,event): self.pointer(event)
    def mousePressEvent(self,event):
        self.setFocus()
        self.pointer(event)
    def mouseReleaseEvent(self,event): self.pointer(event)

    def wheelEvent(self,event):
        if not self.input_enabled:
            event.accept()
            return
        x,y,mask = self.last_pointer
        bit = 8 if event.angleDelta().y()>0 else 16
        for _ in range(max(1,min(8,abs(event.angleDelta().y())//120))):
            self.sink.send('pointer',x,y,mask|bit)
            self.sink.send('pointer',x,y,mask)
        event.accept()

    def keyPressEvent(self,event):
        if not self.input_enabled:
            event.accept()
            return
        key = event.key()
        code = SPECIAL.get(key)
        if key == Qt.Key.Key_Control and self.ctrl_as_command:
            code = KEYSYM_COMMAND
        elif self.swap_alt_win and key == Qt.Key.Key_Alt:
            code = KEYSYM_COMMAND
        elif self.swap_alt_win and key == Qt.Key.Key_Meta:
            code = KEYSYM_OPTION
        if code is None:
            text = event.text()
            if event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
                                    | Qt.KeyboardModifier.AltModifier) and 32 <= key <= 126:
                text = chr(key).lower()
            if text:
                code = keysym_for_char(text[0])
        if code is not None:
            self.held[key] = code
            self.sink.send('key',code,True)
        event.accept()

    def keyReleaseEvent(self,event):
        if not event.isAutoRepeat():
            code = self.held.pop(event.key(),None)
            if code is not None:
                self.sink.send('key',code,False)
        event.accept()

    def release_input(self):
        """Release every held key and pointer button on the remote computer."""
        for code in self.held.values():
            self.sink.send('key',code,False)
        self.held.clear()
        x,y,mask = self.last_pointer
        if mask:
            self.sink.send('pointer',x,y,0)
            self.last_pointer = (x,y,0)

    def focusOutEvent(self,event):
        self.release_input()
        super().focusOutEvent(event)

    def focusNextPrevChild(self,next):
        return False  # Tab belongs to the remote computer while the screen has focus.


class PasswordDialog(QDialog):
    """Asks for a password again (passwords are never stored)."""
    def __init__(self, parent, user, address):
        super().__init__(parent)
        self.setWindowTitle('Reconnect')
        self.setModal(True)
        layout = QVBoxLayout(self); layout.setContentsMargins(24,20,24,20); layout.setSpacing(10)
        layout.addWidget(QLabel(f'Password for {user} on {address}' if user else f'VNC password for {address}'))
        self.field = QLineEdit(); self.field.setEchoMode(QLineEdit.EchoMode.Password)
        self.field.setAccessibleName('Password'); layout.addWidget(self.field)
        show = QCheckBox('Show password')
        show.toggled.connect(lambda on: self.field.setEchoMode(QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password))
        layout.addWidget(show)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText('Reconnect')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def password(self):
        return self.field.text()


class SessionWindow(QMainWindow):
    """Remote desktop in its own window; closing it ends only this session."""
    closed = Signal(object)
    state_changed = Signal(object)

    def __init__(self, controller, title, view='fit'):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.controller = None
        self.title_text = title
        self.host, self.port, self.user, self.auth = controller.host, controller.port, controller.user, controller.auth
        self.template = controller     # remembers the kind of connection, for Reconnect
        self.state = 'Connecting'
        self.system_id = None          # library entry, once saved
        self.saved_secret = None       # optional callable returning the saved password, so Reconnect needs no prompt
        self._saved_rejected = False   # the saved password was refused: prompt next time
        self._reconnect_used_saved = False
        self.resize(1180,780)
        self.setMinimumSize(480,360)
        self.screen_widget = Screen(self)
        self.scroll = QScrollArea(); self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setObjectName('screenScroll')       # dark around the remote pixels, as in macOS
        self.scroll.setWidget(self.screen_widget); self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.banner = QFrame(); self.banner.setObjectName('banner'); self.banner.hide()
        bar = QHBoxLayout(self.banner); bar.setContentsMargins(16,8,16,8)
        self.banner_text = QLabel(); self.banner_text.setWordWrap(True); bar.addWidget(self.banner_text,1)
        self.banner_button = push_button('Reconnect…'); self.banner_button.clicked.connect(self.reconnect); bar.addWidget(self.banner_button)
        central = QWidget(); column = QVBoxLayout(central); column.setContentsMargins(0,0,0,0); column.setSpacing(0)
        column.addWidget(self.banner); column.addWidget(self.scroll,1)
        self.setCentralWidget(central)
        self.info_label = QLabel(); self.info_label.setObjectName('hint')
        self.info_label.setToolTip('Ctrl+Alt+Enter toggles full screen')
        self.statusBar().addPermanentWidget(self.info_label)
        # Toolbar: symbols only (names live in the tooltips), in segmented groups.
        self.control_action = self._action('Control', self.set_control, 'pointer', checkable=True,
                                           tip='Control this computer. Turn off to only watch.')
        self.control_action.setChecked(True)
        self.clip_action = self._action('Share clipboard', self.set_clipboard_sync, 'clipboard', checkable=True,
                                        tip='Copy text between this computer and the Mac automatically')
        self.clip_action.setChecked(True)
        self.type_action = self._action('Type clipboard', self.type_clipboard, 'keyboard',
                                        tip='Type the clipboard text as keystrokes. Works for any text, including characters '
                                            'outside Latin-1, and in places where pasting is not possible.')
        self.launchpad_action = self._action('Launchpad', self.launchpad, 'launchpad',
                                             tip='Open Launchpad (through Spotlight)')
        self.mission_action = self._action('Mission Control', self.mission_control, 'mission',
                                           tip='Mission Control (Control+Up Arrow)')
        self.desktop_action = self._action('Show Desktop', self.show_desktop, 'desktop',
                                           tip='Show Desktop (F11); press again to bring the windows back')
        self.zoom_out_action = self._action('Zoom out', self.zoom_out, 'zoom-out', tip='Zoom out')
        self.actual = self._action('Actual size', self.set_actual_size, 'zoom-actual', checkable=True,
                                   tip='Actual size: one remote pixel per screen pixel (push the pointer against an edge to scroll)')
        self.zoom_in_action = self._action('Zoom in', self.zoom_in, 'zoom-in', tip='Zoom in')
        self.fit_action = self._action('Fit to window', lambda _checked=True: self.set_zoom(None), 'zoom-fit', checkable=True,
                                       tip='Fit the whole screen in this window')
        self.refresh_action = self._action('Refresh Screen', lambda: self.send('request', True), 'refresh')
        self.full_action = self._action('Full Screen', self.toggle_fullscreen, 'fullscreen', tip='Ctrl+Alt+Enter')
        self.reconnect_action = self._action('Reconnect…', self.reconnect, 'refresh')
        self.reconnect_action.setVisible(False)
        self.ctrl_action = self._action('Ctrl = Command', self.set_ctrl_as_command, 'control', checkable=True,
                                        tip='Send the Ctrl key as Command, so Ctrl+C, Ctrl+V and Ctrl+Z work as on Windows')
        self.swap_action = self._action('Swap Alt/Win', self.set_swap_alt_win, 'swap', checkable=True,
                                        tip='Send Alt as Command and the Windows key as Option, instead of Alt as Option '
                                            'and the Windows key as Command')
        self.disconnect_action = self._action('Disconnect', self.close, 'power',
                                              tip='Disconnect and close this window')
        self.toolbar = surface('strong' if is_glass() else 'well', 'header', radius=0)
        row = QHBoxLayout(self.toolbar); row.setContentsMargins(12, 6, 12, 6); row.setSpacing(12)
        row.addStretch(1)
        row.addWidget(self._segment(self.control_action, self.clip_action, self.type_action))
        row.addWidget(self._segment(self.launchpad_action, self.mission_action, self.desktop_action))
        row.addStretch(1)
        row.addWidget(self._segment(self.zoom_out_action, self.actual, self.zoom_in_action, self.fit_action))
        self.more_button = tool_button(); self.more_button.setIcon(glyph('more'))
        self.more_button.setToolTip('More'); self.more_button.setAccessibleName('More')
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_menu = QMenu(self.more_button)
        for action in (self.refresh_action, self.full_action, self.reconnect_action):
            self.more_menu.addAction(action)
        self.more_menu.addSeparator()
        for action in (self.ctrl_action, self.swap_action):
            self.more_menu.addAction(action)
        self.more_menu.addSeparator()
        self.more_menu.addAction(self.disconnect_action)
        self.more_button.setMenu(self.more_menu)
        row.addWidget(self._segment_widget(self.more_button))
        column.insertWidget(0, self.toolbar)
        self._typing = collections.deque()    # keysyms still to be typed
        self.type_timer = QTimer(self); self.type_timer.setInterval(8)
        self.type_timer.timeout.connect(self._type_tick)
        self._clip_text = ''          # last text exchanged, so a copy never echoes back and forth
        self.pan_timer = QTimer(self); self.pan_timer.setInterval(16)
        self.pan_timer.timeout.connect(lambda: self.pan_step(QCursor.pos()))
        QApplication.clipboard().dataChanged.connect(self.on_local_clipboard)
        # Only one shortcut: others would steal keys meant for the remote computer.
        for keys in ('Ctrl+Alt+Return', 'Ctrl+Alt+Enter'):
            QShortcut(QKeySequence(keys),self).activated.connect(self.toggle_fullscreen)
        self.set_zoom(1.0 if view == 'actual' else None)
        self.attach(controller)

    def _note(self, text, ms=6000):
        """Feedback for an action: shown for a few seconds, never overwritten by screen updates."""
        self.statusBar().showMessage(text, ms)

    def _action(self, text, slot, icon, checkable=False, tip=''):
        action = QAction(glyph(icon), text, self); action.setCheckable(checkable)
        action.setToolTip(tip or text)
        action.triggered.connect(lambda checked=False: (slot(checked) if checkable else slot(), self.screen_widget.setFocus()))
        return action

    def _button(self, action):
        button = tool_button(); button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        button.setAccessibleName(action.text())
        button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        return button

    @staticmethod
    def _segment_widget(*buttons):
        segment = surface('segment', 'segment')
        layout = QHBoxLayout(segment); layout.setContentsMargins(3, 2, 3, 2); layout.setSpacing(1)
        for button in buttons:
            layout.addWidget(button)
        return segment

    def _segment(self, *actions):
        return self._segment_widget(*(self._button(action) for action in actions))

    # ── controller binding ───────────────────────────────────────────
    def attach(self, controller):
        """Bind to `controller` (first connect or after a reconnect)."""
        self.detach()
        self.controller = controller
        controller.phase.connect(self.on_phase)
        controller.frame.connect(self.on_frame)
        controller.failed.connect(self.on_failed)
        controller.clipboard.connect(self.on_clipboard)
        self.banner.hide(); self.reconnect_action.setVisible(False)
        if controller.is_established:
            self.on_frame(controller.last_image)
        else:
            self._set_state('Connecting')
            self._note('Connecting…')

    def detach(self):
        controller, self.controller = self.controller, None
        if controller is not None:
            for signal, slot in ((controller.phase, self.on_phase), (controller.frame, self.on_frame),
                                 (controller.failed, self.on_failed), (controller.clipboard, self.on_clipboard)):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            controller.stop()

    def _set_state(self, state):
        self.state = state
        self.setWindowTitle(f'{self.title_text} — {state} — iShareScreen')
        self.state_changed.emit(self)

    @property
    def is_live(self):
        return self.state == 'Connected'

    def on_phase(self, text):
        self._note(text)

    # ── clipboard ────────────────────────────────────────────────────
    def on_clipboard(self, text):
        """Remote -> local."""
        if self.clip_action.isChecked() and text != self._clip_text:
            self._clip_text = text
            QApplication.clipboard().setText(text)

    def on_local_clipboard(self):
        """Local -> remote, only while this window is the one in use."""
        if self.isActiveWindow():
            self.push_clipboard()

    def push_clipboard(self):
        controller = self.controller
        if not self.clip_action.isChecked() or not self.can_control or controller is None \
                or not controller.is_established:
            return
        text = QApplication.clipboard().text()
        if not text or text == self._clip_text:
            return
        if len(text) > StandardClient.MAX_CLIPBOARD:
            self._note('Clipboard not sent: it is larger than 1 MB.')
            return
        self._clip_text = text
        controller.send('send_clipboard', text)
        if not text.isascii() and text.encode('latin-1','replace').decode('latin-1') != text:
            self._note('Clipboard sent. Characters outside Latin-1 were replaced with “?”; '
                                         'use Type clipboard to send them exactly.')

    TYPE_LIMIT = 20000      # characters per "Type clipboard"
    TYPE_PER_TICK = 20      # keystrokes sent every 8 ms, so the window stays responsive

    def type_clipboard(self):
        self.type_text(QApplication.clipboard().text())

    def type_text(self, text):
        """Type `text` on the remote computer as Unicode keystrokes."""
        controller = self.controller
        if controller is None or not controller.is_established or self.state != 'Connected':
            self._note('Not connected.')
            return
        if not self.can_control:
            self._note('Control is off: turn it on to type.')
            return
        keysyms = [k for k in map(keysym_for_char, text.replace('\r\n', '\n')) if k]
        astral = sum(1 for ch in text if ord(ch) > 0xFFFF)
        if not keysyms:
            self._note('Nothing to type: the clipboard has no text that can be typed.' if astral
                                         else 'Nothing to type: the clipboard has no text.')
            return
        self.screen_widget.release_input()      # no held modifier may change what is typed
        self._typing.clear()
        self._typing.extend(keysyms[:self.TYPE_LIMIT])
        self._typed = 0
        self._type_note = f' (only the first {self.TYPE_LIMIT:,} of {len(keysyms):,})' if len(keysyms) > self.TYPE_LIMIT else ''
        if astral:
            self._type_note += (f'. {astral:,} emoji or other character{"s" if astral != 1 else ""} outside the '
                                'Basic Multilingual Plane could not be typed and were skipped')
        self.type_timer.start()

    def _type_tick(self):
        controller = self.controller
        if controller is None or controller.stopping or not self._typing:
            self._stop_typing()
            return
        for _ in range(min(self.TYPE_PER_TICK, len(self._typing))):
            keysym = self._typing.popleft()
            controller.send('key', keysym, True)
            controller.send('key', keysym, False)
            self._typed += 1
        if not self._typing:
            self.type_timer.stop()
            self._note(f'Typed {self._typed:,} characters{self._type_note}.')

    def _stop_typing(self):
        self._typing.clear()
        self.type_timer.stop()

    def set_swap_alt_win(self, on):
        self.screen_widget.release_input()      # a key held across the switch is released as it was pressed
        self.screen_widget.swap_alt_win = on
        self._note('Alt is sent as Command and the Windows key as Option' if on
                                     else 'Alt is sent as Option and the Windows key as Command')

    def set_ctrl_as_command(self, on):
        self.screen_widget.release_input()
        self.screen_widget.ctrl_as_command = on
        self._note('Ctrl is sent as Command' if on else 'Ctrl is sent as Control')

    def set_clipboard_sync(self, on):
        if on:
            self.push_clipboard()
        self._note('Clipboard sharing ' + ('on' if on else 'off'))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.push_clipboard()      # copied elsewhere, then switched to this session

    def on_frame(self, image):
        self.screen_widget.set_image(image)
        first = self.state != 'Connected'
        if first:
            self._set_state('Connected')
            self._saved_rejected = self._reconnect_used_saved = False
            self.screen_widget.setFocus()
            # Whatever is on the clipboard when the session starts is not shared;
            # only text copied afterwards is.
            self._clip_text = QApplication.clipboard().text()
        controller = self.controller
        # Connection info lives in its own label, so it never overwrites an action's message.
        self.info_label.setText(f'{format_address(self.host, self.port)} · {image.width()} × {image.height()} · '
                                f'{self.template.mode_label} · {controller.frames if controller else 0} updates')
        if first:
            self._note('Connected')

    def on_failed(self, kind, message):
        self._launching = False
        self._stop_typing()
        self.screen_widget.release_input()
        self._set_state('Disconnected')
        if kind == 'auth' and self._reconnect_used_saved:
            self._saved_rejected = True
            message = 'The saved password was rejected. Choose Reconnect to enter a new one.'
        self._reconnect_used_saved = False
        self.banner_text.setText(('Connection lost: ' if kind not in ('auth',) else 'Could not reconnect: ') + message)
        self.banner.show(); self.reconnect_action.setVisible(True)
        self._note('Disconnected. Use Reconnect to sign in again.')

    # ── actions ──────────────────────────────────────────────────────
    def send(self, method, *args):
        if self.controller:
            self.controller.send(method, *args)

    # ── control, zoom ────────────────────────────────────────────────
    ZOOM_STEPS = (0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)

    def fit_zoom(self):
        """The scale (device px per remote px) that 'fit to window' currently uses."""
        image = self.screen_widget.image
        if image.isNull():
            return 1.0
        viewport = self.scroll.viewport().size()
        scale = self.screen_widget.devicePixelRatioF() or 1.0
        return max(0.05, min(viewport.width() / image.width(), viewport.height() / image.height()) * scale)

    def set_zoom(self, zoom):
        """None fits the window; a number is device pixels per remote pixel (1.0 = actual size)."""
        self.screen_widget.set_zoom(zoom)
        self.scroll.setWidgetResizable(zoom is None)
        if zoom is None:
            self.pan_timer.stop()
        else:
            self.pan_timer.start()
        self.actual.setChecked(zoom is not None and abs(zoom - 1.0) < 1e-9)
        self.fit_action.setChecked(zoom is None)
        shown = 'Fit to window' if zoom is None else f'Zoom {round(zoom * 100)}%'
        if self.state == 'Connected':
            self._note(shown)

    def set_actual_size(self, on):
        self.set_zoom(1.0 if on else None)

    def zoom_in(self):
        now = self.screen_widget.zoom or self.fit_zoom()
        bigger = [s for s in self.ZOOM_STEPS if s > now * 1.02]
        if bigger:
            self.set_zoom(bigger[0])
        else:
            self._note('Already at the largest zoom')   # never shrink when "in" has nowhere to go

    def zoom_out(self):
        now = self.screen_widget.zoom or self.fit_zoom()
        smaller = [s for s in self.ZOOM_STEPS if s < now / 1.02]
        self.set_zoom(smaller[-1] if smaller else self.ZOOM_STEPS[0])

    def set_control(self, on):
        """Off = view only: nothing typed, clicked or pasted reaches the Mac."""
        self.screen_widget.release_input()
        self.screen_widget.input_enabled = on
        if not on:
            self._stop_typing()
        self._note('Controlling this computer' if on else 'View only: input is not sent')

    @property
    def can_control(self):
        return self.control_action.isChecked()

    # ── Mac shortcuts: Mission Control, Show Desktop, Launchpad ──────
    # Measured on an Intel iMac and an Apple Silicon Mac mini: Control+Up Arrow is
    # Mission Control and F11 is Show Desktop (a toggle). Neither Mac answers a
    # plain key for Launchpad, so it is opened through Spotlight.
    KEY_CONTROL, KEY_UP, KEY_F11, KEY_SUPER, KEY_RETURN, KEY_SPACE = 0xffe3, 0xff52, 0xffc8, 0xffeb, 0xff0d, 0x20

    def _chord(self, modifiers, key, then=None):
        """Press `modifiers` + `key` and release them, with the short holds macOS
        needs to notice a shortcut. Uses raw keysyms, so the Ctrl/Alt/Win options
        never change what these buttons do."""
        controller = self.controller
        if controller is None or not controller.is_established or self.state != 'Connected':
            self._note('Not connected.')
            return False
        if not self.can_control:
            self._note('Control is off: turn it on to use this.')
            return False
        self.screen_widget.release_input()
        steps = [(0, m, True) for m in modifiers] + [(40, key, True), (140, key, False)] + \
                [(170, m, False) for m in reversed(modifiers)]
        for delay, keysym, down in steps:
            QTimer.singleShot(delay, lambda k=keysym, d=down, c=controller: None if c.stopping else c.send('key', k, d))
        if then:
            QTimer.singleShot(200, then)
        return True

    def mission_control(self):
        self._chord([self.KEY_CONTROL], self.KEY_UP)

    def show_desktop(self):
        self._chord([], self.KEY_F11)

    LAUNCH_GRID = (40, 24)      # sample points over the middle of the screen

    def _signature(self):
        image = self.controller.last_image if self.controller else QImage()
        if image.isNull():
            return None
        cols, rows = self.LAUNCH_GRID
        return [image.pixel(int(image.width() * (.2 + .6 * (c + .5) / cols)), int(image.height() * (.2 + .6 * (r + .5) / rows)))
                for r in range(rows) for c in range(cols)]

    def launchpad(self):
        """Open Launchpad: Command+Space, type "Launchpad", Return. Nothing is typed unless
        Spotlight is seen to open, so a Mac with another Spotlight shortcut, or a screen too
        busy to tell, is left alone."""
        if getattr(self, '_launching', False):
            return
        first = self._signature()
        if first is None:
            return
        if not self._chord([self.KEY_SUPER], self.KEY_SPACE):
            return
        self._launching = True
        QTimer.singleShot(150, lambda: self._launch_check(first, self._signature()))

    def _launch_check(self, first, second):
        """`first`/`second` were taken before the keys and shortly after, so points that
        move on their own (video, a clock) can be ignored."""
        QTimer.singleShot(900, lambda: self._launch_decide(first, second))

    def _launch_decide(self, first, second):
        controller, after = self.controller, self._signature()
        if controller is None or controller.stopping or after is None or first is None or second is None:
            self._launching = False
            return
        steady = [i for i, (a, b) in enumerate(zip(first, second)) if a == b]
        changed = sum(1 for i in steady if after[i] != first[i])
        if len(steady) < len(first) * .6:
            note = 'The screen is changing too much to confirm that Spotlight opened, so Launchpad was not started.'
        elif changed < max(8, len(first) * .012):
            note = 'Spotlight did not open (is Command+Space its shortcut?), so Launchpad was not started.'
        else:
            note = ''
        if note:
            self._note(note)
            self._launching = False
            return
        keysyms = [ord(ch) for ch in 'Launchpad']
        for i, keysym in enumerate(keysyms):
            QTimer.singleShot(i * 30, lambda k=keysym, c=controller: None if c.stopping else (c.send('key', k, True), c.send('key', k, False)))
        QTimer.singleShot(len(keysyms) * 30 + 700, lambda c=controller: self._launch_finish(c))

    def _launch_finish(self, controller):
        if not controller.stopping:
            controller.send('key', self.KEY_RETURN, True)
            controller.send('key', self.KEY_RETURN, False)
            self._note('Opened Launchpad')
        self._launching = False

    PAN_MARGIN = 40     # px from the window edge where the view starts to scroll
    PAN_SPEED = 24      # max px per tick, reached at the very edge

    def pan_step(self, global_pos):
        """Zoomed or actual-size view: scroll toward the edge the cursor is pushing against,
        then re-send the pointer, since the image moved under a still cursor."""
        if not self.isActiveWindow() or self.screen_widget.view == 'fit':
            return
        viewport = self.scroll.viewport()
        pos = viewport.mapFromGlobal(global_pos)
        if not viewport.rect().contains(pos):
            return
        moved = False
        for bar, near, far in ((self.scroll.horizontalScrollBar(), pos.x(), viewport.width()-pos.x()),
                               (self.scroll.verticalScrollBar(), pos.y(), viewport.height()-pos.y())):
            if bar.maximum() <= 0:
                continue
            if near < self.PAN_MARGIN:
                delta = -round(self.PAN_SPEED*(self.PAN_MARGIN-near)/self.PAN_MARGIN)
            elif far < self.PAN_MARGIN:
                delta = round(self.PAN_SPEED*(self.PAN_MARGIN-far)/self.PAN_MARGIN)
            else:
                continue
            before = bar.value()
            bar.setValue(before+delta)
            moved = moved or bar.value() != before
        if moved:
            self.screen_widget.sync_pointer(self.screen_widget.mapFromGlobal(global_pos))

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.toolbar.show(); self.statusBar().show()
        else:
            self.toolbar.hide(); self.statusBar().hide()
            self.showFullScreen()
        self.screen_widget.setFocus()

    def reconnect(self):
        controller = self.controller
        if controller is not None and not controller.isFinished() and controller.is_established:
            return
        secret = ''
        self._reconnect_used_saved = False
        if self.auth != 'none':
            if self.saved_secret is not None and not self._saved_rejected:
                secret = self.saved_secret() or ''
                self._reconnect_used_saved = bool(secret)
            if not secret:
                dialog = PasswordDialog(self, self.user, format_address(self.host, self.port))
                if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.password():
                    return
                secret = dialog.password()
        fresh = self.template.clone(secret)          # same kind (Standard or High Performance) as the original
        self.attach(fresh)
        fresh.start()

    def showEvent(self, event):
        first = not getattr(self, '_shown', False)
        self._shown = True
        super().showEvent(event)
        if first:
            fade_in(self)

    def closeEvent(self, event):
        self._launching = False
        self._stop_typing()
        self.pan_timer.stop()
        try:
            QApplication.clipboard().dataChanged.disconnect(self.on_local_clipboard)
        except (RuntimeError, TypeError):
            pass
        self.screen_widget.release_input()
        self.detach()
        self.closed.emit(self)
        event.accept()
