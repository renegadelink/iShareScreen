"""New-connection dialogs, in the style of macOS Screen Sharing: "Connect To:" with
one field, then a sign-in dialog, then progress. The address is checked for a
VNC server before anything asks for credentials. On success the live session
controller is handed to its owner and the dialog closes; on any failure it returns
to the sign-in dialog with the address, username and settings intact."""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import Qt, QObject, Signal, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, QFrame,
    QStackedWidget, QProgressBar, QComboBox, QCheckBox, QToolButton)

from ...proxy.address import ProbeError, ProbeResult, parse_address, probe_server, format_address
from .credentials import UnavailableStore
from .hp import HPController, RESOLUTIONS
from .icons import glyph
from .library import SavedSystem, MODE_LABELS
from .session import SessionController, close_socket
from .theme import fade_in, is_glass, label
from .widgets import paint_backdrop, push_button, reveal, surface, tool_button

HP_NOTE = 'High Performance is not available.'

_PROBES: set = set()   # probe threads that have not finished yet
CONTENT_WIDTH = 456


@dataclass
class WizardResult:
    host: str
    port: int
    user: str
    name: str
    mode: str
    view: str
    auth: str = 'account'      # 'account', 'password' or 'none'
    remember: bool = False     # the user asked to save the password
    password: str = field(default='', repr=False)   # only set when `remember`; never printed
    resolution: str = 'auto'   # High Performance only


class ProbeWorker(QObject):
    """Checks an address on a daemon thread. A daemon thread (not a QThread) so that
    quitting mid-check, or a lookup that is slow to give up, can never block or crash
    shutdown. Both signals carry the token the wizard gave this probe, so results of a
    probe that was cancelled or superseded are recognised and dropped."""
    found = Signal(int, object)        # token, ProbeResult
    failed = Signal(int, str, str)     # token, kind, message

    def __init__(self, host, port, token):
        super().__init__()
        self.host, self.port, self.token = host, port, token
        self.sock = None
        self.cancelled = False

    def start(self):
        _PROBES.add(self)                       # keeps this object alive until the thread is done with it
        threading.Thread(target=self._run, name='address-check', daemon=True).start()

    def _on_socket(self, sock):
        self.sock = sock
        if self.cancelled:
            close_socket(sock)

    def cancel(self):
        self.cancelled = True
        close_socket(self.sock)

    def _run(self):
        try:
            result = probe_server(self.host, self.port, on_socket=self._on_socket)
        except ProbeError as error:
            if not self.cancelled:
                self.failed.emit(self.token, error.kind, error.message)
        except Exception as error:
            if not self.cancelled:
                self.failed.emit(self.token, 'unreachable', str(error) or type(error).__name__)
        else:
            if not self.cancelled:
                self.found.emit(self.token, result)
        finally:
            _PROBES.discard(self)


def _status(label: QLabel, text: str = '', kind: str = '') -> None:
    """Text plus a glyph, so state never relies on colour alone."""
    mark = {'ok': '✓ ', 'error': '✕ '}.get(kind, '')
    label.setText(mark + text if text else '')
    label.setProperty('kind', kind)
    label.style().unpolish(label); label.style().polish(label)


class ConnectionWizard(QWidget):
    connected = Signal(object, object)    # SessionController, WizardResult
    closed = Signal(object)

    ADDRESS, CREDENTIALS, CONNECTING = 0, 1, 2

    def __init__(self, prefill: Optional[SavedSystem] = None, host: str = '', user: str = '',
                 password: str = '', auto_check: bool = False, credentials=None, remember: bool = False,
                 hp_available=None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle('iShareScreen')
        self.probe: Optional[ProbeWorker] = None
        self._probe_token = 0
        self.controller: Optional[SessionController] = None
        self.verified: Optional[ProbeResult] = None
        self.handed_off = False
        self._preset_password = password
        self._pending_secret = ''             # the password of the attempt in progress, for saving on success
        self.credentials = credentials if credentials is not None else UnavailableStore()
        self._remember_initial = remember
        self._hp_available = hp_available     # callable(ProbeResult) -> (bool, note), or None for "never"
        self._saved_mode = prefill.mode if prefill else 'standard'
        self._auto_check = auto_check or bool(prefill)
        if prefill:
            host, user = format_address(prefill.host, prefill.port), prefill.user
        self._build(prefill)
        self.address.setText(host)
        self.username.setText(user)
        self.go(self.ADDRESS)

    # ── construction ─────────────────────────────────────────────────
    # Space the window needs around the page: the macOS sheet's margins, or under Liquid Glass the
    # field margin plus the frosted card's padding.
    def _chrome(self) -> tuple[int, int]:
        return (2 * 24 + 2 * 26, 24 + 28 + 24 + 22) if is_glass() else (44, 38)

    def _build(self, prefill):
        root = QVBoxLayout(self)
        self.stack = QStackedWidget()
        if is_glass():
            root.setContentsMargins(24, 24, 24, 28)
            self.card = surface('strong', 'card', shadow=True)
            card = QVBoxLayout(self.card); card.setContentsMargins(26, 24, 26, 22)
            card.addWidget(self.stack)
            root.addWidget(self.card)
        else:
            root.setContentsMargins(22, 20, 22, 18)
            root.addWidget(self.stack)
        self.stack.addWidget(self._address_page())
        self.stack.addWidget(self._credentials_page(prefill))
        self.stack.addWidget(self._connecting_page())

    def paintEvent(self, event):
        paint_backdrop(self)                       # the field (under Liquid Glass) or the plain window colour

    def _address_page(self):
        glass = is_glass()
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        if glass:
            eyebrow = QLabel('NEW CONNECTION'); eyebrow.setObjectName('eyebrow')
            eyebrow_font = QFont(eyebrow.font()); eyebrow_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.1)
            eyebrow.setFont(eyebrow_font)
            heading = QLabel('Connect to a computer'); heading.setObjectName('pageTitle')
            sub = QLabel('Enter its hostname or IP address. We check that a VNC server answers before asking you to sign in.')
            sub.setObjectName('pageSub'); sub.setWordWrap(True)
            for widget in (eyebrow, heading, sub):
                layout.addWidget(widget)
            layout.addSpacing(6)
        row = QVBoxLayout() if glass else QHBoxLayout(); row.setSpacing(6 if glass else 10)
        caption = QLabel(label('Connect To:', 'Hostname or IP address')); caption.setObjectName('fieldLabel')
        row.addWidget(caption)
        self.address = QLineEdit(); self.address.setPlaceholderText('Hostname or IP address')
        self.address.setAccessibleName('Connect To: hostname or IP address')
        caption.setBuddy(self.address)
        self.address.textChanged.connect(self._address_edited)
        self.address.returnPressed.connect(self.check_address)
        row.addWidget(self.address, 1)
        layout.addLayout(row)
        self.address_status = QLabel(); self.address_status.setWordWrap(True)
        self.address_status.setAccessibleName('Connection check result')
        layout.addWidget(self.address_status)
        layout.addSpacing(6)
        buttons = QHBoxLayout(); buttons.setSpacing(8); buttons.addStretch(1)
        self.stop_check = push_button('Stop'); self.stop_check.clicked.connect(self.cancel_check); self.stop_check.hide()
        buttons.addWidget(self.stop_check)
        cancel = push_button('Cancel'); cancel.clicked.connect(self.close); buttons.addWidget(cancel)
        self.continue_button = push_button('Connect', 'primary')
        self.continue_button.setDefault(True); self.continue_button.clicked.connect(self.check_address)
        buttons.addWidget(self.continue_button)
        layout.addLayout(buttons)
        return page

    def _credentials_page(self, prefill):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        head = QHBoxLayout(); head.setSpacing(14)
        icon = QLabel(); icon.setPixmap(glyph('monitor').pixmap(48, 48)); icon.setFixedSize(52, 52)
        icon.setAlignment(Qt.AlignmentFlag.AlignTop)
        head.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)
        text = QVBoxLayout(); text.setSpacing(4)
        self.message = QLabel(); self.message.setObjectName('heading'); self.message.setWordWrap(True); text.addWidget(self.message)
        self.instruction = QLabel('Enter the user name and password to continue.'); self.instruction.setWordWrap(True)
        text.addWidget(self.instruction)
        self.found_label = QLabel(); self.found_label.setObjectName('hint'); self.found_label.setWordWrap(True)
        text.addWidget(self.found_label)
        head.addLayout(text, 1)
        layout.addLayout(head)

        # What is asked for follows the server: account (Mac), password only, or nothing.
        # macOS puts each label left of its control; Liquid Glass stacks label over control.
        glass = is_glass()
        grid = QGridLayout(); grid.setHorizontalSpacing(10); grid.setVerticalSpacing(6 if glass else 8)
        if not glass:
            grid.setColumnMinimumWidth(0, 120)
        cursor = [0]

        def labelled(text):
            widget = QLabel(text); widget.setObjectName('fieldLabel')
            widget.setAlignment((Qt.AlignmentFlag.AlignLeft if glass else Qt.AlignmentFlag.AlignRight) | Qt.AlignmentFlag.AlignVCenter)
            return widget

        def place(caption, control):
            row = cursor[0]
            if glass:
                grid.addWidget(caption, row, 0, 1, 2); grid.addWidget(control, row + 1, 0, 1, 2); cursor[0] += 2
            else:
                grid.addWidget(caption, row, 0); grid.addWidget(control, row, 1); cursor[0] += 1

        def place_wide(widget):
            grid.addWidget(widget, cursor[0], 0 if glass else 1, 1, 2 if glass else 1); cursor[0] += 1

        self.account_box = QWidget(); account = QHBoxLayout(self.account_box); account.setContentsMargins(0, 0, 0, 0)
        self.username_label = labelled(label('User Name:', 'User name'))
        self.username = QLineEdit(); self.username.setAccessibleName('User Name')
        self.username_label.setBuddy(self.username)
        account.addWidget(self.username)
        place(self.username_label, self.account_box)
        self.password_box = QWidget(); box = QHBoxLayout(self.password_box); box.setContentsMargins(0, 0, 0, 0); box.setSpacing(6)
        self.password_label = labelled(label('Password:', 'Password'))
        self.password = QLineEdit(); self.password.setEchoMode(QLineEdit.EchoMode.Password); self.password.setAccessibleName('Password')
        self.password_label.setBuddy(self.password)
        self.username.returnPressed.connect(self.password.setFocus)
        self.password.returnPressed.connect(self.connect_session)
        box.addWidget(self.password)
        self.show_password = tool_button(); self.show_password.setCheckable(True); self.show_password.setIcon(glyph('eye'))
        self.show_password.setToolTip('Show password'); self.show_password.setAccessibleName('Show password')
        self.show_password.toggled.connect(self._toggle_password)
        box.addWidget(self.show_password)
        place(self.password_label, self.password_box)
        self.remember = QCheckBox('Remember password')
        self.remember.setChecked(self._remember_initial)
        self.remember.setVisible(self.credentials.available)
        self.remember.setToolTip(f'Kept in {self.credentials.description}, never in the library file. '
                                 'Double-clicking this computer will then connect without asking.')
        place_wide(self.remember)
        self.auth_note = QLabel('This computer does not ask for a password.'); self.auth_note.setObjectName('hint')
        self.auth_note.setWordWrap(True); self.auth_note.hide(); place_wide(self.auth_note)
        type_label = labelled(label('Screen Sharing Type:', 'Screen sharing type'))
        self.mode = QComboBox(); self.mode.setAccessibleName('Screen Sharing Type')
        self.mode.addItem(MODE_LABELS['standard'], 'standard'); self.mode.addItem(MODE_LABELS['high_performance'], 'high_performance')
        type_label.setBuddy(self.mode)
        place(type_label, self.mode)
        self.resolution_label = labelled(label('Resolution:', 'Resolution'))
        self.resolution = QComboBox(); self.resolution.setAccessibleName('Resolution')
        for value, text in RESOLUTIONS:
            self.resolution.addItem(text, value)
        index = self.resolution.findData(prefill.resolution if prefill else 'auto')
        self.resolution.setCurrentIndex(max(0, index))
        self.resolution_label.setBuddy(self.resolution)
        self.hp_note = QLabel(); self.hp_note.setObjectName('hint'); self.hp_note.setWordWrap(True)
        for widget in (self.resolution_label, self.resolution, self.hp_note):
            widget.hide()
        place(self.resolution_label, self.resolution)
        place_wide(self.hp_note)
        self._set_hp(False, HP_NOTE)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        layout.addLayout(grid)

        self.options = push_button('More ▸', 'link'); self.options.setCheckable(True)
        self.options.setCursor(Qt.CursorShape.PointingHandCursor); self.options.toggled.connect(self._show_options)
        layout.addWidget(self.options, 0, Qt.AlignmentFlag.AlignLeft)
        self.more_box = QWidget(); more = QGridLayout(self.more_box); more.setContentsMargins(0, 0, 0, 0)
        more.setHorizontalSpacing(10); more.setVerticalSpacing(6 if glass else 8)
        if not glass:
            more.setColumnMinimumWidth(0, 120)
        name_label = labelled(label('Name:', 'Name'))
        self.name = QLineEdit(prefill.name if prefill else ''); self.name.setPlaceholderText('Optional, e.g. Studio Mac')
        self.name.setAccessibleName('Name (optional)'); name_label.setBuddy(self.name)
        view_label = labelled(label('Viewing:', 'Viewing'))
        self.view = QComboBox(); self.view.addItem('Fit to window', 'fit'); self.view.addItem('Actual size', 'actual')
        self.view.setAccessibleName('Viewing'); view_label.setBuddy(self.view)
        if prefill and prefill.view == 'actual':
            self.view.setCurrentIndex(1)
        if glass:
            more.addWidget(name_label, 0, 0); more.addWidget(self.name, 1, 0)
            more.addWidget(view_label, 2, 0); more.addWidget(self.view, 3, 0)
        else:
            more.addWidget(name_label, 0, 0); more.addWidget(self.name, 0, 1)
            more.addWidget(view_label, 1, 0); more.addWidget(self.view, 1, 1)
        self.more_box.hide(); layout.addWidget(self.more_box)

        self.cred_status = QLabel(); self.cred_status.setWordWrap(True); self.cred_status.setAccessibleName('Sign-in result')
        layout.addWidget(self.cred_status)
        buttons = QHBoxLayout(); buttons.setSpacing(8); buttons.addStretch(1)
        back = push_button('Cancel'); back.clicked.connect(self.close); buttons.addWidget(back)
        self.connect_button = push_button(label('Sign In', 'Sign in'), 'primary')
        self.connect_button.setDefault(True); self.connect_button.clicked.connect(self.connect_session)
        buttons.addWidget(self.connect_button)
        layout.addLayout(buttons)
        return page

    def _connecting_page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(10)
        self.connecting_title = QLabel('Connecting…'); self.connecting_title.setObjectName('heading'); layout.addWidget(self.connecting_title)
        self.connecting_target = QLabel(); self.connecting_target.setObjectName('hint'); layout.addWidget(self.connecting_target)
        bar = QProgressBar(); bar.setRange(0, 0); bar.setTextVisible(False); layout.addWidget(bar)  # busy: no invented percentage
        self.phase = QLabel('Starting…'); self.phase.setAccessibleName('Connection progress'); layout.addWidget(self.phase)
        layout.addSpacing(4)
        buttons = QHBoxLayout(); buttons.addStretch(1)
        self.cancel_connect = push_button('Cancel'); self.cancel_connect.clicked.connect(self.cancel_connection)
        buttons.addWidget(self.cancel_connect); layout.addLayout(buttons)
        return page

    # ── navigation ───────────────────────────────────────────────────
    def _fit(self):
        """Size the window to the current page, like a macOS sheet."""
        page = self.stack.currentWidget()
        layout = page.layout()
        layout.activate()
        # sizeHint() lays word-wrapped labels out at their narrowest, so ask for the real width.
        height = layout.totalHeightForWidth(CONTENT_WIDTH) if layout.hasHeightForWidth() else layout.sizeHint().height()
        extra_width, extra_height = self._chrome()
        self.setFixedSize(CONTENT_WIDTH + extra_width, height + extra_height)

    def go(self, stage):
        changed = self.stack.currentIndex() != stage
        self.stack.setCurrentIndex(stage)
        self._fit()
        if changed and is_glass() and self.isVisible():
            reveal([self.stack.currentWidget()], stagger=False)     # a short fade between steps
        if stage == self.ADDRESS:
            self.address.setFocus()
        elif stage == self.CREDENTIALS:
            auth = self.verified.auth if self.verified else 'account'
            if auth == 'none':
                self.connect_button.setFocus()
            else:
                (self.password if auth == 'password' or self.username.text() else self.username).setFocus()
        else:
            self.cancel_connect.setFocus()

    def _show_options(self, on):
        self.more_box.setVisible(on)
        self.options.setText('More ▾' if on else 'More ▸')
        self._fit()

    def _set_hp(self, available: bool, note: str = ''):
        """Enable the High Performance entry only when this computer can do it."""
        self._hp_text = note
        self.mode.model().item(1).setEnabled(available)
        self.mode.setItemData(1, note, Qt.ItemDataRole.ToolTipRole)
        if not available and self.mode.currentIndex() == 1:
            self.mode.setCurrentIndex(0)
        self.mode.setToolTip('' if available else note)
        self._mode_changed()

    def _mode_changed(self, *_):
        """Resolution and the experimental note only matter for High Performance."""
        hp = self.mode.currentData() == 'high_performance'
        for widget in (self.resolution_label, self.resolution, self.hp_note):
            widget.setVisible(hp)
        self.hp_note.setText(getattr(self, '_hp_text', '') if hp else '')
        if self.stack.currentIndex() == self.CREDENTIALS:
            self._fit()

    def _toggle_password(self, shown):
        self.password.setEchoMode(QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password)
        self.show_password.setIcon(glyph('eye-off' if shown else 'eye'))
        self.show_password.setToolTip('Hide password' if shown else 'Show password')
        self.show_password.setAccessibleName('Hide password' if shown else 'Show password')

    def _address_edited(self):
        self.verified = None
        if not self.is_checking:
            _status(self.address_status)
            self.continue_button.setText('Connect')
            if self.stack.currentIndex() == self.ADDRESS:
                self._fit()

    # ── stage 1: verify the address ──────────────────────────────────
    @property
    def is_checking(self):
        return self.probe is not None

    def check_address(self):
        if self.is_checking or self.stack.currentIndex() != self.ADDRESS:
            return
        try:
            host, port = parse_address(self.address.text())
        except ValueError as error:
            _status(self.address_status, str(error), 'error')
            self._fit()
            self.address.setFocus()
            return
        _status(self.address_status, f'Checking {format_address(host, port)}…')
        self.continue_button.setEnabled(False); self.continue_button.setText('Checking…')
        self.stop_check.show()
        self._fit()
        self._probe_token += 1
        probe = ProbeWorker(host, port, self._probe_token)
        self.probe = probe
        probe.found.connect(self._probe_found)       # bound methods: delivered on the GUI thread
        probe.failed.connect(self._probe_failed)
        probe.start()

    def _end_check(self):
        self.probe = None
        self.continue_button.setEnabled(True)
        self.stop_check.hide()

    def cancel_check(self):
        probe = self.probe
        if probe:
            probe.cancel()
            self._end_check()
            _status(self.address_status, 'Check stopped.')
            self.continue_button.setText('Connect')
            self._fit()
            self.address.setFocus()

    def _probe_found(self, token, result):
        if self.probe is None or token != self._probe_token:
            return
        self._end_check()
        self.verified = result
        self.continue_button.setText('Connect')
        _status(self.address_status)
        shown = format_address(result.host, result.port)
        self.message.setText(self._sign_in_message(result.auth, shown))
        self.instruction.setText({'account': 'Enter the user name and password to continue.',
                                  'password': 'Enter the password to continue.'}.get(result.auth, ''))
        self.instruction.setVisible(result.auth != 'none')
        self.found_label.setText(f'✓ VNC server found (RFB {result.version})')
        self.account_box.setVisible(result.auth == 'account'); self.username_label.setVisible(result.auth == 'account')
        self.password_box.setVisible(result.auth != 'none'); self.password_label.setVisible(result.auth != 'none')
        self.remember.setVisible(self.credentials.available and result.auth != 'none')
        self.auth_note.setVisible(result.auth == 'none')
        self.password_label.setText(label('Password:', 'Password') if result.auth == 'account'
                                    else label('VNC Password:', 'VNC password'))
        self.password.setAccessibleName('Password' if result.auth == 'account' else 'VNC Password')
        available, note = self._hp_available(result) if self._hp_available else (False, HP_NOTE)
        self._set_hp(bool(available), note)
        if self._saved_mode == 'high_performance' and available:
            self.mode.setCurrentIndex(1)
        _status(self.cred_status)
        self.go(self.CREDENTIALS)
        ready = result.auth == 'none' or (self.username.text().strip() or result.auth == 'password')
        if self._preset_password and ready:
            QTimer.singleShot(0, self._connect_preset)

    @staticmethod
    def _sign_in_message(auth: str, shown: str) -> str:
        if auth == 'none':
            return f'Connect to “{shown}”.'
        what = 'a user name and password' if auth == 'account' else 'a password'
        return f'Screen Sharing requires {what} to sign in to “{shown}”.'

    def _probe_failed(self, token, kind, message):
        if self.probe is None or token != self._probe_token:
            return
        self._end_check()
        self.verified = None
        _status(self.address_status, message, 'error')
        self.continue_button.setText(label('Try Again', 'Try again'))
        self._fit()
        self.address.setFocus()

    # ── stages 2 and 3: sign in and connect ──────────────────────────
    def _connect_preset(self):
        self.password.setText(self._preset_password)
        self._preset_password = ''
        self.connect_session()

    def connect_session(self):
        if self.controller is not None or self.verified is None or self.stack.currentIndex() != self.CREDENTIALS:
            return        # a connection attempt is already running, or the address is unverified
        verified = self.verified
        auth = verified.auth
        user = self.username.text().strip() if auth == 'account' else ''
        secret = self.password.text() if auth != 'none' else ''
        if (auth == 'account' and not user) or (auth != 'none' and not secret):
            what = {'account': 'the user name and password', 'password': 'the password'}[auth]
            _status(self.cred_status, f'Enter {what}.', 'error')
            self._fit()
            (self.username if auth == 'account' and not user else self.password).setFocus()
            return
        _status(self.cred_status)
        self.connect_button.setEnabled(False)
        self._pending_secret = secret
        who = f'{user} on ' if user else ''
        self.connecting_target.setText(f'{who}{format_address(verified.host, verified.port)}')
        self.phase.setText('Starting…')
        if self.mode.currentData() == 'high_performance' and auth == 'account':
            controller = HPController(verified.host, verified.port, user, secret, auth=auth,
                                      resolution=self.resolution.currentData())
        else:
            controller = SessionController(verified.host, verified.port, user, secret, auth=auth)
        self.controller = controller
        controller.phase.connect(self._on_phase)
        controller.established.connect(self._established)
        controller.failed.connect(self._connect_failed)
        self.go(self.CONNECTING)
        controller.start()

    def _on_phase(self, text):
        self.phase.setText(text)

    def _release_controller(self):
        controller, self.controller = self.controller, None
        if controller is not None:
            for signal, slot in ((controller.phase, self._on_phase), (controller.established, self._established),
                                 (controller.failed, self._connect_failed)):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        return controller

    def cancel_connection(self):
        controller = self._release_controller()
        self._pending_secret = ''
        if controller:
            controller.stop()
        self.connect_button.setEnabled(True)
        _status(self.cred_status, 'Connection cancelled.')
        self.go(self.CREDENTIALS)

    def _connect_failed(self, kind, message):
        controller = self._release_controller()
        self._pending_secret = ''
        if controller:
            controller.stop()
        self.connect_button.setEnabled(True)
        _status(self.cred_status, message, 'error')
        self.go(self.CREDENTIALS)
        if kind == 'auth':
            self.password.clear()
            self.password.setFocus()

    def _established(self, image):
        controller = self._release_controller()
        if controller is None:
            return
        self.password.clear()
        self.handed_off = True
        secret, self._pending_secret = self._pending_secret, ''
        remember = bool(self.credentials.available and self.verified.auth != 'none' and self.remember.isChecked())
        result = WizardResult(host=self.verified.host, port=self.verified.port, user=controller.user,
                              name=self.name.text().strip(), mode=self.mode.currentData(), view=self.view.currentData(),
                              auth=self.verified.auth, remember=remember, password=secret if remember else '',
                              resolution=self.resolution.currentData())
        self.connected.emit(controller, result)
        self.close()

    # ── window events ────────────────────────────────────────────────
    def showEvent(self, event):
        first = not getattr(self, '_shown', False)
        self._shown = True
        super().showEvent(event)
        if first:
            fade_in(self)
            if is_glass():
                reveal([self.card], stagger=False)
            if self._auto_check and self.address.text().strip():
                QTimer.singleShot(0, self.check_address)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.is_checking:
                self.cancel_check()
            elif self.controller is not None:
                self.cancel_connection()
            else:
                self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if self.probe:
            self.probe.cancel(); self.probe = None
        controller = self._release_controller()
        if controller:
            controller.stop()          # closed mid-connect: abandon the partial session
        self._pending_secret = self._preset_password = ''
        self.password.clear()
        self.closed.emit(self)
        event.accept()
