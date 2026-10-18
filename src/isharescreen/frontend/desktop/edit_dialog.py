"""The ⓘ sheet for a saved computer: name, server address, username, Screen Sharing
Type and port, laid out like macOS Screen Sharing's. The password is never shown;
it can only be forgotten here, and changing the address or account forgets it
automatically (see LibraryStore.update)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QFrame,
                               QMessageBox)

from ...proxy.address import parse_address, DEFAULT_PORT
from .hp import RESOLUTIONS
from .icons import glyph
from .library import SavedSystem, MODE_LABELS, describe_last_connected

HELP = ('Name: how this computer is listed.\n'
        'Server Address: its IP address or hostname.\n'
        'Username: the account to sign in with. Leave it empty for servers that need only a password.\n'
        'Screen Sharing Type: Standard works with any Mac. High Performance needs a Mac that supports it.\n'
        'Port: 5900 unless the computer uses another.\n\n'
        'Changing the address, port or username forgets a saved password.')


def resolve_address(address_text: str, port_text: str) -> tuple[str, int]:
    """Combine an address (which may carry its own `:port`) with a separate Port field."""
    host, in_address = parse_address(address_text, default_port=0)
    _, in_field = parse_address(f'x:{port_text.strip() or DEFAULT_PORT}')      # validates the Port field
    if in_address and in_field != DEFAULT_PORT and in_address != in_field:
        raise ValueError('The address and the Port field name different ports. Use one of them.')
    return host, in_address or in_field


def _row(label: str, widget, section_layout, first: bool):
    if not first:
        divider = QFrame(); divider.setObjectName('sectionDivider'); section_layout.addWidget(divider)
    row = QHBoxLayout(); row.setContentsMargins(14, 7, 14, 7)
    text = QLabel(label); text.setBuddy(widget); row.addWidget(text); row.addStretch(1); row.addWidget(widget)
    section_layout.addLayout(row)


def _section():
    frame = QFrame(); frame.setObjectName('section')
    layout = QVBoxLayout(frame); layout.setContentsMargins(0, 2, 0, 2); layout.setSpacing(0)
    return frame, layout


class EditSystemDialog(QDialog):
    def __init__(self, parent, system: SavedSystem, hp_enabled: bool = False, hp_note: str = ''):
        super().__init__(parent)
        self.system = system
        self.forget_password = False
        self.connect_after = False
        self.setWindowTitle(system.title)
        self.setModal(True)
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 18, 18, 14); layout.setSpacing(10)

        def field(text, placeholder=''):
            edit = QLineEdit(text); edit.setObjectName('bare'); edit.setAlignment(Qt.AlignmentFlag.AlignRight)
            edit.setPlaceholderText(placeholder); edit.setMinimumWidth(230)
            return edit
        self.name = field(system.name, system.host); self.name.setAccessibleName('Name')
        self.address = field(system.host); self.address.setAccessibleName('Server Address')
        self.user = field(system.user, 'Optional'); self.user.setAccessibleName('Username')
        self.mode = QComboBox(); self.mode.setObjectName('bare'); self.mode.setAccessibleName('Screen Sharing Type')
        self.mode.addItem(MODE_LABELS['standard'], 'standard')
        self.mode.addItem(MODE_LABELS['high_performance'], 'high_performance')
        if not hp_enabled:
            self.mode.model().item(1).setEnabled(False)
            self.mode.setItemData(1, hp_note or 'High Performance is not available.', Qt.ItemDataRole.ToolTipRole)
        self.mode.setCurrentIndex(1 if system.mode == 'high_performance' else 0)   # a saved HP choice stays visible
        self.resolution = QComboBox(); self.resolution.setObjectName('bare'); self.resolution.setAccessibleName('Resolution')
        for value, text in RESOLUTIONS:
            self.resolution.addItem(text, value)
        self.resolution.setCurrentIndex(max(0, self.resolution.findData(system.resolution)))
        self.resolution.setEnabled(self.mode.currentData() == 'high_performance')      # only High Performance has one
        self.mode.currentIndexChanged.connect(lambda _: self.resolution.setEnabled(self.mode.currentData() == 'high_performance'))
        self.port = field('' if system.port == DEFAULT_PORT else str(system.port), str(DEFAULT_PORT))
        self.port.setAccessibleName('Port'); self.port.setMaximumWidth(120)
        self.view = QComboBox(); self.view.setObjectName('bare'); self.view.setAccessibleName('Viewing')
        self.view.addItem('Fit to window', 'fit'); self.view.addItem('Actual size', 'actual')
        self.view.setCurrentIndex(1 if system.view == 'actual' else 0)

        box, inner = _section(); _row('Name', self.name, inner, True); layout.addWidget(box)
        box, inner = _section(); _row('Server Address', self.address, inner, True)
        _row('Username', self.user, inner, False); layout.addWidget(box)
        box, inner = _section(); _row('Screen Sharing Type', self.mode, inner, True)
        _row('Resolution', self.resolution, inner, False)
        _row('Port', self.port, inner, False); _row('Viewing', self.view, inner, False); layout.addWidget(box)

        self.last = QLabel(describe_last_connected(system.last_connected)); self.last.setObjectName('hint')
        layout.addWidget(self.last)
        row = QHBoxLayout()
        self.saved_note = QLabel(); self.saved_note.setWordWrap(True); self.saved_note.setObjectName('hint')
        row.addWidget(self.saved_note, 1)
        self.forget = QPushButton('Forget Password'); self.forget.clicked.connect(self._forget)
        row.addWidget(self.forget)
        layout.addLayout(row)
        self.error = QLabel(); self.error.setWordWrap(True); self.error.setProperty('kind', 'error')
        layout.addWidget(self.error)

        buttons = QHBoxLayout(); buttons.setSpacing(8)
        help_button = QPushButton(); help_button.setObjectName('round'); help_button.setIcon(glyph('help'))
        help_button.setToolTip('Help'); help_button.setAccessibleName('Help')
        help_button.clicked.connect(lambda: QMessageBox.information(self, 'Edit computer', HELP))
        buttons.addWidget(help_button); buttons.addStretch(1)
        cancel = QPushButton('Cancel'); cancel.clicked.connect(self.reject); buttons.addWidget(cancel)
        self.connect_button = QPushButton('Connect'); self.connect_button.clicked.connect(self._connect)
        buttons.addWidget(self.connect_button)
        self.save = QPushButton('Done'); self.save.setObjectName('primary'); self.save.setDefault(True)
        self.save.clicked.connect(self.accept); buttons.addWidget(self.save)
        layout.addLayout(buttons)
        for edit in (self.address, self.user, self.port):
            edit.textChanged.connect(self._refresh_note)
        self._refresh_note()

    def _identity_changed(self) -> bool:
        try:
            host, port = resolve_address(self.address.text(), self.port.text())
        except ValueError:
            return False
        return (host.lower(), port, self.user.text().strip()) != (self.system.host.lower(), self.system.port, self.system.user)

    def _refresh_note(self):
        saved = self.system.password_saved and not self.forget_password
        self.forget.setVisible(saved)
        if not self.system.password_saved or self.forget_password:
            self.saved_note.setText('No password is saved for this computer.')
        elif self._identity_changed():
            self.saved_note.setText('The saved password will be forgotten, because the address or account changed.')
        else:
            self.saved_note.setText('A password is saved. Double-click connects without asking.')

    def _forget(self):
        self.forget_password = True
        self._refresh_note()

    def _connect(self):
        self.connect_after = True
        self.accept()
        if self.result() != QDialog.DialogCode.Accepted:      # validation failed: stay open
            self.connect_after = False

    def accept(self):
        try:
            host, port = resolve_address(self.address.text(), self.port.text())
        except ValueError as error:
            self.error.setText('✕ ' + str(error))
            self.address.setFocus()
            self.style().unpolish(self.error); self.style().polish(self.error)
            return
        self.error.clear()
        self.result_values = dict(host=host, port=port, user=self.user.text().strip(), name=self.name.text().strip(),
                                  view=self.view.currentData(), mode=self.mode.currentData(),
                                  resolution=self.resolution.currentData(),
                                  forget_password=self.forget_password, connect=self.connect_after)
        super().accept()

    def values(self) -> dict:
        return self.result_values
