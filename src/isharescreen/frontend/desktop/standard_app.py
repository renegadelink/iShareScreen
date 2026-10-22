"""Native Qt desktop application for standard macOS Screen Sharing (TCP RFB).

A library window lists saved computers. Each new session runs in its own
connection wizard and, once the remote screen is up, in its own session window.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

from PySide6.QtCore import QObject, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox, QLineEdit

from ...proxy.address import format_address
from .credentials import CredentialError, CredentialStore, default_store
from .edit_dialog import EditSystemDialog
from .hp import hp_capability
from .library import LibraryStore, LibraryError, SavedSystem
from .settings import Settings
from .library_window import LibraryWindow
from .session import SessionWindow, Screen, SessionController, shutdown_all   # Screen re-exported for compatibility
from .theme import apply_theme
from .wizard import ConnectionWizard, WizardResult
from .rvdp import RVDPSessionWindow

__all__ = ['AppController', 'Screen', 'main']


class _NoResult:
    """Stands in for a probe result when only the capability *reason* is wanted."""
    auth, security = 'account', (33,)


_NO_RESULT = _NoResult()


class AppController(QObject):
    """Owns the library and every wizard/session window. Each session keeps its
    own controller, so opening, closing or failing one never touches another."""

    def __init__(self, store: LibraryStore | None = None, credentials: CredentialStore | None = None,
                 settings: Settings | None = None):
        super().__init__()
        self.store = store or LibraryStore()
        self.credentials = credentials if credentials is not None else default_store()
        self.settings = settings if settings is not None else Settings()
        self.wizards: list[ConnectionWizard] = []
        self.sessions: list[SessionWindow] = []
        self.library = LibraryWindow()
        self.library.new_session.connect(self.open_wizard)
        self.library.open_system.connect(self.open_system)
        self.library.edit_system.connect(self.edit_system)
        self.library.rename_system.connect(self.rename_system)
        self.library.forget_password.connect(self.forget_password)
        self.library.remove_system.connect(self.remove_system)
        self.library.hp_toggled.connect(self.set_experimental_hp)
        self.library.set_hp_checked(self.settings.experimental_hp)
        # High Performance is not released: its switch stays out of the menu unless a developer
        # opts in (ISS_EXPERIMENTAL_HP=1) or had already turned it on.
        self.library.hp_action.setVisible(self.settings.experimental_hp
                                          or os.environ.get('ISS_EXPERIMENTAL_HP') == '1')
        self.library.quit_requested.connect(self.request_quit)
        self.refresh_library()
        if self.store.load_notice:
            self.notify(self.store.load_notice)

    # ── library ──────────────────────────────────────────────────────
    def refresh_library(self):
        self.library.set_systems(self.store.sorted())
        self.update_active()

    def update_active(self):
        active: dict[str, int] = {}
        for window in self.sessions:
            if window.system_id and window.is_live:
                active[window.system_id] = active.get(window.system_id, 0) + 1
        self.library.set_active(active)

    def notify(self, text: str):
        """A recoverable notice: shown in the library status bar and a non-modal box."""
        self.library.statusBar().showMessage(text)
        box = QMessageBox(QMessageBox.Icon.Warning, 'iShareScreen', text, QMessageBox.StandardButton.Ok, self.library)
        box.setAttribute(box.WidgetAttribute.WA_DeleteOnClose)
        box.open()

    # ── wizard ───────────────────────────────────────────────────────
    # ── High Performance (experimental) ──────────────────────────────
    def hp_available(self, result):
        return hp_capability(result, self.settings.experimental_hp)

    def _confirm_experimental(self) -> bool:
        answer = QMessageBox.warning(
            self.library, 'High Performance (Experimental)',
            'High Performance is experimental. In testing it did not deliver video from every Mac, and a failed '
            'attempt can leave that Mac’s Screen Sharing unresponsive for a couple of minutes.\n\n'
            'Standard mode is not affected. Turn High Performance on anyway?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def set_experimental_hp(self, on: bool):
        if on and not self._confirm_experimental():
            self.library.set_hp_checked(False)
            return
        if not self.settings.set('experimental_hp', on):
            self.notify('The setting could not be saved, so it will not be remembered.')
        self.library.set_hp_checked(self.settings.experimental_hp)

    def open_wizard(self, prefill: SavedSystem | None = None, **fields):
        fields.setdefault('credentials', self.credentials)
        fields.setdefault('hp_available', self.hp_available)
        wizard = ConnectionWizard(prefill=prefill if isinstance(prefill, SavedSystem) else None, **fields)
        wizard.connected.connect(self.on_connected)
        wizard.closed.connect(lambda w: self.wizards.remove(w) if w in self.wizards else None)
        self.wizards.append(wizard)
        wizard.show()
        return wizard

    def saved_secret(self, system_id: str | None) -> str | None:
        """The saved password for a library entry, or None when there is none or
        the vault cannot be read (a missing secret clears the stale flag)."""
        system = self.store.get(system_id) if system_id else None
        if system is None or not system.password_saved:
            return None
        try:
            secret = self.credentials.get(system.id)
        except CredentialError as error:
            self.notify(f'The saved password for “{system.title}” could not be read. {error}')
            return None
        if secret is None:
            try:
                self.store.set_password_saved(system.id, False)
            except LibraryError:
                pass
            self.refresh_library()
        return secret

    def open_system(self, system_id: str):
        """Double-click / Enter: with a saved password the wizard verifies the
        address and signs in by itself, still showing progress and any error."""
        system = self.store.get(system_id)
        if not system:
            return
        secret = self.saved_secret(system_id)
        if secret:
            self.open_wizard(prefill=system, password=secret, remember=True, auto_check=True)
        else:
            self.open_wizard(prefill=system)

    def _apply_password_choice(self, saved: SavedSystem, result: WizardResult) -> str:
        """Save or forget the password as the user chose. Returns a problem
        description, or ''. Never raises: the live session must not depend on it."""
        try:
            if result.remember and result.password:
                self.credentials.set(saved.id, result.password, result.user)
                try:
                    self.store.set_password_saved(saved.id, True)
                except LibraryError:
                    try:
                        self.credentials.delete(saved.id)      # keep vault and library consistent
                    except CredentialError:
                        pass
                    raise
            elif not result.remember and saved.password_saved:
                self.credentials.delete(saved.id)
                self.store.set_password_saved(saved.id, False)
        except (CredentialError, LibraryError) as error:
            return f'The password was not saved. {error}'
        return ''

    def on_connected(self, controller: SessionController, result: WizardResult):
        """The wizard verified a usable first screen update: open the session
        window, then record the library entry (and the password, if the user
        asked). Saving can fail without affecting the live session."""
        saved = None
        problems = []
        try:
            saved = self.store.record_success(result.host, result.port, result.user, result.mode,
                                              name=result.name, view=result.view, resolution=result.resolution)
        except LibraryError as error:
            problems.append(f'Connected, but this computer was not saved to the library. {error}')
        if saved is not None:
            problem = self._apply_password_choice(saved, result)
            if problem:
                problems.append(problem)
        elif result.remember:
            problems.append('The password was not saved either.')
        title = saved.title if saved else (result.name or format_address(result.host, result.port))
        window_class = RVDPSessionWindow if result.mode == 'rvdp' else SessionWindow
        window = window_class(controller, title, view=result.view)
        window.system_id = saved.id if saved else None
        window.saved_secret = lambda w=window: self.saved_secret(w.system_id)   # lets Reconnect skip the prompt
        window.closed.connect(self.on_session_closed)
        window.state_changed.connect(lambda _w: self.update_active())
        self.sessions.append(window)
        window.show()
        self.refresh_library()
        if problems:
            self.notify(' '.join(problems))

    def on_session_closed(self, window: SessionWindow):
        if window in self.sessions:
            self.sessions.remove(window)
        self.update_active()

    # ── library actions ──────────────────────────────────────────────
    def _ask_name(self, system: SavedSystem) -> str | None:
        text, ok = QInputDialog.getText(self.library, 'Rename', 'Name:', QLineEdit.EchoMode.Normal,
                                        system.name or system.title)
        return text if ok else None

    def _confirm_remove(self, system: SavedSystem) -> bool:
        answer = QMessageBox.question(
            self.library, 'Remove from library',
            f'Remove “{system.title}” from the library?\n\nOpen sessions to it stay connected. '
            'Only the saved address and account name are deleted.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_quit(self, count: int) -> bool:
        answer = QMessageBox.question(
            self.library, 'Quit iShareScreen',
            f'Quit and disconnect {count} open session{"s" if count != 1 else ""}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Yes

    def rename_system(self, system_id: str):
        system = self.store.get(system_id)
        name = self._ask_name(system) if system else None
        if name is None:
            return
        try:
            self.store.rename(system_id, name)
        except LibraryError as error:
            self.notify(str(error))
        self.refresh_library()

    def remove_system(self, system_id: str):
        system = self.store.get(system_id)
        if not system or not self._confirm_remove(system):
            return
        try:
            self.store.remove(system_id)
        except LibraryError as error:
            self.notify(str(error))
            self.refresh_library()
            return
        if system.password_saved:
            self._delete_secret(system)
        self.refresh_library()

    def _delete_secret(self, system: SavedSystem):
        try:
            self.credentials.delete(system.id)
        except CredentialError as error:
            self.notify(f'The saved password for “{system.title}” could not be removed. {error}')

    def forget_password(self, system_id: str):
        system = self.store.get(system_id)
        if not system or not system.password_saved:
            return
        self._delete_secret(system)
        try:
            self.store.set_password_saved(system_id, False)
        except LibraryError as error:
            self.notify(str(error))
        self.library.statusBar().showMessage(f'Forgot the saved password for “{system.title}”.')
        self.refresh_library()

    def _ask_edit(self, system: SavedSystem) -> dict | None:
        dialog = EditSystemDialog(self.library, system, hp_enabled=self.settings.experimental_hp,
                                  hp_note=hp_capability(_NO_RESULT, self.settings.experimental_hp)[1])
        return dialog.values() if dialog.exec() == QDialog.DialogCode.Accepted else None

    def edit_system(self, system_id: str):
        """Change name, address, account or view. A changed address or account
        forgets the saved password, so it can never reach a different destination."""
        system = self.store.get(system_id)
        values = self._ask_edit(system) if system else None
        if values is None:
            return
        had_password = system.password_saved
        try:
            changed = self.store.update(system_id, host=values['host'], port=values['port'], user=values['user'],
                                        name=values['name'], view=values['view'],
                                        mode=values.get('mode', system.mode), resolution=values.get('resolution'))
        except LibraryError as error:
            self.notify(str(error))
            return
        if had_password and (changed or values['forget_password']):
            self._delete_secret(system)
            try:
                self.store.set_password_saved(system_id, False)
            except LibraryError as error:
                self.notify(str(error))
        self.refresh_library()
        if values.get('connect'):
            self.open_system(system_id)

    # ── quitting ─────────────────────────────────────────────────────
    def request_quit(self):
        if self.sessions and not self._confirm_quit(len(self.sessions)):
            return
        self.quit_now()

    def quit_now(self):
        """Close every wizard and session, wait for their network threads, quit."""
        for wizard in list(self.wizards):
            wizard.close()
        for window in list(self.sessions):
            window.close()
        shutdown_all()
        QApplication.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description='iShareScreen native standard-VNC desktop')
    parser.add_argument('--version',action='version',version='iShareScreen Desktop 0.2.0')
    parser.add_argument('--host',default='',help='open a new-session window for this address')
    parser.add_argument('-u','--user',default='',help='prefill the username for --host')
    parser.add_argument('--password-stdin',action='store_true',help='read the password for --host from stdin')
    parser.add_argument('--smoke-test',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--style',choices=('macos','glass'),
                        help='look of the app: macos (the default) or glass (Liquid Glass); ISS_STYLE does the same')
    args = parser.parse_args(argv)
    if args.style:
        os.environ['ISS_STYLE'] = args.style
    secret = sys.stdin.readline().rstrip('\r\n') if args.password_stdin else ''
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName('iShareScreen')
    app.setQuitOnLastWindowClosed(False)   # quitting is explicit, via AppController
    apply_theme(app)                       # macOS-style light/dark, following the system
    icon = Path(__file__).parents[2]/'gui'/'static'/'icon.svg'
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    if args.smoke_test:
        # A packaged build must contain everything the optional High Performance path imports.
        try:
            import importlib
            importlib.import_module('isharescreen.proxy.session')
            importlib.import_module('av')
            importlib.import_module('isharescreen.frontend.desktop.rvdp')
            importlib.import_module('aiohttp')
        except Exception as error:
            print(f'smoke test: High Performance dependencies are missing: {type(error).__name__}: {error}', file=sys.stderr)
            return 3
    controller = AppController()
    controller.library.show()
    if args.host:
        controller.open_wizard(host=args.host, user=args.user, password=secret, auto_check=True)
    if args.smoke_test:
        QTimer.singleShot(250, controller.quit_now)
    result = app.exec()
    shutdown_all()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
