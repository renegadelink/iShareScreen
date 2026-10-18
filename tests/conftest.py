"""Fixtures for the desktop GUI tests: an offscreen Qt app, an AppController with
a throw-away library, and deterministic local VNC servers."""
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['ISS_REDUCE_MOTION'] = '1'

from fake_vnc import FakeVNC  # noqa: E402


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    from isharescreen.frontend.desktop.theme import apply_theme
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    app.setQuitOnLastWindowClosed(False)
    return app


@pytest.fixture
def app(qapp, tmp_path):
    from isharescreen.frontend.desktop import session as S
    from isharescreen.frontend.desktop.library import LibraryStore
    from isharescreen.frontend.desktop.standard_app import AppController
    from isharescreen.frontend.desktop.credentials import MemoryCredentialStore
    from isharescreen.frontend.desktop.settings import Settings
    # An in-memory vault and a throw-away settings file, so no test touches the real
    # Windows Credential Manager or the user's settings.
    controller = AppController(LibraryStore(tmp_path / 'library.json'), MemoryCredentialStore(),
                               Settings(tmp_path / 'settings.json'))
    controller.notices = []
    controller.notify = controller.notices.append       # no modal boxes under test
    yield controller
    for wizard in list(controller.wizards):
        wizard.close()
    for window in list(controller.sessions):
        window.close()
    S.shutdown_all()
    controller.library.close = lambda: None
    qapp.processEvents()


@pytest.fixture
def server():
    made = []

    def make(**kwargs):
        made.append(FakeVNC(**kwargs))
        return made[-1]
    yield make
    for item in made:
        item.close()
        assert not item.errors, item.errors
