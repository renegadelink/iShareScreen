"""Helpers shared by the desktop GUI tests."""
import time

USER, PASSWORD = 'fixture-user', 'fixture-password'


def pump(qapp, condition, timeout=8.0):
    """Run the Qt event loop until `condition()` holds or the timeout passes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return bool(condition())


def verify(qapp, wizard, server):
    """Type the fake server's address into the wizard and wait for credentials."""
    wizard.address.setText(f'127.0.0.1:{server.port}')
    wizard.check_address()
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS)


def sign_in(wizard, user=USER, password=PASSWORD):
    wizard.username.setText(user)
    wizard.password.setText(password)
    wizard.connect_session()


def open_session(qapp, app, server, **fields):
    before = len(app.sessions)
    wizard = app.open_wizard()
    verify(qapp, wizard, server)
    sign_in(wizard, **fields)
    assert pump(qapp, lambda: len(app.sessions) == before + 1 and app.sessions[-1].is_live)
    return app.sessions[-1]
