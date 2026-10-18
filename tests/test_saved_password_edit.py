"""Saved passwords (in a credential vault, never the library file) and editing entries."""
import json
import sys
import uuid

import pytest

from desktop_helpers import USER, PASSWORD, pump, verify, sign_in
from isharescreen.frontend.desktop import session as S
from isharescreen.frontend.desktop.credentials import (CredentialError, MemoryCredentialStore, UnavailableStore,
                                                       WindowsCredentialStore, default_store)
from isharescreen.frontend.desktop.edit_dialog import EditSystemDialog
from isharescreen.frontend.desktop.library import LibraryError, LibraryStore
from isharescreen.frontend.desktop.wizard import WizardResult


def connect(qapp, app, server, remember=False, user=USER, password=PASSWORD):
    """Run the wizard against `server`, optionally ticking 'save password'; returns the session window."""
    before = len(app.sessions)
    wizard = app.open_wizard()
    verify(qapp, wizard, server)
    wizard.remember.setChecked(remember)
    sign_in(wizard, user, password)
    assert pump(qapp, lambda: len(app.sessions) == before + 1 and app.sessions[-1].is_live)
    return app.sessions[-1]


# ── the vault ────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform != 'win32', reason='Windows Credential Manager')
def test_windows_credential_manager_round_trip():
    store = default_store()
    assert isinstance(store, WindowsCredentialStore) and store.available
    key = 'pytest-' + uuid.uuid4().hex                      # a throwaway entry, always removed
    try:
        assert store.get(key) is None
        store.set(key, 'pässwörd 日本 p@ss w0rd!', 'someone')
        assert store.get(key) == 'pässwörd 日本 p@ss w0rd!'
        store.set(key, 'second')
        assert store.get(key) == 'second'
        store.delete(key)
        assert store.get(key) is None
        store.delete(key)                                    # deleting again is quiet
        with pytest.raises(CredentialError):
            store.set(key, 'x' * 2000)
    finally:
        store.delete(key)


def test_unavailable_store_refuses_to_save():
    store = UnavailableStore('no vault here')
    assert not store.available and store.get('k') is None
    store.delete('k')
    with pytest.raises(CredentialError):
        store.set('k', 'secret')


# ── library flag ─────────────────────────────────────────────────────

def test_library_stores_only_the_flag_and_ignores_a_forged_one(tmp_path):
    path = tmp_path / 'library.json'
    store = LibraryStore(path)
    item = store.record_success('a.local', 5900, 'ana', password_saved=True)
    assert item.password_saved and LibraryStore(path).items[0].password_saved
    assert 'secret' not in path.read_text()
    raw = json.loads(path.read_text())
    raw['systems'][0]['password_saved'] = 'yes'              # not the boolean True
    path.write_text(json.dumps(raw))
    assert LibraryStore(path).items[0].password_saved is False
    assert store.record_success('a.local', 5900, 'ana').password_saved                 # None keeps it
    store.set_password_saved(item.id, False)
    assert not LibraryStore(path).items[0].password_saved


def test_update_forgets_the_password_flag_when_the_destination_or_account_changes(tmp_path):
    store = LibraryStore(tmp_path / 'library.json')
    item = store.record_success('a.local', 5900, 'ana', name='A', password_saved=True)
    assert store.update(item.id, host='a.local', port=5900, user='ana', name='Renamed', view='actual') is False
    assert item.password_saved and item.name == 'Renamed' and item.view == 'actual'    # cosmetic edits keep it
    assert store.update(item.id, host='A.LOCAL', port=5900, user='ana', name='', view='fit') is False   # case only
    for change in (dict(host='b.local', port=5900, user='ana'), dict(host='b.local', port=5901, user='ana'),
                   dict(host='b.local', port=5901, user='bo')):
        item.password_saved = True
        assert store.update(item.id, name='', view='fit', **change) is True
        assert item.password_saved is False


def test_update_rejects_a_clash_and_rolls_back_a_failed_write(tmp_path, monkeypatch):
    store = LibraryStore(tmp_path / 'library.json')
    first = store.record_success('a.local', 5900, 'ana')
    second = store.record_success('b.local', 5900, 'ana', name='Second')
    with pytest.raises(LibraryError, match='Second'):
        store.update(first.id, host='b.local', port=5900, user='ana', name='', view='fit')
    assert first.host == 'a.local'                                                    # untouched
    monkeypatch.setattr(store, 'save', lambda: (_ for _ in ()).throw(LibraryError('disk full')))
    with pytest.raises(LibraryError):
        store.update(first.id, host='c.local', port=5900, user='ana', name='X', view='fit')
    assert store.get(first.id).host == 'a.local' and store.get(first.id).name == ''   # rolled back
    with pytest.raises(LibraryError):
        store.update('missing', host='c.local', port=5900, user='ana', name='', view='fit')


# ── wizard and saving ────────────────────────────────────────────────

def test_save_password_option_follows_the_vault_and_the_server(qapp, app, server):
    wizard = app.open_wizard()
    verify(qapp, wizard, server())
    assert wizard.remember.isVisibleTo(wizard) and not wizard.remember.isChecked()    # explicit opt-in
    assert wizard.remember.text() == 'Remember password' and 'never in the library file' in wizard.remember.toolTip()
    none = app.open_wizard()
    verify(qapp, none, server(security='none'))
    assert not none.password_box.isVisibleTo(none)                                    # nothing to save
    app.credentials = UnavailableStore()
    bare = app.open_wizard()
    verify(qapp, bare, server())
    assert not bare.remember.isVisibleTo(bare)                                        # no vault, no option


def test_remembered_password_goes_to_the_vault_not_the_library(qapp, app, server):
    fake = server()
    window = connect(qapp, app, fake, remember=True)
    saved = app.store.items[0]
    assert app.credentials.get(saved.id) == PASSWORD and saved.password_saved
    assert app.credentials.items[saved.id][1] == USER
    assert PASSWORD not in app.store.path.read_text()
    assert window.saved_secret() == PASSWORD
    assert 'password saved' in app.library.list.item(0).data(app.library.list.item(0).data.__self__ and 0x0100 + 2)


def test_without_the_option_nothing_is_stored(qapp, app, server):
    connect(qapp, app, server(), remember=False)
    assert app.credentials.items == {} and not app.store.items[0].password_saved


def test_unticking_the_option_on_a_later_connection_deletes_the_saved_password(qapp, app, server):
    fake = server()
    window = connect(qapp, app, fake, remember=True)
    saved_id = app.store.items[0].id
    window.close()
    wizard = app.open_wizard(prefill=app.store.get(saved_id), password='', remember=True)
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS)
    assert wizard.remember.isChecked()                                               # remembered choice is shown
    wizard.remember.setChecked(False)
    wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: len(app.sessions) == 1 and app.sessions[0].is_live)
    assert app.credentials.items == {} and not app.store.items[0].password_saved


def test_double_click_connects_by_itself_with_the_saved_password(qapp, app, server):
    fake = server()
    connect(qapp, app, fake, remember=True).close()
    saved_id = app.store.items[0].id
    assert pump(qapp, lambda: not app.sessions)
    app.open_system(saved_id)                                                         # what double-click does
    assert pump(qapp, lambda: len(app.sessions) == 1 and app.sessions[0].is_live)
    assert pump(qapp, lambda: not app.wizards)                                        # no wizard left behind
    assert fake.authenticated == 2 and len(app.store.items) == 1
    assert app.sessions[0].system_id == saved_id and app.credentials.get(saved_id) == PASSWORD


def test_a_stale_saved_password_fails_visibly_and_is_replaceable(qapp, app, server):
    fake = server()
    connect(qapp, app, fake, remember=True).close()
    saved_id = app.store.items[0].id
    app.credentials.set(saved_id, 'out-of-date', USER)                               # the Mac's password changed
    app.open_system(saved_id)
    wizard = app.wizards[-1]
    assert pump(qapp, lambda: wizard.stack.currentIndex() == wizard.CREDENTIALS and wizard.cred_status.text())
    assert 'rejected' in wizard.cred_status.text() and wizard.password.text() == ''
    assert wizard.remember.isChecked() and not app.sessions
    assert app.credentials.get(saved_id) == 'out-of-date'                            # kept until replaced
    wizard.password.setText(PASSWORD); wizard.connect_session()
    assert pump(qapp, lambda: len(app.sessions) == 1 and app.sessions[0].is_live)
    assert app.credentials.get(saved_id) == PASSWORD                                 # replaced with the good one


def test_a_vault_that_cannot_save_does_not_disturb_the_session(qapp, app, server):
    app.credentials.fail_with = 'The vault is locked.'
    window = connect(qapp, app, server(), remember=True)
    assert window.is_live and app.credentials.items == {} and not app.store.items[0].password_saved
    assert any('not saved' in note and 'locked' in note for note in app.notices)


def test_an_unreadable_vault_falls_back_to_the_normal_wizard(qapp, app, server):
    connect(qapp, app, server(), remember=True).close()
    saved = app.store.items[0]

    class Broken(MemoryCredentialStore):
        def get(self, key): raise CredentialError('Credential Manager is unavailable.')
    app.credentials = Broken()
    app.open_system(saved.id)
    wizard = app.wizards[-1]
    assert any('could not be read' in note for note in app.notices)
    assert wizard.address.text().startswith('127.0.0.1:') and wizard._preset_password == ''     # asks as usual


def test_a_missing_vault_entry_clears_the_stale_flag(qapp, app, server):
    connect(qapp, app, server(), remember=True).close()
    saved = app.store.items[0]
    app.credentials.items.clear()                                                    # deleted in Credential Manager itself
    app.open_system(saved.id)
    assert app.store.items[0].password_saved is False and app.wizards[-1]._preset_password == ''


def test_removing_an_entry_and_forgetting_delete_the_secret(qapp, app, server):
    one, two = server(), server()
    connect(qapp, app, one, remember=True).close(); connect(qapp, app, two, remember=True).close()
    first, second = app.store.items
    assert set(app.credentials.items) == {first.id, second.id}
    app.forget_password(first.id)
    assert set(app.credentials.items) == {second.id} and not app.store.get(first.id).password_saved
    assert app.store.get(first.id) is not None                                       # the entry itself stays
    app._confirm_remove = lambda system: True
    app.remove_system(second.id)
    assert app.credentials.items == {} and app.store.get(second.id) is None


def test_wizard_result_never_prints_the_password():
    result = WizardResult('h', 5900, 'u', '', 'standard', 'fit', remember=True, password='top-secret')
    assert 'top-secret' not in repr(result) and result.password == 'top-secret'


def test_password_only_servers_can_be_remembered_too(qapp, app, server):
    fake = server(security='vnc', users={'': 'sekret'})
    wizard = app.open_wizard()
    verify(qapp, wizard, fake)
    wizard.remember.setChecked(True); wizard.password.setText('sekret'); wizard.connect_session()
    assert pump(qapp, lambda: app.sessions and app.sessions[0].is_live)
    saved = app.store.items[0]
    assert saved.user == '' and app.credentials.get(saved.id) == 'sekret'
    app.sessions[0].close()
    app.open_system(saved.id)
    assert pump(qapp, lambda: len(app.sessions) == 1 and app.sessions[0].is_live)


# ── reconnect ────────────────────────────────────────────────────────

class NoPrompt:
    def __init__(self, *args): raise AssertionError('no password prompt expected')


def test_reconnect_uses_the_saved_password_without_asking(qapp, app, server, monkeypatch):
    fake = server()
    window = connect(qapp, app, fake, remember=True)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    monkeypatch.setattr(S, 'PasswordDialog', NoPrompt)
    window.reconnect()
    assert pump(qapp, lambda: window.state == 'Connected') and fake.authenticated == 2


def test_a_rejected_saved_password_means_the_next_reconnect_asks(qapp, app, server, monkeypatch):
    fake = server()
    window = connect(qapp, app, fake, remember=True)
    app.credentials.set(app.store.items[0].id, 'out-of-date', USER)
    fake.drop_all()
    assert pump(qapp, lambda: window.state == 'Disconnected')
    window.reconnect()
    assert pump(qapp, lambda: 'saved password was rejected' in window.banner_text.text())

    asked = []

    class Dialog:
        def __init__(self, parent, user, address): asked.append(user)
        def exec(self): return S.QDialog.DialogCode.Accepted
        def password(self): return PASSWORD
    monkeypatch.setattr(S, 'PasswordDialog', Dialog)
    window.reconnect()
    assert pump(qapp, lambda: window.state == 'Connected') and asked == [USER]       # prompted this time
    assert window._saved_rejected is False


# ── edit ─────────────────────────────────────────────────────────────

def test_context_menu_offers_edit_and_forget_only_when_relevant(qapp, app):
    plain = app.store.record_success('a.local', 5900, 'ana', name='A')
    keeper = app.store.record_success('b.local', 5900, 'ana', name='B', password_saved=True)
    app.refresh_library()
    emitted = []
    for signal in ('edit_system', 'forget_password'):
        getattr(app.library, signal).connect(lambda sid, name=signal: emitted.append((name, sid)))
    with_pw = app.library.build_menu(keeper.id)
    without = app.library.build_menu(plain.id)
    labels = lambda menu: [a.text() for a in menu.actions() if a.text()]
    assert labels(without) == ['Connect', 'Edit…', 'Rename…', 'Remove from library…']
    assert labels(with_pw) == ['Connect', 'Edit…', 'Rename…', 'Forget saved password', 'Remove from library…']
    subtitles = [app.library.list.item(i).data(0x0100 + 2) for i in range(app.library.list.count())]
    assert sum('password saved' in text for text in subtitles) == 1                 # only the entry that has one
    app._ask_edit = lambda system: None                                             # the real dialog is modal
    next(a for a in with_pw.actions() if a.text() == 'Edit…').trigger()
    next(a for a in with_pw.actions() if a.text() == 'Forget saved password').trigger()
    assert emitted == [('edit_system', keeper.id), ('forget_password', keeper.id)]
    assert app.store.get(keeper.id).password_saved is False                         # the menu action really forgets


def edit_values(system, **changes):
    values = dict(host=system.host, port=system.port, user=system.user, name=system.name, view=system.view,
                  forget_password=False)
    values.update(changes)
    return values


def test_editing_name_and_view_keeps_the_saved_password(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', name='Old', password_saved=True)
    app.credentials.set(saved.id, 'pw', 'ana')
    app._ask_edit = lambda system: edit_values(system, name='New name', view='actual')
    app.edit_system(saved.id)
    item = app.store.get(saved.id)
    assert (item.name, item.view, item.password_saved) == ('New name', 'actual', True)
    assert app.credentials.get(saved.id) == 'pw' and app.library.list.item(0).text() == 'New name'


def test_editing_the_address_or_account_forgets_the_saved_password(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', password_saved=True)
    app.credentials.set(saved.id, 'pw', 'ana')
    app._ask_edit = lambda system: edit_values(system, host='other.local', port=5901)
    app.edit_system(saved.id)
    item = app.store.get(saved.id)
    assert (item.host, item.port, item.password_saved) == ('other.local', 5901, False)
    assert app.credentials.items == {}                                               # never follows to a new host
    app.credentials.set(saved.id, 'pw2', 'ana'); app.store.set_password_saved(saved.id, True)
    app._ask_edit = lambda system: edit_values(system, user='someone-else')
    app.edit_system(saved.id)
    assert app.credentials.items == {} and not app.store.get(saved.id).password_saved


def test_edit_can_forget_the_password_on_request_and_reports_a_clash(qapp, app):
    one = app.store.record_success('a.local', 5900, 'ana', password_saved=True)
    app.store.record_success('b.local', 5900, 'ana', name='Taken')
    app.credentials.set(one.id, 'pw', 'ana')
    app._ask_edit = lambda system: edit_values(system, host='b.local')
    app.edit_system(one.id)
    assert any('Taken' in note for note in app.notices)
    assert app.store.get(one.id).host == 'a.local' and app.credentials.get(one.id) == 'pw'   # nothing half-applied
    app._ask_edit = lambda system: edit_values(system, forget_password=True)
    app.edit_system(one.id)
    assert app.credentials.items == {} and not app.store.get(one.id).password_saved


def test_edit_dialog_validates_and_describes_what_will_happen(qapp, app):
    saved = app.store.record_success('a.local', 5900, 'ana', name='A', password_saved=True)
    dialog = EditSystemDialog(None, saved)
    assert dialog.address.text() == 'a.local' and dialog.forget.isVisibleTo(dialog)
    assert dialog.save.text() == 'Done' and dialog.connect_button.text() == 'Connect'      # like macOS's sheet
    assert dialog.mode.currentData() == 'standard' and not dialog.mode.model().item(1).isEnabled()
    assert 'Double-click connects' in dialog.saved_note.text()
    dialog.address.setText('b.local:5901')
    assert 'will be forgotten' in dialog.saved_note.text()                            # warned before saving
    dialog.address.setText('bad address')
    dialog.accept()
    assert dialog.error.text().startswith('✕') and dialog.result() != dialog.DialogCode.Accepted
    dialog.address.setText('b.local:5901'); dialog.user.setText('bo'); dialog.name.setText(' Studio ')
    dialog.view.setCurrentIndex(1)
    dialog.accept()
    assert dialog.values() == dict(host='b.local', port=5901, user='bo', name='Studio', view='actual',
                                   mode='standard', resolution='auto', forget_password=False, connect=False)
    fresh = EditSystemDialog(None, app.store.record_success('c.local', 5900, 'u'))
    assert not fresh.forget.isVisibleTo(fresh) and 'No password is saved' in fresh.saved_note.text()
    fresh.forget_password = True; fresh._refresh_note()
    assert 'No password is saved' in fresh.saved_note.text()
