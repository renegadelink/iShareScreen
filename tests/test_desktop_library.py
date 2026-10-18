"""Saved-computer library: persistence, identity, failure behaviour."""
import json
import os

import pytest

from isharescreen.frontend.desktop import library as L
from isharescreen.frontend.desktop.library import LibraryError, LibraryStore


@pytest.fixture
def path(tmp_path):
    return tmp_path / 'state' / 'library.json'


def test_record_success_creates_entry_without_any_password(path):
    store = LibraryStore(path)
    item = store.record_success('Studio.Local', 5900, 'ana', name='Studio', view='actual', when=1000.0)
    raw = json.loads(path.read_text())
    assert raw['version'] == L.SCHEMA_VERSION
    assert raw['systems'][0] == {'id': item.id, 'host': 'Studio.Local', 'port': 5900, 'user': 'ana',
                                 'name': 'Studio', 'mode': 'standard', 'view': 'actual',
                                 'last_connected': 1000.0, 'created': item.created, 'password_saved': False,
                                 'resolution': 'auto'}
    assert [key for key in raw['systems'][0] if 'password' in key] == ['password_saved']   # a flag, never a secret


def test_reconnect_updates_the_same_entry(path):
    store = LibraryStore(path)
    first = store.record_success('studio.local', 5900, 'ana', name='Studio', when=1.0)
    again = store.record_success('STUDIO.LOCAL', 5900, 'ana', when=2.0)
    assert again.id == first.id and len(store.items) == 1
    assert again.name == 'Studio' and again.last_connected == 2.0     # a blank name keeps the old one
    store.record_success('studio.local', 5901, 'ana')                 # other port
    store.record_success('studio.local', 5900, 'bo')                  # other account
    assert len(store.items) == 3
    assert len(LibraryStore(path).items) == 3                          # and it round-trips


def test_sorted_puts_recent_first(path):
    store = LibraryStore(path)
    store.record_success('old.local', 5900, 'a', when=1.0)
    store.record_success('new.local', 5900, 'a', when=9.0)
    assert [i.host for i in store.sorted()] == ['new.local', 'old.local']


def test_rename_and_remove(path):
    store = LibraryStore(path)
    item = store.record_success('a.local', 5900, 'a')
    store.rename(item.id, '  Office  ')
    assert LibraryStore(path).items[0].title == 'Office'
    store.remove(item.id)
    assert LibraryStore(path).items == []


def test_failed_write_leaves_disk_and_memory_unchanged(path, monkeypatch):
    store = LibraryStore(path)
    store.record_success('a.local', 5900, 'a', when=1.0)
    before = path.read_bytes()

    def broken(*_):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(os, 'replace', broken)
    with pytest.raises(LibraryError):
        store.record_success('b.local', 5900, 'b')
    with pytest.raises(LibraryError):
        store.record_success('a.local', 5900, 'a', name='Renamed', when=5.0)
    with pytest.raises(LibraryError):
        store.remove(store.items[0].id)
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert [(i.host, i.name, i.last_connected) for i in store.items] == [('a.local', '', 1.0)]
    assert not [p for p in path.parent.iterdir() if p.suffix == '.tmp']      # no temp file left behind


def test_corrupt_file_is_set_aside_not_fatal(path):
    path.parent.mkdir(parents=True)
    path.write_text('{not json')
    store = LibraryStore(path)
    assert store.items == [] and 'unreadable' in store.load_notice and not store.read_only
    assert path.with_name('library.json.corrupt').read_text() == '{not json'
    store.record_success('a.local', 5900, 'a')
    assert len(LibraryStore(path).items) == 1


def test_newer_schema_is_read_only_and_never_overwritten(path):
    path.parent.mkdir(parents=True)
    original = json.dumps({'version': 99, 'systems': [{'id': 'x', 'host': 'future.local', 'port': 5900}]})
    path.write_text(original)
    store = LibraryStore(path)
    assert store.read_only and [i.host for i in store.items] == ['future.local']
    with pytest.raises(LibraryError):
        store.record_success('b.local', 5900, 'b')
    assert path.read_text() == original
    assert [i.host for i in store.items] == ['future.local']


def test_load_skips_malformed_entries_and_discards_unknown_keys(path):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'version': 1, 'systems': [
        'junk', {'host': ''}, {'host': 'a.local', 'port': 'x'}, {'host': 'a.local', 'port': 70000},
        {'host': 'ok.local', 'port': 5900, 'user': 'u', 'password': 'hunter2', 'mode': 'hp', 'view': 'weird'},
        {'host': 'OK.local', 'port': 5900, 'user': 'u'},      # duplicate identity
    ]}))
    store = LibraryStore(path)
    assert [i.host for i in store.items] == ['ok.local']
    assert store.items[0].mode == 'standard' and store.items[0].view == 'fit'
    store.save()
    assert 'hunter2' not in path.read_text()                  # a hand-added password is never carried forward


def test_missing_file_is_an_empty_library(path):
    store = LibraryStore(path)
    assert store.items == [] and not store.read_only and store.load_notice == ''
    assert not path.exists()                                  # loading alone writes nothing
