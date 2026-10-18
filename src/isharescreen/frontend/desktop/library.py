"""Saved-computer library: a versioned local JSON file, written atomically.

Only connection metadata is stored. A password is never written here: at most a
`password_saved` flag records that the user chose to keep one in the operating
system's credential vault (credentials.py). Unknown keys read from disk are
discarded rather than carried forward, so a hand-edited file cannot smuggle a
secret back in.
"""
from __future__ import annotations
import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from ...proxy.address import DEFAULT_PORT, format_address
from ...tui.storage import state_dir

log = logging.getLogger("iss.desktop.library")

SCHEMA_VERSION = 1
MODES = ('standard', 'high_performance')
MODE_LABELS = {'standard': 'Standard', 'high_performance': 'High Performance'}
VIEWS = ('fit', 'actual')


def describe_last_connected(when) -> str:
    if not when:
        return 'Not connected yet'
    return 'Last connected ' + time.strftime('%b %d, %Y at %I:%M %p', time.localtime(when)).replace(' 0', ' ')


class LibraryError(Exception):
    """The library could not be read or written; the message is user-facing."""


@dataclass
class SavedSystem:
    id: str
    host: str
    port: int
    user: str
    name: str = ''
    mode: str = 'standard'
    view: str = 'fit'                       # local viewing choice: fit | actual
    last_connected: Optional[float] = None  # Unix time of last successful setup
    created: float = field(default_factory=time.time)
    # True when the user chose to save this entry's password. Only the flag lives
    # here; the secret itself is in the OS credential vault (see credentials.py).
    password_saved: bool = False
    resolution: str = 'auto'                # High Performance only: 'auto' or 'WxH'

    @property
    def title(self) -> str:
        return self.name or self.host

    @property
    def address(self) -> str:
        return format_address(self.host, self.port)

    @property
    def identity(self) -> tuple:
        return _identity(self.host, self.port, self.user, self.mode)


def valid_resolution(value) -> str:
    """'auto' or a 'WxH' inside what a Mac will accept, else 'auto'."""
    if isinstance(value, str) and re.fullmatch(r'\d{3,4}x\d{3,4}', value):
        width, height = (int(part) for part in value.split('x'))
        if 640 <= width <= 1920 and 480 <= height <= 1080:
            return value
    return 'auto'


def _identity(host: str, port: int, user: str, mode: str) -> tuple:
    # One entry per host+port+account+mode; the hostname is case-insensitive.
    return (host.strip().lower(), int(port), user, mode)


def library_path() -> Path:
    override = os.environ.get('ISHARESCREEN_LIBRARY')
    return Path(override) if override else state_dir() / 'library.json'


def _entry(raw) -> Optional[SavedSystem]:
    if not isinstance(raw, dict):
        return None
    host = raw.get('host')
    port = raw.get('port', DEFAULT_PORT)
    if not isinstance(host, str) or not host.strip() or isinstance(port, bool) or not isinstance(port, int) \
            or not 1 <= port <= 65535:
        return None
    mode = raw.get('mode') if raw.get('mode') in MODES else 'standard'
    view = raw.get('view') if raw.get('view') in VIEWS else 'fit'
    last = raw.get('last_connected')
    created = raw.get('created')
    return SavedSystem(
        id=str(raw.get('id') or uuid.uuid4().hex), host=host.strip(), port=port,
        user=str(raw.get('user') or ''), name=str(raw.get('name') or ''), mode=mode, view=view,
        last_connected=float(last) if isinstance(last, (int, float)) and not isinstance(last, bool) else None,
        created=float(created) if isinstance(created, (int, float)) and not isinstance(created, bool) else time.time(),
        password_saved=raw.get('password_saved') is True, resolution=valid_resolution(raw.get('resolution')))


class LibraryStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else library_path()
        self.items: list[SavedSystem] = []
        self.read_only = False      # set when the file was written by a newer version
        self.load_notice = ''       # human-readable note when the file had to be set aside
        self.load()

    # ── reading ──────────────────────────────────────────────────────
    def load(self) -> None:
        self.items, self.read_only, self.load_notice = [], False, ''
        try:
            text = self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return
        except OSError as error:
            self.read_only = True
            self.load_notice = f'The library could not be read ({error.strerror or error}).'
            return
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or not isinstance(data.get('systems'), list):
                raise ValueError('unexpected structure')
            version = data.get('version')
            if not isinstance(version, int) or version < 1:
                raise ValueError('missing version')
        except ValueError as error:
            self._set_aside(error)
            return
        if version > SCHEMA_VERSION:
            self.read_only = True
            self.load_notice = 'The library was saved by a newer version of iShareScreen and is read-only here.'
        seen = set()
        for raw in data['systems']:
            item = _entry(raw)
            if item and item.identity not in seen and item.id not in {i.id for i in self.items}:
                seen.add(item.identity)
                self.items.append(item)

    def _set_aside(self, error) -> None:
        backup = self.path.with_name(self.path.name + '.corrupt')
        try:
            os.replace(self.path, backup)
            self.load_notice = f'The saved library was unreadable and was moved to {backup.name}.'
        except OSError:
            self.read_only = True
            self.load_notice = 'The saved library is unreadable and could not be moved aside.'
        log.warning('library %s unreadable: %s', self.path, error)

    # ── writing ──────────────────────────────────────────────────────
    def save(self) -> None:
        """Write atomically; raises LibraryError and leaves the old file intact."""
        if self.read_only:
            raise LibraryError(self.load_notice or 'The library is read-only.')
        payload = {'version': SCHEMA_VERSION, 'systems': [asdict(item) for item in self.items]}
        temp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix='.library-', suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            temp = None
        except OSError as error:
            raise LibraryError(f'Could not save the library ({error.strerror or error}).') from error
        finally:
            if temp:
                try:
                    os.unlink(temp)
                except OSError:
                    pass

    def _commit(self, snapshot: list[SavedSystem]) -> None:
        """Save; on failure restore `snapshot` so memory matches disk."""
        try:
            self.save()
        except LibraryError:
            self.items = snapshot
            raise

    # ── operations ───────────────────────────────────────────────────
    def get(self, system_id: str) -> Optional[SavedSystem]:
        return next((item for item in self.items if item.id == system_id), None)

    def find(self, host: str, port: int, user: str, mode: str = 'standard') -> Optional[SavedSystem]:
        key = _identity(host, port, user, mode)
        return next((item for item in self.items if item.identity == key), None)

    def sorted(self) -> list[SavedSystem]:
        return sorted(self.items, key=lambda i: (-(i.last_connected or 0), i.title.lower()))

    def record_success(self, host: str, port: int, user: str, mode: str = 'standard',
                       name: str = '', view: str = 'fit', when: Optional[float] = None,
                       password_saved: Optional[bool] = None, resolution: Optional[str] = None) -> SavedSystem:
        """Add or refresh an entry after a session was really established.

        An existing entry (same host, port, account and mode) keeps its id and
        its name unless a new name is supplied. `password_saved` None leaves the
        flag alone. If the write fails the in-memory library is rolled back and
        LibraryError is raised for the caller to show.
        """
        snapshot = [SavedSystem(**asdict(i)) for i in self.items]
        when = time.time() if when is None else when
        item = self.find(host, port, user, mode)
        if item is None:
            item = SavedSystem(id=uuid.uuid4().hex, host=host.strip(), port=port, user=user, mode=mode)
            self.items.append(item)
        if name.strip():
            item.name = name.strip()
        item.view = view if view in VIEWS else 'fit'
        item.last_connected = when
        if password_saved is not None:
            item.password_saved = password_saved
        if resolution is not None:
            item.resolution = valid_resolution(resolution)
        self._commit(snapshot)
        return item

    def set_password_saved(self, system_id: str, saved: bool) -> None:
        item = self.get(system_id)
        if item and item.password_saved != saved:
            snapshot = [SavedSystem(**asdict(i)) for i in self.items]
            item.password_saved = saved
            self._commit(snapshot)

    def update(self, system_id: str, *, host: str, port: int, user: str, name: str, view: str,
               mode: str = 'standard', resolution: Optional[str] = None) -> bool:
        """Edit an entry. Returns True when its destination or account (host, port
        or user) changed, in which case the saved-password flag is cleared: a saved
        password must never follow an entry to a different destination or account.
        Changing only the mode, name or view keeps it. Raises LibraryError if
        another entry already has the new identity."""
        item = self.get(system_id)
        if item is None:
            raise LibraryError('That computer is no longer in the library.')
        host = host.strip()
        mode = mode if mode in MODES else 'standard'
        clash = self.find(host, port, user, mode)
        if clash is not None and clash.id != system_id:
            raise LibraryError(f'“{clash.title}” already uses that address and account.')
        snapshot = [SavedSystem(**asdict(i)) for i in self.items]
        changed = item.identity[:3] != _identity(host, port, user, mode)[:3]
        item.host, item.port, item.user, item.mode = host, port, user, mode
        item.name = name.strip()
        item.view = view if view in VIEWS else 'fit'
        if resolution is not None:
            item.resolution = valid_resolution(resolution)
        if changed:
            item.password_saved = False
        self._commit(snapshot)
        return changed

    def rename(self, system_id: str, name: str) -> None:
        item = self.get(system_id)
        if item:
            snapshot = [SavedSystem(**asdict(i)) for i in self.items]
            item.name = name.strip()
            self._commit(snapshot)

    def remove(self, system_id: str) -> None:
        snapshot = list(self.items)
        self.items = [i for i in self.items if i.id != system_id]
        self._commit(snapshot)
