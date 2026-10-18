"""Small per-user app settings, in a JSON file next to the library."""
from __future__ import annotations
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from ...tui.storage import state_dir

log = logging.getLogger('iss.desktop.settings')


def settings_path() -> Path:
    override = os.environ.get('ISHARESCREEN_SETTINGS')
    return Path(override) if override else state_dir() / 'settings.json'


class Settings:
    """Only booleans the app itself defines are read back; anything else in the
    file is ignored, and a damaged file just means the defaults."""
    DEFAULTS = {'experimental_hp': False}

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else settings_path()
        self.values = dict(self.DEFAULTS)
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                for key in self.DEFAULTS:
                    if isinstance(data.get(key), bool):
                        self.values[key] = data[key]
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as error:
            log.warning('settings %s unreadable, using defaults: %s', self.path, error)

    @property
    def experimental_hp(self) -> bool:
        return self.values['experimental_hp']

    def set(self, key: str, value: bool) -> bool:
        """Store and write atomically. Returns False (value unchanged on disk) if the write failed."""
        if key not in self.DEFAULTS:
            raise KeyError(key)
        previous, self.values[key] = self.values[key], bool(value)
        temp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=self.path.parent, prefix='.settings-', suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(self.values, handle, indent=2)
            os.replace(temp, self.path)
            temp = None
            return True
        except OSError as error:
            log.warning('could not save settings: %s', error)
            self.values[key] = previous
            return False
        finally:
            if temp:
                try:
                    os.unlink(temp)
                except OSError:
                    pass
