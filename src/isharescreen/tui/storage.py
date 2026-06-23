"""Per-user persisted state for the TUI.

Just one file today: `~/.iss/last.json` (or the platform-equivalent),
holding the most recently submitted connect form (sans password). On
launch the TUI prefills the connect form from it. Passwords are
deliberately NOT persisted — getting that wrong becomes a security
incident; the convenience win is not worth the risk.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from .connect_screen import ConnectFormValues


log = logging.getLogger("iss.tui.storage")


def _state_dir() -> Path:
    """Per-user dir for iss TUI state. POSIX prefers XDG_STATE_HOME,
    falls back to ~/.iss. Windows uses %LOCALAPPDATA%\\iss."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "iss"
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "iss"
    return Path.home() / ".iss"


def _last_path() -> Path:
    return _state_dir() / "last.json"


def load_last() -> Optional[ConnectFormValues]:
    """Return the most recently saved form, or None if none exists / the
    file is malformed."""
    p = _last_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("could not read %s: %s", p, e)
        return None
    if not isinstance(data, dict):
        return None
    return ConnectFormValues(
        host=str(data.get("host", "")),
        user=str(data.get("user", "")),
        password="",  # passwords are never persisted -- see docstring
        advertise=str(data.get("advertise", "1920x1080")),
        audio=bool(data.get("audio", True)),
        curtain=bool(data.get("curtain", True)),
        hdr=bool(data.get("hdr", False)),
        share_console=bool(data.get("share_console", False)),
        alt_session=bool(data.get("alt_session", False)),
        frontend=str(data.get("frontend", "browser")),
        dynamic_resolution=bool(data.get("dynamic_resolution", False)),
        hidpi=str(data.get("hidpi", "auto")),
        decoder=str(data.get("decoder", "auto")),
    )


def save_last(values: ConnectFormValues) -> None:
    """Persist the connect form (sans password). Best effort -- failure
    just means next launch won't prefill."""
    p = _last_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "host": values.host,
            "user": values.user,
            "advertise": values.advertise,
            "audio": values.audio,
            "curtain": values.curtain,
            "hdr": values.hdr,
            "share_console": values.share_console,
            "alt_session": values.alt_session,
            "frontend": values.frontend,
            "dynamic_resolution": values.dynamic_resolution,
            "hidpi": values.hidpi,
            "decoder": values.decoder,
        }
        p.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(p, 0o600)
        except OSError:
            pass
    except OSError as e:
        log.warning("could not save %s: %s", p, e)


def bug_snapshot_path() -> Path:
    """Timestamped path for a bug-report dump. Caller ensures the dir
    exists by writing to it."""
    import time
    ts = time.strftime("%Y%m%d-%H%M%S")
    return _state_dir() / "snapshots" / f"snapshot-{ts}.json"


__all__ = ["load_last", "save_last", "bug_snapshot_path"]
