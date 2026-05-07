"""Pre-flight connect prompt: host / username / password.

A small stdin/stdout prompt that runs once before the viewer starts.
Anything supplied on the CLI is honoured without re-asking; anything
still missing is asked interactively (passwords via getpass so they
aren't echoed). No persistence — every run asks for whatever the CLI
didn't supply.
"""
from __future__ import annotations

import getpass
import logging
from dataclasses import dataclass
from typing import Optional

from ..proxy.protocol.negotiation import AdvertiseDims


_RESOLUTION_PRESETS: list[tuple[str, AdvertiseDims]] = [
    ("3840 × 2160 (4K UHD)",  AdvertiseDims(3840, 2160)),
    ("3440 × 1440 (UWQHD)",   AdvertiseDims(3440, 1440)),
    ("2560 × 1600",           AdvertiseDims(2560, 1600)),
    ("2560 × 1440 (QHD)",     AdvertiseDims(2560, 1440)),
    ("1920 × 1200 (WUXGA)",   AdvertiseDims(1920, 1200)),
    ("1920 × 1080 (FHD)",     AdvertiseDims(1920, 1080)),
    ("1680 × 1050",           AdvertiseDims(1680, 1050)),
    ("1600 ×  900",           AdvertiseDims(1600,  900)),
    ("1366 ×  768",           AdvertiseDims(1366,  768)),
    ("1280 ×  720 (HD)",      AdvertiseDims(1280,  720)),
    ("1024 ×  768 (XGA)",     AdvertiseDims(1024,  768)),
    (" 800 ×  600 (SVGA)",    AdvertiseDims( 800,  600)),
]
_DEFAULT_RESOLUTION_IDX = 4  # 1920 × 1200 (WUXGA)

log = logging.getLogger("iss.connect")


@dataclass(slots=True)
class ConnectChoice:
    """User's answer to the prompt (or the CLI's pre-supplied values)."""
    host: str = ""
    username: str = ""
    password: str = ""
    advertise: Optional[AdvertiseDims] = None

    @property
    def is_complete(self) -> bool:
        """True if host+user+password are all set; the prompt can be skipped."""
        return bool(self.host and self.username and self.password)


class UserCancelled(Exception):
    """The user cancelled the prompt (Ctrl-C / Ctrl-D / empty host)."""


def _ask_resolution() -> AdvertiseDims:
    """Show the numbered preset list and return the user's pick. Empty
    input or anything unparseable picks the default (WUXGA 1920×1200)."""
    print("\nResolution:")
    for i, (label, _) in enumerate(_RESOLUTION_PRESETS):
        print(f"  {i + 1:2d}. {label}")
    raw = _ask(
        f"Pick [1-{len(_RESOLUTION_PRESETS)}, "
        f"default={_DEFAULT_RESOLUTION_IDX + 1}]: "
    )
    try:
        idx = int(raw) - 1 if raw else _DEFAULT_RESOLUTION_IDX
        if not 0 <= idx < len(_RESOLUTION_PRESETS):
            idx = _DEFAULT_RESOLUTION_IDX
    except ValueError:
        idx = _DEFAULT_RESOLUTION_IDX
    return _RESOLUTION_PRESETS[idx][1]


def _ask(label: str) -> str:
    """input() wrapper that raises UserCancelled on EOF (Ctrl-D, closed
    stdin) so callers can distinguish "user cancelled" from "user
    typed an empty string"."""
    try:
        return input(label).strip()
    except EOFError as e:
        raise UserCancelled() from e


def prompt(prefill: Optional[ConnectChoice] = None) -> ConnectChoice:
    """Resolve a complete ConnectChoice. Asks interactively for any
    field the prefill doesn't supply — including resolution unless
    `prefill.advertise` is already set (i.e. `--advertise` was
    passed on the CLI).

    Raises UserCancelled if the user dismisses (Ctrl-C / Ctrl-D /
    empty host).
    """
    prefill = prefill or ConnectChoice()
    if prefill.is_complete and prefill.advertise is not None:
        return prefill

    host = prefill.host or _ask("Connect to: ")
    if not host:
        raise UserCancelled()

    user = prefill.username or _ask(f"Username for {host}: ")
    if not user:
        raise UserCancelled()

    pw = prefill.password
    if not pw:
        try:
            pw = getpass.getpass(f"Password for {user}@{host}: ")
        except EOFError as e:
            raise UserCancelled() from e
    if not pw:
        raise UserCancelled()

    # Resolution picker — only when the CLI didn't pass --advertise.
    advertise = prefill.advertise
    if advertise is None:
        advertise = _ask_resolution()

    return ConnectChoice(
        host=host, username=user, password=pw,
        advertise=advertise,
    )


__all__ = ["ConnectChoice", "UserCancelled", "prompt"]
