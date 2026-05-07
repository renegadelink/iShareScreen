"""InputController — sends user input back to the host over the RFB control sock.

Frontends instantiate one and route their UI events through it. The
controller serialises wire events with the active stream cipher and
ships them over the shared TCP socket.

Coordinates arrive in the source-stream pixel space
(0..server_w, 0..server_h); no scaling is performed here.

When `alt_session=True`, mouse events are wrapped in msg 0x10
HandleEncryptedEventMessage instead of the standard msg 0x05
PointerEvent. The daemon's HID filter gates msg 0x05 PointerEvents
by uid match against the console user; the alt-user uid never
matches, so the daemon silently drops cmd=2 mouse events via the
standard path. Msg 0x10 takes a different code path that bypasses
the uid gate. Keyboard input is unaffected because Apple uses
CGEventPost with no per-uid routing, so we keep the standard
msg 0x04 KeyEvent for both paths.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Optional

from Crypto.Cipher import AES

from .protocol.enc1103 import StreamCipher
from .protocol.rfb import (
    BTN_SCROLL_DOWN,
    BTN_SCROLL_UP,
    build_key_event,
    build_pointer_event,
)


log = logging.getLogger(__name__)


def _build_msg10_pointer(
    cbc_key: bytes, buttons: int, x: int, y: int,
) -> bytes:
    """Build an 18-byte msg 0x10 HandleEncryptedEventMessage wrapping
    one pointer event. Reverse-engineered from screensharingd
    HandleEncryptedEventMessage.

    Wire format (18 bytes):
        [0]        0x10                msg type
        [1]        pad
        [2..17]    16B AES-128-ECB ciphertext

    Plaintext (16 bytes), encrypted with cbc_key (the post-toggle
    cryptor that replaced the SRP-K-derived ecb_key inside
    HandleSetEncryptionMessage when iss sent msg 0x12):
        [0..9]     pad/timing — 10 bytes (daemon byte-swaps as 2×u32
                                but doesn't enforce values)
        [10]       0xff sentinel  (REQUIRED — the daemon checks this
                                byte and drops the event if not 0xff)
        [11]       button mask    (Apple-style: bit0=L, bit1=R, bit2=M)
        [12..13]   u16 BE x       (in server-screen pixels)
        [14..15]   u16 BE y
    """
    plaintext = (
        b"\x00" * 10
        + b"\xff"
        + bytes([buttons & 0xff])
        + struct.pack(">HH", x & 0xffff, y & 0xffff)
    )
    assert len(plaintext) == 16, len(plaintext)
    aes = AES.new(cbc_key[:16], AES.MODE_ECB)
    ct = aes.encrypt(plaintext)
    return bytes([0x10, 0x00]) + ct


class InputController:
    """Thread-safe wrapper around the RFB control socket.

    All three event methods are non-blocking from the caller's perspective and
    swallow per-call socket errors — a stale socket during hot reconnect must
    not propagate up into the render loop. Resolution is only used to clamp
    out-of-range pointer coordinates; accurate clamping prevents wrap-around
    bugs in the macOS pointer handler.
    """

    def __init__(
        self,
        sock: socket.socket,
        cipher: Optional[StreamCipher],
        *,
        server_width: int,
        server_height: int,
        alt_session: bool = False,
    ) -> None:
        self._sock = sock
        self._cipher = cipher
        self._w = server_width
        self._h = server_height
        self._alt_session = alt_session
        self._lock = threading.Lock()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True

    # ── public event API ──────────────────────────────────────────────

    def pointer_event(self, buttons: int, x: int, y: int) -> None:
        cx = max(0, min(self._w - 1, int(x)))
        cy = max(0, min(self._h - 1, int(y)))
        if self._alt_session and self._cipher is not None:
            # cmd=2 alt-user path: wrap in msg 0x10 with AES-ECB(cbc_key)
            # so the daemon dispatches via HandleEncryptedEventMessage
            # instead of the uid-gated msg 0x05 path that would silently
            # drop our event.
            msg = _build_msg10_pointer(
                self._cipher.cbc_key, buttons, cx, cy,
            )
        else:
            msg = build_pointer_event(buttons=buttons, x=cx, y=cy)
        self._send(msg)

    def scroll_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Apple emulates a wheel via pointer events with bits 3/4 of buttons.

        Each `(press, release)` pair = one wheel tick. `abs(dy)` is the number
        of ticks; the sign picks up vs down. The host's HID handler edge-
        triggers, so back-to-back press/release pairs at line rate are fine —
        Mac UI elements (Safari, Finder) eat them at >100 Hz.

        Horizontal scroll isn't natively supported in RFB; we ignore dx."""
        if dy == 0:
            return
        bit = BTN_SCROLL_UP if dy < 0 else BTN_SCROLL_DOWN
        for _ in range(abs(int(dy))):
            self.pointer_event(bit, x, y)
            self.pointer_event(0, x, y)

    def key_event(self, down: bool, keysym: int) -> None:
        if not keysym:
            return
        self._send(build_key_event(down=bool(down), keysym=int(keysym)))

    # ── internals ─────────────────────────────────────────────────────

    def _send(self, payload: bytes) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._sock.sendall(self._cipher.encrypt_message(payload) if self._cipher else payload)
            except OSError as e:
                log.debug("input send dropped: %s", e)
