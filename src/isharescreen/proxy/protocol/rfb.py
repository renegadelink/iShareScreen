"""RFB 3.889 protocol: version handshake + the client → server messages we
actually send during an HP session.

Apple's RFB dialect is RFC 6143 plus private message types in the 0x10–0x2f
range. We only build what we send; incoming server messages route by their
first byte at the call site (after enc1103 decryption).

Apple's pointer-button mask is non-standard: bit 1 is right-click and
bit 2 is middle, swapped from the RFC.
"""
from __future__ import annotations

import socket
import struct
import time


PROTOCOL_VERSION = b"RFB 003.889\n"


# ── message types ─────────────────────────────────────────────────────

# Server → client (incoming, after enc1103 decryption).
SRV_FRAMEBUFFER_UPDATE = 0x00
SRV_SET_COLOR_MAP_ENTRIES = 0x01
SRV_BELL = 0x02
SRV_SERVER_CUT_TEXT = 0x03
SRV_USER_SESSION_CHANGED = 0x14   # Apple-private; lock/login/desktop transitions

# Client → server (outgoing).
CLI_SET_ENCODINGS = 0x02
CLI_KEY_EVENT = 0x04
CLI_POINTER_EVENT = 0x05
CLI_CLIENT_CUT_TEXT = 0x06
CLI_POST_ENCRYPTION_TOGGLE = 0x12
CLI_MEDIA_NEGOTIATION = 0x1C
CLI_VIRTUAL_DISPLAY = 0x1D
CLI_VIEWER_INFO = 0x21


# Apple HP encoding list. Sent twice during handshake: once plaintext before
# the enc1103 toggle, once encrypted after.
HP_ENCODINGS_FULL: tuple[int, ...] = (
    1010, 1011, 1002, 6, 16, -239, 1104, 1100,
    -223, 1101, 1105, 1107, 1109, 1110,
)


# Apple's non-standard pointer button mask (bits 1 and 2 swapped vs. RFC 6143).
BTN_LEFT = 1 << 0
BTN_RIGHT = 1 << 1
BTN_MIDDLE = 1 << 2
BTN_SCROLL_UP = 1 << 3
BTN_SCROLL_DOWN = 1 << 4


# ── socket helper ────────────────────────────────────────────────────

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Block until exactly `n` bytes have been received from `sock`. Raises
    `ConnectionError` if the peer closes mid-read."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed during recv_exact")
        buf += chunk
    return bytes(buf)


# ── version handshake ────────────────────────────────────────────────

def do_protocol_handshake(sock: socket.socket) -> bytes:
    """Run version negotiation + read the security-types list. Returns the
    raw security-types bytes for the auth code to interpret."""
    recv_exact(sock, 12)
    sock.sendall(PROTOCOL_VERSION)
    nt = recv_exact(sock, 1)[0]
    return recv_exact(sock, nt)


def warmup_tcp(host: str, port: int, *, dwell_seconds: float = 1.4) -> None:
    """Apple Screen Sharing's pre-session TCP probe (the first of two TCPs).

    Open, exchange the ProtocolVersion banner, drop. Completing the version
    handshake registers something with screensharingd that lets the real
    session (TCP #2) survive user-context transitions (lock → login →
    desktop) without the server tearing the TCP at every transition.

    Apple holds the warmup TCP open for ~1.4 s after the banner exchange
    before closing — replicating the dwell gives screensharingd time to
    register the session before TCP #2 opens.
    """
    sock = socket.create_connection((host, port), timeout=10)
    try:
        sock.settimeout(5)
        recv_exact(sock, 12)
        sock.sendall(PROTOCOL_VERSION)
        nt = recv_exact(sock, 1)[0]
        recv_exact(sock, nt)
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
    if dwell_seconds > 0:
        time.sleep(dwell_seconds)


# ── client → server messages we actually send ────────────────────────

def build_set_encodings(encodings: tuple[int, ...] = HP_ENCODINGS_FULL) -> bytes:
    """SetEncodings (RFC 6143 §7.5.2) carrying Apple's HP encoding list."""
    return struct.pack(">BBH", CLI_SET_ENCODINGS, 0, len(encodings)) + b"".join(
        struct.pack(">i", e) for e in encodings
    )


def build_post_encryption_toggle() -> bytes:
    """0x12 PostEncryptionToggle — sent once after enc1103 setup completes,
    tells screensharingd we're ready for the 0x1c media offer."""
    return bytes.fromhex("1200000200010000")


def build_key_event(*, down: bool, keysym: int) -> bytes:
    """KeyEvent (msg 4): X11 keysym down/up. Apple expects X11 keysyms on the
    wire, not raw JS keyCode or platform scancodes."""
    return struct.pack(">BBxxI", CLI_KEY_EVENT, int(bool(down)), keysym)


def build_pointer_event(*, buttons: int, x: int, y: int) -> bytes:
    """PointerEvent (msg 5). Coordinates clamp to u16."""
    return struct.pack(
        ">BBHH",
        CLI_POINTER_EVENT,
        buttons & 0xFF,
        max(0, min(0xFFFF, x)),
        max(0, min(0xFFFF, y)),
    )


def build_client_cut_text(text: str) -> bytes:
    """ClientCutText (msg 6). Encoded latin-1 with replacement; matches how
    Apple's Screen Sharing.app handles wide characters."""
    body = text.encode("latin-1", errors="replace")
    return struct.pack(">BxxxI", CLI_CLIENT_CUT_TEXT, len(body)) + body


def build_viewer_info(
    *,
    app_id: int = 2,
    app_ver: tuple[int, int, int] = (6, 1, 0),
    os_ver: tuple[int, int, int] = (15, 3, 0),
    command_mask: bytes,
    extra: bytes = b"",
) -> bytes:
    """0x21 HandleViewerInfoMessage. Sent immediately after enc1103 setup.

    The 32-byte `command_mask` sets per-viewer feature gates on
    screensharingd. Pass `protocol.apple.APPLE_VIEWER_COMMAND_MASK`
    for byte-identical parity with Screen Sharing.app's shipping handshake.
    """
    if len(command_mask) != 32:
        raise ValueError("command_mask must be exactly 32 bytes")
    head = struct.pack(
        ">BBHHIIIIIII",
        CLI_VIEWER_INFO, 0,
        0x3E + len(extra),                  # msgSize: bytes after the msgSize field
        1,                                  # msgVersion
        app_id,
        app_ver[0], app_ver[1], app_ver[2],
        os_ver[0], os_ver[1], os_ver[2],
    )
    return head + command_mask + extra


def build_virtual_display(
    *,
    width: int,
    height: int,
    hidpi_scale: int = 2,
    width_mm: float = 300.0,
    height_mm: float = 200.0,
    hdr: bool = False,
    display_name: str = "iShareScreen Virtual Display",
    mode_count: int = 5,
    alt_user_login: bool = False,
) -> bytes:
    """0x1d HandleSetDisplayConfiguration — engages curtain mode and sets the
    virtual framebuffer geometry.

    Wire format reverse-engineered from screensharingd
    `ViewerMessages.c:HandleSetDisplayConfiguration`. All multi-byte fields
    are big-endian on the wire; the daemon byte-swaps several fields BE→LE
    in place after rx, which is why agent-side captures *look* little-endian
    but the wire format is BE throughout.

    `hidpi_scale` advertises Retina-style geometry: width/height become the
    logical (point) size while widthPix/heightPix carry the pixel dimensions.
    `hidpi_scale=1` flat-encodes at the requested resolution.

    `alt_user_login=True` flips a single byte at displayInfo+0x99 to 0x07.
    Captured Apple Screen Sharing traffic for the cmd=2 alt-user path
    sets this byte; with it clear, the daemon's `createLoginWindow=1`
    branch leaves `virtualDisplayCount=0` and the encoder targets the
    *console* user's screen instead of the alt-user vdisplay we just
    spawned. Other deltas in Apple's captured cmd=2 SDC (specific name
    string, MBP-shaped width_mm/height_mm, heterogeneous 5-mode list)
    appear to be informational; this one byte is the magic bit.
    """
    pts_w = width * hidpi_scale
    pts_h = height * hidpi_scale
    pix_w = width
    pix_h = height
    di_size = 0x9C + 28 * mode_count

    di = bytearray(di_size)
    struct.pack_into(">H", di, 0x00, di_size)
    name_bytes = display_name.encode("utf-8")[:121]  # leave 1 byte for NUL at +0x79
    di[0x02:0x02 + len(name_bytes)] = name_bytes
    struct.pack_into(">f", di, 0x82, width_mm)
    struct.pack_into(">f", di, 0x86, height_mm)
    struct.pack_into(">I", di, 0x8A, pts_w)
    struct.pack_into(">I", di, 0x8E, pts_h)
    struct.pack_into(">H", di, 0x9A, mode_count)
    if alt_user_login:
        di[0x99] = 0x07

    mode_flags = 1 if hdr else 0
    for i in range(mode_count):
        m = 0x9C + 28 * i
        struct.pack_into(">IIII", di, m + 0x00, pts_w, pts_h, pix_w, pix_h)
        struct.pack_into(">d", di, m + 0x10, 60.0)
        struct.pack_into(">I", di, m + 0x18, mode_flags)

    msg_size = 8 + di_size
    header = struct.pack(">BBHHHI", CLI_VIRTUAL_DISPLAY, 0, msg_size, 1, 1, 0)
    return header + bytes(di)


__all__ = [
    "BTN_LEFT", "BTN_MIDDLE", "BTN_RIGHT", "BTN_SCROLL_DOWN", "BTN_SCROLL_UP",
    "CLI_CLIENT_CUT_TEXT", "CLI_KEY_EVENT", "CLI_MEDIA_NEGOTIATION",
    "CLI_POINTER_EVENT", "CLI_POST_ENCRYPTION_TOGGLE", "CLI_SET_ENCODINGS",
    "CLI_VIEWER_INFO", "CLI_VIRTUAL_DISPLAY",
    "HP_ENCODINGS_FULL",
    "PROTOCOL_VERSION",
    "SRV_BELL", "SRV_FRAMEBUFFER_UPDATE", "SRV_SERVER_CUT_TEXT",
    "SRV_SET_COLOR_MAP_ENTRIES", "SRV_USER_SESSION_CHANGED",
    "build_client_cut_text", "build_key_event", "build_pointer_event",
    "build_post_encryption_toggle", "build_set_encodings",
    "build_viewer_info", "build_virtual_display",
    "do_protocol_handshake", "recv_exact", "warmup_tcp",
]
