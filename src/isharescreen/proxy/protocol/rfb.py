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
CLI_POST_ENCRYPTION_TOGGLE = 0x12
CLI_MEDIA_NEGOTIATION = 0x1C
CLI_VIRTUAL_DISPLAY = 0x1D
CLI_VIEWER_INFO = 0x21


# Apple HP encoding list. Sent twice during handshake: once plaintext before
# the enc1103 toggle, once encrypted after.
HP_ENCODINGS_FULL: tuple[int, ...] = (
    1010, 1011, 1002, 6, 16, 1104, 1100,
    -223, 1101, 1105, 1107, 1109, 1110,
)
# Note: -239 (RichCursor, legacy pre-cache cursor pseudo-encoding) is
# intentionally not advertised. We use 1104 (Apple's cached cursor
# pseudo-encoding) exclusively.


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
    hdr: bool = False,
    display_name: str = "iShareScreen Virtual Display",
    mode_count: int = 5,
    alt_user_login: bool = False,
) -> bytes:
    """0x1d HandleSetDisplayConfiguration — sets the virtual-display
    geometry. Used both for the initial handshake and for mid-session
    dynamic-resolution requests (the message is identical; the server
    treats a steady-state 0x1d as a resize and answers with 0x451).

    Wire format reverse-engineered from screensharingd
    `ViewerMessages.c:HandleSetDisplayConfiguration`, then byte-aligned to
    native Screen Sharing.app. All multi-byte fields are big-endian on the
    wire; the daemon byte-swaps several fields BE→LE in place after rx,
    which is why agent-side captures *look* little-endian but the wire
    format is BE throughout.

    The descriptor mirrors what native sends on every 0x1d:
    `display_flags = 0x01` (DYNAMIC_RESOLUTION), `display_type = 4`
    (virtual display), `reserved = 7`, physical dims of an MBP panel,
    `max_width/height = 3840×2160`, and a heterogeneous mode table scaled
    to the target. A mid-session 0x1d MUST carry these or the server
    won't treat it as a dynamic-resolution request. (RFC §7.1/§7.3.)

    `hidpi_scale` advertises Retina-style geometry: width/height become the
    logical (point) size while the mode table carries the pixel dimensions.
    `hidpi_scale=1` flat-encodes at the requested resolution.

    `alt_user_login=True` marks the cmd=2 alt-user path. The "magic byte"
    Apple's captured cmd=2 traffic sets is a single `0x07` at
    displayInfo+0x99 — which is the low byte of the `reserved` u32 at
    +0x96. With that byte clear, the daemon's `createLoginWindow=1` branch
    leaves `virtualDisplayCount=0` and the encoder targets the *console*
    user's screen instead of the alt-user vdisplay we just spawned. The
    other deltas in Apple's cmd=2 SDC (specific display-name string,
    MBP-shaped physical dimensions, heterogeneous mode list) appear to be
    informational. Now that we always emit `reserved = 7`, displayInfo+0x99
    is already `0x07` for every 0x1d, so this flag is currently a no-op at
    the byte level — kept for call-site intent and in case the reserved
    value ever changes.
    """
    # `width`/`height` are the LOGICAL (point / scaled) target. `hidpi_scale`
    # is the backing:point ratio the mode table advertises: 2 => Retina
    # (backing = 2× points), 1 => flat (backing = points, ~1/4 the pixels =
    # much less bandwidth; the right choice on non-Retina clients where a 2×
    # stream renders the UI half-size). The host honors whichever ratio the
    # mode table carries (verified against 24G231: a 1× mode table returns
    # backing == scaled).
    pts_w = width
    pts_h = height
    di_size = 0x9C + 28 * mode_count

    di = bytearray(di_size)
    struct.pack_into(">H", di, 0x00, di_size)
    name_bytes = display_name.encode("utf-8")[:121]  # leave 1 byte for NUL at +0x79
    di[0x02:0x02 + len(name_bytes)] = name_bytes

    # display_flags: always set DYNAMIC_RESOLUTION (0x01) — matches
    # native Screen Sharing.app which sets this on every 0x1d.
    struct.pack_into(">I", di, 0x7A, 1)

    # display_type = 4 (virtual display).
    struct.pack_into(">I", di, 0x7E, 4)

    # Physical dimensions from native macOS capture.
    struct.pack_into(">f", di, 0x82, 369.4545593261719)
    struct.pack_into(">f", di, 0x86, 207.81817626953125)

    struct.pack_into(">H", di, 0x92, 0)   # current_mode_index
    struct.pack_into(">H", di, 0x94, 0)   # preferred_mode_index
    struct.pack_into(">I", di, 0x96, 7)   # reserved — native sends 7
    struct.pack_into(">H", di, 0x9A, mode_count)

    # Heterogeneous mode table (like native kModes), scaled to target.
    _NATIVE_MODES = [
        (3840, 2160, 1920, 1080),
        (2880, 1800, 1440,  900),
        (3840, 2160, 1920, 1080),
        (2880, 1620, 1440,  810),
        (2624, 1696, 1312,  848),
    ]
    sx = pts_w / 1920.0 if pts_w else 1.0
    sy = pts_h / 1080.0 if pts_h else 1.0
    mode_flags = 1 if hdr else 0

    for i in range(mode_count):
        base = _NATIVE_MODES[i % len(_NATIVE_MODES)]
        # point (scaled) dims from the native template, scaled to the target;
        # pixel (backing) dims = point × hidpi_scale, so the mode table
        # advertises a 2× (Retina) or 1× (flat) backing per the caller.
        msw = int(base[2] * sx + 0.5)
        msh = int(base[3] * sy + 0.5)
        mw = msw * hidpi_scale
        mh = msh * hidpi_scale
        m = 0x9C + 28 * i
        struct.pack_into(">IIII", di, m + 0x00, mw, mh, msw, msh)
        struct.pack_into(">d", di, m + 0x10, 60.0)
        struct.pack_into(">I", di, m + 0x18, mode_flags)

    # max_width/height = 3840×2160, matching native app. The server
    # caps the virtual display at this backing size regardless of
    # the mode-table entries. Larger requests get a server-side
    # fallback (typically to a safe minimum). Keeping the client
    # request ≤ this bound avoids triggering the fallback path.
    struct.pack_into(">I", di, 0x8A, 3840)
    struct.pack_into(">I", di, 0x8E, 2160)

    msg_size = 8 + di_size
    header = struct.pack(">BBHHHI", CLI_VIRTUAL_DISPLAY, 0, msg_size, 1, 1, 0)
    return header + bytes(di)


__all__ = [
    "BTN_LEFT", "BTN_MIDDLE", "BTN_RIGHT", "BTN_SCROLL_DOWN", "BTN_SCROLL_UP",
    "CLI_KEY_EVENT", "CLI_MEDIA_NEGOTIATION",
    "CLI_POINTER_EVENT", "CLI_POST_ENCRYPTION_TOGGLE", "CLI_SET_ENCODINGS",
    "CLI_VIEWER_INFO", "CLI_VIRTUAL_DISPLAY",
    "HP_ENCODINGS_FULL",
    "PROTOCOL_VERSION",
    "SRV_BELL", "SRV_FRAMEBUFFER_UPDATE", "SRV_SERVER_CUT_TEXT",
    "SRV_SET_COLOR_MAP_ENTRIES", "SRV_USER_SESSION_CHANGED",
    "build_key_event", "build_pointer_event",
    "build_post_encryption_toggle", "build_set_encodings",
    "build_viewer_info", "build_virtual_display",
    "do_protocol_handshake", "recv_exact", "warmup_tcp",
]
