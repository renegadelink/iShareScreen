"""TCP-side handshake driver: from `socket.connect()` to the SRTP keys.

Sequence (per Apple HP-mode RE):
    1. Open TCP, run version + security-types preamble
    2. Auth: SRP with non-SRP RSA1+AES fallback
    3. ClientInit + ServerInit
    4. Plaintext: ViewerInfo + Apple 0x12 follow-up
    5. Plaintext: VirtualDisplay + first SetEncodings
    6. Read 1010/1011/1103 reply burst → build enc1103 stream cipher
    7. PostEncryptionToggle (0x12)
    8. Encrypted: second SetEncodings
    9. Encrypted: 0x1c MediaStreamConfiguration carrying SRTP master keys
   10. Read 0x1c answer → canvas dims; on degenerate canvas, re-query × N

UDP receivers must be bound *before* this returns — the session-start media
burst lands within a fraction of a second of the 0x1c answer.

The driver is decomposed into four phase functions (`_phase_auth`,
`_phase_handshake_plaintext`, `_phase_enable_enc1103`, `_phase_media_offer`).
The orchestrator `connect_and_negotiate` reads top-to-bottom as the protocol
does, with each phase responsible for one wire-defined chunk.
"""
from __future__ import annotations

import logging
import os
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from .apple import (
    APPLE_0X12_FOLLOWUP,
    APPLE_VIEWER_COMMAND_MASK, APPLE_VIEWER_OS_VER,
)
from .auth import AuthError, do_nonsrp_auth, do_srp_auth
from .enc1103 import StreamCipher
from .offers import create_offers, extract_canvas_dims
from .rfb import (
    HP_ENCODINGS_FULL,
    build_post_encryption_toggle, build_set_encodings,
    build_viewer_info, build_virtual_display,
    do_protocol_handshake, recv_exact,
)
from .srtp import SRTP_KEY_BLOB_LEN, SRTPDecryptor


log = logging.getLogger(__name__)


# Inter-phase timing. Each delay is between two specific wire events; the
# values came from observing how long Apple's Screen Sharing.app waits between the same
# pair of events. Tuning down breaks the handshake on slow / lossy links;
# tuning up just makes connect feel sluggish.
_POST_VIEWERINFO_SETTLE_S = 0.1
_POST_TOGGLE_SETTLE_S = 0.2

# Degenerate-canvas retry behaviour: if the encoder hasn't populated the
# media blob yet (common right after a server-side agent transition), we
# re-send the 0x1c on the same TCP rather than forcing a reconnect.
_DEGENERATE_RETRY_INTERVAL_S = 0.2
_DEGENERATE_RETRY_LIMIT = 16  # bumped to compensate for shorter interval


# ── public data carriers ─────────────────────────────────────────────

@dataclass(slots=True)
class AdvertiseDims:
    """Display geometry advertised in the 0x1d VirtualDisplay message.

    Core uses a safe universal default. The CLI's `--advertise WxH[@HIDPI]`
    flag and the connect-prompt resolution picker both construct this
    with explicit values. Keeping it purely declarative means the
    protocol layer has no glfw / display-server dependency.
    """
    width: int = 1920
    height: int = 1200
    hidpi_scale: int = 1
    width_mm: float = 300.0
    height_mm: float = 200.0


@dataclass(slots=True)
class NegotiationKeys:
    """SRTP master keys exchanged in the 0x1c message. Each blob is 32 B
    cipher key + 14 B salt = 46 B; pass to `SRTP*.from_blob` rather than
    slicing manually."""
    audio_key_v: bytes  # akv (audio, viewer→server)
    audio_key_s: bytes  # aks (audio, server→viewer)
    video_key_v: bytes  # vkv (video, viewer→server)
    video_key_s: bytes  # vks (video, server→viewer)


@dataclass
class NegotiationResult:
    """Returned by `connect_and_negotiate`. The session driver uses these to
    bring up UDP receivers and SRTP decoders."""
    sock: socket.socket
    cipher: StreamCipher
    keys: NegotiationKeys
    server_width: int
    server_height: int
    canvas_width: int
    canvas_height: int
    canvas_tiles: int
    video_decryptor: SRTPDecryptor


# ── 0x1c message ──────────────────────────────────────────────────────

def random_negotiation_keys() -> NegotiationKeys:
    """Fresh SRTP master+salt blobs for each direction."""
    return NegotiationKeys(
        audio_key_v=os.urandom(SRTP_KEY_BLOB_LEN),
        audio_key_s=os.urandom(SRTP_KEY_BLOB_LEN),
        video_key_v=os.urandom(SRTP_KEY_BLOB_LEN),
        video_key_s=os.urandom(SRTP_KEY_BLOB_LEN),
    )


def build_0x1c(
    audio_offer: bytes, video_offer: bytes, keys: NegotiationKeys,
    *, alt_session: bool = False,
) -> bytes:
    """Construct the 0x1c MediaStreamConfiguration message — matches
    AVConference's `_buildOfferMessage` byte-for-byte.

    Layout (little-endian sizes are u16; offsets are absolute into `buf`):

        +0x00   u8           msg type = 0x1c
        +0x02   u16 BE       MS = audio_size + video_size + 0xd8
        +0x04   u16 BE       version = 3
        +0x06   u32 BE       reserved = 3
        +0x0a   u16 BE       audio_size
        +0x0c   u16 BE       video_size
        +0x14   16 bytes     CallID UUID
        +0x24   46 bytes     audio_key_v (akv)
        +0x52   46 bytes     audio_key_s (aks)
        +0x80   audio_size   audio offer plist
        + var   46 bytes     video_key_v (vkv)
        + 0x2e  46 bytes     video_key_s (vks)
        + 0x5c  video_size   video offer plist
    """
    AS = len(audio_offer)
    VS = len(video_offer)
    MS = AS + VS + 0xD8

    buf = bytearray(MS + 4)
    buf[0] = 0x1C
    struct.pack_into(">H", buf, 2, MS)
    struct.pack_into(">H", buf, 4, 3)
    # body[6..9] = config bitmask passed to the agent's
    # SetServerStreamConfiguration_rpc:
    #   bit 0 = 60 fps stream 1 (always on)
    #   bit 1 = 60 fps stream 2 (extra quality tier)
    #   bit 2 = "no cursor in framebuffer" — agent stops baking the
    #           system cursor texture into encoded frames. iss's host-OS
    #           cursor floats over the wgpu/canvas naturally so we don't
    #           need it baked in.
    # Without bit 2 on the cmd=2 alt-session path, the agent bakes the
    # alt-user's cursor into the framebuffer at top-left and never
    # updates it — visible as a cursor stuck in the corner of every
    # tile. We set bit 2 only for alt-session because the standard
    # console-user path's cursor is expected to update via the encoder
    # (the local mouse drives it).
    config_flags = 3
    if alt_session:
        config_flags = (config_flags & ~2) | 4   # drop bit 1, set bit 2
    struct.pack_into(">I", buf, 6, config_flags)
    struct.pack_into(">H", buf, 10, AS)
    struct.pack_into(">H", buf, 12, VS)
    buf[0x14:0x24] = uuid.uuid4().bytes
    buf[0x24:0x52] = keys.audio_key_v
    buf[0x52:0x80] = keys.audio_key_s
    buf[0x80:0x80 + AS] = audio_offer
    vo = 0x80 + AS
    buf[vo:vo + 0x2E] = keys.video_key_v
    buf[vo + 0x2E:vo + 0x5C] = keys.video_key_s
    buf[vo + 0x5C:vo + 0x5C + VS] = video_offer
    return bytes(buf)


# ── phase 1: open + auth ─────────────────────────────────────────────

_AuthFunc = Callable[[socket.socket, str, str], bytes]


def _open_socket_and_handshake(host: str, port: int) -> socket.socket:
    """TCP connect + RFB version + security-types preamble. Leaves the socket
    positioned right before c2s1."""
    sock = socket.create_connection((host, port), timeout=15)
    sock.settimeout(15)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    do_protocol_handshake(sock)
    return sock


def _phase_auth(
    host: str, port: int, username: str, password: str, mode: str,
) -> tuple[socket.socket, bytes]:
    """Open TCP, run auth (with fallback on AuthError), return live socket
    and the 16-byte enc1103 master key."""
    primary, fallback = (
        (do_srp_auth, do_nonsrp_auth) if mode == "srp"
        else (do_nonsrp_auth, do_srp_auth)
    )

    sock = _open_socket_and_handshake(host, port)
    try:
        return sock, primary(sock, username, password)
    except AuthError as e:
        log.warning(
            "%s failed (%s); falling back to %s",
            primary.__name__, e, fallback.__name__,
        )
        sock.close()

    sock = _open_socket_and_handshake(host, port)
    return sock, fallback(sock, username, password)


# ── phase 2: ClientInit + ServerInit ──────────────────────────────────

def _phase_client_init(sock: socket.socket) -> tuple[int, int]:
    """Send ClientInit (0xc1, shared-flag), read ServerInit (24-byte fixed
    header + variable-length name). Returns (server_width, server_height)."""
    sock.sendall(b"\xc1")
    head = recv_exact(sock, 24)
    name_len = struct.unpack(">I", head[20:24])[0]
    if name_len:
        recv_exact(sock, name_len)
    w, h = struct.unpack(">HH", head[0:4])
    log.info("ServerInit: %dx%d", w, h)
    return w, h


# ── phase 2b: SessionSelect (cmd=0 share-console / cmd=2 alt-session) ────

# Magic at body[0..3] of the 76-byte SessionSelect_SessionInfo struct that
# the daemon appends to ServerInit when auth user ≠ console user. We peek
# 4 bytes after ServerInit and compare against this; presence is how we
# know we're in SessionSelect state.
_SS_MAGIC = b"\x00\x4a\x00\x01"
# How long the daemon will park waiting for either the SessionSelect-ack handshake
# or, in the cmd=0 share-console path, the human click on the
# GuestAccessAskUser popup.
_SS_POPUP_TIMEOUT_S = 60.0


def _build_ss_cmd0_body(console_user: str) -> bytes:
    """Build the 72-byte SessionSelect cmd=0 body. Triggers
    `SSAgent_GuestAccessAskUser_rpc` on the console user's screen — they
    click Allow → this viewer joins their session in observe-only mode."""
    machine = b"\x00"
    body = (
        struct.pack(">HH", 2, 0x0000)
        + b"\x00\x00"
        + bytes([0]) + b"\x00"             # cmd byte at body[6] = 0
        + console_user.encode() + b"\x00"
        + machine
    )
    return body.ljust(72, b"\x00")[:72]


def _build_ss_cmd2_body(
    username: str, password: str, ecb_key: bytes,
) -> bytes:
    """Build the 200-byte SessionSelect cmd=2 body with AES-128-ECB-
    encrypted username + password.

    Apple's daemon checks `body[0..1]` (BE u16) is `>= 2` — we send 2.
    If that and the other gates all pass, the daemon decrypts
    username (body+0x48, 64 B) and password (body+0x88, 64 B) under
    the SRP-K-derived `ecb_key` cryptor, copies the plaintext into
    its viewer state, and clears the `createLoginWindow` flag —
    auto-logging the alt-user without a fresh loginwindow stub.

    Layout (200 bytes):
        +0..1     u16 BE  version = 2
        +2..3     u16 BE  reserved = 0
        +4..5     2 B     reserved
        +6        u8      cmd = 2
        +7        u8      pad
        +8..0x47  56 B    reserved (zero)
        +0x48..0x87  64 B AES-ECB encrypted USERNAME
        +0x88..0xc7  64 B AES-ECB encrypted PASSWORD
    """
    from Crypto.Cipher import AES  # noqa: PLC0415
    aes = AES.new(ecb_key[:16], AES.MODE_ECB)
    user_buf = (username.encode("utf-8") + b"\x00" * 64)[:64]
    pass_buf = (password.encode("utf-8") + b"\x00" * 64)[:64]
    enc_user = aes.encrypt(user_buf)
    enc_pass = aes.encrypt(pass_buf)
    body = (
        struct.pack(">HH", 2, 0x0000)      # body[0..3]
        + b"\x00\x00"                      # body[4..5]
        + bytes([2]) + b"\x00"             # body[6..7]: cmd=2 + pad
        + b"\x00" * 64                     # body[8..0x47] (= 0x48 − 8)
        + enc_user                         # body[0x48..0x87]
        + enc_pass                         # body[0x88..0xc7]
    )
    assert len(body) == 200, len(body)
    return body


def _read_ss_prompt(sock: socket.socket) -> Optional[tuple[int, str]]:
    """Peek 4 bytes after ServerInit. If they're the SessionSelect magic, read the
    rest of the 76-byte prompt and return `(flags, console_user)`. If
    not, log a warning and return None — those 4 bytes are then lost
    (the caller has already moved past them and there's no way to push
    them back into Python's socket recv buffer)."""
    sock.settimeout(0.5)
    try:
        peek = recv_exact(sock, 4)
    except (TimeoutError, socket.timeout, OSError):
        sock.settimeout(15)
        return None
    if peek != _SS_MAGIC:
        sock.settimeout(15)
        log.warning(
            "session-select: magic missing (got %s); first 4 bytes after "
            "ServerInit will be lost", peek.hex(),
        )
        return None
    sock.settimeout(15)
    rest = recv_exact(sock, 72)
    ssi = peek + rest
    flags = struct.unpack(">I", ssi[4:8])[0]
    console_user = ssi[0xc:].split(b"\x00", 1)[0].decode("utf-8", "replace")
    return flags, console_user


def _phase_session_select(
    sock: socket.socket,
    *,
    mode: str,            # "share_console" or "alt_session"
    username: str,
    password: str,
    ecb_key: bytes,
    ss_wait: float = _SS_POPUP_TIMEOUT_S,
) -> bool:
    """Detect + handle the SessionSelect prompt. Returns True if a prompt
    was processed, False if the server didn't emit one (caller continues
    with the normal handshake).

    For `mode="share_console"` (cmd=0): fires the popup at the console
    user, blocks until they click Allow.

    For `mode="alt_session"` (cmd=2 with encrypted creds): tells the
    daemon to auto-log the auth user into a fresh virtual display. No
    popup; the daemon decrypts the creds we just sent, strlcpys them
    into the viewer struct, and clears `createLoginWindow`.

    Either way, after the SessionSelect-ack lands the caller can
    continue with the normal post-toggle handshake — except cmd=2
    also needs the alt-user-login variant of the SetDisplayConfiguration
    (`build_virtual_display(..., alt_user_login=True)`), otherwise the
    daemon's encoder targets the console user's screen instead of our
    newly-spawned alt-vdisplay.
    """
    prompt = _read_ss_prompt(sock)
    if prompt is None:
        log.warning(
            "session-select: no prompt — server may not be in "
            "session-select state (auth user matches console, or "
            "VNCSelectSession off)"
        )
        return False
    flags, console_user = prompt
    log.info(
        "session-select: prompt — flags=0x%x console_user=%r mode=%s",
        flags, console_user, mode,
    )

    if mode == "share_console":
        cmd, body = 0, _build_ss_cmd0_body(console_user)
    elif mode == "alt_session":
        cmd, body = 2, _build_ss_cmd2_body(username, password, ecb_key)
    else:
        raise ValueError(f"unknown SessionSelect mode: {mode!r}")
    if not (flags >> cmd) & 1:
        log.warning(
            "session-select: cmd=%d not in flags mask 0x%x; daemon will "
            "likely reject", cmd, flags,
        )

    msg = struct.pack(">H", len(body)) + body
    sock.sendall(msg)
    log.info(
        "session-select: sent cmd=%d (%dB) — waiting up to %.0fs for ack",
        cmd, len(msg), ss_wait,
    )

    sock.settimeout(ss_wait)
    try:
        ack = sock.recv(65536)
    except socket.timeout:
        sock.settimeout(15)
        raise RuntimeError(
            f"session-select: timed out after {ss_wait:.0f}s waiting for "
            f"server ack — for cmd=0 the popup was likely never clicked, "
            f"for cmd=2 the daemon rejected the creds"
        )
    sock.settimeout(15)
    if not ack:
        raise RuntimeError(
            "session-select: server closed before responding — popup "
            "denied or daemon rejected the cmd"
        )
    log.info(
        "session-select: ack received (%dB) — proceeding with normal "
        "post-toggle handshake", len(ack),
    )
    return True


# ── phase 3: pre-toggle plaintext + enc1103 setup ─────────────────────

def _phase_handshake_plaintext(
    sock: socket.socket, advertise: AdvertiseDims, hdr: bool, curtain: bool,
) -> None:
    """Send the messages that must arrive plaintext, before enc1103 takes
    effect: ViewerInfo (+ 0x12 / 0x0a Apple follow-ups), the optional
    VirtualDisplay (skipped when `curtain=False`), and the first
    SetEncodings. Returns nothing — purely sends.

    The VirtualDisplay (msg 0x1d SetDisplayConfiguration) is what
    flips the daemon into virtual-framebuffer mode and engages the
    SkyLight curtain on the host. Skipping it makes the daemon stream
    the actual physical display — both the viewer and anyone at the
    host see the same thing."""
    sock.sendall(
        build_viewer_info(
            app_id=2, app_ver=(6, 1, 0),
            os_ver=APPLE_VIEWER_OS_VER,
            command_mask=APPLE_VIEWER_COMMAND_MASK,
            extra=b"",
        )
        + APPLE_0X12_FOLLOWUP
    )
    time.sleep(_POST_VIEWERINFO_SETTLE_S)

    if curtain:
        sock.sendall(build_virtual_display(
            width=advertise.width,
            height=advertise.height,
            hidpi_scale=advertise.hidpi_scale,
            width_mm=advertise.width_mm,
            height_mm=advertise.height_mm,
            hdr=hdr,
        ))
    else:
        log.info("curtain=off — skipping SetDisplayConfiguration; "
                 "host's physical screen will mirror the stream")
    sock.sendall(build_set_encodings(HP_ENCODINGS_FULL))


def _phase_enable_enc1103(
    sock: socket.socket, ecb_key: bytes, *, first_byte_timeout: float = 10.0,
) -> StreamCipher:
    """Drain framebuffer-update messages until we see the 1103 encoding entry,
    build a `StreamCipher` from its 36-byte body, then send the
    PostEncryptionToggle and drain pending traffic so the cipher's receive
    counter stays aligned with the server."""
    cipher = _read_until_enc1103(sock, ecb_key, first_byte_timeout=first_byte_timeout)
    sock.sendall(build_post_encryption_toggle())
    time.sleep(_POST_TOGGLE_SETTLE_S)
    _drain_through_cipher(sock, cipher, timeout=0.5)
    return cipher


# ── phase 4: encrypted media offer + answer ───────────────────────────

def _phase_media_offer(
    sock: socket.socket, cipher: StreamCipher,
    audio_offer: bytes, video_offer: bytes,
    *, alt_session: bool = False,
) -> tuple[NegotiationKeys, tuple[int, int, int]]:
    """Send the encrypted second SetEncodings + 0x1c, read the answer. On
    degenerate canvas (encoder still warming up after an agent transition),
    re-query up to `_DEGENERATE_RETRY_LIMIT` times. Returns (keys, canvas)."""
    sock.sendall(cipher.encrypt_message(build_set_encodings(HP_ENCODINGS_FULL)))

    keys = random_negotiation_keys()
    msg_1c = build_0x1c(audio_offer, video_offer, keys, alt_session=alt_session)
    sock.sendall(cipher.encrypt_message(msg_1c))
    log.info("0x1c sent (encrypted)")

    canvas = _read_video_answer(sock, cipher)
    if canvas[0] and canvas[1]:
        return keys, canvas

    for attempt in range(_DEGENERATE_RETRY_LIMIT):
        time.sleep(_DEGENERATE_RETRY_INTERVAL_S)
        try:
            sock.sendall(cipher.encrypt_message(msg_1c))
        except OSError as e:
            log.warning("0x1c re-query send failed: %s", e)
            break
        log.info(
            "0x1c re-query %d/%d (degenerate canvas)",
            attempt + 1, _DEGENERATE_RETRY_LIMIT,
        )
        canvas = _read_video_answer(sock, cipher)
        if canvas[0] and canvas[1]:
            break
    return keys, canvas


# ── orchestrator ─────────────────────────────────────────────────────

def connect_and_negotiate(
    host: str,
    port: int,
    username: str,
    password: str,
    *,
    auth_mode: str = "srp",
    advertise: Optional[AdvertiseDims] = None,
    hdr: bool = False,
    curtain: bool = True,
    audio_offer: Optional[bytes] = None,
    video_offer: Optional[bytes] = None,
    share_console: bool = False,
    alt_session: bool = False,
) -> NegotiationResult:
    """Drive the full handshake. The returned socket is in encrypted-RFB
    mode; subsequent control-channel sends must wrap with
    `result.cipher.encrypt_message(...)`.

    `share_console` and `alt_session` are mutually exclusive
    SessionSelect modes; both kick in when the auth user differs from
    the host's console user. The daemon won't show a SessionSelect
    prompt if they're the same user, so these flags are no-ops in that
    case.

    `curtain=True` (default) sends the VirtualDisplay message that
    blanks the host's physical screen. `curtain=False` skips it so the
    physical screen mirrors the stream. The cmd=2 alt-session path
    bypasses both: it always sends Apple's canned SDC instead, which
    creates the alt-user vdisplay without curtaining the console user.
    """
    if share_console and alt_session:
        raise ValueError("share_console and alt_session are mutually exclusive")
    if audio_offer is None or video_offer is None:
        v_off, a_off = create_offers()
        audio_offer = audio_offer or a_off
        video_offer = video_offer or v_off

    advertise = advertise or AdvertiseDims()

    sock, ecb_key = _phase_auth(host, port, username, password, auth_mode)
    server_w, server_h = _phase_client_init(sock)

    if share_console or alt_session:
        # Apple Screen Sharing's two-TCP convention for the cmd=2 alt-session
        # path: open conn1, do SRP, see the SessionSelect prompt, close
        # conn1, then open a fresh conn2 and redo SRP from scratch
        # before sending cmd=2. The point is to avoid daemon-side state
        # bleed where conn1's SRP-K-derived cryptor sticks around and
        # confuses subsequent decryption. In practice: console user
        # stays uncurtained and display routing doesn't combine. We do
        # the same here.
        # cmd=0 share-console only needs one TCP — no encrypted creds
        # to worry about — so it skips this dance.
        if alt_session:
            log.info("alt-session: closing conn1, opening conn2 for cmd=2")
            try:
                sock.close()
            except Exception:
                pass
            sock, ecb_key = _phase_auth(host, port, username, password, auth_mode)
            server_w, server_h = _phase_client_init(sock)

        # Read the (re-)issued SessionSelect prompt and send our cmd
        # response. For cmd=2 the body carries
        # AES-128-ECB(ecb_key, username/password) at body+0x48/+0x88;
        # the daemon decrypts under conn2's freshly-derived ecb_key.
        # For cmd=0 it's a 72-byte plain login_info.
        mode = "alt_session" if alt_session else "share_console"
        _phase_session_select(
            sock,
            mode=mode,
            username=username, password=password, ecb_key=ecb_key,
        )

    if alt_session:
        # cmd=2 path: send a SetDisplayConfiguration with the alt-user
        # login hint set (the 0x07 byte at displayInfo+0x99 — captured
        # from Apple Screen Sharing.app traffic). Without that hint the
        # daemon's `createLoginWindow=1` branch leaves
        # `virtualDisplayCount=0` and the encoder targets the *console*
        # user's screen instead of the alt-user vdisplay we just
        # spawned. With it set, count=1 and the encoder picks up the
        # correct `displayID to capture`.
        sock.sendall(
            build_viewer_info(
                app_id=2, app_ver=(6, 1, 0),
                os_ver=APPLE_VIEWER_OS_VER,
                command_mask=APPLE_VIEWER_COMMAND_MASK,
                extra=b"",
            )
            + APPLE_0X12_FOLLOWUP
        )
        time.sleep(_POST_VIEWERINFO_SETTLE_S)
        sock.sendall(build_virtual_display(
            width=advertise.width,
            height=advertise.height,
            hidpi_scale=advertise.hidpi_scale,
            width_mm=advertise.width_mm,
            height_mm=advertise.height_mm,
            hdr=hdr,
            alt_user_login=True,
        ))
        sock.sendall(build_set_encodings(HP_ENCODINGS_FULL))
    else:
        _phase_handshake_plaintext(sock, advertise, hdr, curtain=curtain)

    cipher = _phase_enable_enc1103(sock, ecb_key)
    keys, (canvas_w, canvas_h, canvas_tiles) = _phase_media_offer(
        sock, cipher, audio_offer, video_offer,
        alt_session=alt_session,
    )

    return NegotiationResult(
        sock=sock,
        cipher=cipher,
        keys=keys,
        server_width=server_w,
        server_height=server_h,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        canvas_tiles=canvas_tiles,
        video_decryptor=SRTPDecryptor.from_blob(keys.video_key_s),
    )


# ── helpers ──────────────────────────────────────────────────────────

def _read_until_enc1103(
    sock: socket.socket, ecb_key: bytes, *, first_byte_timeout: float = 10.0,
) -> StreamCipher:
    """Buffer recv until we see a 1103 encoding entry, then construct a
    `StreamCipher` from its 36-byte body. Tolerates 0x14 UserSessionChanged
    notifications that can interleave with the ServerInit reply burst.

    The default `first_byte_timeout` of 10 s covers a normal handshake.
    Share-console / alt-session flows wait for the SessionSelect prompt
    earlier in `_phase_session_select`, so by the time we get here the
    connection is already past any human-confirm gate.
    """
    sock.settimeout(1.0)
    init = bytearray()
    deadline = time.monotonic() + first_byte_timeout
    while time.monotonic() < deadline:
        try:
            chunk = sock.recv(65536)
            if not chunk:
                break
            init += chunk
        except socket.timeout:
            if init:
                break
    sock.settimeout(15)

    pos = 0
    cipher: Optional[StreamCipher] = None
    while pos < len(init):
        b = init[pos]
        if b == 0x14 and pos + 8 <= len(init):
            pos += 8
            continue
        if b == 0x00 and pos + 4 <= len(init):
            n_rects = struct.unpack(">H", init[pos + 2:pos + 4])[0]
            p = pos + 4
            for _ in range(n_rects):
                if p + 12 > len(init):
                    break
                enc = struct.unpack(">i", init[p + 8:p + 12])[0]
                p += 12
                if enc == 1103 and p + 36 <= len(init):
                    cipher = StreamCipher(bytes(init[p:p + 36]), ecb_key=ecb_key)
                    p += 36
                elif enc in (1010, 1011) and p + 2 <= len(init):
                    sz = struct.unpack(">H", init[p:p + 2])[0]
                    p += 2 + sz
                else:
                    break
            pos = p
            break
        break

    if cipher is None:
        sock.close()
        raise RuntimeError(
            "server did not advertise enc1103. Most often this is rate-limiting "
            "on the server — wait 10-15 seconds and retry."
        )
    log.info("enc1103 OK")

    if pos < len(init):
        cipher.decrypt_stream(bytes(init[pos:]))  # discard tail; we only want the side-effect of advancing the recv counter
    return cipher


def _drain_through_cipher(
    sock: socket.socket, cipher: StreamCipher, *, timeout: float,
) -> None:
    """Read whatever's queued on the socket and feed it through `decrypt_stream`
    so the cipher's receive counter stays aligned with the server."""
    sock.settimeout(timeout)
    pre = bytearray()
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            pre += chunk
    except socket.timeout:
        pass
    sock.settimeout(15)
    if pre:
        cipher.decrypt_stream(bytes(pre))  # discard tail; we only want the side-effect of advancing the recv counter


def _read_video_answer(
    sock: socket.socket, cipher: StreamCipher,
) -> tuple[int, int, int]:
    """Read + decrypt the server's 0x1c answer; return (canvas_w, canvas_h,
    canvas_tiles). Returns zeros if the answer hasn't arrived or the encoder
    sent a degenerate canvas — caller should retry."""
    sock.settimeout(5.0)
    answer = bytearray()
    try:
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            answer += chunk
            if len(answer) > 100:
                break
    except socket.timeout:
        pass
    sock.settimeout(15)

    if not answer:
        return 0, 0, 0
    msgs, _ = cipher.decrypt_stream(bytes(answer))
    for msg in msgs:
        log.debug("0x1c answer msg cmd=0x%02x len=%d", msg[0], len(msg))
        cw, ch, ct = extract_canvas_dims(msg)
        if cw and ch:
            log.info("encoder canvas: %dx%d (%d tiles)", cw, ch, ct)
            return cw, ch, ct
    return 0, 0, 0


__all__ = [
    "AdvertiseDims",
    "NegotiationKeys",
    "NegotiationResult",
    "build_0x1c",
    "connect_and_negotiate",
    "random_negotiation_keys",
]
