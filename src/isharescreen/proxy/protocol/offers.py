"""AVCMediaStreamNegotiator offer build + answer parse.

The negotiation protobuf shape is reverse-engineered from
`-[AVCMediaStreamNegotiator createOffer]`. We rebuild it from scratch in
pure Python so the package works on Linux and Windows (Apple's framework
is macOS-only). Output is byte-identical to AVConference modulo three
per-call dynamic fields: session_id, timestamp, and the CallID UUID.
"""
from __future__ import annotations

import logging
import platform
import plistlib
import secrets
import time
import uuid
import zlib
from typing import Optional

from ... import __version__


log = logging.getLogger(__name__)


# ── protobuf helpers ──────────────────────────────────────────────────

def _varint(v: int) -> bytes:
    out = bytearray()
    while v > 0x7F:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v & 0x7F)
    return bytes(out)


def _field_varint(field_num: int, value: int) -> bytes:
    return _varint((field_num << 3) | 0) + _varint(value)


def _field_bytes(field_num: int, value: bytes) -> bytes:
    return _varint((field_num << 3) | 2) + _varint(len(value)) + value


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return val, pos


# ── audio f9 codec entries ────────────────────────────────────────────

# Apple's canonical f9 audio-config tier list. f1 is the entry kind:
#   0       primary tier with a network bitrate cap (f2 = bps, f3 = buffer cap)
#   16/4/1  codec-specific markers (CELT-NB, SILK, etc.)
#   4074    header marker
# 75M and 100M are HP tiers Sequoia's stock AVConference omits; we always
# emit them so the server picks an HP tier.
_AUDIO_F9_TIERS: tuple[tuple[int, int, Optional[int]], ...] = (
    (0,    40_000_000,  12288),       # 40M
    (0,     6_000_000, 131072),       # 6M
    (4074,        0,    16384),       # header marker
    (16,        4100,    None),       # CELT-NB 4100
    (0,    75_000_000, 524288),       # 75M CELT-FB ← HP tier
    (0,    20_000_000,  98304),       # 20M
    (4,         6500,    None),       # SILK 6500
    (0,    60_000_000, 262144),       # 60M
    (1,          299,    None),       # 299
    (0,   100_000_000, 1048576),      # 100M CELT-FB ← HP tier
)


def _build_audio_f9_entry(f1: int, f2: int, f3: Optional[int]) -> bytes:
    body = b"\x08" + _varint(f1) + b"\x10" + _varint(f2)
    if f3 is not None:
        body += b"\x18" + _varint(f3)
    return b"\x4a" + _varint(len(body)) + body


_APPLE_AUDIO_F9 = b"".join(_build_audio_f9_entry(*t) for t in _AUDIO_F9_TIERS)


# ── HEVC + AVC parameter strings ──────────────────────────────────────

_HEVC_PARAMS = (
    b"FLS;MS:-1;LF:-1;LTR;CABAC;POS:0;EOD:1;HTS:2;RR:3;"
    b"AR:16/9,5/8;XR:16/9,5/8;"
)
_AVC_PARAMS = (
    b"FLS;LF:-1;POS:5;EOD:1;HTS:2;RR:3;POSE:4;"
    b"AR:16/9,5/8;XR:16/9,5/8;"
)


def _build_remote_endpoint_info() -> bytes:
    """RemoteEndpointInfo protobuf: derive hw_model/os_build at runtime so we
    don't masquerade as the recording host. AVConference treats this as
    informational; populates opportunistically."""
    hw_model = "Generic"
    avc_version = "1.0.0"
    os_build = "0"
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            import subprocess
            hw_model = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], text=True, timeout=1
            ).strip() or hw_model
            os_build = subprocess.check_output(
                ["sw_vers", "-buildVersion"], text=True, timeout=1
            ).strip() or os_build
        elif sys_name in ("Linux", "Windows"):
            hw_model = f"{sys_name}-{platform.machine()}"
            os_build = platform.release()
    except Exception as e:
        log.debug("RemoteEndpointInfo probe failed: %s", e)

    def _str(tag: int, s: str) -> bytes:
        b = s.encode("utf-8")[:127]
        return bytes([tag, len(b)]) + b

    return (
        b"\x08\x00"        # f1 = 0
        + b"\x10\x01"      # f2 = 1
        + _str(0x1A, hw_model)
        + _str(0x22, avc_version)
        + _str(0x2A, os_build)
    )


_REMOTE_ENDPOINT_INFO = _build_remote_endpoint_info()


# ── offer construction ───────────────────────────────────────────────

def _build_mediablob(mode: int, session_id: int, timestamp: int) -> bytes:
    """Build the MediaBlob protobuf. mode 7 = video, mode 8 = audio. Output
    matches Apple's createOffer modulo the dynamic fields."""
    if mode == 7:
        res_entry = _field_varint(1, 1) + _field_varint(2, 1) + _field_varint(3, 50115) + _field_varint(4, 0)
        res_entry_alt = _field_varint(1, 1) + _field_varint(2, 2) + _field_varint(3, 50115) + _field_varint(4, 0)
        hevc_bank = (
            _field_varint(1, 123)
            + _field_bytes(2, res_entry) + _field_bytes(2, res_entry_alt)
            + _field_bytes(2, res_entry) + _field_bytes(2, res_entry_alt)
            + _field_bytes(3, _HEVC_PARAMS)
            + _field_varint(4, 1)
        )
        avc_bank = (
            _field_varint(1, 100)
            + _field_bytes(2, res_entry) + _field_bytes(2, res_entry_alt)
            + _field_bytes(3, _AVC_PARAMS)
            + _field_varint(4, 14)
        )
        desc = (
            _field_varint(1, session_id) + _field_varint(2, 0)
            + _field_bytes(3, hevc_bank) + _field_bytes(3, avc_bank)
            + _field_varint(6, 4) + _field_varint(7, 1) + _field_varint(8, 63)
            + _field_varint(9, 1) + _field_varint(12, 1)
        )
        desc_field = _field_bytes(5, desc)
    elif mode == 8:
        desc = (
            _field_varint(1, session_id) + _field_varint(2, 0)
            + _field_varint(3, 0) + _field_varint(4, 24191)
            + _field_varint(5, 0) + _field_varint(6, 0)
        )
        desc_field = _field_bytes(3, desc)
    else:
        raise ValueError(f"unsupported negotiation mode {mode}")

    return (
        _field_varint(1, 1) + _field_varint(2, 1)
        + desc_field
        + _field_bytes(6, f"iShareScreen {__version__}".encode("ascii"))
        + _field_varint(8, 0)
        + _APPLE_AUDIO_F9
        + _field_varint(13, timestamp)
        + _field_varint(14, 2) + _field_varint(16, 0) + _field_varint(18, 1)
    )


def create_offers() -> tuple[bytes, bytes]:
    """Generate fresh (video, audio) offer plists. Each call produces a new
    session_id, timestamp, and CallID UUID."""

    def _plist(mode: int) -> bytes:
        session_id = secrets.randbits(32)
        timestamp = time.time_ns()
        blob = _build_mediablob(mode, session_id, timestamp)
        plist = {
            "avcMediaStreamOptionRemoteEndpointInfo": _REMOTE_ENDPOINT_INFO,
            "avcMediaStreamNegotiatorMode": mode,
            "avcMediaStreamNegotiatorMediaBlob": zlib.compress(blob),
            "avcMediaStreamOptionCallID": str(uuid.uuid4()).upper(),
        }
        return plistlib.dumps(plist, fmt=plistlib.FMT_BINARY)

    return _plist(7), _plist(8)


# ── answer parsing ────────────────────────────────────────────────────

def extract_offer_ssrc(offer_plist: bytes, *, is_video: bool) -> Optional[int]:
    """Pull our advertised SSRC from a freshly-built offer.

    AVConference accepts only RTCP/RTP from the SSRC we negotiated; using a
    different one (e.g. 0x00DECADE) silently triggers `noRemotePacketsTimeout`
    after ~30 s. Video SSRC lives in field 5→1; audio in field 3→1.
    """
    plist = plistlib.loads(offer_plist)
    blob = zlib.decompress(plist["avcMediaStreamNegotiatorMediaBlob"])
    target = 5 if is_video else 3
    pos = 0
    while pos < len(blob):
        tag, pos = _read_varint(blob, pos)
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            _, pos = _read_varint(blob, pos)
        elif wt == 2:
            length, pos = _read_varint(blob, pos)
            if fn == target:
                inner = blob[pos:pos + length]
                ipos = 0
                inner_tag, ipos = _read_varint(inner, ipos)
                if (inner_tag & 7) == 0 and (inner_tag >> 3) == 1:
                    ssrc, _ = _read_varint(inner, ipos)
                    return ssrc & 0xFFFFFFFF
            pos += length
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            break
    return None


def extract_canvas_dims(answer_msg: bytes) -> tuple[int, int, int]:
    """Pull (canvas_w, canvas_h, num_tiles) from the server's 0x1c answer.

    The video media-stream answer's protobuf sits inside an embedded bplist.
    Top-level F5 carries the video config, with F4=canvas_width,
    F5=canvas_height (luma samples) and F6=tile_count. Returns zeros if not
    found — the caller should treat that as "encoder not ready, retry".
    """
    if not answer_msg or answer_msg[0] != 0x00:
        return 0, 0, 0
    idx = 0
    while True:
        idx = answer_msg.find(b"bplist", idx)
        if idx < 0:
            return 0, 0, 0
        plist_obj = None
        for end in range(idx + 1, len(answer_msg) + 1, 2):
            try:
                plist_obj = plistlib.loads(answer_msg[idx:end])
                break
            except Exception:
                plist_obj = None
        if not isinstance(plist_obj, dict):
            idx += 6
            continue
        blob = plist_obj.get("avcMediaStreamNegotiatorMediaBlob")
        if not blob:
            idx += 6
            continue
        try:
            dec = zlib.decompress(blob)
        except Exception:
            idx += 6
            continue
        cw = ch = ct = 0
        pos = 0
        while pos < len(dec):
            tag, pos = _read_varint(dec, pos)
            fn = tag >> 3
            wt = tag & 7
            if wt == 0:
                _, pos = _read_varint(dec, pos)
            elif wt == 2:
                ln, pos = _read_varint(dec, pos)
                if fn == 5:
                    sub = dec[pos:pos + ln]
                    sp = 0
                    while sp < len(sub):
                        st, sp = _read_varint(sub, sp)
                        sf = st >> 3
                        sw = st & 7
                        if sw == 0:
                            v, sp = _read_varint(sub, sp)
                            if sf == 4:
                                cw = v
                            elif sf == 5:
                                ch = v
                            elif sf == 6:
                                ct = v
                        elif sw == 2:
                            sl, sp = _read_varint(sub, sp)
                            sp += sl
                        elif sw == 1:
                            sp += 8
                        elif sw == 5:
                            sp += 4
                        else:
                            break
                pos += ln
            elif wt == 1:
                pos += 8
            elif wt == 5:
                pos += 4
            else:
                break
        if cw and ch:
            return cw, ch, ct
        idx += 6


__all__ = ["create_offers", "extract_canvas_dims", "extract_offer_ssrc"]
