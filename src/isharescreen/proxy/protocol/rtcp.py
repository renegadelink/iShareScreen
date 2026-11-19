"""RTCP feedback packet builders and parsers.

The host is an AVConference RTCP peer that expects standard SR/RR/PLI/FIR/NACK
packets. We only ever build receiver-side feedback (we're never the encoder),
plus an empty SR so the AVConference liveness check accepts us as a sender.

Packet types used here:
    PT=200  Sender Report
    PT=201  Receiver Report
    PT=204  Application-defined (Apple's LTR-ACK — subtype "5"; rate-control
            feedback — name "RCTL")
    PT=205  Transport-layer feedback (NACK = FMT 1)
    PT=206  Payload-specific feedback (PLI = FMT 1, FIR = FMT 4)
"""
from __future__ import annotations

import struct
import time
from typing import Iterable, Mapping, Optional


_NTP_EPOCH_DELTA = 2208988800  # seconds between 1900-01-01 and 1970-01-01


def build_fir(sender_ssrc: int, target_ssrc: int, seq_nr: int) -> bytes:
    """Full Intra Request (RFC 5104 §4.3.1.1). Forces an IDR on `target_ssrc`."""
    return (
        struct.pack(">BBH", 0x80 | 4, 206, 4)
        + struct.pack(">II", sender_ssrc, 0)
        + struct.pack(">I", target_ssrc)
        + struct.pack(">B3x", seq_nr & 0xFF)
    )


def build_fir_legacy(target_ssrc: int) -> bytes:
    """Legacy Full-INTRA-frame Request (RFC 2032 §5.2.1, PT=192). This is the
    keyframe request the *native* Screen Sharing viewer sends; Apple's
    screensharingd answers it with a fresh IDR, where it often ignores the
    AVPF PT=206 FIR. 8 bytes: header + the SSRC of the source to refresh."""
    return struct.pack(">BBHI", 0x80, 192, 1, target_ssrc & 0xFFFFFFFF)


def build_pli(sender_ssrc: int, media_ssrc: int) -> bytes:
    """Picture Loss Indication (RFC 4585 §6.3.1). Lighter than FIR."""
    return (
        struct.pack(">BBH", 0x80 | 1, 206, 2)
        + struct.pack(">II", sender_ssrc, media_ssrc)
    )


def build_nack(sender_ssrc: int, media_ssrc: int, lost_seqs: Iterable[int]) -> bytes:
    """Generic NACK (RFC 4585 §6.2.1). Coalesces consecutive losses into BLP entries."""
    seqs = sorted({s & 0xFFFF for s in lost_seqs})
    if not seqs:
        return b""

    fcis = bytearray()
    i = 0
    while i < len(seqs):
        pid = seqs[i]
        blp = 0
        j = i + 1
        while j < len(seqs):
            diff = (seqs[j] - pid) & 0xFFFF
            if 1 <= diff <= 16:
                blp |= 1 << (diff - 1)
                j += 1
            else:
                break
        fcis += struct.pack(">HH", pid, blp)
        i = j

    n_fcis = len(fcis) // 4
    length_words = 2 + n_fcis  # sender_ssrc + media_ssrc + FCI list
    return (
        struct.pack(">BBH", 0x80 | 1, 205, length_words)
        + struct.pack(">II", sender_ssrc, media_ssrc)
        + bytes(fcis)
    )


def build_rtcp_app_ltrp(sender_ssrc: int, ltr_id: int) -> bytes:
    """Apple's RTCP_APP_LTRP — long-term reference picture acknowledgment.

    Wire format (16 bytes pre-SRTCP, 30 bytes post-SRTCP):
        [0]    0x80           V=2, P=0, RC/subtype=0
        [1]    0xcc           PT=204 (APP)
        [2:4]  BE u16  3      length (3 32-bit words after header)
        [4:8]  BE u32         sender SSRC
        [8:12] BE u32  5      APP "name" / subtype field
        [12:16] BE u32        LTR frame id (monotonic per decoded frame)

    The host emits ~30 of these per second during a healthy HP session; the
    server uses the last-acked LTR as the next P-frame's reference on loss,
    skipping a full fresh-IDR round-trip.
    """
    return (
        struct.pack(">BBHI", 0x80, 204, 3, sender_ssrc)
        + struct.pack(">I", 5)
        + struct.pack(">I", ltr_id & 0xFFFFFFFF)
    )


RCTL_NAME = 0x5243544C  # b"RCTL"


def build_rtcp_app_rctl(
    sender_ssrc: int,
    *,
    echo: int,
    echo_delay_ms: int,
    now_1024: int,
    bwe_kbps: int,
    owrd: int = 0,
    burst_loss: int = 0,
    loss: int = 0,
) -> bytes:
    """AVConference receiver rate-control feedback: RTCP APP named "RCTL".

    Apple's Screen Sharing viewer sends one every 50 ms on the video port;
    the host's video encoder targets the bandwidth estimate it carries.

    Wire format (32 bytes pre-SRTCP, 46 bytes post-SRTCP):
        [0]      0x80          V=2, P=0, subtype=0
        [1]      0xcc          PT=204 (APP)
        [2:4]    BE u16  7     length (7 32-bit words after header)
        [4:8]    BE u32        sender SSRC (our video SSRC)
        [8:12]   "RCTL"        APP name
        [12]     0x85          media-control info header (0x80 | flags 5)
        [13]     0x00
        [14:16]  BE u16  4     info length in words - 1 (20-byte info)
        [16:18]  BE u16        echo: low 16 bits of the newest video RTP ts
        [18:22]  0
        [22:24]  BE u16        ms since that packet arrived
        [24:26]  BE u16        receiver clock, 1/1024 s
        [26:28]  BE u16        one-way relative delay (0 = not measured)
        [28:30]  BE u16        burst_loss << 12 | loss (windowed, percent)
        [30:32]  BE u16        receiver bandwidth estimate, kbps
    """
    return (
        struct.pack(">BBHII", 0x80, 204, 7, sender_ssrc & 0xFFFFFFFF, RCTL_NAME)
        + struct.pack(">BBH", 0x85, 0, 4)
        + struct.pack(">HI", echo & 0xFFFF, 0)
        + struct.pack(
            ">HHHHH",
            min(max(echo_delay_ms, 0), 0xFFFF),
            now_1024 & 0xFFFF,
            owrd & 0xFFFF,
            (min(max(burst_loss, 0), 15) << 12) | (min(max(loss, 0), 0xFFF)),
            min(max(bwe_kbps, 0), 0xFFFF),
        )
    )


def build_empty_sr(sender_ssrc: int) -> bytes:
    """Empty Sender Report so AVConference accepts us as a live sender."""
    now = time.time()
    ntp_sec = int(now) + _NTP_EPOCH_DELTA
    ntp_frac = int((now - int(now)) * (1 << 32)) & 0xFFFFFFFF
    rtp_ts = int(now * 90000) & 0xFFFFFFFF
    return (
        struct.pack(">BBH", 0x80, 200, 6)
        + struct.pack(">IIIIII", sender_ssrc, ntp_sec, ntp_frac, rtp_ts, 0, 0)
    )


def build_rr(
    sender_ssrc: int,
    source_ssrcs: Optional[Iterable[int]] = None,
    ssrc_stats: Optional[Mapping[int, Mapping[str, int]]] = None,
    sr_data: Optional[Mapping[int, tuple[int, float]]] = None,
) -> bytes:
    """Receiver Report. Up to 31 report blocks. Empty form is a 1-word RR."""
    sources = list(source_ssrcs or ())
    if not sources:
        return struct.pack(">BBHI", 0x80, 201, 1, sender_ssrc)

    rc = min(len(sources), 31)
    length = 1 + rc * 6
    out = bytearray(struct.pack(">BBHI", 0x80 | rc, 201, length, sender_ssrc))

    now = time.time()
    for ssrc in sources[:rc]:
        stats = (ssrc_stats or {}).get(ssrc, {})
        max_seq = stats.get("max_seq", 0)
        roc = stats.get("roc", 0)
        ext_seq = ((roc & 0xFFFF) << 16) | (max_seq & 0xFFFF)

        lsr = dlsr = 0
        if sr_data and ssrc in sr_data:
            lsr, sr_arrival = sr_data[ssrc]
            dlsr = int((now - sr_arrival) * 65536) & 0xFFFFFFFF

        out += struct.pack(
            ">IIIIII",
            ssrc,
            0,            # fraction lost (8) | cumulative lost (24)
            ext_seq,
            0,            # interarrival jitter
            lsr,
            dlsr,
        )
    return bytes(out)


def parse_sr_arrivals(data: bytes) -> list[tuple[int, int, float]]:
    """Walk a compound RTCP buffer and return SR arrivals as (ssrc, ntp_mid32, now)."""
    out: list[tuple[int, int, float]] = []
    now = time.time()
    pos = 0
    while pos + 4 <= len(data):
        b0, pt, length = struct.unpack(">BBH", data[pos:pos + 4])
        pkt_len = (length + 1) * 4
        if pos + pkt_len > len(data):
            break
        if pt == 200 and pkt_len >= 28:
            ssrc, ntp_sec, ntp_frac = struct.unpack(">III", data[pos + 4:pos + 16])
            mid32 = ((ntp_sec & 0xFFFF) << 16) | ((ntp_frac >> 16) & 0xFFFF)
            out.append((ssrc, mid32, now))
        pos += pkt_len
    return out


def compound_with_rr(sender_ssrc: int, payload: bytes) -> bytes:
    """Prefix `payload` with an empty RR, since some peers reject feedback that
    isn't part of a compound packet starting with SR or RR."""
    return build_rr(sender_ssrc) + payload


__all__ = [
    "build_empty_sr",
    "build_fir",
    "build_nack",
    "build_pli",
    "build_rr",
    "build_rtcp_app_ltrp",
    "build_rtcp_app_rctl",
    "compound_with_rr",
    "parse_sr_arrivals",
]
