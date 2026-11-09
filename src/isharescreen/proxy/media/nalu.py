"""HEVC NAL reassembly from Apple's RTP payload format.

Apple's multi-tile mode carries decoding-order fields:
  - Aggregation Packets (type 48) have a 2-byte DONL once after the NAL
    header, with **no** DOND between sub-NALUs.
  - Fragmentation Units (type 49) carry a 2-byte DONL **inside every
    fragment**, not just the start fragment.
  - Single NALUs ride with a 2-byte DONL prefix that must be stripped
    before handing the NALU to a decoder.

Single-picture streams omit these fields; callers pass `has_donl=False`.
"""
from __future__ import annotations

import struct
from typing import Iterable, Optional


# HEVC NAL unit types we care about.
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AGGREGATION = 48
NAL_FRAGMENTATION = 49

# IDR / IRAP NAL unit types: BLA_W_LP (16) through CRA_NUT (21).
IDR_RANGE = range(16, 22)


def reassemble_group(payloads: Iterable[bytes], *, has_donl: bool = True) -> list[bytes]:
    """Turn the RTP payloads belonging to one timestamp group into clean
    NALUs. Handles AP, FU, and single-NAL packets with Apple's DONL
    conventions. Drops malformed entries silently — UDP loss makes that
    routine, and the decoder gets to error on what survives."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False
    donl_size = 2 if has_donl else 0

    for pay in payloads:
        if len(pay) < 2:
            continue
        nt = (pay[0] >> 1) & 0x3F

        if nt == NAL_AGGREGATION:
            # header(2) + DONL(2) + [size(2) + data]...
            pos = 2 + donl_size
            n = len(pay)
            while pos + 2 <= n:
                size = struct.unpack(">H", pay[pos:pos + 2])[0]
                pos += 2
                if size == 0 or pos + size > n:
                    break
                out.append(bytes(pay[pos:pos + size]))
                pos += size

        elif nt == NAL_FRAGMENTATION:
            # header(2) + FU_hdr(1) + DONL(2) + payload
            body_start = 3 + donl_size
            if len(pay) <= body_start:
                continue
            fu_hdr = pay[2]
            start = bool(fu_hdr & 0x80)
            end = bool(fu_hdr & 0x40)
            inner_type = fu_hdr & 0x3F
            if start:
                # Reconstruct the inner NAL header from the FU NAL header.
                hdr0 = (pay[0] & 0x81) | (inner_type << 1)
                fu_buf = bytearray([hdr0, pay[1]])
                fu_buf += pay[body_start:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[body_start:]
            if fu_active and end:
                out.append(bytes(fu_buf))
                fu_active = False

        else:
            # Single NAL with leading 2-byte DONL.
            if len(pay) < 2 + donl_size:
                continue
            out.append(bytes(pay[:2]) + bytes(pay[2 + donl_size:]))

    return out


def first_donl(payloads: Iterable[bytes]) -> Optional[int]:
    """Extract the DONL (HEVC Decoding Order Number, 16-bit) carried in the
    first packet of a timestamp group. The host stamps every access unit with
    a DONL that increments +1 per frame in decoding order, and its encoder
    keys the long-term-reference ring on exactly this value — so the LTRP ack
    must echo this DONL for the host to map the ack to a valid encoder
    reference token; see Session._send_ltr_ack.

    Returns None if no packet carries a usable DONL. Per Apple's RFC 7798
    deviation the DONL sits after the 2-byte NAL header (AP / single NAL) or
    after the 1-byte FU header (FU), so at payload offset 2 or 3 respectively."""
    for pay in payloads:
        if len(pay) < 2:
            continue
        nt = (pay[0] >> 1) & 0x3F
        if nt == NAL_FRAGMENTATION:
            if len(pay) >= 5:
                return struct.unpack(">H", pay[3:5])[0]
        elif len(pay) >= 4:
            return struct.unpack(">H", pay[2:4])[0]
    return None


__all__ = [
    "IDR_RANGE",
    "NAL_AGGREGATION",
    "NAL_FRAGMENTATION",
    "NAL_PPS",
    "NAL_SPS",
    "NAL_VPS",
    "first_donl",
    "reassemble_group",
]
