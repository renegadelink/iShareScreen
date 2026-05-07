"""HEVC NAL reassembly from Apple's RTP payload format.

Apple deviates from RFC 7798 in two places:
  - Aggregation Packets (type 48) have a 2-byte DONL once after the NAL
    header, with **no** DOND between sub-NALUs.
  - Fragmentation Units (type 49) carry a 2-byte DONL **inside every
    fragment**, not just the start fragment.
  - Single NALUs ride with a 2-byte DONL prefix that must be stripped
    before handing the NALU to a decoder.
"""
from __future__ import annotations

import struct
from typing import Iterable


# HEVC NAL unit types we care about.
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AGGREGATION = 48
NAL_FRAGMENTATION = 49

# IDR / IRAP NAL unit types: BLA_W_LP (16) through CRA_NUT (21).
IDR_RANGE = range(16, 22)


def reassemble_group(payloads: Iterable[bytes]) -> list[bytes]:
    """Turn the RTP payloads belonging to one timestamp group into clean
    NALUs. Handles AP, FU, and single-NAL packets with Apple's DONL
    conventions. Drops malformed entries silently — UDP loss makes that
    routine, and the decoder gets to error on what survives."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False

    for pay in payloads:
        if len(pay) < 2:
            continue
        nt = (pay[0] >> 1) & 0x3F

        if nt == NAL_AGGREGATION:
            # header(2) + DONL(2) + [size(2) + data]...
            pos = 4
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
            if len(pay) < 6:
                continue
            fu_hdr = pay[2]
            start = bool(fu_hdr & 0x80)
            end = bool(fu_hdr & 0x40)
            inner_type = fu_hdr & 0x3F
            if start:
                # Reconstruct the inner NAL header from the FU NAL header.
                hdr0 = (pay[0] & 0x81) | (inner_type << 1)
                fu_buf = bytearray([hdr0, pay[1]])
                fu_buf += pay[5:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[5:]
                if end:
                    out.append(bytes(fu_buf))
                    fu_active = False

        else:
            # Single NAL with leading 2-byte DONL.
            if len(pay) < 4:
                continue
            out.append(bytes(pay[:2]) + bytes(pay[4:]))

    return out


__all__ = [
    "IDR_RANGE",
    "NAL_AGGREGATION",
    "NAL_FRAGMENTATION",
    "NAL_PPS",
    "NAL_SPS",
    "NAL_VPS",
    "reassemble_group",
]
