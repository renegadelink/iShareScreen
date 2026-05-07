"""Apple-specific magic byte sequences sent during the RFB handshake.

Reverse-engineered from a live Apple Screen Sharing handshake. Where
we understand the fields well enough to generate the bytes, we do
that — only blobs whose layout isn't fully un-reversed remain as
captured-hex constants. Each constant carries a comment with what we
do and don't know about it, so future maintainers can keep peeling.
"""
from __future__ import annotations

import struct


# 32-byte command-mask appended to the 0x21 ViewerInfo message. Each
# bit gates a viewer-side feature on the daemon (image clipboard,
# certain input subtypes, etc.). An all-zero mask passes the core High
# Performance handshake: the slim build doesn't ship any of the
# bit-gated features, so we emit zeros and the daemon turns those
# code paths off — exactly what we want.
#
# For reference, the bits Apple's shipping Screen Sharing.app sets
# (captured live, semantics partially un-reversed):
#     byte 2 = 0xb0   (bits 4, 5, 7)
#     byte 4 = 0x0c   (bits 2, 3)
#     byte 5 = 0x03   (bits 0, 1)
#     byte 6 = 0x90   (bits 4, 7)
#     byte 12 = 0x40  (bit 6)
APPLE_VIEWER_COMMAND_MASK: bytes = bytes(32)


# macOS (major, minor, patch) we send in the 0x21 ViewerInfo `os_ver`
# slot regardless of the actual local OS. The server uses this to
# pick the code path for our viewer; we want every supported host
# (Ventura through Tahoe and beyond) to recognise it and route us to
# the modern High Performance path. (15, 3, 0) is what shipping
# Screen Sharing.app advertised in the capture we replayed against.
APPLE_VIEWER_OS_VER: tuple[int, int, int] = (15, 3, 0)


# 12-byte 0x12 follow-up Apple sends in the plaintext phase right
# after the 0x21 ViewerInfo message. State-3 path interprets this
# with cmd=0x100, distinct from the post-toggle cmd=1/2 SetEncryption
# codes. Field values appear load-bearing: an all-zero variant fails
# the handshake.
#
# Layout (12 bytes total):
#     [0]      u8   msg type    = 0x12
#     [1]      u8   pad         = 0
#     [2..3]   u16  field A     = 1   (BE; load-bearing, all-zero fails)
#     [4..5]   u16  field B     = 1   (BE; load-bearing)
#     [6..7]   u16  field C     = 1   (BE; load-bearing)
#     [8..11]  u32  trailer     = 1   (BE)
#
# Building this from struct.pack instead of fromhex makes the
# structure explicit and the magic-number all-1s pattern visible.
APPLE_0X12_FOLLOWUP: bytes = struct.pack(
    ">BBHHHI", 0x12, 0,  1, 1, 1,  1,
)


__all__ = [
    "APPLE_0X12_FOLLOWUP",
    "APPLE_VIEWER_COMMAND_MASK",
    "APPLE_VIEWER_OS_VER",
]
