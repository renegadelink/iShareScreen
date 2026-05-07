"""`TileFrame` — the decoder→consumer protocol.

Both the FFmpeg HEVC decoder and downstream consumers agree on this
shape. Frontends should not assume any other internal type from the
media subpackage.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TileFrame:
    """One decoded video tile in YUV planar form.

    Two on-the-wire chroma layouts are represented:

    * **Planar** (the common case): `u` and `v` are separate U and V
      planes. Source can be 4:2:0 (`chroma_width == width // 2`,
      `chroma_height == height // 2`) or 4:4:4 (`chroma_width == width`,
      `chroma_height == height`). Apple's HEVC RExt is 4:4:4.

    * **NV12 passthrough**: `v is None` and `u` carries the NV12-style
      interleaved UV plane verbatim. Reserved for fast paths where the
      consumer wants to skip the deinterleave; not currently produced by
      the FFmpeg decoder (it always emits planar).
    """
    y: bytes
    u: bytes
    v: bytes | None
    width: int
    height: int
    y_stride: int
    uv_stride: int
    chroma_width: int
    chroma_height: int

    @property
    def is_nv12_passthrough(self) -> bool:
        return self.v is None

    @property
    def is_yuv444(self) -> bool:
        return self.chroma_width == self.width and self.chroma_height == self.height


__all__ = ["TileFrame"]
