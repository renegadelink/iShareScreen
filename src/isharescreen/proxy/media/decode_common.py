"""Decode helpers shared by the HEVC and AVC decoders.

Holds the codec-independent pieces both `hevc.py` and `avc.py` need — the
NAL start code, libavcodec flag constants, the per-tile output slot, and the
`av.VideoFrame → TileFrame` conversion — so the AVC path doesn't import from
`hevc.py` (and vice versa). Codec-specific logic (HEVC RPS/POC, the AVC NAL
handling, decoder lifecycles) stays in the respective modules.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import av
import numpy as np

from .tiles import TileFrame


log = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────

_NAL_START_CODE = b"\x00\x00\x00\x01"

# libavcodec flags. AV_CODEC_FLAG_LOW_DELAY = 0x80000 disables frame
# reordering buffers. AV_CODEC_FLAG2_FAST = 0x400000 enables non-bitexact
# but faster decode paths (acceptable for live screen-share).
_CODEC_FLAG_LOW_DELAY = 0x00080000
_CODEC_FLAG2_FAST = 0x00400000


# ── per-tile output state ────────────────────────────────────────────

@dataclass(slots=True)
class _TileSlot:
    """The latest decoded frame for one tile + its bookkeeping. Locked so
    the decode worker (writer) and the consumer's `get_frame()` (reader)
    don't tear."""
    raw_frame: Optional[av.VideoFrame] = None
    good_count: int = 0
    # Frames published WITHOUT a libav decode-error flag — i.e. actually
    # clean video, not concealed gray. `good_count` counts every published
    # frame including concealed ones, so a fully-gray tile still shows a
    # healthy good_count/rate; `clean_count` is the honest "is this tile
    # showing real picture" signal. Compare the two in the profile log.
    clean_count: int = 0
    last_evaluated_count: int = 0
    # Per-tile flag, kept for diagnostics. The actual publish gate is
    # the session-wide `_dpb_has_idr` because Apple's stream uses
    # cross-tile DPB references (tile-1/2/3 P-frames reference tile-0's
    # IDR) — so as soon as ANY tile delivers an IDR the shared codec
    # context has a usable DPB for every tile.
    saw_idr_since_reset: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


# ── frame extraction (av.VideoFrame → TileFrame) ─────────────────────

# libavcodec format names that mean "this frame lives on the GPU; reformat
# before extracting planes."
_HW_FRAME_FORMATS = frozenset({
    "vaapi", "vaapi_vld",
    "d3d11", "d3d11va", "dxva2_vld",
    "cuda",
    "drm_prime",
    "mediacodec",
    "videotoolbox", "videotoolbox_vld",
})


def _av_frame_to_tile(
    frame: av.VideoFrame,
    reformatter_holder: list[Optional[av.video.reformatter.VideoReformatter]],
    seen_fmts: set[str],
) -> tuple[Optional[TileFrame], bool]:
    """Convert an `av.VideoFrame` to our `TileFrame`.

    Returns `(tile, had_decode_error)`. `had_decode_error` is True when
    libavcodec set `decode_error_flags` or marked the frame corrupt —
    the cheapest signal we have for "the decoder concealed missing
    reference data". The tile is still returned (publish-it-anyway
    policy: a momentary visible artifact + a fast FIR is more useful
    to the operator than a frozen tile)."""
    err = getattr(frame, "decode_error_flags", 0)
    flg = getattr(frame, "flags", 0)
    had_error = bool(err) or bool(flg & 0x01)
    if had_error:
        log.debug("decode error: decode_error_flags=%d flags=0x%x", err, flg)
    fmt = frame.format.name
    width = frame.width
    height = frame.height
    if fmt not in seen_fmts:
        seen_fmts.add(fmt)
        log.info("decoded frame format: %s (%dx%d)", fmt, width, height)

    if fmt in _HW_FRAME_FORMATS:
        if reformatter_holder[0] is None:
            from av.video.reformatter import VideoReformatter
            reformatter_holder[0] = VideoReformatter()
        # Apple's HEVC stream is RExt 4:4:4 — reformatting to 4:2:0
        # (nv12) throws away three-quarters of the chroma and produces
        # visible "pixelated gray" fringing on text and rectangles.
        # Preserve full chroma. yuv444p is universally supported by
        # libswscale; the GPU upload path already handles it as planar
        # full-resolution Y/U/V.
        frame = reformatter_holder[0].reformat(frame, format="yuv444p")
        fmt = frame.format.name

    if fmt in ("nv12", "nv21"):
        # Biplanar 4:2:0: Y plane + half-resolution interleaved UV.
        yp = frame.planes[0]
        uvp = frame.planes[1]
        chroma_h = height // 2
        uv_view = (
            np.frombuffer(bytes(uvp), dtype=np.uint8)
            .reshape(chroma_h, uvp.line_size)[:, :width]
        )
        if fmt == "nv12":
            u_bytes = uv_view[:, 0::2].tobytes()
            v_bytes = uv_view[:, 1::2].tobytes()
        else:
            v_bytes = uv_view[:, 0::2].tobytes()
            u_bytes = uv_view[:, 1::2].tobytes()
        return TileFrame(
            y=bytes(yp), u=u_bytes, v=v_bytes,
            width=width, height=height,
            y_stride=yp.line_size,
            uv_stride=width // 2,
            chroma_width=width // 2,
            chroma_height=chroma_h,
        ), had_error

    if fmt in ("yuv420p", "yuvj420p"):
        yp, up, vp = frame.planes
        return TileFrame(
            y=bytes(yp), u=bytes(up), v=bytes(vp),
            width=width, height=height,
            y_stride=yp.line_size,
            uv_stride=up.line_size,
            chroma_width=width // 2,
            chroma_height=height // 2,
        ), had_error

    if fmt in ("yuv444p", "yuvj444p"):
        yp, up, vp = frame.planes
        return TileFrame(
            y=bytes(yp), u=bytes(up), v=bytes(vp),
            width=width, height=height,
            y_stride=yp.line_size,
            uv_stride=up.line_size,
            chroma_width=width,
            chroma_height=height,
        ), had_error

    log.warning("unsupported decoded frame format: %s", fmt)
    return None, had_error


__all__ = [
    "_NAL_START_CODE",
    "_CODEC_FLAG_LOW_DELAY",
    "_CODEC_FLAG2_FAST",
    "_TileSlot",
    "_HW_FRAME_FORMATS",
    "_av_frame_to_tile",
]
