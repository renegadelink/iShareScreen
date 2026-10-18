"""Decode helpers shared by the HEVC and AVC decoders.

Holds the codec-independent pieces both `hevc.py` and `avc.py` need — the
NAL start code, libavcodec flag constants, the per-tile output slot, and the
`av.VideoFrame → TileFrame` conversion — so the AVC path doesn't import from
`hevc.py` (and vice versa). Codec-specific logic (HEVC RPS/POC, the AVC NAL
handling, decoder lifecycles) stays in the respective modules.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import av

from .tiles import TileFrame


log = logging.getLogger(__name__)


# ── chroma-signature diagnostic (ISS_CHROMA_TRACE=1) ─────────────────────
# Off by default (a single bool check per frame). When on, logs the luma and
# chroma-plane mean/std ~2×/s. Purpose: the "bluish tint that sticks" artifact
# is a SILENT corruption — libav sets no decode-error flag and the pixel gate
# doesn't flag a detailed-but-colour-shifted frame — so nothing detects it. A
# bad reference shifts the chroma DC uniformly (U=Cb rises for blue) and it
# PERSISTS/drifts until an IDR, whereas a legitimately-blue window has stable,
# content-shaped chroma. This trace captures that signature so a detector can
# be built that distinguishes the two instead of FIR-storming on blue content.
_CHROMA_TRACE = os.environ.get("ISS_CHROMA_TRACE") == "1"
_chroma_last_log_t: list[float] = [0.0]


def chroma_trace(tile_idx: int, tf: TileFrame, had_error: bool) -> None:
    """Throttled per-frame chroma-plane stats. No-op unless ISS_CHROMA_TRACE=1."""
    if not _CHROMA_TRACE or tf is None:
        return
    now = time.monotonic()
    if now - _chroma_last_log_t[0] < 0.5:      # ~2 Hz — the expensive part
        return
    _chroma_last_log_t[0] = now
    try:
        import numpy as np

        def _stats(buf: bytes, stride: int, w: int, h: int) -> tuple[float, float]:
            a = np.frombuffer(buf, dtype=np.uint8)
            if stride >= w and len(a) >= stride * h:   # strip row padding
                a = a.reshape(h, stride)[:, :w]
            return float(a.mean()), float(a.std())

        ym, ys = _stats(tf.y, tf.y_stride, tf.width, tf.height)
        um, us = _stats(tf.u, tf.uv_stride, tf.chroma_width, tf.chroma_height)
        vm, vs = (_stats(tf.v, tf.uv_stride, tf.chroma_width, tf.chroma_height)
                  if tf.v is not None else (-1.0, -1.0))
        log.info(
            "CHROMA_TRACE tile=%d err=%s Y=%.1f/%.1f U=%.1f/%.1f V=%.1f/%.1f "
            "(mean/std; U,V neutral=128, a persistent U/V-mean drift = the "
            "silent chroma corruption)",
            tile_idx, had_error, ym, ys, um, us, vm, vs,
        )
    except Exception as e:
        log.debug("chroma_trace failed: %s", e)


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

# Packed 4:4:4 outputs. vuyx = V,U,Y,X(pad); Intel QSV's hevc_qsv emits these
# for Apple's HEVC RExt 4:4:4 stream. They carry FULL chroma (no subsample),
# so the reformat to planar yuv444p is a lossless de-interleave.
_PACKED_444_FORMATS = frozenset({"vuyx", "vuya", "ayuv", "uyva", "xyuv"})


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

    if fmt in _PACKED_444_FORMATS:
        # Intel QSV 4:4:4 (vuyx etc.): de-interleave to planar yuv444p. Full
        # chroma preserved — the full-quality HEVC 4:4:4 path. Falls through to
        # the yuv444p planar-extraction branch below.
        if reformatter_holder[0] is None:
            from av.video.reformatter import VideoReformatter
            reformatter_holder[0] = VideoReformatter()
        frame = reformatter_holder[0].reformat(frame, format="yuv444p")
        fmt = frame.format.name

    if fmt in ("nv12", "nv21"):
        # Biplanar 4:2:0 (VideoToolbox's AVC output). Deinterleaving the UV
        # plane in numpy is a strided gather — ~5.7ms/frame at HiDPI, enough
        # to blow the frame budget and stutter. libswscale does the same
        # nv12/nv21 → planar deinterleave in SIMD (~0.6ms), so reformat to
        # yuv420p and fall through to the contiguous-copy planar branch.
        if reformatter_holder[0] is None:
            from av.video.reformatter import VideoReformatter
            reformatter_holder[0] = VideoReformatter()
        frame = reformatter_holder[0].reformat(frame, format="yuv420p")
        fmt = frame.format.name
        width = frame.width
        height = frame.height

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


def available_hwaccels(candidates: "tuple[str, ...]") -> "tuple[str, ...]":
    """Keep only the hwaccel device types this PyAV build supports.

    Requesting a device type PyAV was built without (e.g. ``vaapi`` from the PyPI
    wheel, which bundles no libva) does not fail: libav quietly hands back a
    software context, so the type looks usable when it is not. Filtering on
    ``hwdevices_available()`` lets the next candidate (e.g. ``cuda``) be tried.
    It does not detect hardware: a supported type with no device behind it still
    fails at context creation, which callers already handle."""
    try:
        from av.codec.hwaccel import hwdevices_available
        present = set(hwdevices_available())
    except Exception:
        return candidates
    return tuple(c for c in candidates if c in present)


def context_is_hwaccel(ctx: object) -> "bool | None":
    """``CodecContext.is_hwaccel`` where PyAV provides it, else ``None`` (unknown)."""
    value = getattr(ctx, "is_hwaccel", None)
    return value if isinstance(value, bool) else None
