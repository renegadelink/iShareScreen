"""H.264 (AVC) decoder for Apple's screen stream.

Apple sends H.264 4:2:0 (yuvj420p) when the client advertises the field1=123
codec bank. Reverse-engineered from live captures:

  * The 4 tiles are decoded as ONE timestamp-ordered H.264 stream through a
    single shared `av.CodecContext`. This was settled empirically: feeding all
    tiles' NALs to one context in arrival (timestamp) order decodes clean at
    60fps, whereas per-tile contexts or out-of-order feeding conceal/gray.
    Output frames are routed back to the tile that fed them via a FIFO, which
    is correct because there is no B-frame reordering (one slice = one AU,
    emitted in order).
  * Apple does NOT emit type-5 IDR NALs. Keyframes are intra (I) slices carried
    in ordinary type-1 NALs; it re-keys by spinning up a fresh SSRC generation.
    So "have we got a keyframe" can't be detected from the NAL type — instead
    the first frame the decoder actually EMITS is the keyframe signal.

H.264 4:2:0 is hardware-decodable everywhere (the point of this path vs HEVC
4:4:4), though this decoder is software-only for now. Frame extraction + the
quality gate are reused verbatim from the HEVC path.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from collections import deque
from typing import Callable, Optional

import av

from .avc_nalu import h264_nal_type
from .decode_common import (
    _CODEC_FLAG_LOW_DELAY, _CODEC_FLAG2_FAST,
    _NAL_START_CODE, _TileSlot, _av_frame_to_tile,
)
from .tiles import TileFrame
from .quality_gate import FrameQualityGate

log = logging.getLogger(__name__)

# Hardware decoders to try per platform for H.264. Unlike Apple's HEVC RExt
# 4:4:4 (which DXVA2/D3D11VA/VAAPI cannot decode, so the "HW" context silently
# falls back to software), H.264 4:2:0 is the universally hardware-decodable
# profile — these accelerators DO bind, which is the whole point of the AVC
# path: real GPU decode on the Windows/Linux boxes where 4:4:4 can't.
_H264_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "dxva2"),
    "linux": ("vaapi",),
    "*": (),
}


def _h264_hwaccels() -> tuple[str, ...]:
    return _H264_HWACCELS.get(sys.platform, _H264_HWACCELS["*"])


class AvcDecoder:
    def __init__(
        self,
        num_tiles: int,
        *,
        prefer_hwaccel: bool = True,
        enable_quality_gate: bool = True,
        on_frame_published: Optional[Callable[[int], None]] = None,
    ) -> None:
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.num_tiles = num_tiles
        # ISS_FORCE_SW_DECODE=1 forces software decode — removes the
        # platform-specific HW decoder (e.g. VideoToolbox) as a variable when
        # validating the protocol/stream itself.
        if os.environ.get("ISS_FORCE_SW_DECODE", "0") == "1":
            prefer_hwaccel = False
        self._prefer_hwaccel = prefer_hwaccel
        self._on_frame_published = on_frame_published
        self._tiles = [_TileSlot() for _ in range(num_tiles)]
        # ONE shared context: the 4 tiles are decoded as a single timestamp-
        # ordered stream (verified — feeding all tiles' NALs to one context in
        # ts order is what decodes clean; separate contexts / out-of-order
        # feeding conceal). Output frames are routed back to the tile that fed
        # them via a FIFO, since H.264 here has no B-frame reordering.
        self._codec: Optional[av.codec.context.CodecContext] = None
        # Guards every _codec access. feed_nalu runs on the video-process
        # thread while restart()/close() can fire from the stall-watchdog
        # thread — without this lock a teardown could free the context mid-
        # decode (use-after-free). Mirrors HevcDecoder._codec_lock.
        self._codec_lock = threading.Lock()
        self._fed_tiles: deque = deque()
        self._reformatter: list = [None]
        self._seen_fmts: set = set()
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._sps = b""
        self._pps = b""
        # Opens on the first emitted frame (Apple has no type-5 IDR — keyframes
        # are intra slices in type-1 NALs); until then output is cold-DPB fill.
        self._dpb_ready = False
        self._hw_name: Optional[str] = None  # software-only for now
        self.nalu_counts_per_tile: list[dict[int, int]] = [
            {} for _ in range(num_tiles)
        ]
        # LTRP is HEVC-only; keep the attribute so the session's LTR-ack path
        # is a no-op instead of crashing.
        self.last_clean_donl: list[Optional[int]] = [None] * num_tiles

    # -- setup ---------------------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict) -> None:
        """Install SPS/PPS. `vps` is ignored (H.264 has none); all_pps is the
        {pps_id: pps_nal} map harvested from an avcC config. Tiles share the
        same geometry so one SPS/PPS seeds every tile context."""
        self._sps = sps or b""
        self._pps = next(iter(all_pps.values())) if all_pps else b""

    def _build_extradata(self) -> bytes:
        return _NAL_START_CODE + self._sps + _NAL_START_CODE + self._pps

    def _ensure_codec_locked(self) -> Optional[av.codec.context.CodecContext]:
        """Build the shared context if needed (HW accel first, SW fallback).
        Caller MUST hold _codec_lock."""
        if self._codec is not None or not (self._sps and self._pps):
            return self._codec
        extradata = self._build_extradata()
        if self._prefer_hwaccel:
            for hw_type in _h264_hwaccels():
                ctx = self._try_hwaccel_locked(hw_type, extradata)
                if ctx is not None:
                    self._codec = ctx
                    self._hw_name = hw_type
                    log.info("AVC decode: hardware (%s)", hw_type)
                    return self._codec
        self._codec = self._make_sw_context(extradata)
        self._hw_name = None
        log.info("AVC decode: software")
        return self._codec

    def _try_hwaccel_locked(
        self, hw_type: str, extradata: bytes,
    ) -> Optional[av.codec.context.CodecContext]:
        try:
            from av.codec.hwaccel import HWAccel
            hw = HWAccel(device_type=hw_type)
            c = av.CodecContext.create("h264", "r", hwaccel=hw)
            c.extradata = extradata
            c.flags = _CODEC_FLAG_LOW_DELAY
            c.flags2 = _CODEC_FLAG2_FAST
            c.open()
            return c
        except Exception as e:
            log.info("AVC hwaccel %s unavailable: %s", hw_type, e)
            return None

    @staticmethod
    def _make_sw_context(extradata: bytes) -> av.codec.context.CodecContext:
        # SLICE threading (parallelise within a frame, no reordering/latency)
        # so software H.264 keeps pace with the 60fps stream on slower CPUs.
        c = av.CodecContext.create("h264", "r")
        c.extradata = extradata
        c.thread_type = "SLICE"
        c.thread_count = 0
        c.flags = _CODEC_FLAG_LOW_DELAY
        c.flags2 = _CODEC_FLAG2_FAST
        c.open()
        return c

    def start(self) -> None:
        if not (self._sps and self._pps):
            raise RuntimeError("set_params() must be called before start()")
        with self._codec_lock:
            self._ensure_codec_locked()

    # -- feed ----------------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict) -> None:
        for ti, nalus in tile_nalu_cache.items():
            for nalu in nalus:
                self.feed_nalu(nalu, ti)

    def feed_nalu(self, nalu: bytes, tile_idx: int, donl: Optional[int] = None) -> None:
        if not nalu or not (0 <= tile_idx < self.num_tiles):
            return
        t = h264_nal_type(nalu[0])
        bucket = self.nalu_counts_per_tile[tile_idx]
        bucket[t] = bucket.get(t, 0) + 1
        if t in (7, 8):  # SPS/PPS already in extradata
            return
        # Each Apple tile-frame is one slice = one access unit. Feed through the
        # parser (ctx.parse) so libav frames AUs + establishes the I keyframe;
        # raw decode() per NAL loses references and conceals. Route each emitted
        # frame back to the tile that fed it via the FIFO (no B-frame reorder).
        # The whole parse+decode runs under _codec_lock so a concurrent
        # restart()/close() can't free the context mid-decode.
        published: list[int] = []
        with self._codec_lock:
            ctx = self._ensure_codec_locked()
            if ctx is None:
                return
            self._fed_tiles.append(tile_idx)
            try:
                packets = ctx.parse(_NAL_START_CODE + bytes(nalu))
            except Exception:
                return
            for pkt in packets:
                try:
                    frames = ctx.decode(pkt)
                except Exception:
                    if self._fed_tiles:
                        self._gate.mark_decode_error(self._fed_tiles.popleft())
                    continue
                ti = self._fed_tiles.popleft() if self._fed_tiles else tile_idx
                if frames and not self._dpb_ready:
                    self._dpb_ready = True
                    for _t in range(self.num_tiles):
                        self._gate.mark_idr_observed(_t)
                for frame in frames:
                    slot = self._tiles[ti]
                    with slot.lock:
                        slot.raw_frame = frame
                        slot.good_count += 1
                    published.append(ti)
        # Notify outside the codec lock to avoid holding it across the callback.
        if self._on_frame_published is not None:
            for ti in published:
                self._on_frame_published(ti)

    # -- consume -------------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        if not self._dpb_ready:
            return None
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.raw_frame
            count = slot.good_count
            already = count <= slot.last_evaluated_count
        if frame is None or already:
            return None
        tile_frame, had_error = _av_frame_to_tile(
            frame, self._reformatter, self._seen_fmts,
        )
        with slot.lock:
            slot.last_evaluated_count = count
        if tile_frame is None:
            return None
        if had_error:
            self._gate.mark_decode_error(tile_idx)
        else:
            self._gate.mark_clean(tile_idx)
            with slot.lock:
                slot.clean_count += 1
        if not self._gate.should_publish(tile_idx, tile_frame):
            return None
        return tile_frame

    def consume_fir_request(self) -> set:
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int):
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def bad_tiles(self) -> set:
        return self._gate.bad_tiles

    @property
    def good_counts(self) -> list:
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list:
        return [t.clean_count for t in self._tiles]

    def restart(self) -> None:
        """Tear down + rebuild the shared codec context. May fire from the
        stall-watchdog thread, so it takes _codec_lock to avoid freeing the
        context while feed_nalu is mid-decode. Resets the gate too (HEVC does
        this in _teardown) so post-restart publish/FIR decisions start clean."""
        with self._codec_lock:
            if self._codec is not None:
                try:
                    self._codec.close()
                except Exception:
                    pass
                self._codec = None
            self._fed_tiles.clear()
            self._dpb_ready = False
            self._gate.reset()
            if self._sps and self._pps:
                self._ensure_codec_locked()

    def close(self) -> None:
        with self._codec_lock:
            if self._codec is not None:
                try:
                    self._codec.close()
                except Exception:
                    pass
                self._codec = None
