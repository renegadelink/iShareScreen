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
import time
from typing import Callable, Optional

import av

from .avc_nalu import h264_nal_type
from .decode_common import (
    _CODEC_FLAG_LOW_DELAY, _CODEC_FLAG2_FAST,
    _NAL_START_CODE, _TileSlot, _av_frame_to_tile, chroma_trace,
)
from .tiles import TileFrame
from .quality_gate import FrameQualityGate

log = logging.getLogger(__name__)


def _patch_avc_sps_dpb(sps: bytes) -> bytes:
    """Raise the SPS DPB ceiling so libav can hold Apple's full reference set.

    Apple's H.264 stream declares ``max_num_ref_frames = 8`` and ``level 5.2``
    (whose DPB caps at 8 frames at share resolutions), but its encoder actually
    references up to 9 pictures — 7 short-term + 2 long-term (LTRP). Strict
    libav enforces the declared 8, discards the 9th, and later frames then
    reference that now-"missing" picture, producing a smear until the next
    keyframe ("reference picture missing" / "number of reference frames (7+2)
    exceeds max (8)"). This is H.264-only: the HEVC path sizes its DPB
    differently and we ack its LTRP. Rewrite the SPS to level 6.0 +
    ``max_num_ref_frames = 16`` so libav's DPB holds everything Apple sends.

    Bit-exact surgery on the SPS RBSP; any parse surprise (e.g. an unexpected
    scaling matrix) returns the SPS untouched. Disable with ISS_AVC_SPS_PATCH=0.
    """
    if os.environ.get("ISS_AVC_SPS_PATCH") == "0":
        return sps
    try:
        if len(sps) < 4 or (sps[0] & 0x1f) != 7:
            return sps
        bits = [(byte >> i) & 1 for byte in sps[1:] for i in range(7, -1, -1)]
        pos = 0

        def u(n):
            nonlocal pos
            v = 0
            for _ in range(n):
                v = (v << 1) | bits[pos]
                pos += 1
            return v

        def ue():
            nonlocal pos
            start = pos
            z = 0
            while bits[pos] == 0:
                z += 1
                pos += 1
            pos += 1
            val = (1 << z) - 1
            if z:
                val += u(z)
            return start, pos, val

        profile = u(8)
        u(8)                      # constraint flags
        level_start = pos
        u(8)                      # level_idc
        ue()                      # seq_parameter_set_id
        if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
            _, _, cfi = ue()      # chroma_format_idc
            if cfi == 3:
                u(1)              # separate_colour_plane_flag
            ue()                  # bit_depth_luma_minus8
            ue()                  # bit_depth_chroma_minus8
            u(1)                  # qpprime_y_zero_transform_bypass_flag
            if u(1):              # seq_scaling_matrix_present_flag
                return sps        # scaling lists present — don't attempt surgery
        ue()                      # log2_max_frame_num_minus4
        _, _, poc = ue()          # pic_order_cnt_type
        if poc == 0:
            ue()                  # log2_max_pic_order_cnt_lsb_minus4
        elif poc == 1:
            u(1)
            ue()
            ue()
            n = ue()[2]
            for _ in range(n):
                ue()
        ms, me, mnrf = ue()       # max_num_ref_frames

        # macOS VideoToolbox HW decode refuses more than 10 refs: use 10 when it
        # covers Apple's declared refs + 1, else 16 (software).
        target_refs = 10 if sys.platform == "darwin" and mnrf < 10 else 16
        new_bits = bits
        if mnrf < target_refs:
            code = target_refs + 1  # Exp-Golomb encode ue(target_refs)
            nz = code.bit_length() - 1
            enc = [0] * nz + [(code >> (nz - j)) & 1 for j in range(nz + 1)]
            new_bits = bits[:ms] + enc + bits[me:]
        for i in range(8):        # level_idc → 6.0 (60) so the level doesn't cap the DPB
            new_bits[level_start + i] = (60 >> (7 - i)) & 1

        out = bytearray([sps[0]])
        for i in range(0, len(new_bits), 8):
            chunk = new_bits[i:i + 8]
            byte = 0
            for b in chunk:
                byte = (byte << 1) | b
            if len(chunk) < 8:
                byte <<= (8 - len(chunk))
            out.append(byte)
        return bytes(out)
    except Exception as e:
        log.debug("SPS DPB patch skipped: %s", e)
        return sps

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


def _vt_accepts_h264(sps: bytes, pps: bytes) -> bool:
    """True if VideoToolbox can hardware-decode these parameter sets.

    Checked up front so a refused canvas uses software without disabling
    hardware for later resizes. Unknown returns True (libav tries, as before).
    """
    try:
        import CoreMedia as _CM
        import VideoToolbox as _VT
        st, fmt = _CM.CMVideoFormatDescriptionCreateFromH264ParameterSets(
            None, 2, (bytes(sps), bytes(pps)), (len(sps), len(pps)), 4, None)
        if st != 0 or fmt is None:
            return True
        spec = {_VT.kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder: True}
        st, session = _VT.VTDecompressionSessionCreate(None, fmt, spec, None, None, None)
        if st != 0 or session is None:
            return False
        _VT.VTDecompressionSessionInvalidate(session)
        return True
    except Exception as e:
        log.debug("VideoToolbox H.264 pre-check unavailable: %s", e)
        return True


# pts→tile routing map bounds. Each fed slice gets a monotonic pts and the map
# is drained as frames emit (one in → one out under low-delay), so it normally
# holds ≤ num_tiles entries. The cap only guards a pathological non-emitting
# decoder from unbounded growth; prune keeps the most-recent window.
_PTS_MAP_SOFT_MAX = 256
_PTS_MAP_PRUNE_KEEP = 64


def _nal_is_keyframe(nalu: bytes) -> bool:
    """True if a raw H.264 NAL re-roots the DPB (a keyframe). Type-5 is a
    real IDR; Apple's HP encoder also keys via intra (I) slices in ordinary
    type-1 NALs, detected from the slice header's slice_type ∈ {2,7} (I /
    I-only-picture). The slice header after the 1-byte NAL header is
    first_mb_in_slice ue(v), slice_type ue(v) on the EPB-stripped RBSP."""
    if len(nalu) < 2:
        return False
    t = nalu[0] & 0x1F
    if t == 5:               # NAL_SLICE_IDR
        return True
    if t == 1:               # NAL_SLICE_NONIDR — intra if slice_type is I
        from .bitstream import BitReader, remove_emulation_prevention
        try:
            br = BitReader(remove_emulation_prevention(nalu[1:]))
            br.read_ue()                  # first_mb_in_slice
            return br.read_ue() in (2, 7)  # slice_type
        except Exception:
            return False
    return False


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
        # them via a pts→tile map (each slice gets a monotonic pts), since
        # H.264 here has no B-frame reordering.
        self._codec: Optional[av.codec.context.CodecContext] = None
        # Guards every _codec access. feed_nalu runs on the video-process
        # thread while restart()/close() can fire from the stall-watchdog
        # thread — without this lock a teardown could free the context mid-
        # decode (use-after-free). Mirrors HevcDecoder._codec_lock.
        self._codec_lock = threading.Lock()
        # pts→tile routing (mirrors HevcDecoder): each fed slice gets a
        # monotonic pts; emitted frames are mapped back to their source tile
        # via frame.pts. Robust against any decoder drop/reorder, unlike a
        # strict in/out FIFO.
        self._next_pts = 0
        self._pts_to_tile: dict[int, int] = {}
        self._pts_submit_t: dict[int, float] = {}
        self._reformatter: list = [None]
        self._seen_fmts: set = set()
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._sps = b""
        self._pps = b""
        # Opens on the first emitted frame (Apple has no type-5 IDR — keyframes
        # are intra slices in type-1 NALs); until then output is cold-DPB fill.
        self._dpb_ready = False
        # After a restart the DPB is empty; drop slices until the first keyframe
        # re-roots it (see feed_nalu). Armed at construction so the initial
        # burst's IDR opens the gate.
        self._await_key = True
        self._hw_name: Optional[str] = None
        # PyAV can open a nominal HWAccel context, then have FFmpeg reject the
        # actual stream/device combination on first decode and continue through
        # its allowed software fallback. D3D11VA can also accept the stream but
        # later return persistently corrupt frames after a confirmed reference
        # break. Both paths set this per-decoder/session latch so the next
        # context rebuild uses software instead of retrying broken hardware.
        self._hw_failed: bool = False
        self.nalu_counts_per_tile: list[dict[int, int]] = [
            {} for _ in range(num_tiles)
        ]
        # LTRP is HEVC-only; keep the attribute so the session's LTR-ack path
        # is a no-op instead of crashing.
        self.last_clean_donl: list[Optional[int]] = [None] * num_tiles
        # Decode latency monitoring: EMA of submit→frame round-trip (ms).
        self._decode_latency_ms: float = 0.0
        self._queue_full_drops: int = 0
        # Event-time diagnostics for rare, hours-later DPB failures. These are
        # deliberately cheap counters so the libav log callback can snapshot
        # decoder history without enabling frame-by-frame debug logging.
        self._nalus_fed: int = 0
        # Decodable pictures fed since the codec context/DPB was created.
        # Unlike `_frames_since_keyframe`, this deliberately does not reset on
        # Apple's non-IDR intra pictures: those do not necessarily clear a
        # D3D11VA reference buffer. The session uses this counter to rebuild
        # d3d11va before H.264's picture-order counter can wrap silently.
        self._frames_since_context_reset: int = 0
        self._keyframes_seen: int = 0
        self._frames_since_keyframe: int = 0
        self._last_keyframe_t: float = 0.0
        self._restart_count: int = 0
        self._sps_patch_applied: bool = False
        # Apple's AVC recovery pictures are non-IDR intra slices (NAL type 1),
        # so libav does not automatically flush a poisoned DPB when one lands.
        # A confirmed reference-chain break arms this flag from libav's log
        # callback. We then drop deltas and rebuild the codec context exactly
        # when the next complete intra slice arrives, using that independently
        # decodable picture to seed an empty DPB.
        self._reference_reset_pending: bool = False
        self._reference_reset_count: int = 0
        self._reference_break_t: float = 0.0
        self._reference_break_trigger: str = ""

    # -- setup ---------------------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict) -> None:
        """Install SPS/PPS. `vps` is ignored (H.264 has none); all_pps is the
        {pps_id: pps_nal} map harvested from an avcC config. Tiles share the
        same geometry so one SPS/PPS seeds every tile context."""
        self._sps = sps or b""
        self._pps = next(iter(all_pps.values())) if all_pps else b""

    def _build_extradata(self) -> bytes:
        patched_sps = _patch_avc_sps_dpb(self._sps)
        self._sps_patch_applied = patched_sps != self._sps
        self._patched_sps = patched_sps
        return _NAL_START_CODE + patched_sps + _NAL_START_CODE + self._pps

    def _ensure_codec_locked(self) -> Optional[av.codec.context.CodecContext]:
        """Build the shared context if needed (HW accel first, SW fallback).
        Caller MUST hold _codec_lock."""
        if self._codec is not None or not (self._sps and self._pps):
            return self._codec
        extradata = self._build_extradata()
        hw_allowed = self._prefer_hwaccel and not self._hw_failed
        if (hw_allowed and sys.platform == "darwin"
                and not _vt_accepts_h264(self._patched_sps, self._pps)):
            log.info("AVC: VideoToolbox won't hardware-decode this canvas "
                     "(reference-frame count); software until the next resize")
            hw_allowed = False
        if hw_allowed:
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
            # Apple's AVC stream carries params out-of-band in the 0x92 avcC
            # config, NOT in-band — verified: no type-7 arrives on resize. So
            # there's nothing to capture here; the session re-harvests the
            # avcC config and calls set_params()+restart() when the geometry
            # changes (see Session._maybe_reharvest_avc_config).
            return
        nb = nalu if isinstance(nalu, bytes) else bytes(nalu)
        is_key = _nal_is_keyframe(nb)
        # Post-restart / broken-reference keyframe gate. After a restart the
        # DPB is empty; after a confirmed reference miss it is poisoned. In
        # either case, feeding inter slices before a fresh intra picture merely
        # extends the failure. Apple's AVC "keyframe" is a non-IDR type-1 I
        # slice, so the broken-reference case additionally needs an explicit
        # context rebuild when that slice arrives (performed under the codec
        # lock below). Until then, freeze on the last good frame.
        if self._await_key and not is_key:
            return
        self._nalus_fed += 1
        if t in (1, 5):
            self._frames_since_context_reset += 1
        if is_key:
            self._keyframes_seen += 1
            self._frames_since_keyframe = 0
            self._last_keyframe_t = time.monotonic()
        elif t in (1, 5):
            self._frames_since_keyframe += 1
        # Each Apple tile-frame is one slice = one complete access unit, so we
        # build the av.Packet directly and decode() it — NO ctx.parse(). The
        # libav H.264 parser can't tell an AU is complete until the *next* slice
        # delimits it (Apple sends no AUD), so it holds every frame back ~one
        # frame-interval (~45ms @ 22fps) — invisible to the decode queue but
        # felt as input lag. Direct-decode emits immediately under LOW_DELAY.
        # Emitted frames route back to their source tile via frame.pts (a map,
        # not an in/out FIFO), surviving any decoder drop/reorder. Mirrors
        # HevcDecoder._decode_one. Runs under _codec_lock so a concurrent
        # restart()/close() can't free the context mid-decode.
        import time as _time
        published: list[int] = []
        with self._codec_lock:
            reset_for_this_key = False
            if is_key and self._reference_reset_pending:
                waited = (
                    _time.monotonic() - self._reference_break_t
                    if self._reference_break_t > 0.0 else -1.0
                )
                trigger = self._reference_break_trigger
                self._close_codec_context_locked()
                self._reference_reset_pending = False
                self._reference_reset_count += 1
                reset_for_this_key = True
                log.warning(
                    "AVC recovery: reset poisoned decoder DPB on fresh intra "
                    "frame (wait=%.3fs reset=%d trigger=%s)",
                    waited,
                    self._reference_reset_count,
                    trigger[:120] or "reference-chain break",
                )
            ctx = self._ensure_codec_locked()
            if ctx is None:
                return  # no params yet — don't register a pts that never drains
            if is_key:
                self._await_key = False
            # Every independently decodable intra frame is a recovery
            # observation for all tiles, so re-mark the quality gate (mirrors
            # HevcDecoder). The explicit context reset above, not the non-IDR
            # NAL itself, is what actually clears a poisoned libav DPB. Without
            # this gate observation
            # the gate only ever marks the first frame: a tile that drops into
            # `keyframe_required` mid-stream (a post-SSRC-adoption P-frame
            # error) can never satisfy mark_clean's IDR-observed condition, so
            # it FIR-storms / grays out forever even as fresh IDRs arrive.
            if is_key:
                for _t in range(self.num_tiles):
                    self._gate.mark_idr_observed(_t)
            pkt = av.Packet(_NAL_START_CODE + nb)
            pts = self._next_pts
            pkt.pts = pts
            pkt.dts = pts
            self._next_pts += 1
            self._pts_to_tile[pts] = tile_idx
            self._pts_submit_t[pts] = _time.monotonic()
            if len(self._pts_to_tile) > _PTS_MAP_SOFT_MAX:
                cutoff = pts - _PTS_MAP_PRUNE_KEEP
                self._pts_to_tile = {
                    k: v for k, v in self._pts_to_tile.items() if k > cutoff
                }
                self._pts_submit_t = {
                    k: v for k, v in self._pts_submit_t.items() if k > cutoff
                }
            try:
                frames = ctx.decode(pkt)
            except Exception:
                # Decode raised → no frame will carry this pts; drop it so the
                # map doesn't leak the in-flight entry.
                self._pts_to_tile.pop(pts, None)
                self._pts_submit_t.pop(pts, None)
                if reset_for_this_key:
                    # The recovery intra picture itself was unusable. Keep
                    # deltas gated and wait for the sticky FIR loop's retry
                    # instead of feeding an empty DPB immediately afterward.
                    self.mark_reference_chain_broken(
                        "fresh intra frame failed after DPB reset",
                    )
                self._gate.mark_decode_error(tile_idx)
                return
            for frame in frames:
                ti = self._pts_to_tile.pop(frame.pts, tile_idx)
                submit_t = self._pts_submit_t.pop(frame.pts, None)
                # The libav callback can arm recovery synchronously from inside
                # ctx.decode(pkt). Never publish that same concealed frame:
                # hold the last known-good picture while deltas are gated and
                # the requested intra recovery frame is in flight.
                if self._reference_reset_pending:
                    continue
                if submit_t is not None:
                    latency_ms = (_time.monotonic() - submit_t) * 1000
                    self._decode_latency_ms = (
                        0.1 * latency_ms + 0.9 * self._decode_latency_ms
                    )
                if not self._dpb_ready:
                    self._dpb_ready = True
                    for _t in range(self.num_tiles):
                        self._gate.mark_idr_observed(_t)
                slot = self._tiles[ti]
                with slot.lock:
                    slot.raw_frame = frame
                    slot.good_count += 1
                published.append(ti)
        # Notify outside the codec lock to avoid holding it across the callback.
        if self._on_frame_published is not None:
            for ti in published:
                self._on_frame_published(ti)

    def mark_hwaccel_failed(self, trigger: str = "") -> None:
        """Record FFmpeg's explicit late hardware-initialization failure.

        PyAV deliberately transfers successful hardware frames back to a CPU
        pixel format, so the returned frame format cannot distinguish hardware
        decode from software fallback. The libav error is authoritative. Keep
        the already-running fallback context; only make the label truthful and
        skip the broken accelerator on future context rebuilds.
        """
        if self._hw_failed:
            return
        failed = self._hw_name or "requested accelerator"
        self._hw_failed = True
        self._hw_name = None
        log.warning(
            "AVC hwaccel %r failed during stream setup; continuing in "
            "software and disabling HW retries for this session (%s)",
            failed, trigger[:120] or "libav hardware initialization error",
        )

    def mark_hwaccel_reference_failure(self, trigger: str = "") -> None:
        """Fall back only after an active D3D11VA decoder proves unreliable.

        Keep ``_hw_name`` until the already-armed reference recovery reaches a
        fresh intra frame. That preserves truthful event diagnostics
        (``decoder=d3d11va``); the ensuing codec rebuild sees ``_hw_failed`` and
        creates a software context for the remainder of this connection.
        """
        if self._hw_failed or self._hw_name != "d3d11va":
            return
        self._hw_failed = True
        log.warning(
            "AVC d3d11va produced a confirmed broken reference chain; "
            "the fresh-intra recovery will switch this session to software "
            "(hardware remains the default for new sessions; %s)",
            trigger[:120] or "libav reference-picture error",
        )

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
        chroma_trace(tile_idx, tile_frame, had_error)
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
    def decode_latency_ms(self) -> float:
        return self._decode_latency_ms

    @property
    def decode_queue_depth(self) -> int:
        return len(self._pts_to_tile)

    @property
    def decode_queue_cap(self) -> int:
        return 512

    @property
    def decode_queue_drops(self) -> int:
        return self._queue_full_drops

    @property
    def recovery_diagnostics(self) -> dict[str, object]:
        """Small lock-free snapshot safe to read from libav's log callback."""
        last_key_age = (
            time.monotonic() - self._last_keyframe_t
            if self._last_keyframe_t > 0.0 else -1.0
        )
        return {
            "decoder": self._hw_name or "software",
            "nalus_fed": self._nalus_fed,
            "frames_since_context_reset": self._frames_since_context_reset,
            "keyframes_seen": self._keyframes_seen,
            "frames_since_keyframe": self._frames_since_keyframe,
            "last_keyframe_age_s": last_key_age,
            "restarts": self._restart_count,
            "sps_patch": self._sps_patch_applied,
            "await_key": self._await_key,
            "reference_reset_pending": self._reference_reset_pending,
            "reference_resets": self._reference_reset_count,
        }

    @property
    def good_counts(self) -> list:
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list:
        return [t.clean_count for t in self._tiles]

    def mark_reference_chain_broken(self, trigger: str = "") -> None:
        """Arm loss recovery without touching libav from its log callback.

        This method is intentionally lock-free: PyAV invokes the session's
        concealment handler synchronously while ``ctx.decode()`` holds
        ``_codec_lock``, so closing the codec here would deadlock. The next
        ``feed_nalu`` drops deltas until a complete intra frame arrives, then
        performs the actual context rebuild under the normal decode lock.
        """
        if not self._reference_reset_pending:
            self._reference_break_t = time.monotonic()
            self._reference_break_trigger = trigger
        self._reference_reset_pending = True
        self._await_key = True

    def _close_codec_context_locked(self) -> None:
        """Clear only codec/DPB state; caller holds ``_codec_lock``.

        The quality gate deliberately survives this operation so its sticky
        FIR request cannot clear until the newly seeded decoder publishes a
        clean frame.
        """
        if self._codec is not None:
            try:
                self._codec.close()
            except Exception:
                pass
            self._codec = None
        self._pts_to_tile.clear()
        self._pts_submit_t.clear()
        self._dpb_ready = False
        self._await_key = True
        self._decode_latency_ms = 0.0
        self._frames_since_context_reset = 0

    def restart(self) -> None:
        """Tear down + rebuild the shared codec context. May fire from the
        stall-watchdog thread, so it takes _codec_lock to avoid freeing the
        context while feed_nalu is mid-decode. Resets the gate too (HEVC does
        this in _teardown) so post-restart publish/FIR decisions start clean."""
        with self._codec_lock:
            self._restart_count += 1
            self._close_codec_context_locked()
            self._reference_reset_pending = False
            self._reference_break_t = 0.0
            self._reference_break_trigger = ""
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
