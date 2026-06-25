"""HEVC 4:4:4 decoder via Intel Quick Sync (`hevc_qsv`).

Apple's screen-share default is HEVC RExt 4:4:4. libav's generic `hevc` +
d3d11va hwaccel does NOT implement the 4:4:4 profile, so it silently falls
back to (far too slow) CPU decode — which is why the app drops to the H.264
4:2:0 AVC bank on Windows. But Intel's dedicated `hevc_qsv` decoder DOES
hardware-decode Apple's 4:4:4 stream on Gen11+/Xe iGPUs (verified on UHD 770:
all frames, zero errors, `vuyx` output, clean past the POC wrap that breaks
the AMD d3d11va H.264 path). This module drives that path.

Two things differ from the libav `HevcDecoder`:
  * **Access-unit feeding.** `hevc_qsv` is a black-box hardware decoder that
    wants one complete access unit (parameter sets + slice) per packet, not
    the NALU-at-a-time feed the slice-based libav decoder accepts. We batch:
    parameter-set NALUs accumulate, and the next VCL slice flushes the whole
    AU as one packet. Apple uses one slice per tile picture, so each VCL NAL
    is a complete picture.
  * **No RPS pre-check / IDR-size heuristics.** QSV manages its own DPB and
    cross-tile references internally; it just works when fed valid AUs, so the
    libav-specific concealment scaffolding isn't needed here.

Output is `vuyx` (packed 4:4:4); `_av_frame_to_tile` de-interleaves it to
planar yuv444p for the existing full-resolution GPU upload.
"""
from __future__ import annotations

import logging
import queue
import threading
from fractions import Fraction
from typing import Callable, Optional

import av

from .tiles import TileFrame
from .quality_gate import FrameQualityGate, TileVisState
from .hevc import _TileSlot, _av_frame_to_tile

log = logging.getLogger(__name__)

_NAL_START = b"\x00\x00\x00\x01"
_QUEUE_MAX = 512
_WORKER_DEQUEUE_TIMEOUT_S = 0.2


def _hevc_nal_type(b0: int) -> int:
    return (b0 >> 1) & 0x3F


def _is_vcl(nal_type: int) -> bool:
    # HEVC VCL NAL unit types are 0..31; 32+ are non-VCL (VPS/SPS/PPS/SEI/...).
    return nal_type <= 31


def qsv_hevc444_available() -> bool:
    """True iff `hevc_qsv` hardware-decodes a HEVC 4:4:4 sample to a hardware
    (packed) format. Mirrors hwcaps' probe but for the QSV decoder."""
    # A 64x64 HEVC Main 4:4:4 8-bit IDR access unit (same sample hwcaps uses).
    sample = bytes.fromhex(
        "0000000140010c01ffff0408000003009e280000030000baba0240000000014201"
        "010408000003009e280000030000ba90041020b2dd25261734040000030004003d"
        "090020000000014401c070306011200000012801ade0d117ffd39173238b80")
    try:
        ctx = av.CodecContext.create("hevc_qsv", "r")
        frames = list(ctx.decode(av.Packet(sample))) + list(ctx.decode(None))
        if not frames:
            return False
        # Software fallback would yield yuv444p/gbrp; a packed format means QSV.
        ok = frames[0].format.name not in ("yuv444p", "yuvj444p", "gbrp")
        log.info("hevc_qsv 4:4:4 probe: %s (fmt=%s)",
                 "HW-decoded" if ok else "software", frames[0].format.name)
        return ok
    except Exception as e:
        log.info("hevc_qsv 4:4:4 probe: unavailable (%s)", e)
        return False


class QsvHevcDecoder:
    """Drop-in for HevcDecoder using Intel Quick Sync `hevc_qsv` for 4:4:4."""

    def __init__(
        self,
        num_tiles: int,
        *,
        prefer_hwaccel: bool = True,   # accepted for interface parity (always HW)
        enable_quality_gate: bool = True,
        on_frame_published: Optional[Callable[[int], None]] = None,
    ) -> None:
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.num_tiles = num_tiles
        self._on_frame_published = on_frame_published
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._tiles: list[_TileSlot] = [_TileSlot() for _ in range(num_tiles)]
        # Parallel had-error flag per tile, set by _publish_frame and read by
        # get_frame. Protected by the corresponding slot.lock.
        self._tile_had_error: list[bool] = [False] * num_tiles
        self.nalu_counts_per_tile: list[dict[int, int]] = [{} for _ in range(num_tiles)]
        self._reformatter: list[Optional[av.video.reformatter.VideoReformatter]] = [None]
        self._seen_fmts: set[str] = set()

        self._codec: Optional[av.codec.context.CodecContext] = None
        self._codec_lock = threading.Lock()
        # Serializes the WHOLE start/restart/close lifecycle. Distinct from
        # _codec_lock, which only guards a single decode() against the
        # _codec=None swap. This one prevents two threads — the video-process
        # thread (SSRC adoption / param harvest) and the tx thread (stall /
        # saturation / FIR-exhaust watchdogs) — from running
        # _teardown()+_create_codec() concurrently. Two overlapping teardowns
        # double-free the QSV/MFX hardware session and start two qsv-decode
        # workers on the same D3D device → STATUS_ACCESS_VIOLATION in native
        # code. RLock so a future re-entrant lifecycle call can't self-deadlock;
        # the decode worker never takes this lock, so the join() in
        # _stop_worker() can't deadlock against a holder.
        self._lifecycle_lock = threading.RLock()
        self._hw_name: Optional[str] = "qsv"
        self._dpb_has_idr = False

        self._vps: Optional[bytes] = None
        self._sps: Optional[bytes] = None
        self._all_pps: dict[int, bytes] = {}

        # Access-unit batching state (only touched by the feeding thread / worker).
        self._au_buf = bytearray()
        self._au_has_params = False

        # PTS routing: hevc_qsv buffers ~17 frames before its first output, so
        # FIFO-by-submit-order misroutes every frame by the pipeline depth.
        # Instead we stamp each AU's packet with a monotonic pts, map pts→tile,
        # and look the tile up from the decoded frame's (preserved) pts. Exact
        # regardless of decode latency or any reordering.
        self._next_pts = 0
        self._pts_to_tile: dict[int, int] = {}
        self._pts_submit_t: dict[int, float] = {}
        self._decode_latency_ms: float = 0.0   # EMA; 0.0 until first frame

        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sync_decode_mode = False
        self._queue_full_drops: int = 0

    # -- configuration / lifecycle ------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict[int, bytes]) -> None:
        self._vps, self._sps, self._all_pps = vps, sps, dict(all_pps)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._codec is None:
                self._create_codec()
            self._start_worker()

    def _build_extradata(self) -> bytes:
        ed = bytearray()
        for ps in (self._vps, self._sps):
            if ps:
                ed += _NAL_START + ps
        for pid in sorted(self._all_pps):
            ed += _NAL_START + self._all_pps[pid]
        return bytes(ed)

    def _create_codec(self) -> None:
        # No extradata: hevc_qsv doesn't reliably accept Annex-B parameter sets
        # as extradata (yields "Picture size 0x0"). Instead the parameter sets
        # ride inline in each IRAP access unit (see feed_nalu), which is what
        # QSV's bitstream parser wants. low_delay trims decode buffering so the
        # first frames surface promptly instead of after a multi-frame delay.
        c = av.CodecContext.create("hevc_qsv", "r")
        try:
            c.flags |= 0x00080000  # AV_CODEC_FLAG_LOW_DELAY
        except Exception:
            pass
        try:
            # async_depth controls how many Access Units QSV keeps in flight
            # inside its internal pipeline simultaneously.
            #
            # The decode latency formula is:
            #   latency ≈ (async_depth - 1) × frame_interval
            # where frame_interval = 1000 ms / (tiles × fps) = 1000/(4×60) ≈ 4 ms.
            #
            # async_depth=1 — lowest latency (~0 ms pipeline overhead), but QSV
            #   must complete each frame before starting the next. At 240 AU/s the
            #   worker can starve the hardware between frames, causing the Python
            #   decode queue to fill up under sustained load. Before the QSV surface
            #   race was fixed (surface pool exhaustion blocked decode(), worsening
            #   the stall), this caused queue overflow at depth=1 reliably within
            #   ~30 s. After the fix it is more stable but still the tightest margin.
            #
            # async_depth=4 — very stable (4-frame buffer absorbs bursts and render-
            #   thread scheduling jitter), but pipeline latency ≈ 3×4 ms = 12 ms.
            #   Was the safe fallback before the surface race fix; latency too high
            #   for interactive use when compared to the software path (~2 ms).
            #
            # async_depth=2 — tested balance: latency ≈ 1×4 ms = 4 ms, throughput
            #   headroom enough to stay queue-free under 4-tile 60 fps after the
            #   surface race fix. Current setting.
            c.options = {"async_depth": "2", "gpu_copy": "on"}
        except Exception:
            pass
        c.open()
        self._codec = c
        self._next_pts = 0
        self._pts_to_tile = {}
        self._au_buf = bytearray()
        log.debug("hevc_qsv codec (re)created")

    def restart(self) -> None:
        with self._lifecycle_lock:
            self._teardown()
            if self._sps:
                self._create_codec()
                self._start_worker()

    def close(self) -> None:
        with self._lifecycle_lock:
            self._teardown()

    def _teardown(self) -> None:
        self._stop_worker()
        # Release all av.VideoFrame references BEFORE dropping the codec.
        # avcodec_free_context (triggered when self._codec ref-count reaches 0)
        # may free QSV's internal surface pool. If slot.raw_frame still holds a
        # live av.VideoFrame at that point, the frame's AVBufferRef points into
        # freed pool memory; subsequent GC of the frame object triggers a
        # double-free that corrupts the Windows heap (ntdll crash).
        for ti, slot in enumerate(self._tiles):
            with slot.lock:
                slot.raw_frame = None
                slot.good_count = 0
                slot.last_evaluated_count = 0
                self._tile_had_error[ti] = False
        with self._codec_lock:
            # Do NOT flush (decode(None)) before releasing the codec — hevc_qsv
            # intermittently crashes (STATUS_ACCESS_VIOLATION) in avcodec_send_packet
            # during flush when the QSV session is in certain states (e.g. after an
            # SSRC-group adoption or on session close).  avcodec_free_context, called
            # when the Python object is collected, performs the necessary hardware
            # teardown without triggering the flush code path.
            self._codec = None
        self._pts_to_tile = {}
        self._pts_submit_t = {}
        self._decode_latency_ms = 0.0
        self._next_pts = 0
        self._au_buf = bytearray()
        self._gate.reset()
        self._dpb_has_idr = False

    # -- feeding -------------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict[int, list[bytes]]) -> None:
        """Decode the session-start burst synchronously (round-robin across
        tiles, matching the live arrival order) so the consumer can start
        immediately. Always hardware here — no software-fallback probe."""
        if self._codec is None:
            self._create_codec()
        max_burst = max((len(v) for v in tile_nalu_cache.values()), default=0)
        if max_burst == 0:
            return
        self._sync_decode_mode = True
        fed = 0
        try:
            for idx in range(max_burst):
                for ti, nalus in tile_nalu_cache.items():
                    if idx < len(nalus):
                        self.feed_nalu(nalus[idx], ti)
                        fed += 1
        finally:
            self._sync_decode_mode = False
        good = sum(t.good_count for t in self._tiles)
        log.info("qsv burst complete: fed %d NALUs, decoded %d frames", fed, good)

    def feed_nalu(self, nalu: bytes, tile_idx: int) -> None:
        """Accumulate NALUs into an access unit; submit the AU when its VCL
        slice arrives (Apple = one slice per tile picture). Parameter-set NALUs
        ride inline with the slice."""
        if not nalu:
            return
        t = _hevc_nal_type(nalu[0])
        if 0 <= tile_idx < len(self.nalu_counts_per_tile):
            self.nalu_counts_per_tile[tile_idx][t] = (
                self.nalu_counts_per_tile[tile_idx].get(t, 0) + 1)
        if t in (32, 33, 34):          # VPS / SPS / PPS arrived inline
            self._au_has_params = True
        self._au_buf += _NAL_START + bytes(nalu)
        if _is_vcl(t):
            au = self._au_buf
            # An IRAP (IDR/BLA/CRA, types 16..23) must carry parameter sets for
            # hevc_qsv to size and root its DPB. The session harvests VPS/SPS/PPS
            # out of band, so burst IDR slices arrive bare — prepend the stored
            # params. Steady-state IDRs already include them inline (then
            # _au_has_params is set and we don't duplicate).
            if 16 <= t <= 23:
                # IRAP (IDR/BLA/CRA): tell the gate a keyframe landed so its
                # keyframe-required state clears and frames publish again. Apple
                # uses a cross-tile shared DPB, so any tile's IDR re-roots all
                # tiles — mark them all (mirrors HevcDecoder).
                for gi in range(self.num_tiles):
                    self._gate.mark_idr_observed(gi, suspicious=False)
                if not self._au_has_params:
                    params = self._build_extradata()
                    if params:
                        au = bytearray(params) + au
            au_bytes = bytes(au)
            self._au_buf = bytearray()
            self._au_has_params = False
            self._submit_au(au_bytes, tile_idx)

    def _submit_au(self, au: bytes, tile_idx: int) -> None:
        if self._sync_decode_mode:
            self._decode_au(au, tile_idx)
            return
        if self._queue is None or self._worker is None or not self._worker.is_alive():
            self._start_worker()
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait((au, tile_idx))
        except queue.Full:
            self._queue_full_drops += 1
            n = self._queue_full_drops
            if n in (1, 10, 100) or n % 1000 == 0:
                log.warning("qsv decoder queue FULL — dropped AU for tile %d (n=%d)",
                            tile_idx, n)

    # -- worker / decode ----------------------------------------------

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._worker_loop, name="qsv-decode", daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker is None:
            return
        self._stop.set()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if threading.current_thread() is not self._worker:
            self._worker.join(timeout=2.0)
        self._worker = None
        self._queue = None

    def _worker_loop(self) -> None:
        q = self._queue
        if q is None:
            return
        while not self._stop.is_set():
            try:
                item = q.get(timeout=_WORKER_DEQUEUE_TIMEOUT_S)
            except queue.Empty:
                continue
            if item is None:
                break
            au, ti = item
            try:
                self._decode_au(au, ti)
            except Exception as e:
                log.warning("qsv-decode worker swallowed error: %s", e)

    def _decode_au(self, au: bytes, tile_idx: int) -> None:
        codec = self._codec
        if codec is None:
            return
        import time as _time
        pts = self._next_pts
        self._next_pts += 1
        self._pts_to_tile[pts] = tile_idx
        self._pts_submit_t[pts] = _time.monotonic()
        if len(self._pts_to_tile) > 256:
            cutoff = pts - 128
            self._pts_to_tile = {k: v for k, v in self._pts_to_tile.items() if k > cutoff}
            self._pts_submit_t = {k: v for k, v in self._pts_submit_t.items() if k > cutoff}
        pkt = av.Packet(au)
        pkt.pts = pkt.dts = pts
        pkt.time_base = Fraction(1, 90000)
        try:
            with self._codec_lock:
                if self._codec is None:
                    return
                frames = self._codec.decode(pkt)
                # Publish inside the lock: _teardown() holds the same lock
                # when it sets self._codec = None and triggers
                # avcodec_free_context. Processing frames outside the lock
                # would let the codec be freed while we still iterate the
                # returned frame list, causing use-after-free in QSV's
                # internal surface pool -> ntdll heap corruption.
                for frame in frames:
                    self._publish_frame(frame)
        except Exception as e:
            log.debug("qsv decode error (tile %d): %s", tile_idx, e)

    def _publish_frame(self, frame: av.VideoFrame) -> None:
        import time as _time
        ti = self._pts_to_tile.pop(frame.pts, None)
        submit_t = self._pts_submit_t.pop(frame.pts, None)
        if ti is None:
            return
        if submit_t is not None:
            latency_ms = (_time.monotonic() - submit_t) * 1000
            # EMA with α=0.1 — ~10-frame smoothing window (~42 ms at 240 AU/s).
            self._decode_latency_ms = (0.1 * latency_ms
                                       + 0.9 * self._decode_latency_ms)
        if not self._dpb_has_idr:
            self._dpb_has_idr = True
        # Convert av.VideoFrame → TileFrame (pure Python bytes) while still
        # inside _codec_lock (called from _decode_au which holds the lock).
        # QSV's surface pool cannot be reclaimed during a concurrent decode()
        # call while we hold the lock, so reading frame pixel data here is
        # safe. Storing a TileFrame (not av.VideoFrame) in the slot means the
        # render thread never touches a QSV surface, eliminating the
        # STATUS_ACCESS_VIOLATION in _av_frame_to_tile at get_frame() time.
        tile_frame, had_err = _av_frame_to_tile(frame, self._reformatter, self._seen_fmts)
        if tile_frame is None:
            return
        slot = self._tiles[ti]
        with slot.lock:
            slot.raw_frame = tile_frame  # type: ignore[assignment]
            self._tile_had_error[ti] = had_err
            slot.good_count += 1
        if self._on_frame_published is not None:
            try:
                self._on_frame_published(ti)
            except Exception as e:
                log.debug("on_frame_published raised: %s", e)

    # -- consumer API -------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        if not self._dpb_has_idr:
            return None
        slot = self._tiles[tile_idx]
        with slot.lock:
            tile_frame = slot.raw_frame  # TileFrame stored by _publish_frame
            count = slot.good_count
            already = count <= slot.last_evaluated_count
            had_err = self._tile_had_error[tile_idx]
            self._tile_had_error[tile_idx] = False
        if tile_frame is None or already:
            return None
        with slot.lock:
            slot.last_evaluated_count = count
        if had_err:
            self._gate.mark_decode_error(tile_idx)
        else:
            self._gate.mark_clean(tile_idx)
        if not self._gate.should_publish(tile_idx, tile_frame):  # type: ignore[arg-type]
            return None
        return tile_frame  # type: ignore[return-value]

    def consume_fir_request(self) -> set[int]:
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int) -> TileVisState:
        return self._gate.tile_state(tile_idx)

    @property
    def decode_latency_ms(self) -> float:
        """EMA of time from AU submit to frame output (ms). Reflects async
        pipeline depth: async_depth=1 ≈ hardware decode time; higher values
        ≈ hardware decode time + (depth-1) × frame interval."""
        return self._decode_latency_ms

    @property
    def decode_queue_depth(self) -> int:
        q = self._queue
        return q.qsize() if q is not None else 0

    @property
    def decode_queue_cap(self) -> int:
        return _QUEUE_MAX

    @property
    def decode_queue_drops(self) -> int:
        return self._queue_full_drops

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def good_counts(self) -> list[int]:
        return [t.good_count for t in self._tiles]


__all__ = ["QsvHevcDecoder", "qsv_hevc444_available"]
