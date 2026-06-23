"""H.264 / AVC decoder for Apple's HP stream (the AVC codec bank, RTP PT 123).

When the client advertises AVC instead of HEVC (``ISS_VIDEO_CODEC=avc`` /
``--codec avc``), the host streams **H.264 High@4.2, 4:2:0** — the only chroma
the host's H.264 encoder produces (it has no 4:4:4 H.264 profile). The win:
4:2:0 H.264 is hardware-decodable on essentially every GPU (Windows D3D11VA,
Linux VAAPI, macOS VideoToolbox), whereas Apple's HEVC RExt **4:4:4** forces a
CPU fallback on consumer GPUs that only do HEVC Main 4:2:0.

This mirrors `HevcDecoder`'s public interface (so `Session` uses them
interchangeably) and its proven machinery — one shared `av.CodecContext`
fed all tiles in interleaved order, a single decode-worker thread, PTS→tile
dispatch, HW→SW fallback measured during the start burst, and native-aligned
no-flush wedge recovery — but is deliberately leaner: Apple's H.264 stream
has none of the HEVC quirks (cross-tile RExt POC references, suspicious
minimal-IDRs, the libav-VT-hwaccel DPB stall) that justify HevcDecoder's RPS
tracker and suspicious-IDR heuristics, so those are omitted here.

Frame extraction (`_av_frame_to_tile`) and the FIR/quality gate are shared
with the HEVC path; only the codec name, NAL parsing, and extradata differ.
"""
from __future__ import annotations

import errno
import logging
import os
import queue
import threading
from typing import Callable, Optional

import av

from .tiles import TileFrame
from .nalu import H264_NAL_IDR
from .quality_gate import FrameQualityGate, TileVisState
from .hevc import (
    _av_frame_to_tile, _TileSlot, _platform_hwaccels,
    _NAL_START_CODE, _CODEC_FLAG_LOW_DELAY, _CODEC_FLAG2_FAST,
    _QUEUE_MAX, _WORKER_DEQUEUE_TIMEOUT_S, _WORKER_JOIN_TIMEOUT_S,
    _HWACCEL_SILENT_NALU_LIMIT, _HWACCEL_BURST_ERROR_THRESHOLD,
    _HWACCEL_BURST_ERROR_WINDOW, _HWACCEL_BURST_MIN_FRAMES,
    _PTS_MAP_SOFT_MAX, _PTS_MAP_PRUNE_KEEP,
)


log = logging.getLogger(__name__)

# Same diagnostic env as hevc.py: dump every fed NALU (Annex-B) for offline
# inspection with a stock H.264 player.
_NALU_DUMP_PATH = os.environ.get("ISS_NALU_DUMP")
_nalu_dump_f = open(_NALU_DUMP_PATH, "wb") if _NALU_DUMP_PATH else None

def _h264_type(b: int) -> int:
    return b & 0x1F


class AvcDecoder:
    """N-tile H.264 decoder with one shared codec context and HW→SW fallback.

    Public interface matches `HevcDecoder`/`VTHevcDecoder` so `Session` can
    swap it in: set_params / start / feed_burst / feed_nalu / get_frame /
    consume_fir_request / tile_state / restart / close / hw_accel /
    good_counts / nalu_counts_per_tile.
    """

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
        self._prefer_hwaccel = prefer_hwaccel
        self._on_frame_published = on_frame_published

        self._codec: Optional[av.codec.context.CodecContext] = None
        self._codec_lock = threading.Lock()
        # Serializes the whole start/restart/close/fallback lifecycle (see the
        # matching note in hevc.py). restart() is driven from both the
        # video-process thread and the tx-loop watchdogs; without this, two
        # concurrent _teardown()+_create_codec() sequences corrupt codec/worker
        # state and can double-free the native context. RLock: re-entrant-safe;
        # the decode worker never takes it, so _stop_worker()'s join can't
        # deadlock against a holder.
        self._lifecycle_lock = threading.RLock()
        self._reformatter: list = [None]
        self._seen_fmts: set[str] = set()
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._tiles: list[_TileSlot] = [_TileSlot() for _ in range(num_tiles)]
        self.nalu_counts_per_tile: list[dict[int, int]] = [
            {} for _ in range(num_tiles)
        ]

        # Parameter sets. For H.264, `vps` is unused (kept in the signature
        # for interface parity); `sps` and the PPS pool seed the extradata.
        self._sps: Optional[bytes] = None
        self._all_pps: dict[int, bytes] = {}

        self._hw_name: Optional[str] = None
        self._hw_failed = False

        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sync_decode_mode = False

        self._next_pts = 0
        self._pts_to_tile: dict[int, int] = {}
        self._eagain_streak = 0
        self._dpb_has_idr = False
        self._burst_cache: dict[int, list[bytes]] = {}
        self._silent_nalus = 0
        self._recovery_in_progress = False
        # Frames fed since the last IDR (any source). The session uses this to
        # request a fresh IDR before poc_lsb wraps (~8192) on the AVC hardware
        # path — the rollover the d3d11va decoder mishandles. See _maybe_reanchor.
        self.frames_since_idr = 0

    # -- public configuration ------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict[int, bytes]) -> None:
        """Install parameter sets. `vps` is ignored (H.264 has none)."""
        self._sps = sps
        self._all_pps = dict(all_pps)

    def start(self) -> None:
        if not (self._sps and self._all_pps):
            raise RuntimeError("set_params() must be called before start()")
        with self._lifecycle_lock:
            self._create_codec(force_software=False)

    # -- decoder feed --------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict[int, list[bytes]]) -> None:
        """Decode the session-start IDR cache synchronously in round-robin
        tile order, measuring hwaccel health so we can drop to software
        before the consumer starts asking for frames."""
        self._burst_cache = {ti: list(n) for ti, n in tile_nalu_cache.items()}
        max_burst = max((len(v) for v in tile_nalu_cache.values()), default=0)
        if max_burst == 0:
            log.info("feed_burst: empty cache — gate stays armed")
            self._gate.reset()
            return

        self._sync_decode_mode = True
        is_hw = self._hw_name is not None
        fed = 0
        per_tile_seen_idr = {ti: False for ti in tile_nalu_cache}
        per_tile_expected = 0
        good_at_start = sum(t.good_count for t in self._tiles)
        try:
            for idx in range(max_burst):
                for ti, nalus in tile_nalu_cache.items():
                    if idx < len(nalus):
                        nalu = nalus[idx]
                        if nalu and _h264_type(nalu[0]) == H264_NAL_IDR:
                            per_tile_seen_idr[ti] = True
                        if per_tile_seen_idr[ti]:
                            per_tile_expected += 1
                        self._decode_one(nalu, ti)
                        fed += 1
                if is_hw and (idx + 1) % 8 == 0 and per_tile_expected > 0:
                    decoded = sum(t.good_count for t in self._tiles) - good_at_start
                    errs = max(0, per_tile_expected - decoded)
                    if (errs > _HWACCEL_BURST_ERROR_THRESHOLD
                            and per_tile_expected > _HWACCEL_BURST_ERROR_WINDOW):
                        log.warning("hwaccel burst failing (%d/%d) — switching to software",
                                    errs, per_tile_expected)
                        self._fallback_to_software(tile_nalu_cache)
                        return
        finally:
            self._sync_decode_mode = False

        good = sum(t.good_count for t in self._tiles)
        log.info("burst complete: fed %d NALUs, expected %d, decoded %d (%s)",
                 fed, per_tile_expected, good, self._hw_name or "software")
        if (is_hw and per_tile_expected > _HWACCEL_BURST_ERROR_WINDOW
                and good < _HWACCEL_BURST_MIN_FRAMES):
            log.warning("hwaccel produced only %d frames from %d expected — software",
                        good, per_tile_expected)
            self._fallback_to_software(tile_nalu_cache)

    def feed_nalu(self, nalu: bytes, tile_idx: int) -> None:
        """Queue one NALU for the decode worker (or decode synchronously during
        the start burst). H.264 has no RPS pre-check — we rely on libav's
        post-decode concealment flags, surfaced via the quality gate."""
        if nalu and 0 <= tile_idx < len(self.nalu_counts_per_tile):
            t = _h264_type(nalu[0])
            self.nalu_counts_per_tile[tile_idx][t] = (
                self.nalu_counts_per_tile[tile_idx].get(t, 0) + 1
            )
        if self._sync_decode_mode:
            self._decode_one(nalu, tile_idx)
            return
        if self._queue is None or self._worker is None or not self._worker.is_alive():
            self._start_worker()
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait((nalu, tile_idx))
        except queue.Full:
            self._queue_full_drops = getattr(self, "_queue_full_drops", 0) + 1
            n = self._queue_full_drops
            if n in (1, 10, 100, 1000) or n % 1000 == 0:
                log.warning("decoder queue FULL — dropped slice for tile %d (n=%d)",
                            tile_idx, n)

    # -- consumer API --------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        """Latest decoded frame for `tile_idx`, or None. Returns None until the
        shared context has decoded an IDR (output before that is cold-DPB fill)."""
        if not self._dpb_has_idr:
            return None
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.raw_frame
            count = slot.good_count
            already = count <= slot.last_evaluated_count
        if frame is None or already:
            return None
        tile_frame, had_err = _av_frame_to_tile(frame, self._reformatter, self._seen_fmts)
        with slot.lock:
            slot.last_evaluated_count = count
        if tile_frame is None:
            return None
        if had_err:
            self._gate.mark_decode_error(tile_idx)
        else:
            self._gate.mark_clean(tile_idx)
        if not self._gate.should_publish(tile_idx, tile_frame):
            return None
        return tile_frame

    def consume_fir_request(self) -> set[int]:
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int) -> TileVisState:
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def good_counts(self) -> list[int]:
        return [t.good_count for t in self._tiles]

    # -- lifecycle -----------------------------------------------------

    def restart(self) -> None:
        with self._lifecycle_lock:
            self._teardown()
            if self._sps and self._all_pps:
                self._create_codec(force_software=self._hw_failed)

    def soft_reset(self) -> None:
        """Reset DPB tracking without flushing the codec context.

        On plain SSRC group adoption (same canvas resolution), the decoder's
        DPB frames from the prior SSRC group remain valid references for the
        new stream. Apple's FIR responses are 'P-IDR' NALUs — nal_unit_type=5
        (IDR) in the header but a non-intra slice that references existing DPB
        entries. A full restart() wipes the DPB, so every P-IDR fails with
        "A non-intra slice in an IDR NAL unit", libav fires "no frame!", the
        concealment handler sends another FIR, and the encoder restarts again
        every ~2 s — a sustained FIR storm. With an intact DPB the P-IDRs
        decode immediately, frames flow, and the encoder's next natural I-IDR
        re-anchors the DPB cleanly."""
        self._try_recovery()
        self._recovery_in_progress = False

    def close(self) -> None:
        with self._lifecycle_lock:
            self._teardown()

    # -- internals: decode --------------------------------------------

    def _decode_one(self, nalu: bytes, tile_idx: int) -> None:
        codec = self._codec
        if codec is None or not _is_decodable_h264(nalu):
            return
        nal_type = _h264_type(nalu[0])
        if nal_type == H264_NAL_IDR:
            # An IDR resets the stream's frame_num/poc_lsb to 0, so reset our
            # since-IDR frame counter too. The session watches this counter to
            # request a fresh IDR before poc_lsb would wrap (~8192 frames) — the
            # rollover the d3d11va decoder mishandles. Counts every IDR source:
            # startup burst, SSRC adoption, and our own re-anchor requests.
            self.frames_since_idr = 0
            slot = self._tiles[tile_idx]
            with slot.lock:
                slot.saw_idr_since_reset = True
            for ti in range(len(self._tiles)):
                self._gate.mark_idr_observed(ti, suspicious=False)
            if not self._dpb_has_idr:
                self._dpb_has_idr = True
        elif not self._dpb_has_idr:
            # Drop P-frames until the first IDR roots the DPB.
            self._pre_idr_drops = getattr(self, "_pre_idr_drops", 0) + 1
            return
        else:
            # A non-IDR frame that will be fed: advance the since-IDR counter.
            # Tracks the stream's poc/frame_num progression (Apple steps both by
            # 1 per frame) so the session can re-anchor before the poc wrap.
            self.frames_since_idr = getattr(self, "frames_since_idr", 0) + 1

        if _nalu_dump_f is not None:
            _nalu_dump_f.write(_NAL_START_CODE + bytes(nalu))

        pkt = av.Packet(_NAL_START_CODE + bytes(nalu))
        pts = self._next_pts
        pkt.pts = pkt.dts = pts
        self._pts_to_tile[pts] = tile_idx
        self._next_pts += 1
        if len(self._pts_to_tile) > _PTS_MAP_SOFT_MAX:
            cutoff = pts - _PTS_MAP_PRUNE_KEEP
            self._pts_to_tile = {k: v for k, v in self._pts_to_tile.items() if k > cutoff}

        try:
            with self._codec_lock:
                if self._codec is None:
                    return
                frames = self._codec.decode(pkt)
            published = False
            for frame in frames:
                self._publish_frame(frame)
                published = True
            self._eagain_streak = 0
            if published:
                self._silent_nalus = 0
                self._recovery_in_progress = False
            else:
                self._silent_nalus += 1
                if (self._silent_nalus > _HWACCEL_SILENT_NALU_LIMIT
                        and not self._recovery_in_progress):
                    log.warning("%s silent for %d NALUs — drop P-frames + FIR (no flush)",
                                self._hw_name or "software", self._silent_nalus)
                    self._recovery_in_progress = True
                    self._try_recovery()
                    self._silent_nalus = 0
        except Exception as e:
            self._handle_decode_error(tile_idx, nalu, e)

    def _publish_frame(self, frame: av.VideoFrame) -> None:
        ti = self._pts_to_tile.pop(frame.pts, None)
        if ti is None:
            return
        slot = self._tiles[ti]
        with slot.lock:
            slot.raw_frame = frame
            slot.good_count += 1
        if self._on_frame_published is not None:
            try:
                self._on_frame_published(ti)
            except Exception as e:
                log.debug("on_frame_published raised: %s", e)

    def _handle_decode_error(self, tile_idx: int, nalu: bytes, exc: Exception) -> None:
        """Native-aligned: FIR on a genuine error, never flush; only a sustained
        wedge (no output across a full frame's NALUs) escalates to no-flush
        recovery. EAGAIN is pure backpressure — don't FIR on it."""
        err_no = getattr(exc, "errno", 0) or 0
        if err_no != errno.EAGAIN:
            self._gate.mark_decode_error(tile_idx)
        self._eagain_streak += 1
        if (not self._recovery_in_progress
                and self._eagain_streak >= _HWACCEL_SILENT_NALU_LIMIT):
            self._recovery_in_progress = True
            self._gate.mark_decode_error(tile_idx)
            log.warning("%s wedged (errno=%d, %d consecutive errors) — FIR + drain (no flush)",
                        self._hw_name or "software", err_no, self._eagain_streak)
            self._try_recovery()

    def _try_recovery(self) -> None:
        """No-flush recovery: stop feeding P-frames until the next IDR re-roots
        the DPB, and drain the stale backlog so the worker reaches that IDR
        fast. Never `flush_buffers()` — that wipes the shared DPB and cascades."""
        self._dpb_has_idr = False
        for slot in self._tiles:
            with slot.lock:
                slot.saw_idr_since_reset = False
        self._eagain_streak = 0
        q = self._queue
        if q is not None:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    # -- internals: worker --------------------------------------------

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(target=self._worker_loop, name="avc-decode", daemon=True)
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
            self._worker.join(timeout=_WORKER_JOIN_TIMEOUT_S)
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
            nalu, ti = item
            try:
                self._decode_one(nalu, ti)
            except Exception as e:
                log.warning("avc-decode worker swallowed error: %s", e)

    # -- internals: codec construction --------------------------------

    def _build_extradata(self) -> bytes:
        """SPS + all PPSes as Annex B — H.264 has no VPS."""
        ed = bytearray()
        ed += _NAL_START_CODE + self._sps  # type: ignore[operator]
        for pid in sorted(self._all_pps.keys()):
            ed += _NAL_START_CODE + self._all_pps[pid]
        return bytes(ed)

    def _create_codec(self, *, force_software: bool) -> None:
        extradata = self._build_extradata()
        if not force_software and self._prefer_hwaccel and not self._hw_failed:
            for hw_type in _platform_hwaccels():
                ctx = self._try_hwaccel(hw_type, extradata)
                if ctx is not None:
                    self._install_codec(ctx, hw_name=hw_type)
                    return
        self._install_codec(self._make_sw_context(extradata), hw_name=None)

    def _try_hwaccel(self, hw_type: str, extradata: bytes):
        try:
            from av.codec.hwaccel import HWAccel
            # ISS_D3D11_ADAPTER selects which GPU does d3d11va decode by DXGI
            # adapter index (0 = default/primary, usually the discrete GPU). Use
            # this to pick the Intel iGPU over a discrete AMD/NVIDIA card — e.g.
            # to dodge the AMD VCN H.264 POC-wrap bug. Applies to d3d11va only.
            device = os.environ.get("ISS_D3D11_ADAPTER")
            if device and hw_type in ("d3d11va", "d3d11"):
                hw = HWAccel(device_type=hw_type, device=device)
                log.info("d3d11va: using DXGI adapter index %s", device)
            else:
                hw = HWAccel(device_type=hw_type)
            c = av.CodecContext.create("h264", "r", hwaccel=hw)
            c.extradata = extradata
            c.thread_type = "NONE"
            c.thread_count = 1
            c.flags = _CODEC_FLAG_LOW_DELAY
            c.flags2 = _CODEC_FLAG2_FAST
            c.open()
            return c
        except Exception as e:
            log.info("hwaccel %s unavailable: %s", hw_type, e)
            return None

    def _make_sw_context(self, extradata: bytes):
        c = av.CodecContext.create("h264", "r")
        c.extradata = extradata
        c.thread_type = "NONE"
        c.thread_count = 1
        c.flags = _CODEC_FLAG_LOW_DELAY
        c.flags2 = _CODEC_FLAG2_FAST
        c.open()
        return c

    def _install_codec(self, codec, *, hw_name: Optional[str]) -> None:
        self._codec = codec
        self._hw_name = hw_name
        self._reformatter[0] = None
        self._seen_fmts.clear()
        self._next_pts = 0
        self._pts_to_tile = {}
        log.info("H.264 decoder: shared context (%s)", hw_name or "software")
        if _nalu_dump_f is not None:
            _nalu_dump_f.write(self._build_extradata())
            _nalu_dump_f.flush()

    def _fallback_to_software(self, tile_nalu_cache: Optional[dict[int, list[bytes]]]) -> None:
        log.warning("falling back from %s to software decode", self._hw_name)
        with self._lifecycle_lock:
            self._hw_failed = True
            self._teardown()
            self._create_codec(force_software=True)
            if tile_nalu_cache:
                self._sync_decode_mode = True
                try:
                    max_burst = max((len(v) for v in tile_nalu_cache.values()), default=0)
                    for idx in range(max_burst):
                        for ti, nalus in tile_nalu_cache.items():
                            if idx < len(nalus):
                                self._decode_one(nalus[idx], ti)
                    log.info("software fallback burst: %d frames", sum(t.good_count for t in self._tiles))
                finally:
                    self._sync_decode_mode = False

    def _teardown(self) -> None:
        self._stop_worker()
        with self._codec_lock:
            if self._codec is not None:
                try:
                    list(self._codec.decode(None))
                except Exception:
                    pass
                try:
                    self._codec.flush_buffers()
                except Exception:
                    pass
                self._codec = None
        for slot in self._tiles:
            with slot.lock:
                slot.raw_frame = None
                slot.good_count = 0
                slot.last_evaluated_count = 0
                slot.saw_idr_since_reset = False
        self._gate.reset()
        self._eagain_streak = 0
        self._reformatter[0] = None
        self._next_pts = 0
        self._pts_to_tile = {}
        self._dpb_has_idr = False
        self._pre_idr_drops = 0


def _is_decodable_h264(nalu: bytes) -> bool:
    """Accept real H.264 NAL types (1..23); drop empties and the >23
    application/reserved range libav doesn't want."""
    if len(nalu) < 1:
        return False
    t = nalu[0] & 0x1F
    return 1 <= t <= 23


__all__ = ["AvcDecoder"]
