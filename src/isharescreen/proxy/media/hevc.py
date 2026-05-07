"""HEVC decoder for Apple's HP stream.

ONE shared `av.CodecContext` for the whole session, fed all tiles' NALUs
in `(timestamp, tile_idx)` interleaved order. Output frames are dispatched
back to the right tile via the PTS we attach on input.

Why one shared context: Apple's stream has cross-tile POC references —
P-frames in tile N reference POCs assigned to frames in tiles M ≠ N. A
per-tile-per-context design (one `av.CodecContext` per tile) gives each
tile its own DPB, so cross-tile references resolve to nothing → bad
output. libavcodec's HEVC decoder is one DPB per `av.CodecContext`, so
the only correct architecture is a single context per session.

Decode runs on a single decoder thread that pulls from a queue. The Session
RX threads enqueue NALUs and return; libavcodec releases the GIL inside
``codec.decode()`` so the rest of the process stays responsive while the
HEVC heavy lifting happens.

Hardware acceleration is requested per-platform (VideoToolbox on macOS,
D3D11VA on Windows, VAAPI on Linux). Software libavcodec is the universal
fallback. The HW path matches Apple Screen Sharing perf when the underlying
silicon supports HEVC RExt 4:4:4 (RDNA3+, Intel Arc/12th gen+, NVIDIA Ada+).

The frame-stats quality gate (gray / black / green-pop detection + FIR
escalation) lives in `quality_gate.py`.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional


# Diagnostic: when set, every NALU fed to libavcodec is also written
# (Annex-B framed) to this path. Pipe the resulting file through any
# stock HEVC player (`mpv file.h265`, `ffplay -f hevc -i file.h265`)
# to confirm whether the bitstream itself decodes cleanly outside our
# wgpu/render path. ~No overhead when unset.
_NALU_DUMP_PATH = os.environ.get("ISS_NALU_DUMP")
_nalu_dump_f = open(_NALU_DUMP_PATH, "wb") if _NALU_DUMP_PATH else None

import av
import numpy as np

from .tiles import TileFrame
from .nalu import IDR_RANGE
from .quality_gate import FrameQualityGate, TileVisState


log = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────

_NAL_START_CODE = b"\x00\x00\x00\x01"

# libavcodec flags. AV_CODEC_FLAG_LOW_DELAY = 0x80000 disables frame
# reordering buffers. AV_CODEC_FLAG2_FAST = 0x400000 enables non-bitexact
# but faster decode paths (acceptable for live screen-share).
_CODEC_FLAG_LOW_DELAY = 0x00080000
_CODEC_FLAG2_FAST = 0x00400000

# Decode-worker queue + lifecycle.
_QUEUE_MAX = 512                           # NALUs in flight to the decoder
_WORKER_DEQUEUE_TIMEOUT_S = 0.5
_WORKER_JOIN_TIMEOUT_S = 2.0

# HW-accel fallback thresholds — empirical, tuned against Apple Silicon
# + Linux iGPU.
# 10 was too aggressive: VideoToolbox routinely emits 10+ EAGAINs during
# warm-up (codec buffering input before producing output). 50 lets VT
# settle without being mistaken for a broken accelerator.
_HWACCEL_EAGAIN_STREAK_LIMIT = 50
# Number of NALUs we feed to a hwaccel context with zero output frames
# before forcing a fallback to software. VideoToolbox can decode the
# initial burst successfully and then go silent on Apple HEVC RExt
# 4:4:4 — `decode()` returns no error but emits no frames. EAGAIN
# logic doesn't catch this; we'd otherwise wait for the SSRC-adoption
# stall window (10-15s) before realising decode is dead.
_HWACCEL_SILENT_NALU_LIMIT = 60  # one source frame's worth of NALUs at 60 fps
_HWACCEL_BURST_ERROR_THRESHOLD = 20
_HWACCEL_BURST_ERROR_WINDOW = 40
_HWACCEL_BURST_MIN_FRAMES = 5

# PTS map size. Each fed NALU records `pts → tile_idx`; output frames
# look up by pts to dispatch to the right tile slot. We prune to keep
# the dict bounded on lossy streams where some PTSes never produce
# output frames.
_PTS_MAP_SOFT_MAX = 4000
_PTS_MAP_PRUNE_KEEP = 2000

# Per-platform HW accel candidate list, in priority order.
_PLATFORM_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "d3d12va"),
    "*": ("vaapi", "cuda"),
}


def _platform_hwaccels() -> tuple[str, ...]:
    return _PLATFORM_HWACCELS.get(sys.platform, _PLATFORM_HWACCELS["*"])


# ── per-tile output state ────────────────────────────────────────────

@dataclass(slots=True)
class _TileSlot:
    """The latest decoded frame for one tile + its bookkeeping. Locked so
    the decode worker (writer) and the consumer's `get_frame()` (reader)
    don't tear."""
    raw_frame: Optional[av.VideoFrame] = None
    good_count: int = 0
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
    "videotoolbox_vld",
    "cuda",
    "drm_prime",
    "mediacodec",
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

    if fmt in ("nv24", "nv42"):
        # Biplanar 4:4:4: Y plane + full-resolution interleaved UV. This is
        # what VideoToolbox delivers for HEVC RExt 4:4:4 8-bit streams (the
        # libavcodec name for the CV format `444f`/`kCVPixelFormatType_
        # 444YpCbCr8BiPlanarFullRange`). Apple's screen-share is the
        # canonical example. Same deinterleave as NV12 but at full chroma
        # resolution.
        yp = frame.planes[0]
        uvp = frame.planes[1]
        uv_view = (
            np.frombuffer(bytes(uvp), dtype=np.uint8)
            .reshape(height, uvp.line_size)[:, :width * 2]
        )
        if fmt == "nv24":
            u_bytes = uv_view[:, 0::2].tobytes()
            v_bytes = uv_view[:, 1::2].tobytes()
        else:
            v_bytes = uv_view[:, 0::2].tobytes()
            u_bytes = uv_view[:, 1::2].tobytes()
        return TileFrame(
            y=bytes(yp), u=u_bytes, v=v_bytes,
            width=width, height=height,
            y_stride=yp.line_size,
            uv_stride=width,
            chroma_width=width,
            chroma_height=height,
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


# ── main decoder ─────────────────────────────────────────────────────

class HevcDecoder:
    """N-tile HEVC decoder with one shared codec context and HW→SW fallback.

    Lifecycle:
        dec = HevcDecoder(num_tiles=4)
        dec.set_params(vps, sps, all_pps)         # from gather_initial_burst
        dec.start()
        dec.feed_burst(tile_nalus)                # session-start IDRs (sync)
        # … steady state …
        for nalu, ti in stream:
            dec.feed_nalu(nalu, ti)
        # … render thread asks …
        frame = dec.get_frame(ti)
        dec.close()

    Threading:
        - `feed_nalu` / `feed_burst` may be called from any thread.
        - During burst, decode runs synchronously on the calling thread.
        - In steady state, NALUs are queued; one decoder worker consumes.
        - `get_frame` may be called from any thread; per-tile locks make
          reads consistent with worker writes.
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
        # Held during every codec.decode() call. `restart()` (called
        # from the RX thread on SSRC adoption) must acquire this lock
        # before tearing down the codec, otherwise the worker thread
        # can be mid-decode when we null `_codec` and libavcodec
        # SEGVs on the next dereference. We've seen this crash named
        # `libavcodec...+0x43a9d0` from the `hevc-decode` thread.
        self._codec_lock = threading.Lock()
        self._reformatter: list[Optional[av.video.reformatter.VideoReformatter]] = [None]
        self._seen_fmts: set[str] = set()
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._tiles: list[_TileSlot] = [_TileSlot() for _ in range(num_tiles)]
        # RPS-based pre-decode error detector. We feed every slice
        # NALU through the tracker before queueing it for decode; if
        # the slice references a POC we never saw, we know the
        # decoder will conceal and we mark the tile as needing FIR.
        from .hevc_rps import HevcRpsTracker  # local to avoid cycles
        self._rps_tracker = HevcRpsTracker()

        # Codec parameter sets, set by `set_params`.
        self._vps: Optional[bytes] = None
        self._sps: Optional[bytes] = None
        self._all_pps: dict[int, bytes] = {}

        # HW state.
        self._hw_name: Optional[str] = None
        self._hw_failed = False

        # Decode worker + queue.
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # When True (during feed_burst), feed_nalu runs decode synchronously
        # on the calling thread instead of queueing.
        self._sync_decode_mode = False

        # PTS bookkeeping. Only the active decoder thread (sync mode = main
        # caller; async mode = worker) reads/writes these.
        self._next_pts = 0
        self._pts_to_tile: dict[int, int] = {}
        self._eagain_streak = 0
        # Session-wide IDR seen since last decoder reset. Cross-tile DPB
        # references mean every tile's frames become visually meaningful
        # the moment ANY tile delivers an IDR.
        self._dpb_has_idr = False
        # Burst cache for fallback re-feed (set by feed_burst).
        self._burst_cache: dict[int, list[bytes]] = {}
        # Counts how many NALUs we've fed since the last successful
        # publish — used to spot a silent-but-not-erroring hwaccel
        # context. See `_HWACCEL_SILENT_NALU_LIMIT`.
        self._silent_nalus = 0
        # True between the moment we observe a decode error and the
        # moment the next successful publish lands. Prevents the
        # flush+IDR-wait recovery cycle from re-triggering on every
        # subsequent EAGAIN we get while waiting for the FIR-driven
        # IDR to arrive.
        self._recovery_in_progress = False

    # -- public configuration ------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict[int, bytes]) -> None:
        """Install the parameter sets harvested from the initial burst."""
        self._vps = vps
        self._sps = sps
        self._all_pps = dict(all_pps)
        # Feed the SPS into the RPS tracker so it can interpret
        # subsequent slice headers. SPS bytes here are the full NAL
        # payload (header + RBSP); strip the 2-byte NAL header.
        if len(sps) > 2:
            self._rps_tracker.feed_sps(sps[2:])

    def start(self) -> None:
        """Build the codec context (HW first if preferred + available, SW
        fallback otherwise). The decoder worker spins up lazily on the
        first `feed_nalu` outside burst mode."""
        if not (self._vps and self._sps and self._all_pps):
            raise RuntimeError("set_params() must be called before start()")
        self._create_codec(force_software=False)

    # -- decoder feed --------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict[int, list[bytes]]) -> None:
        """Decode the session-start IDR cache synchronously, in
        `(timestamp_index, tile_idx)` interleaved order so cross-tile
        POC references resolve in the shared DPB.

        Synchronous decode lets us measure burst success: if hwaccel is
        broken on this host, we fall back to software before the consumer
        starts asking for frames.
        """
        # Cache for re-feeding on later fallbacks (EAGAIN streak,
        # silent NALU). Without this, a mid-session SW fallback
        # produces no IDR until the next live IDR arrives — which can
        # be tens of seconds for stable Apple streams.
        self._burst_cache = {ti: list(nalus) for ti, nalus in tile_nalu_cache.items()}
        max_burst = max(
            (len(tile_nalu_cache[ti]) for ti in tile_nalu_cache),
            default=0,
        )
        if max_burst == 0:
            log.info("feed_burst: empty cache — gate stays armed")
            self._gate.reset()
            return

        self._sync_decode_mode = True
        is_hw = self._hw_name is not None
        fed = 0
        # Per-tile counts of NALUs we *expect* to decode — only NALUs
        # fed AFTER the tile has received an IDR in this burst. P/B
        # frames preceding the tile's IDR have no reference and are
        # always going to fail; counting them as hwaccel errors makes
        # us drop to SW whenever Apple's burst is heavy on TRAIL_R but
        # only carries one IDR (the common case after an SSRC
        # rollover). SW HEVC RExt 4:4:4 produces torn output so this
        # mis-classification is visually catastrophic.
        per_tile_seen_idr: dict[int, bool] = {ti: False for ti in tile_nalu_cache}
        per_tile_expected = 0
        good_at_start = sum(t.good_count for t in self._tiles)

        try:
            # Round-robin: tile-0 NAL i, tile-1 NAL i, tile-2 NAL i, tile-3 NAL i,
            # then move to NAL i+1. This is the order Apple's encoder emits
            # frames in real time, so the shared DPB sees POCs in their
            # natural decode order.
            for idx in range(max_burst):
                for ti, nalus in tile_nalu_cache.items():
                    if idx < len(nalus):
                        nalu = nalus[idx]
                        if nalu and ((nalu[0] >> 1) & 0x3F) in IDR_RANGE:
                            per_tile_seen_idr[ti] = True
                        if per_tile_seen_idr[ti]:
                            per_tile_expected += 1
                        # Drive the RPS tracker on burst NALUs so its
                        # `seen_pocs` is populated for the live P-frames
                        # that follow. Otherwise the first non-burst
                        # slice references a POC the tracker never saw
                        # and gets falsely dropped.
                        if nalu:
                            try:
                                self._rps_tracker.check_slice(nalu)
                            except Exception:
                                pass
                        self._decode_one(nalu, ti)
                        fed += 1

                # Hwaccel sanity check during the burst — only trigger
                # fallback if NALUs we expected to decode (those after
                # an in-burst IDR) actually failed.
                if is_hw and (idx + 1) % 8 == 0 and per_tile_expected > 0:
                    good_now = sum(t.good_count for t in self._tiles)
                    decoded = good_now - good_at_start
                    errs = max(0, per_tile_expected - decoded)
                    if (
                        errs > _HWACCEL_BURST_ERROR_THRESHOLD
                        and per_tile_expected > _HWACCEL_BURST_ERROR_WINDOW
                    ):
                        log.warning(
                            "hwaccel burst failing (%d errors / %d expected "
                            "/ %d fed); switching to software",
                            errs, per_tile_expected, fed,
                        )
                        self._fallback_to_software(tile_nalu_cache)
                        return
        finally:
            self._sync_decode_mode = False

        good = sum(t.good_count for t in self._tiles)
        log.info(
            "burst complete: fed %d NALUs, expected %d, decoded %d frames (%s)",
            fed, per_tile_expected, good, self._hw_name or "software",
        )

        if (
            is_hw
            and per_tile_expected > _HWACCEL_BURST_ERROR_WINDOW
            and good < _HWACCEL_BURST_MIN_FRAMES
        ):
            log.warning(
                "hwaccel produced only %d frames from %d expected NALUs; "
                "switching to software", good, per_tile_expected,
            )
            self._fallback_to_software(tile_nalu_cache)

    def feed_nalu(self, nalu: bytes, tile_idx: int) -> None:
        """Send one NALU to the shared decoder. During `feed_burst` this
        runs synchronously; otherwise it enqueues to the decoder worker
        (libavcodec releases the GIL inside `decode()`).

        Pre-decode RPS check: parse the slice's reference picture set
        and compare it to the POCs we've seen. If any required POC is
        missing, the decoder would conceal — instead we DROP the slice
        outright and request a FIR. Letting libav decode a known-bad
        slice produces a flood of "Could not find ref" log messages
        and a gray frame for that POC; both are useless and trigger
        cascade FIRs. Dropping is silent + free, and we'll resume
        feeding once a fresh IDR arrives.
        """
        if len(nalu) >= 1:
            nal_unit_type = (nalu[0] >> 1) & 0x3F
            # Skip parameter-set NALUs (VPS=32, SPS=33, PPS=34) — only
            # slice NALUs have an RPS we can check. IDRs (16-21) reset
            # the DPB and have no inter-picture refs to validate.
            from .hevc_rps import IRAP_RANGE
            if nal_unit_type < 32 and nal_unit_type not in IRAP_RANGE:
                missing = self._rps_tracker.check_slice(nalu)
                # Commit the POC unconditionally. Apple's stream is
                # 4 tiles sharing one codec context with independent
                # per-tile POC sequences; per-tile teardown isn't
                # something the tracker can model from a global view.
                # Dropping slices here risks false-positive cascades
                # (a tile's IDR resets the global tracker but the
                # codec's DPB still has the other tiles' refs); the
                # post-decode `decode_error_flags` path is the
                # authoritative concealment signal anyway.
                self._rps_tracker.commit_decoded()
                if missing:
                    # Mark for FIR but DO feed the slice. Feeding a
                    # slice with truly-missing refs produces a
                    # concealment-flagged frame; we hide that at the
                    # publish boundary in `get_frame()` so the user
                    # sees the previous-good frame instead of grey
                    # fill. Apple responds to the FIR with a fresh
                    # IDR within ~200 ms which closes the window.
                    self._gate.mark_decode_error(tile_idx)
        if self._sync_decode_mode:
            self._decode_one(nalu, tile_idx)
            return
        if self._queue is None or self._worker is None or not self._worker.is_alive():
            self._start_worker()
        # `_teardown()` (called from `restart()` on the watchdog
        # thread) sets `self._queue = None` mid-flight. We snapshot
        # the reference here before the put and accept that a few
        # NALUs may slip into a queue we're about to discard — the
        # restart's `_start_worker()` call above creates a fresh
        # one, and any orphans land harmlessly in the GC. Without
        # this snapshot the video rx thread crashes on the
        # AttributeError and never reattaches.
        q = self._queue
        if q is None:
            return
        try:
            q.put_nowait((nalu, tile_idx))
        except queue.Full:
            # Queue overflow — drop. The IDR cache is already loaded so
            # losing a TRAIL_R rarely matters; the gate will request a
            # fresh IDR if the lost slice causes a visible artefact.
            pass

    # -- consumer API --------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        """Return the latest decoded frame for `tile_idx`, or None.

        Returns None when (a) there's no new frame since the last call,
        (b) the shared codec context hasn't yet decoded any IDR (output
        before that is concealment fill from a cold DPB), or (c) the
        legacy heuristic gate (if enabled) blocked the frame.
        """
        if not self._dpb_has_idr:
            return None
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.raw_frame
            count = slot.good_count
            already_evaluated = count <= slot.last_evaluated_count

        if frame is None or already_evaluated:
            return None

        tile_frame, had_decode_error = _av_frame_to_tile(
            frame, self._reformatter, self._seen_fmts,
        )
        with slot.lock:
            slot.last_evaluated_count = count

        if tile_frame is None:
            return None
        # libavcodec flagged this frame as concealed / corrupt — fire
        # FIR immediately so the next IDR closes the failure window.
        # The frame is still published so the operator sees the
        # transient artifact rather than a frozen tile.
        if had_decode_error:
            self._gate.mark_decode_error(tile_idx)
        else:
            # Clean frame → reset the per-tile bad_streak so a
            # transient burst of decode errors doesn't accumulate
            # indefinitely and trip the persistent-concealment
            # watchdog after a recovery has already happened.
            self._gate.mark_clean(tile_idx)
        if not self._gate.should_publish(tile_idx, tile_frame):
            return None
        return tile_frame

    def consume_fir_request(self) -> set[int]:
        """Tile indices needing a fresh IDR. Read + cleared each call."""
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int) -> TileVisState:
        """Gate's HUD-state snapshot for one tile."""
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        """The active HW accelerator name, or None on software."""
        return self._hw_name

    @property
    def good_counts(self) -> list[int]:
        """Per-tile decoded-frame totals since last reset."""
        return [t.good_count for t in self._tiles]

    # -- lifecycle -----------------------------------------------------

    def restart(self) -> None:
        """Tear down + rebuild the codec context. Used after canvas changes
        or hot-reconnect.

        Skips rebuild if `set_params` was never called — restart on an
        unconfigured decoder is just a teardown."""
        self._teardown()
        if self._vps and self._sps and self._all_pps:
            self._create_codec(force_software=self._hw_failed)

    def close(self) -> None:
        """Permanent shutdown. Stops the worker, releases the codec context."""
        self._teardown()

    # -- internals: decode --------------------------------------------

    def _decode_one(self, nalu: bytes, tile_idx: int) -> None:
        """Wrap one NALU in an `av.Packet`, attach a fresh PTS, run through
        the shared codec, and dispatch any output frames back to per-tile
        slots via the PTS map."""
        codec = self._codec
        if codec is None:
            return
        if not _is_decodable_nalu(nalu):
            return

        nal_type = (nalu[0] >> 1) & 0x3F
        if nal_type in IDR_RANGE:
            slot = self._tiles[tile_idx]
            with slot.lock:
                already = slot.saw_idr_since_reset
                slot.saw_idr_since_reset = True
            if not already:
                log.info("tile %d IDR (nt=%d) — gate opens", tile_idx, nal_type)
            if not self._dpb_has_idr:
                self._dpb_has_idr = True
        elif not self._dpb_has_idr:
            # Drop P-frames before the first IDR — feeding them to a
            # cold codec context produces concealment-fill output that
            # the rest of the pipeline cannot distinguish from real
            # content. Better to wait for the IDR and start clean.
            self._pre_idr_drops = getattr(self, "_pre_idr_drops", 0) + 1
            if self._pre_idr_drops in (1, 10, 100, 1000):
                log.info(
                    "dropping pre-IDR P-frame for tile %d (drop count=%d)",
                    tile_idx, self._pre_idr_drops,
                )
            return

        if _nalu_dump_f is not None:
            _nalu_dump_f.write(_NAL_START_CODE + (
                nalu if isinstance(nalu, bytes) else bytes(nalu)))

        pkt = av.Packet(_NAL_START_CODE + (nalu if isinstance(nalu, bytes) else bytes(nalu)))
        pts = self._next_pts
        pkt.pts = pts
        pkt.dts = pts
        self._pts_to_tile[pts] = tile_idx
        self._next_pts += 1
        if len(self._pts_to_tile) > _PTS_MAP_SOFT_MAX:
            cutoff = pts - _PTS_MAP_PRUNE_KEEP
            self._pts_to_tile = {k: v for k, v in self._pts_to_tile.items() if k > cutoff}

        try:
            with self._codec_lock:
                # Re-check `_codec` inside the lock — restart() may have
                # nulled it after we read `codec` above and before we
                # got the lock.
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
                if (
                    self._silent_nalus > _HWACCEL_SILENT_NALU_LIMIT
                    and not self._recovery_in_progress
                ):
                    log.warning(
                        "%s silent for %d NALUs without output; "
                        "flushing + waiting for fresh IDRs",
                        self._hw_name or "software", self._silent_nalus,
                    )
                    self._recovery_in_progress = True
                    self._try_recovery()
                    self._silent_nalus = 0
        except Exception as e:
            self._handle_decode_error(tile_idx, nalu, e)

    def _publish_frame(self, frame: av.VideoFrame) -> None:
        """Map `frame.pts` back to its source tile and update that slot."""
        ti = self._pts_to_tile.pop(frame.pts, None)
        if ti is None:
            # PTS aged out of the map (rare on healthy streams; can happen
            # if libavcodec reordered output and our prune fired between).
            log.debug("frame pts=%d not in map", frame.pts)
            return

        slot = self._tiles[ti]
        with slot.lock:
            slot.raw_frame = frame
            slot.good_count += 1

        if self._on_frame_published is not None:
            try:
                self._on_frame_published(ti)
            except Exception as e:
                log.debug("on_frame_published callback raised: %s", e)

    def _handle_decode_error(
        self, tile_idx: int, nalu: bytes, exc: Exception,
    ) -> None:
        """Mid-session decode error. We do NOT fall back to software:
        whatever decoder we picked at startup is the one we keep. A
        single bad slice — usually from packet loss — can wedge
        VideoToolbox in `kVTVideoDecoderBadDataErr` / "reconfig
        pending" state, after which every send_packet returns EAGAIN.
        The fix is to flush libavcodec's buffers (which propagates a
        reset to the hwaccel layer), drop incoming P-frames until each
        tile's IDR arrives via FIR, and resume on the same context.
        """
        err_no = getattr(exc, "errno", 0) or 0
        log.debug(
            "tile %d decode error: errno=%d nalu_len=%d nt=%d err=%s",
            tile_idx, err_no, len(nalu), (nalu[0] >> 1) & 0x3F if nalu else -1, exc,
        )

        # First error since the last good publish triggers a recovery
        # cycle. Subsequent errors during the same cycle are expected
        # (more EAGAINs land while we wait for the FIR-driven IDRs);
        # they don't kick off another flush.
        if not self._recovery_in_progress:
            self._recovery_in_progress = True
            log.warning(
                "%s decode error (errno=%d) — flushing codec, dropping "
                "P-frames until fresh IDRs arrive",
                self._hw_name or "software", err_no,
            )
            self._try_recovery()

        self._gate.mark_decode_error(tile_idx)
        self._eagain_streak += 1


    def _try_recovery(self) -> None:
        """Reset the codec to a clean state without losing the active
        decoder context. Calls `flush_buffers()` to clear libavcodec's
        internal queue (and the hwaccel's wedge state, if any), then
        clears our per-tile IDR-seen flags so `_decode_one` will drop
        every P-frame until the next IDR for that tile arrives. The
        `mark_decode_error` calls in `_handle_decode_error` already
        triggered FIRs for each tile, so fresh IDRs are on the way."""
        with self._codec_lock:
            codec = self._codec
            if codec is not None:
                try:
                    codec.flush_buffers()
                except Exception as e:
                    log.debug("flush_buffers failed: %s", e)
        self._dpb_has_idr = False
        for slot in self._tiles:
            with slot.lock:
                slot.saw_idr_since_reset = False
        self._eagain_streak = 0

    def _drain_codec_to_slots(self) -> None:
        """Pull any frames libavcodec has buffered (None packet flush)."""
        codec = self._codec
        if codec is None:
            return
        try:
            for frame in codec.decode(None):
                self._publish_frame(frame)
        except Exception as e:
            log.debug("drain_codec_to_slots: %s", e)

    # -- internals: worker --------------------------------------------

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._queue = queue.Queue(maxsize=_QUEUE_MAX)
        self._worker = threading.Thread(
            target=self._worker_loop, name="hevc-decode", daemon=True,
        )
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
        # Snapshot the queue once. `_stop_worker()` (called from a
        # restart on another thread) sets `self._queue = None` after
        # signalling stop; if we read self._queue every iteration we
        # crash on the next get(). The local reference keeps draining
        # even after the parent has unlinked it — we exit on the
        # _stop event regardless.
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
            nalu, tile_idx = item
            try:
                self._decode_one(nalu, tile_idx)
            except Exception as e:
                log.warning("hevc-decode worker swallowed error: %s", e)

    # -- internals: codec construction --------------------------------

    def _build_extradata(self) -> bytes:
        """VPS+SPS+all-PPSes concatenated as Annex B — what libavcodec's
        HEVC decoder expects for raw-NALU init."""
        ed = bytearray()
        ed += _NAL_START_CODE + self._vps  # type: ignore[operator]
        ed += _NAL_START_CODE + self._sps  # type: ignore[operator]
        for pid in sorted(self._all_pps.keys()):
            ed += _NAL_START_CODE + self._all_pps[pid]
        return bytes(ed)

    def _create_codec(self, *, force_software: bool) -> None:
        """Build the shared codec context. Tries HW accel first when allowed;
        SW on failure. Updates `self._hw_name`."""
        extradata = self._build_extradata()

        if not force_software and self._prefer_hwaccel and not self._hw_failed:
            for hw_type in _platform_hwaccels():
                ctx = self._try_hwaccel(hw_type, extradata)
                if ctx is not None:
                    self._install_codec(ctx, hw_name=hw_type)
                    return

        self._install_codec(self._make_sw_context(extradata), hw_name=None)

    def _try_hwaccel(
        self, hw_type: str, extradata: bytes,
    ) -> Optional[av.codec.context.CodecContext]:
        try:
            from av.codec.hwaccel import HWAccel

            hw = HWAccel(device_type=hw_type)
            c = av.CodecContext.create("hevc", "r", hwaccel=hw)
            c.extradata = extradata
            # Single-threaded for HEVC: HEVC frame/slice threading can
            # interact awkwardly with cross-frame DPB references at the
            # decoder level. The HW path itself parallelises internally.
            c.thread_type = "NONE"
            c.thread_count = 1
            c.flags = _CODEC_FLAG_LOW_DELAY
            c.flags2 = _CODEC_FLAG2_FAST
            c.open()
            return c
        except Exception as e:
            log.info("hwaccel %s unavailable: %s", hw_type, e)
            return None

    def _make_sw_context(self, extradata: bytes) -> av.codec.context.CodecContext:
        # Single shared context, single thread. Cross-tile POC refs make
        # frame/slice threading risky for Apple's stream — the original
        # rev-eng work landed on NONE for this reason. SW perf at 1440p
        # in this mode is ~15-30 fps on a modest CPU; for higher
        # framerates, HW accel takes over.
        c = av.CodecContext.create("hevc", "r")
        c.extradata = extradata
        c.thread_type = "NONE"
        c.thread_count = 1
        c.flags = _CODEC_FLAG_LOW_DELAY
        c.flags2 = _CODEC_FLAG2_FAST
        c.open()
        return c

    def _install_codec(
        self, codec: av.codec.context.CodecContext, *, hw_name: Optional[str],
    ) -> None:
        self._codec = codec
        self._hw_name = hw_name
        self._reformatter[0] = None
        self._seen_fmts.clear()
        self._next_pts = 0
        self._pts_to_tile = {}
        log.info("HEVC decoder: shared context (%s)", hw_name or "software")
        # NALU dump diagnostic: prepend VPS+SPS+all-PPSes (Annex-B) so a
        # stock player can decode the dump without external param sets.
        if _nalu_dump_f is not None:
            _nalu_dump_f.write(self._build_extradata())
            _nalu_dump_f.flush()

    # -- internals: hwaccel fallback ----------------------------------

    def _fallback_to_software(
        self, tile_nalu_cache: Optional[dict[int, list[bytes]]],
    ) -> None:
        """Tear down the HW codec context, build SW one, optionally re-feed
        the burst NALUs."""
        log.warning("falling back from %s to software decode", self._hw_name)
        self._hw_failed = True
        self._teardown()
        self._create_codec(force_software=True)

        if tile_nalu_cache:
            self._sync_decode_mode = True
            try:
                max_burst = max(
                    (len(tile_nalu_cache[ti]) for ti in tile_nalu_cache),
                    default=0,
                )
                for idx in range(max_burst):
                    for ti, nalus in tile_nalu_cache.items():
                        if idx < len(nalus):
                            self._decode_one(nalus[idx], ti)
                good = sum(t.good_count for t in self._tiles)
                log.info("software fallback burst: %d frames decoded", good)
            finally:
                self._sync_decode_mode = False

    # -- internals: shared teardown -----------------------------------

    def _teardown(self) -> None:
        """Stop worker, drain + invalidate codec, reset all state, reset gate.

        Acquire the codec lock around the destruction so the worker
        thread (if it ignored our stop signal or is still draining)
        can't be mid-decode when we null `_codec` — that's the path
        that produced SIGSEGV in libavcodec's HEVC decoder."""
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
        self._rps_tracker.reset()
        if self._sps and len(self._sps) > 2:
            self._rps_tracker.feed_sps(self._sps[2:])


# ── module-level helpers ─────────────────────────────────────────────

def _is_decodable_nalu(nalu: bytes) -> bool:
    """Quick filter for NALUs libavcodec will accept: drop too-short,
    NAL types > 31 (SEI / EOB / FU control — handled elsewhere), and
    those missing first_slice_segment_in_pic_flag."""
    if len(nalu) < 3:
        return False
    if (nalu[0] >> 1) & 0x3F > 31:
        return False
    return bool(nalu[2] & 0x80)


__all__ = ["HevcDecoder"]
