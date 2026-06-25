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

Hardware acceleration is requested per-platform (D3D11VA on Windows, VAAPI
on Linux); macOS intentionally runs software-only (see _PLATFORM_HWACCELS).
Software libavcodec is the universal fallback and, in practice, the path
nearly everyone hits: Apple's stream is HEVC RExt 4:4:4, which the common
desktop HW decoders (D3D11VA/DXVA2/VAAPI) do not support — they are 4:2:0
only — so the decoder silently stays on software for this stream.

The frame-stats quality gate (gray / black / green-pop detection + FIR
escalation) lives in `quality_gate.py`.
"""
from __future__ import annotations

import errno
import logging
import os
import queue
import sys
import threading
from typing import Callable, Optional


# Diagnostic: when set, every NALU fed to libavcodec is also written
# (Annex-B framed) to this path. Pipe the resulting file through any
# stock HEVC player (`mpv file.h265`, `ffplay -f hevc -i file.h265`)
# to confirm whether the bitstream itself decodes cleanly outside our
# wgpu/render path. ~No overhead when unset.
_NALU_DUMP_PATH = os.environ.get("ISS_NALU_DUMP")
_nalu_dump_f = open(_NALU_DUMP_PATH, "wb") if _NALU_DUMP_PATH else None

import av

from .tiles import TileFrame
from .nalu import IDR_RANGE
from .quality_gate import FrameQualityGate, TileVisState
from .decode_common import (
    _NAL_START_CODE,
    _CODEC_FLAG_LOW_DELAY,
    _CODEC_FLAG2_FAST,
    _TileSlot,
    _HW_FRAME_FORMATS,
    _av_frame_to_tile,
)


log = logging.getLogger(__name__)


# ── constants ────────────────────────────────────────────────────────

# nv24 (Apple's HW VideoToolbox 4:4:4 biplanar output) is delivered to the GPU
# as a passthrough TileFrame (raw interleaved UV, `v is None`) and deinterleaved
# in the fragment shader from an rg8unorm texture. Set ISS_LEGACY_CHROMA=1 to
# fall back to the old CPU-side strided deinterleave (one full UV gather per
# tile per frame — ~half a core at 4-tile/60fps; the passthrough path is ~49×
# cheaper). The escape hatch exists in case the rg8 sampling misbehaves on a
# given GPU; both the decoder and renderer honour it.
_LEGACY_CHROMA = os.environ.get("ISS_LEGACY_CHROMA") == "1"

# Mid-stream wedge recovery granularity. Default (ON): mark ONLY the
# gate-flagged broken tiles (`bad_streak > 0`) as awaiting re-root — the
# healthy tiles keep decoding + publishing from their intact history, and
# only the broken tile(s) wait for the shared tile-0 IDR. This avoids the
# all-tile freeze the alternative causes: with the session-wide
# `_dpb_has_idr = False` gate, ALL four tiles' P-frames are dropped and
# every tile's `get_frame` returns None until the next IDR — and since
# Apple emits IDRs on tile 0 only (~every 1.7 s), tiles 1-3 re-wedge and
# re-recover for ~4 IDR cycles (~8 s) for what is usually a single broken
# tile. Set ISS_PERTILE_RECOVERY=0 to fall back to that session-wide gate.
# Cold start (before the very first IDR) is untouched — `_dpb_has_idr`
# still gates every tile then.
_PERTILE_RECOVERY = os.environ.get("ISS_PERTILE_RECOVERY", "1") != "0"


# Decode-worker queue + lifecycle.
_QUEUE_MAX = 512                           # NALUs in flight to the decoder
_WORKER_DEQUEUE_TIMEOUT_S = 0.5
_WORKER_JOIN_TIMEOUT_S = 2.0

# HW-accel fallback thresholds — empirical (Windows D3D11VA, Linux iGPU
# VAAPI). 10 was too aggressive: a HW decoder routinely emits 10+ EAGAINs
# during warm-up (codec buffering input before producing output). 50 lets it
# settle without being mistaken for a broken accelerator.
_HWACCEL_EAGAIN_STREAK_LIMIT = 50
# Number of NALUs we feed to a hwaccel context with zero output frames
# before forcing a fallback to software. A HW decoder can accept the initial
# burst and then go silent on Apple HEVC RExt 4:4:4 — `decode()` returns no
# error but emits no frames. EAGAIN logic doesn't catch this; we'd otherwise
# wait for the SSRC-adoption stall window (10-15s) before realising decode
# is dead.
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
# macOS deliberately has NO hwaccel: Apple's stream is HEVC RExt 4:4:4 and
# the only macOS HW path for it is VideoToolbox, but a real macOS user would
# run Apple's own Screen Sharing.app, not iss. Forcing software decode on
# macOS keeps it on the exact same libavcodec path as Windows/Linux (where
# 4:4:4 has no HW decoder either — D3D11VA/DXVA2/VAAPI are 4:2:0-only), so
# the macOS box is a faithful test bed for the platforms that matter.
_PLATFORM_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": (),
    "win32": ("d3d11va", "d3d12va"),
    "*": ("vaapi", "cuda"),
}


def _platform_hwaccels() -> tuple[str, ...]:
    return _PLATFORM_HWACCELS.get(sys.platform, _PLATFORM_HWACCELS["*"])


# `_TileSlot`, `_HW_FRAME_FORMATS`, and `_av_frame_to_tile` now live in
# `decode_common.py` (shared with the AVC decoder) and are imported above.



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
        # Per-tile cumulative NALU-type histogram. nal_unit_type is the
        # HEVC 6-bit field at byte0>>1 & 0x3F. Surfaced in the session
        # profile log for post-mortem diagnosis (e.g. "tile 2 stopped
        # receiving IDRs from 14:50 onwards" — visible as nt=20 count
        # plateauing for tile 2 while tile 0 keeps incrementing).
        self.nalu_counts_per_tile: list[dict[int, int]] = [
            {} for _ in range(num_tiles)
        ]
        # Rolling window of recent (non-suspicious) IDR payload sizes.
        # Apple's HP encoder sometimes emits a "minimal" IDR (~25 KB)
        # in response to a FIR that parses as nt=20 but doesn't
        # actually cleanse the decoder's drift; real IDRs are
        # ~60-100 KB. We use the running median to flag suspiciously-
        # small IDRs so the quality gate doesn't mistake them for
        # recovery. Size is content-independent (encoder-side), so it
        # can't false-positive on low-motion real content (which
        # changes the encoder's IDR cadence, not their byte count).
        # An earlier attempt used per-tile parameter-set freshness
        # instead — appealing because it sounds protocol-correct, but
        # parameter sets arrive asynchronously on each tile's SSRC
        # and the relative timing at IDR-decode-time is too racy to
        # be a reliable signal (false-positives caused a FIR storm).
        self._recent_idr_sizes: list[int] = []
        self._IDR_HISTORY_LEN = 10
        self._IDR_FAKE_RATIO = 0.40
        # RPS-based pre-decode error detector. We feed every slice
        # NALU through the tracker before queueing it for decode; if
        # the slice references a POC we never saw, we know the
        # decoder will conceal and we mark the tile as needing FIR.
        from .hevc_rps import HevcRpsTracker  # local to avoid cycles
        self._rps_tracker = HevcRpsTracker()
        # Per-tile DONL (decoding-order number) of the last cleanly-decoded
        # frame (one whose reference picture set is fully present). The host's
        # encoder keys its long-term-reference ring on this per-frame id, and
        # only honours an LTR ack whose value is in that ring — so it's the
        # value the ack must carry. See nalu.first_donl / Session._send_ltr_ack.
        self.last_clean_donl: list[Optional[int]] = [None] * num_tiles

        # Codec parameter sets, set by `set_params`.
        self._vps: Optional[bytes] = None
        self._sps: Optional[bytes] = None
        self._all_pps: dict[int, bytes] = {}

        # HW state.
        self._hw_name: Optional[str] = None
        self._hw_verified = False  # first-frame HW-binding truthfulness check
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
        # Per-tile mid-stream recovery set (ISS_PERTILE_RECOVERY only).
        # On a wedge we add just the gate-flagged broken tiles here
        # instead of clearing the session-wide `_dpb_has_idr`; those
        # tiles' P-frames are dropped and their `get_frame` returns None
        # until the shared tile-0 IDR re-roots the context (which clears
        # the whole set). Empty in the default path — `_dpb_has_idr`
        # remains the sole gate then.
        self._tiles_await_idr: set[int] = set()
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

    def feed_nalu(self, nalu: bytes, tile_idx: int, donl: Optional[int] = None) -> None:
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
            # Tally per-tile NALU-type histogram for diagnostics.
            if 0 <= tile_idx < len(self.nalu_counts_per_tile):
                bucket = self.nalu_counts_per_tile[tile_idx]
                bucket[nal_unit_type] = bucket.get(nal_unit_type, 0) + 1
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
                # Record this frame's DONL for the LTRP ack when the slice's
                # references are all present (best available pre-decode "this
                # frame is good" signal). Post-decode concealment is still
                # caught by decode_error_flags.
                if not missing and donl is not None and 0 <= tile_idx < len(self.last_clean_donl):
                    self.last_clean_donl[tile_idx] = donl
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
            # Queue overflow — drop. DIAGNOSTIC: a dropped slice that a
            # later P-frame references (Apple uses refs ≤8 back) is a
            # silent wedge trigger, so count + surface it.
            self._queue_full_drops = getattr(self, "_queue_full_drops", 0) + 1
            n = self._queue_full_drops
            if n in (1, 10, 100, 1000) or n % 1000 == 0:
                log.warning("decoder queue FULL — dropped slice for tile %d "
                            "(drop count=%d) — worker not keeping up", tile_idx, n)

    # -- consumer API --------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        """Return the latest decoded frame for `tile_idx`, or None.

        Returns None when (a) there's no new frame since the last call,
        (b) the shared codec context hasn't yet decoded any IDR (output
        before that is concealment fill from a cold DPB), or (c) the
        legacy heuristic gate (if enabled) blocked the frame.

        Under ISS_PERTILE_RECOVERY, a tile that wedged mid-stream and is
        still awaiting the re-root IDR also returns None — but only that
        tile; its healthy siblings keep publishing from their own intact
        DPB history.
        """
        if not self._dpb_has_idr:
            return None
        if _PERTILE_RECOVERY and tile_idx in self._tiles_await_idr:
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
    def bad_tiles(self) -> set[int]:
        """Tiles the gate considers gray/concealing and awaiting recovery
        (decoder-agnostic; works on HW decoders too). See gate.bad_tiles."""
        return self._gate.bad_tiles

    @property
    def hw_accel(self) -> Optional[str]:
        """The active HW accelerator name, or None on software."""
        return self._hw_name

    @property
    def good_counts(self) -> list[int]:
        """Per-tile published-frame totals since last reset (INCLUDES
        concealed/gray frames — this is throughput, not visual health)."""
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list[int]:
        """Per-tile CLEAN (non-concealed) published-frame totals. A tile
        whose good_count climbs but clean_count is flat is showing gray."""
        return [t.clean_count for t in self._tiles]

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
            # Suspicious-IDR detection: compare this IDR's byte count
            # against the median of recent IDRs. Real IDRs from the
            # host's HP encoder are typically 60-100 KB; the "fake"
            # minimal IDRs seen in the field are ~25 KB. Below 40 % of
            # the recent median is the conservative threshold. Requires
            # ≥3 samples to avoid flagging the very first IDRs of a
            # session.
            size = len(nalu)
            history = self._recent_idr_sizes
            suspicious = False
            if len(history) >= 3:
                sorted_h = sorted(history)
                median = sorted_h[len(sorted_h) // 2]
                if size < median * self._IDR_FAKE_RATIO:
                    suspicious = True
            # Only push real IDRs into the median window — otherwise a
            # streak of fakes would drag the median down and they'd
            # stop being flagged. Suspicious IDRs don't qualify as
            # "what a real IDR looks like for this stream".
            if not suspicious:
                history.append(size)
                if len(history) > self._IDR_HISTORY_LEN:
                    history.pop(0)
            # Apple's HP encoder only ever emits IDRs on the base SSRC
            # (= tile 0); the other tiles' SSRCs carry P-frames that
            # reference the shared codec context's DPB. So an IDR
            # arriving for ANY tile resets the DPB for ALL tiles, and
            # all tiles' post-IDR grace windows should engage. Without
            # this fan-out, tiles 1-3 would never see their grace
            # engage, would stay perpetually flagged in
            # `keyframe_required`, and we'd FIR them forever waiting
            # for IDRs Apple architecturally won't send.
            for ti in range(len(self._tiles)):
                self._gate.mark_idr_observed(ti, suspicious=suspicious)
            # Diagnostic: every IDR arrival. The "gate opens" log only
            # fires once per reset so it under-reports actual IDR
            # cadence; this DEBUG line is ground truth.
            log.debug("IDR arrival: tile %d nt=%d (size=%d) suspicious=%s",
                      tile_idx, nal_type, size, suspicious)
            if not already:
                log.debug("tile %d IDR (nt=%d) — gate opens", tile_idx, nal_type)
            if not self._dpb_has_idr:
                self._dpb_has_idr = True
            if _PERTILE_RECOVERY and self._tiles_await_idr:
                # The shared codec context's DPB is re-rooted by this IDR
                # (Apple emits IDRs on tile 0 only, but cross-tile refs
                # mean it re-roots every tile). Every tile that was
                # waiting on the wedge can resume.
                self._tiles_await_idr.clear()
        elif not self._dpb_has_idr or (
            _PERTILE_RECOVERY and tile_idx in self._tiles_await_idr
        ):
            # Drop P-frames while the context can't decode them. Two cases:
            #   - cold start (`not _dpb_has_idr`): no IDR has EVER landed,
            #     so the DPB is empty for ALL tiles — feeding P-frames
            #     produces concealment-fill output the pipeline can't
            #     distinguish from real content. (Default path: this is
            #     the only drop condition.)
            #   - mid-stream per-tile recovery (`tile_idx in
            #     _tiles_await_idr`, ISS_PERTILE_RECOVERY only): this tile
            #     wedged and its refs are gone; drop just its P-frames
            #     until the re-root IDR clears the await set. Healthy
            #     siblings (not in the set) fall through and keep decoding.
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
                        "dropping P-frames + waiting for fresh IDRs (no flush)",
                        self._hw_name or "software", self._silent_nalus,
                    )
                    self._recovery_in_progress = True
                    self._try_recovery()
                    self._silent_nalus = 0
        except Exception as e:
            self._handle_decode_error(tile_idx, nalu, e)

    def _publish_frame(self, frame: av.VideoFrame) -> None:
        """Map `frame.pts` back to its source tile and update that slot."""
        # One-time HW-binding truthfulness check: if we requested a hwaccel
        # but the output is a software pixel format, the accel never bound
        # (e.g. D3D11VA on a 4:4:4 stream) — relabel as software so the
        # profile log / `hw_accel` are honest and downstream stops assuming
        # a GPU surface. The decode itself is unaffected; only the label.
        if not self._hw_verified:
            self._hw_verified = True
            if (self._hw_name is not None
                    and frame.format.name not in _HW_FRAME_FORMATS):
                log.warning(
                    "hwaccel %r did not bind for this stream (output=%s); "
                    "decoding in software", self._hw_name, frame.format.name,
                )
                self._hw_name = None

        ti = self._pts_to_tile.pop(frame.pts, None)
        if ti is None:
            # PTS aged out of the map (rare on healthy streams; can happen
            # if libavcodec reordered output and our prune fired between).
            log.debug("frame pts=%d not in map", frame.pts)
            return

        # Concealed (gray) frames carry libav's decode-error flag; count
        # them in good_count (throughput) but NOT clean_count (real video),
        # so the profile can distinguish a healthy tile from a gray one that
        # is still churning concealed frames at full rate.
        err = getattr(frame, "decode_error_flags", 0)
        flg = getattr(frame, "flags", 0)
        had_error = bool(err) or bool(flg & 0x01)

        slot = self._tiles[ti]
        with slot.lock:
            slot.raw_frame = frame
            slot.good_count += 1
            if not had_error:
                slot.clean_count += 1

        if self._on_frame_published is not None:
            try:
                self._on_frame_published(ti)
            except Exception as e:
                log.debug("on_frame_published callback raised: %s", e)

    def _handle_decode_error(
        self, tile_idx: int, nalu: bytes, exc: Exception,
    ) -> None:
        """Mid-session decode error. We do NOT fall back to software:
        whatever decoder we picked at startup is the one we keep.

        Two very different events surface here as a `decode()` raise:

          1. **Transient broken reference chain** (the common case).
             Runtime traces (`ISS_DPB_TRACE`) show the trigger is a
             single recent frame from ONE tile that never reached the
             decoder — a reorder / in-pipeline gap, with `loss_total=0`,
             *not* an aged-out reference (the missing POC sits ~2 frames
             behind head). libavcodec conceals it, the sibling tiles keep
             decoding, and the FIR we raise here pulls a fresh IDR that
             re-roots the chain within ~one round-trip. Flushing in this
             case is actively harmful: `flush_buffers()` wipes the whole
             SHARED DPB — destroying the other three tiles' good
             references — and `_try_recovery` then drops every P-frame
             until an IDR that, for tiles 1-3, architecturally never
             comes, turning a one-frame blip into a multi-second all-tile
             freeze (the exact restart-loop seen in the field).

          2. **Genuine hwaccel wedge.** A bad slice *can* wedge a HW
             decoder in a bad-data / reconfig-pending state, after which
             every send_packet raises with zero output. Only this warrants
             the escalation to `_try_recovery`.

        We distinguish them by output: any successful publish resets
        `_eagain_streak` (see `_decode_one`), so a wedge is the only thing
        that can accumulate `_HWACCEL_SILENT_NALU_LIMIT` consecutive
        raises with nothing decoded in between. Below that, we just FIR
        and let the codec conceal — same policy the libav-concealment
        fast path already uses ("deliberately do NOT call flush_buffers").
        """
        err_no = getattr(exc, "errno", 0) or 0
        log.debug(
            "tile %d decode error: errno=%d nalu_len=%d nt=%d err=%s",
            tile_idx, err_no, len(nalu), (nalu[0] >> 1) & 0x3F if nalu else -1, exc,
        )

        # `errno=35` (EAGAIN) from `avcodec_send_packet()` is pure
        # BACKPRESSURE — the decoder's input queue is full because it hasn't
        # produced output yet (it is simply slower than real time on this
        # content, e.g. incompressible 4:4:4 above the HW decoder's
        # throughput). It is NOT a broken reference chain, so FIR-ing on it
        # is actively harmful: a FIR pulls a fresh IDR, the single heaviest
        # NALU to decode, which deepens the backlog → a death spiral that
        # locks the stream at 0 fps (observed: ~7500 EAGAINs in 45 s under a
        # 60 Mbps noise load, each spawning a giant-IDR FIR). So on EAGAIN we
        # do NOT FIR and do NOT pollute `bad_streak`/the gate; we only bump
        # `_eagain_streak` so a *genuine* wedge — sustained EAGAIN with ZERO
        # output, since any publish resets the streak in `_decode_one` —
        # still escalates to the no-flush recovery below.
        is_backpressure = err_no == errno.EAGAIN
        if not is_backpressure:
            # Genuine decode error (bad data / VT malfunction such as
            # -17694 / -12909): re-root the broken reference chain with a
            # fresh IDR.
            self._gate.mark_decode_error(tile_idx)
        self._eagain_streak += 1

        # Only a sustained wedge (no output for a full source frame's
        # worth of NALUs) escalates to the DPB-wiping flush.
        if (not self._recovery_in_progress
                and self._eagain_streak >= _HWACCEL_SILENT_NALU_LIMIT):
            self._recovery_in_progress = True
            # A genuine wedge (sustained EAGAIN, zero output) needs a fresh
            # IDR to re-root. We deliberately did NOT FIR on the individual
            # EAGAINs above (that storms giant IDRs), so request exactly ONE
            # here on wedge-entry — otherwise nothing pulls a recovery IDR
            # until the 15 s session watchdog, leaving a ~14 s freeze.
            self._gate.mark_decode_error(tile_idx)
            log.warning(
                "%s wedged (errno=%d, %d consecutive decode errors, no "
                "output) — draining backlog + one FIR for a fresh IDR "
                "(no flush)",
                self._hw_name or "software", err_no, self._eagain_streak,
            )
            self._try_recovery()


    def _try_recovery(self) -> None:
        """Native-aligned wedge recovery — do NOT flush the codec.

        Apple's own viewer (AVConference) never flushes or invalidates
        its VTDecompressionSession on a decode error: it flags the bad
        frame, keeps the one session alive, and requests a keyframe
        (FIR/PLI). We mirror that. `flush_buffers()` wipes the SHARED
        DPB; because Apple emits IDRs only on tile 0 — and a tile-0 IDR
        does NOT in practice evict tiles 1-3's reference pictures from
        the working DPB (see `HevcRpsTracker.commit_decoded`) — wiping
        the DPB ourselves orphans those tiles: their refs are gone, no
        per-tile IDR re-roots them, and feeding their next P-frames
        re-wedges the just-flushed decoder → flush → re-wedge → flush, a
        self-sustaining cascade (observed: ~13 s of back-to-back
        `errno=35` flushes that even a full `restart()` couldn't break).

        Instead we just stop feeding P-frames until the next IDR re-roots
        the DPB (`_dpb_has_idr = False`); the caller has requested a fresh
        keyframe (FIR) on wedge-entry. The codec's existing reference
        frames survive, so the tiles that weren't hit resume from their own
        intact history once the keyframe lands. (No `flush_buffers()`, so
        the RPS tracker still mirrors the live DPB and must NOT be reset
        here.)

        Under ISS_PERTILE_RECOVERY we go one step finer: rather than the
        session-wide `_dpb_has_idr = False` (which drops + blanks ALL four
        tiles until the next tile-0 IDR — an ~8 s freeze across ~4 IDR
        cycles for what is usually one broken tile), we mark ONLY the
        gate-flagged broken tiles (`bad_streak > 0`) as awaiting re-root.
        The healthy tiles keep decoding + publishing from their intact
        history; the broken tiles' P-frames are dropped and their frames
        held back until the shared tile-0 IDR clears the await set in
        `_decode_one`. (If no tile is flagged — shouldn't happen on a real
        wedge, but be safe — we mark all so we never get stuck.)

        We also DRAIN the worker queue: at high bitrate the wedge is a
        backlog of hundreds of stale P-frames (each re-EAGAIN-ing), and
        grinding through them all before the incoming IDR is reached is
        what stretches recovery to ~15 s. Dropping the backlog lets the
        worker reach the recovery IDR in ~1 RTT. Safe to empty here: this
        runs on the decode worker thread (via `_decode_one`), the queue's
        only consumer."""
        if _PERTILE_RECOVERY:
            broken = {
                ti for ti in range(len(self._tiles))
                if self._gate._states[ti].bad_streak > 0
            }
            if not broken:
                broken = set(range(len(self._tiles)))
            self._tiles_await_idr |= broken
            for ti in broken:
                with self._tiles[ti].lock:
                    self._tiles[ti].saw_idr_since_reset = False
        else:
            self._dpb_has_idr = False
            for slot in self._tiles:
                with slot.lock:
                    slot.saw_idr_since_reset = False
        self._eagain_streak = 0
        q = self._queue
        if q is not None:
            n = q.qsize()
            for _ in range(n):
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

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
            # SLICE threading (parallelise within a frame), all cores.
            # On Windows, DXVA2/D3D11VA cannot decode HEVC 4:4:4 (FFmpeg's
            # hevc get_format never offers them for YUV444P), so this "HW"
            # context silently software-decodes Apple's 4:4:4 stream. Single
            # thread only hit ~57 fps on busy full-res content (< the 60 fps
            # stream) → backlog → gray; SLICE threading measured ~205 fps.
            # SLICE (not FRAME) adds no latency and no frame reordering, so
            # it's safe for the cross-frame DPB refs Apple's tiles use. When
            # a real HW decoder does bind, it parallelises internally and
            # ignores this setting.
            c.thread_type = "SLICE"
            c.thread_count = 0
            c.flags = _CODEC_FLAG_LOW_DELAY
            c.flags2 = _CODEC_FLAG2_FAST
            c.open()
            return c
        except Exception as e:
            log.info("hwaccel %s unavailable: %s", hw_type, e)
            return None

    def _make_sw_context(self, extradata: bytes) -> av.codec.context.CodecContext:
        # Single shared context, SLICE threading across all cores. SLICE
        # parallelises within a single frame's CTU rows (no frame reordering,
        # no added latency), so the cross-tile/cross-frame DPB refs Apple's
        # stream relies on are decoded in order — unlike FRAME threading,
        # which the original rev-eng work correctly avoided. Single-thread
        # 4:4:4 decode fell behind the 60 fps stream on busy content
        # (~57 fps measured at 2940x1912); SLICE measured ~205 fps.
        c = av.CodecContext.create("hevc", "r")
        c.extradata = extradata
        c.thread_type = "SLICE"
        c.thread_count = 0
        c.flags = _CODEC_FLAG_LOW_DELAY
        c.flags2 = _CODEC_FLAG2_FAST
        c.open()
        return c

    def _install_codec(
        self, codec: av.codec.context.CodecContext, *, hw_name: Optional[str],
    ) -> None:
        self._codec = codec
        self._hw_name = hw_name
        # _try_hwaccel labels the context with the REQUESTED hwaccel, but the
        # accel only binds in get_format. e.g. DXVA2/D3D11VA are never offered
        # for HEVC 4:4:4, so on Windows Apple's stream silently decodes in
        # software through a context still labelled "d3d11va". Re-verify from
        # the first real frame's pixel format (see _publish_frame).
        self._hw_verified = False
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
                slot.clean_count = 0
                slot.last_evaluated_count = 0
                slot.saw_idr_since_reset = False
        self._gate.reset()
        self._eagain_streak = 0
        self._reformatter[0] = None
        self._next_pts = 0
        self._pts_to_tile = {}
        self._dpb_has_idr = False
        self._tiles_await_idr.clear()
        self._pre_idr_drops = 0
        self._rps_tracker.reset()
        self.last_clean_donl = [None] * self.num_tiles
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
