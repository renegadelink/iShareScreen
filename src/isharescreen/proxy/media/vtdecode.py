"""VideoToolbox-native HEVC decoder (macOS) — a drop-in for `HevcDecoder`.

Why this exists: libav's HEVC decoder via its VideoToolbox *hwaccel* glue
mismanages the DPB on Apple's Screen-Sharing stream — it periodically loses a
recent reference and then BLOCKS (`avcodec_send_packet` → EAGAIN) on a missing
ref, which forces our drain/drop recovery, which gaps the stream, which
cascades into a multi-second freeze (see the decode-stall notes). The native
viewer never hits this because it decodes through a `VTDecompressionSession`
directly: VideoToolbox does its own bitstream parsing + DPB and simply conceals
the odd missing-ref frame without ever blocking. This module does the same.

Validated offline against captured feeds: decodes Apple HEVC RExt 4:4:4 to
`kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` (`444f` = nv24) and never
blocks. The two output planes (Y r8 + interleaved UV) map straight onto the
existing nv24 *passthrough* `TileFrame` (v is None) and the biplanar GPU path.

One shared session decodes all tiles' NALUs (same shared-context model as the
libav path — Apple uses cross-tile references). Output is dispatched to the
right tile by a per-decode closure. macOS-only; import guarded.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .tiles import TileFrame
from .nalu import IDR_RANGE
from .quality_gate import FrameQualityGate

log = logging.getLogger(__name__)

# Imported lazily / guarded so the module is importable (and skippable) off
# macOS. `available()` reports whether the native path can be used.
try:
    import VideoToolbox as _VT
    import CoreMedia as _CM
    import Quartz as _Q
    _VT_OK = True
except Exception:  # pragma: no cover - non-macOS / missing pyobjc
    _VT_OK = False

_LOCK_READONLY = 0x00000001                       # kCVPixelBufferLock_ReadOnly
_NV24_FMT = 0x34343466                             # '444f'
_VT_DEBUG = os.environ.get("ISS_VT_DEBUG") == "1"
# Diagnostic: dump every VCL NALU we feed VT (Annex-B framed, in feed order)
# so the feed ORDER can be analysed offline against POC. Same env var as the
# libav path; only the active decoder writes.
_NALU_DUMP_PATH = os.environ.get("ISS_NALU_DUMP")
_nalu_dump_f = open(_NALU_DUMP_PATH, "wb") if _NALU_DUMP_PATH else None


def available() -> bool:
    """True if VideoToolbox + pyobjc are importable (macOS with the deps)."""
    return _VT_OK


@dataclass(slots=True)
class _VTSlot:
    """Latest decoded `TileFrame` for one tile + a monotonic sequence so an
    out-of-order async callback can't replace a newer frame with an older one."""
    frame: Optional[TileFrame] = None
    seq: int = 0
    good_count: int = 0
    last_evaluated_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class VTHevcDecoder:
    """Native VideoToolbox HEVC decoder. Mirrors `HevcDecoder`'s public API:
    `set_params`, `feed_burst`, `feed_nalu`, `get_frame`, `restart`, `close`,
    plus the `_gate` / `good_counts` / `hw_accel` attributes session.py reads."""

    def __init__(
        self,
        num_tiles: int,
        *,
        prefer_hwaccel: bool = True,           # accepted for signature parity
        enable_quality_gate: bool = True,
        on_frame_published: Optional[Callable[[int], None]] = None,
    ) -> None:
        if not _VT_OK:
            raise RuntimeError("VideoToolbox/pyobjc not available")
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.num_tiles = num_tiles
        self._on_frame_published = on_frame_published
        self._tiles = [_VTSlot() for _ in range(num_tiles)]
        # API parity with HevcDecoder: session.py reads last_clean_donl[0] for
        # the LTRP DON-ack. VideoToolbox steers its own references and never
        # uses iss-side LTRP, so this stays all-None → session.py acks no DON.
        self.last_clean_donl: list = [None] * num_tiles
        # Per-tile NAL-unit-type histogram (diagnostic; read by the snapshot).
        self.nalu_counts_per_tile: list = [{} for _ in range(num_tiles)]
        # session.py reads `_decoder._gate._states[ti].bad_streak` (stall
        # watchdog) and `_decoder._rps_tracker` (ISS_DPB_TRACE only — libav
        # log path, never fires for VT). Provide a real gate, a None tracker.
        self._gate = FrameQualityGate(num_tiles)
        self._rps_tracker = None
        # Display name surfaced as the "decoder" field in the control-socket
        # snapshot + the TUI header line. Distinct from the libav hwaccel name
        # ("videotoolbox") so an operator can see which path is live: this
        # direct-VTDecompressionSession path vs libav's VT *hwaccel* glue.
        self._hw_name = "videotoolbox-native"

        self._fmt = None                       # CMVideoFormatDescription
        self._session = None                   # VTDecompressionSession
        self._params: Optional[tuple] = None   # (vps, sps, (pps...))
        self._seq = 0                          # monotonic decode counter
        self._feed_total = 0
        self._feeds_at_last_pub = 0
        self._last_pub_t = 0.0
        self._last_err_fir_t = 0.0
        self._restart_suppressed = 0
        self._decode_errors = 0                # benign VT conceals (diagnostic)
        self._lock = threading.Lock()          # guards session create/teardown
        self._seen_fmt = False
        self._fmt_warned = False                # H1: warn once on non-nv24 output
        # Decode latency monitoring: EMA of submit→frame round-trip (ms).
        # VT decode is synchronous on the RX thread, so this is the HW decode
        # time per NAL. Shared with the control-socket snapshot.
        self._decode_latency_ms: float = 0.0
        self._submit_t: float = 0.0

    # -- params / session ---------------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict) -> None:
        """Build the CMVideoFormatDescription + decompression session from the
        VPS/SPS/PPS harvested at session start. Re-callable (rebuilds)."""
        pps = tuple(bytes(p) for p in all_pps.values()) if all_pps else ()
        self._params = (bytes(vps), bytes(sps), pps)
        with self._lock:
            self._build_session_locked()

    def _build_session_locked(self) -> None:
        if self._params is None:
            return
        vps, sps, pps = self._params
        psets = (vps, sps) + pps
        sizes = tuple(len(p) for p in psets)
        st, fmt = _CM.CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            None, len(psets), psets, sizes, 4, None, None)
        if st != 0 or fmt is None:
            raise RuntimeError(f"CMVideoFormatDescription create failed: {st}")
        # Ask for the native 4:4:4 biplanar output (nv24) so no conversion.
        attrs = {_Q.kCVPixelBufferPixelFormatTypeKey: _NV24_FMT}
        st, session = _VT.VTDecompressionSessionCreate(
            None, fmt, None, attrs, None, None)
        if st != 0 or session is None:
            # fall back to letting VT pick the format
            st, session = _VT.VTDecompressionSessionCreate(
                None, fmt, None, None, None, None)
            if st != 0 or session is None:
                raise RuntimeError(f"VTDecompressionSession create failed: {st}")
        self._teardown_session_locked()
        self._fmt = fmt
        self._session = session
        log.info("VTHevcDecoder: native VideoToolbox session ready (%d tiles)",
                 self.num_tiles)

    def _teardown_session_locked(self) -> None:
        if self._session is not None:
            try:
                _VT.VTDecompressionSessionWaitForAsynchronousFrames(self._session)
                _VT.VTDecompressionSessionInvalidate(self._session)
            except Exception as e:
                log.debug("VT session teardown: %s", e)
            self._session = None

    # -- feed ---------------------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict) -> None:
        """Feed the session-start IDR burst in ROUND-ROBIN (tile-0 NAL i,
        tile-1 NAL i, … then NAL i+1) — the order Apple's encoder emits frames
        in real time, so VT's DPB sees POCs in natural decode order. (Feeding
        each tile's NALUs grouped instead corrupts the cross-tile references.)"""
        if not tile_nalu_cache:
            return
        max_burst = max((len(v) for v in tile_nalu_cache.values()), default=0)
        for idx in range(max_burst):
            for ti in sorted(tile_nalu_cache):
                nalus = tile_nalu_cache[ti]
                if idx < len(nalus):
                    self.feed_nalu(nalus[idx], ti)

    def feed_nalu(self, nalu: bytes, tile_idx: int,
                  donl: Optional[int] = None) -> None:
        """Wrap one NAL unit in a length-prefixed CMSampleBuffer and submit it
        for async decode. The output is dispatched to `tile_idx` by the closure
        below. No RPS pre-check / no drop: VT manages its own DPB and conceals
        missing refs instead of blocking.

        `donl` (the libav path's HEVC decoding-order number, used there for
        DON-based LTRP reference tracking) is accepted for API parity with
        `HevcDecoder.feed_nalu` and intentionally ignored: VideoToolbox owns
        its own DPB + decode ordering, so iss does not steer its references."""
        session = self._session
        if session is None or not nalu:
            return
        t = (nalu[0] >> 1) & 0x3F
        if 0 <= tile_idx < self.num_tiles:
            b = self.nalu_counts_per_tile[tile_idx]
            b[t] = b.get(t, 0) + 1
        if t in IDR_RANGE:
            # A tile-0 IDR re-roots the shared context for ALL tiles (Apple
            # IDRs on tile 0 only; cross-tile refs re-root the rest), so fan
            # out — matches the libav path. This is half of the gate's
            # two-condition `keyframe_required` clear; `mark_clean` on the
            # next good frame is the other half. Without it the gate — armed
            # by an SSRC adoption or a stuck event from session.py, not by VT
            # (VT decodes fine) — stays armed forever and FIR-storms ("4 tiles
            # still need recovery; Apple not responding to FIR").
            for ti in range(self.num_tiles):
                self._gate.mark_idr_observed(ti)
        # Only VCL slices (0-31) are decode frames. VPS/SPS/PPS (32-34), SEI
        # (39/40) etc. reached the decoder inline on the libav path, but VT
        # takes its parameter sets from the CMVideoFormatDescription — feeding
        # a non-VCL NALU as a sample buffer just produces a decode error.
        if t >= 32:
            return
        if _nalu_dump_f is not None:
            _nalu_dump_f.write(b"\x00\x00\x00\x01" + (
                nalu if isinstance(nalu, bytes) else bytes(nalu)))
        self._feed_total += 1
        if _VT_DEBUG and self._feed_total % 200 == 0:
            log.info("VTdbg feed/tile=%s good=%s errs=%d",
                     [sum(d.values()) for d in self.nalu_counts_per_tile],
                     self.good_counts, self._decode_errors)
        sb = self._make_sample(nalu)
        if sb is None:
            return
        self._seq += 1
        seq = self._seq
        self._submit_t = time.monotonic()
        def _handler(status, info_flags, image_buffer, pts, dur,
                     _ti=tile_idx, _seq=seq):
            self._on_decoded(_ti, _seq, status, image_buffer)
        try:
            # Synchronous decode: the output handler fires inline before this
            # returns. (Async mode only delivers frames when
            # WaitForAsynchronousFrames is called, which a live stream never
            # does — that starves the handlers.) Decode is fed from the RX
            # thread; VT HW decode is sub-millisecond per tile.
            _VT.VTDecompressionSessionDecodeFrameWithOutputHandler(
                session, sb, 0, None, _handler)
        except Exception as e:
            log.debug("VT decode submit failed (tile %d): %s", tile_idx, e)

    def _make_sample(self, nalu: bytes):
        payload = struct.pack(">I", len(nalu)) + (
            nalu if isinstance(nalu, bytes) else bytes(nalu))
        st, bb = _CM.CMBlockBufferCreateWithMemoryBlock(
            None, None, len(payload), None, None, 0, len(payload), 0, None)
        if st != 0 or bb is None:
            return None
        _CM.CMBlockBufferReplaceDataBytes(payload, bb, 0, len(payload))
        st, sb = _CM.CMSampleBufferCreateReady(
            None, bb, self._fmt, 1, 0, (), 1, (len(payload),), None)
        if st != 0:
            return None
        return sb

    # -- output -------------------------------------------------------------

    def _on_decoded(self, tile_idx, seq, status, image_buffer) -> None:
        """VT output-handler (runs on a VT dispatch thread). On success, copy
        the Y + interleaved-UV planes into a passthrough TileFrame and publish
        to the tile slot.

        On a decode error we do NOT touch the gate's `bad_streak`: VideoToolbox
        conceals the odd missing-ref frame and recovers on Apple's next IDR
        (~1.7 s) on its own — it never BLOCKS the way libav's hwaccel does. The
        gate's decode-error path is tuned for that libav wedge; driving it from
        VT's benign conceals just makes the session.py Path-B watchdog restart
        the session in a loop. A genuine VT freeze (no frames at all) is still
        caught by the gap-based Path-A watchdog. We only count errors for the
        diagnostic snapshot."""
        if status != 0 or image_buffer is None:
            self._decode_errors += 1
            # Request a recovery keyframe EARLY (throttled) rather than waiting
            # for session.py's 3 s "stuck" watchdog. A single missing/undecodable
            # frame breaks the inter-frame chain → every later frame conceals
            # until an IDR; Apple answers a FIR in ~0.2 s, so FIRing on the first
            # error of a burst turns a ~3 s freeze into a sub-second blip. The
            # gate's FIR send is itself rate-limited, and the 0.5 s throttle here
            # keeps bad_streak low so the Path-B watchdog never trips (and VT's
            # restart() is a no-op regardless) — no storm, no cascade.
            now = time.monotonic()
            if (0 <= tile_idx < self.num_tiles
                    and now - self._last_err_fir_t >= 0.5):
                self._last_err_fir_t = now
                self._gate.mark_decode_error(tile_idx)
            return
        tile = self._cvbuf_to_tile(image_buffer)
        if tile is None:
            return
        slot = self._tiles[tile_idx]
        with slot.lock:
            if seq < slot.seq:                 # a newer frame already landed
                return
            slot.frame = tile
            slot.seq = seq
            slot.good_count += 1
        # Diagnostic: measure the session-wide publish gap. _on_decoded is
        # single-threaded (sync decode on the RX process thread), so the gap
        # between any two calls is the real publish gap. A long gap with MANY
        # feeds-in-between = VT stalled while being fed; with ~0 feeds = the
        # RX/host starved us. This is what the Path-A "decoder stuck" watchdog
        # trips on.
        now = time.monotonic()
        if self._last_pub_t > 0.0:
            gap = now - self._last_pub_t
            if gap > 1.5:
                fed = self._feed_total - self._feeds_at_last_pub
                if fed > 30:
                    # Fed but couldn't decode for >1.5 s = a real decode stall.
                    log.warning("VT decode stall %.2fs — fed %d NALUs, all "
                                "concealed (broken ref chain)", gap, fed)
                elif _VT_DEBUG:
                    # ~0 feeds = host/RX sent nothing (static screen / idle).
                    log.info("VT publish gap %.2fs — host idle (fed %d)", gap, fed)
        self._last_pub_t = now
        self._feeds_at_last_pub = self._feed_total
        if self._submit_t > 0.0:
            latency_ms = (now - self._submit_t) * 1000
            self._decode_latency_ms = 0.1 * latency_ms + 0.9 * self._decode_latency_ms
        # The clean-frame half of the gate's two-condition recovery clear
        # (pairs with mark_idr_observed). Lets `keyframe_required` discharge
        # so the FIR storm stops once a real frame flows again.
        self._gate.mark_clean(tile_idx)
        if self._on_frame_published is not None:
            try:
                self._on_frame_published(tile_idx)
            except Exception as e:
                log.debug("on_frame_published raised: %s", e)

    def _cvbuf_to_tile(self, buf) -> Optional[TileFrame]:
        if _Q.CVPixelBufferLockBaseAddress(buf, _LOCK_READONLY) != 0:
            return None
        try:
            w = _Q.CVPixelBufferGetWidth(buf)
            h = _Q.CVPixelBufferGetHeight(buf)
            if _Q.CVPixelBufferGetPlaneCount(buf) < 2:
                return None
            # H1: only nv24 (444f) is the biplanar 4:4:4 layout the rest of the
            # pipeline assumes. The fallback decode session (set_params) can
            # hand us a VT-chosen format (e.g. 4:2:0 biplanar) that would pass
            # the plane-count guard above and be mislabeled 4:4:4 → garbage
            # chroma. Reject it rather than render garbage.
            fmt = _Q.CVPixelBufferGetPixelFormatType(buf)
            if fmt != _NV24_FMT:
                if not self._fmt_warned:
                    self._fmt_warned = True
                    log.warning("VTHevcDecoder: non-nv24 output format 0x%x — "
                                "dropping frames (expected 444f)", fmt)
                return None
            y_rb = _Q.CVPixelBufferGetBytesPerRowOfPlane(buf, 0)
            uv_rb = _Q.CVPixelBufferGetBytesPerRowOfPlane(buf, 1)
            y_h = _Q.CVPixelBufferGetHeightOfPlane(buf, 0)
            uv_h = _Q.CVPixelBufferGetHeightOfPlane(buf, 1)
            y_base = _Q.CVPixelBufferGetBaseAddressOfPlane(buf, 0)
            uv_base = _Q.CVPixelBufferGetBaseAddressOfPlane(buf, 1)
            if y_base is None or uv_base is None:
                return None
            y_bytes = bytes(y_base.as_buffer(y_rb * y_h))
            uv_bytes = bytes(uv_base.as_buffer(uv_rb * uv_h))
        except Exception as e:
            log.debug("CVPixelBuffer read failed: %s", e)
            return None
        finally:
            _Q.CVPixelBufferUnlockBaseAddress(buf, _LOCK_READONLY)
        if not self._seen_fmt:
            self._seen_fmt = True
            log.info("VTHevcDecoder output: nv24 %dx%d (y_stride=%d uv_stride=%d)",
                     w, h, y_rb, uv_rb)
        # nv24 passthrough: v is None, u carries the interleaved UV plane.
        return TileFrame(
            y=y_bytes, u=uv_bytes, v=None,
            width=w, height=h,
            y_stride=y_rb,
            uv_stride=uv_rb,
            chroma_width=w,
            chroma_height=h,
        )

    # -- consumer -----------------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.frame
            count = slot.good_count
            if frame is None or count <= slot.last_evaluated_count:
                return None
            slot.last_evaluated_count = count
            return frame

    def consume_fir_request(self) -> set:
        """Tile indices needing a fresh IDR (read + cleared) — gate-backed."""
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int):
        """Gate HUD-state snapshot for one tile."""
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def good_counts(self) -> list:
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list:
        """Per-tile CLEAN (non-concealed) published-frame totals, the signal
        session.py uses to flag GRAY tiles (high frame rate, ~0 clean = gray).
        VideoToolbox conceals missing refs internally at Apple quality and does
        not publish the libav-style persistent-gray frames this metric exists
        to catch, so every published (good) frame counts as clean. Aliasing
        good_counts keeps API parity with HevcDecoder.clean_counts."""
        return [t.good_count for t in self._tiles]

    @property
    def bad_tiles(self) -> set:
        """Tiles the gate considers gray/concealing and awaiting recovery.
        Delegates to the gate (decoder-agnostic), matching HevcDecoder."""
        return self._gate.bad_tiles

    @property
    def decode_latency_ms(self) -> float:
        return self._decode_latency_ms

    @property
    def decode_queue_depth(self) -> int:
        return 0  # synchronous VT path has no queue

    @property
    def decode_queue_cap(self) -> int:
        return 0

    @property
    def decode_queue_drops(self) -> int:
        return 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:                   # parity no-op (no worker thread)
        pass

    def restart(self) -> None:
        """DPB-PRESERVING no-op by design. session.py calls restart() on SSRC
        adoption and from the stall watchdog to "reset" the decoder — but for
        VideoToolbox a rebuild WIPES the DPB, producing a ~3 s
        all-frames-conceal burst until the next IDR, which re-trips the
        watchdog → restart → a self-sustaining freeze cascade (the same
        flush-is-the-engine pattern the libav path suffers). VT doesn't need
        it: it re-roots on the IDR the paired request_fir() pulls (or Apple's
        natural cadence) while keeping its DPB intact, so every "restart" path
        degrades to native-aligned FIR-only recovery. Only set_params() — a
        real format/resolution change with a new SPS — rebuilds the session."""
        self._restart_suppressed += 1
        if _VT_DEBUG:
            log.info("VT restart() suppressed (DPB-preserving, #%d) — "
                     "recovering via FIR/IDR, not a session wipe",
                     self._restart_suppressed)

    def close(self) -> None:
        with self._lock:
            self._teardown_session_locked()


__all__ = ["VTHevcDecoder", "available"]
