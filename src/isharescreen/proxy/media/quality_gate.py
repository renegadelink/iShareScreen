"""Per-tile decoder-recovery state.

The recovery logic is *not* heuristic. We do not look at the pixel
contents of decoded frames to guess whether the decoder concealed
something — that approach can never distinguish real gray content
(curtain mode, a settings panel, a dark theme) from concealment
fill, and false-positives on real content trigger unnecessary FIR
storms / decoder restarts.

Instead, recovery is driven by the two signals we *can* trust:

  1. **RTP sequence-number gaps** — tracked at the SRTP RX layer,
     surfaced via NACK retransmits. Ground truth for "we lost a
     packet."
  2. **libavcodec error reports** — `AVFrame.decode_error_flags` on
     each decoded frame, plus libav log callbacks for messages like
     "Could not find ref with POC N" that escape the API but are
     emitted to the log system by the decoder when it had to
     conceal. Ground truth for "the decoder concealed something."

This file just manages the per-tile FIR-pending set + an opaque
`bad_streak` counter for diagnostics. `mark_decode_error(i)` is the
single public hook other code calls when one of the trusted signals
fires.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .tiles import TileFrame  # noqa: F401  (kept for type-annotation parity)


log = logging.getLogger(__name__)


# ── public state (kept for HUD-overlay round-trip compat) ────────────

STATE_INIT = "init"
STATE_OK = "ok"
STATE_HOLD = "hold"


@dataclass(slots=True, frozen=True)
class TileVisState:
    state: str
    mean: float = 0.0
    std: float = 0.0


@dataclass(slots=True)
class _TileState:
    bad_streak: int = 0       # decoder-reported errors since last recovery
    needs_real_frame: bool = False
    vis: TileVisState = field(default_factory=lambda: TileVisState(STATE_INIT))


class FrameQualityGate:
    """Tracks per-tile FIR-pending set. `mark_decode_error(i)` is the
    only mutation hook; `consume_fir_request()` drains the pending set
    each tx-tick.

    Methods are not thread-safe — call from the decoder's frame-publish
    thread (or hold the decoder's lock around them). The FIR set is
    safe to read from another thread *as long as* no `mark_decode_error`
    call is in progress.
    """

    # Minimum gap between successive FIRs for the same tile. Multiple
    # call sites (libav concealment log, RPS pre-decode tracker, post-
    # decode `had_decode_error`, EAGAIN streaks in `_decode_one`) can
    # all funnel into `mark_decode_error` for one logical event, and
    # libav happily re-flags concealed output frames for hundreds of
    # ms after a FIR while the decoder is still catching up. Without
    # this gate the loop produces ~2 FIRs/sec/tile in steady-state
    # (rate-limited only by the 0.5 s tx drain cadence), which is
    # wasteful on the wire and on Apple's encoder. 2 s is comfortably
    # longer than a single FIR round-trip + IDR delivery + decode.
    _PER_TILE_FIR_COOLDOWN_S: float = 2.0

    def __init__(
        self,
        num_tiles: int,
        *,
        enabled: bool = True,           # kept for API back-compat
    ) -> None:
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self._num_tiles = num_tiles
        self._states: list[_TileState] = [_TileState() for _ in range(num_tiles)]
        self._fir_pending: set[int] = set()
        # Last time a FIR was actually emitted (drained) for each tile.
        # Initialised to 0 so the first error per tile fires immediately.
        import time as _time
        self._fir_last_t: list[float] = [0.0] * num_tiles
        self._time = _time
        self.flicker_events = 0  # diagnostic counter

    # -- main "publish" hook --------------------------------------------
    # Always publishes. Kept as a callable for API compatibility with
    # the old gate; no pixel inspection is done.
    def should_publish(self, tile_idx: int, tile: TileFrame) -> bool:
        state = self._states[tile_idx]
        state.vis = TileVisState(STATE_OK)
        return True

    # -- decoder-error path (the only escalation source) ----------------
    def mark_decode_error(self, tile_idx: int) -> None:
        """Trusted signal that libavcodec concealed / failed for this
        tile. Adds the tile to the FIR-pending set; the session will
        send an RTCP FIR on the next tx-tick.

        Per-tile cooldown: a tile that had a FIR emitted within the
        last `_PER_TILE_FIR_COOLDOWN_S` doesn't get re-added. Internal
        streak/`needs_real_frame` bookkeeping still updates so the
        watchdog logic upstream sees the truth, but we don't spam
        Apple with FIRs while the previous one's IDR is still in
        flight or being decoded.
        """
        if tile_idx < 0 or tile_idx >= self._num_tiles:
            return
        state = self._states[tile_idx]
        state.bad_streak += 1
        state.needs_real_frame = True
        if self._time.monotonic() - self._fir_last_t[tile_idx] < self._PER_TILE_FIR_COOLDOWN_S:
            return
        if tile_idx not in self._fir_pending:
            self._fir_pending.add(tile_idx)
            self.flicker_events += 1
            log.info("tile %d decode error → FIR requested", tile_idx)

    # -- "I just published a clean frame" hook --------------------------
    # Decoders call this when a tile produces output without a
    # decode_error_flag set. Resets the streak so per-tile watchdogs
    # don't fire on healthy tiles.
    def mark_clean(self, tile_idx: int) -> None:
        if tile_idx < 0 or tile_idx >= self._num_tiles:
            return
        state = self._states[tile_idx]
        state.bad_streak = 0
        state.needs_real_frame = False

    # -- FIR consumption -------------------------------------------------
    def consume_fir_request(self) -> set[int]:
        if not self._fir_pending:
            return set()
        out = self._fir_pending
        self._fir_pending = set()
        # Stamp last-FIR-emitted time for cooldown bookkeeping.
        now = self._time.monotonic()
        for ti in out:
            if 0 <= ti < self._num_tiles:
                self._fir_last_t[ti] = now
        return out

    # -- introspection ---------------------------------------------------
    def tile_state(self, tile_idx: int) -> TileVisState:
        return self._states[tile_idx].vis

    def needs_real_frame(self, tile_idx: int) -> bool:
        return self._states[tile_idx].needs_real_frame

    # -- lifecycle -------------------------------------------------------
    def reset(self, tile_idx: Optional[int] = None) -> None:
        if tile_idx is None:
            for i in range(self._num_tiles):
                self._states[i] = _TileState()
            self._fir_pending.clear()
        else:
            self._states[tile_idx] = _TileState()
            self._fir_pending.discard(tile_idx)


__all__ = [
    "FrameQualityGate",
    "STATE_HOLD",
    "STATE_INIT",
    "STATE_OK",
    "TileVisState",
]
