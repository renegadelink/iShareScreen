"""Unit tests for per-tile mid-stream wedge recovery (ISS_PERTILE_RECOVERY).

Pure-logic tests, same harness as `test_decode_recovery.py`: a minimal
`HevcDecoder` built via `__new__`, a real `FrameQualityGate`, real
`_TileSlot`s, `_queue=None`. No network, no host, no real decode except a
single tiny in-memory `av.VideoFrame` used to prove the healthy-tile
publish path survives recovery.

The flag is read once at module import into the module-level constant
`hevc._PERTILE_RECOVERY`; we flip it per-test with `monkeypatch.setattr`
so the production default (OFF) is never mutated globally.

What we assert:
  * `_try_recovery` under the flag marks ONLY the gate-flagged broken
    tiles (`bad_streak > 0`) as awaiting re-root, leaves `_dpb_has_idr`
    alone, and still drains the worker queue;
  * a tile with no flagged break is left out of the await set, keeps
    decoding its P-frames, and keeps publishing through `get_frame`;
  * the broken tile's P-frames are dropped and its `get_frame` returns
    None until an IDR arrives;
  * an IDR clears the await set for ALL tiles (shared tile-0 re-root);
  * the cold-start all-tiles drop is preserved under the flag;
  * with the flag OFF the path is byte-identical to today (global
    `_dpb_has_idr = False`, empty await set).
"""
from __future__ import annotations

import queue as _queue

import numpy as np
import av

from isharescreen.proxy.media import hevc
from isharescreen.proxy.media.hevc import HevcDecoder, _TileSlot
from isharescreen.proxy.media.quality_gate import FrameQualityGate


# ── NALU fixtures ────────────────────────────────────────────────────
# HEVC NAL header byte0 = (nal_unit_type << 1) with the forbidden-zero
# bit clear; `_is_decodable_nalu` also needs len>=3 and byte2 high bit
# (first_slice_segment_in_pic_flag) set.
_IDR = b"\x26\x01\x80\x00"    # nal_unit_type 19 (IDR_W_RADL) — in IDR_RANGE
_PFRAME = b"\x02\x01\x80\x00"  # nal_unit_type 1 (TRAIL_R) — a P-frame slice


class _StubCodec:
    """Stands in for the libav CodecContext. `decode()` accepts a packet
    and returns no output frames — enough to exercise `_decode_one`'s
    gating + bookkeeping without a real decoder. Counts packets so a test
    can assert a NALU actually reached the codec (vs being dropped)."""

    def __init__(self) -> None:
        self.packets: list = []

    def decode(self, pkt):
        self.packets.append(pkt)
        return []


def _make_decoder(num_tiles: int = 4, *, with_codec: bool = False) -> HevcDecoder:
    import threading

    d = HevcDecoder.__new__(HevcDecoder)
    d._gate = FrameQualityGate(num_tiles)
    d._tiles = [_TileSlot() for _ in range(num_tiles)]
    d._queue = None
    d._eagain_streak = 0
    d._recovery_in_progress = False
    d._silent_nalus = 0
    d._dpb_has_idr = True
    d._tiles_await_idr = set()
    d._hw_name = "videotoolbox"
    # State touched by `_decode_one`'s IDR / packet paths.
    d._codec = _StubCodec() if with_codec else None
    d._codec_lock = threading.Lock()
    d._next_pts = 0
    d._pts_to_tile = {}
    d._pts_submit_t = {}
    d._decode_latency_ms = 0.0
    d._pre_idr_drops = 0
    d._recent_idr_sizes = []
    d._IDR_HISTORY_LEN = 10
    d._IDR_FAKE_RATIO = 0.40
    d._on_frame_published = None
    return d


def _on(monkeypatch) -> None:
    monkeypatch.setattr(hevc, "_PERTILE_RECOVERY", True)


def _tiny_frame() -> av.VideoFrame:
    """A 2x2 yuv444p frame so `_av_frame_to_tile` produces a real
    TileFrame on the healthy-tile publish path. PyAV wants planar
    `(3, H, W)` for yuv444p."""
    arr = np.zeros((3, 2, 2), dtype=np.uint8)
    return av.VideoFrame.from_ndarray(arr, format="yuv444p")


def _publish_into(d: HevcDecoder, tile_idx: int) -> None:
    """Drop a fresh decoded frame into a tile slot, as `_publish_frame`
    would, so the next `get_frame(tile_idx)` has something to evaluate."""
    slot = d._tiles[tile_idx]
    slot.raw_frame = _tiny_frame()
    slot.good_count += 1


# ── _try_recovery: per-tile marking ──────────────────────────────────

def test_recovery_marks_only_broken_tile(monkeypatch):
    """Under the flag, only tiles with bad_streak>0 enter the await set;
    `_dpb_has_idr` stays True so healthy tiles keep flowing."""
    _on(monkeypatch)
    d = _make_decoder()
    # Tile 2 is the broken one; the rest are healthy.
    d._gate._states[2].bad_streak = 1

    d._try_recovery()

    assert d._tiles_await_idr == {2}
    assert d._dpb_has_idr is True          # NOT the global gate
    assert d._eagain_streak == 0
    # Only the broken tile's per-tile diagnostic flag is cleared.
    assert d._tiles[2].saw_idr_since_reset is False


def test_recovery_marks_multiple_broken_tiles(monkeypatch):
    _on(monkeypatch)
    d = _make_decoder()
    d._gate._states[1].bad_streak = 3
    d._gate._states[3].bad_streak = 1

    d._try_recovery()

    assert d._tiles_await_idr == {1, 3}
    assert d._dpb_has_idr is True


def test_recovery_no_flagged_tiles_falls_back_to_all(monkeypatch):
    """Defensive fallback: a wedge with zero flagged tiles marks every
    tile so the stream can never get permanently stuck."""
    _on(monkeypatch)
    d = _make_decoder(num_tiles=4)
    # No bad_streak set anywhere.

    d._try_recovery()

    assert d._tiles_await_idr == {0, 1, 2, 3}
    assert d._dpb_has_idr is True


def test_recovery_drains_queue(monkeypatch):
    """The stale-P-frame backlog is still drained under the flag."""
    _on(monkeypatch)
    d = _make_decoder()
    d._gate._states[0].bad_streak = 1
    d._queue = _queue.Queue()
    for _ in range(100):
        d._queue.put_nowait((b"x", 0))

    d._try_recovery()

    assert d._queue.qsize() == 0


def test_recovery_accumulates_across_calls(monkeypatch):
    """A second wedge hitting a different tile adds to the set rather than
    replacing it (`|=`), so an earlier broken tile isn't prematurely
    cleared before the re-root IDR."""
    _on(monkeypatch)
    d = _make_decoder()
    d._gate._states[1].bad_streak = 1
    d._try_recovery()
    assert d._tiles_await_idr == {1}

    d._gate._states[1].bad_streak = 0   # tile 1 cleaned, tile 2 now breaks
    d._gate._states[2].bad_streak = 1
    d._try_recovery()
    assert d._tiles_await_idr == {1, 2}


# ── get_frame: per-tile hold ─────────────────────────────────────────

def test_get_frame_holds_awaiting_tile_publishes_healthy(monkeypatch):
    """The awaiting (broken) tile returns None; a healthy sibling with a
    freshly published frame still returns a real TileFrame."""
    _on(monkeypatch)
    d = _make_decoder()
    d._reformatter = [None]
    d._seen_fmts = set()
    d._tiles_await_idr = {2}

    # Both tiles have a fresh decoded frame waiting.
    _publish_into(d, 0)
    _publish_into(d, 2)

    assert d.get_frame(2) is None          # broken tile held back
    healthy = d.get_frame(0)
    assert healthy is not None             # healthy tile still publishes
    assert healthy.width == 2 and healthy.height == 2


def test_get_frame_cold_start_still_blocks_all(monkeypatch):
    """Cold start (no IDR ever): `_dpb_has_idr` False blocks every tile
    regardless of the await set — under the flag too."""
    _on(monkeypatch)
    d = _make_decoder()
    d._reformatter = [None]
    d._seen_fmts = set()
    d._dpb_has_idr = False
    d._tiles_await_idr = set()
    _publish_into(d, 0)

    assert d.get_frame(0) is None


# ── _decode_one: per-tile drop + IDR clear ───────────────────────────

def test_decode_one_drops_awaiting_tile_keeps_healthy(monkeypatch):
    """A P-frame for an awaiting tile is dropped (never reaches the
    codec); a P-frame for a healthy tile decodes normally."""
    _on(monkeypatch)
    d = _make_decoder(with_codec=True)
    d._tiles_await_idr = {2}

    d._decode_one(_PFRAME, 2)              # awaiting tile → dropped
    assert d._pre_idr_drops == 1
    assert len(d._codec.packets) == 0
    assert d._next_pts == 0                # nothing was queued for decode

    d._decode_one(_PFRAME, 0)              # healthy tile → decoded
    assert len(d._codec.packets) == 1
    assert d._next_pts == 1
    assert d._pre_idr_drops == 1           # unchanged


def test_decode_one_idr_clears_await_set(monkeypatch):
    """The shared tile-0 IDR re-roots the context → clears the whole
    await set, so every previously-broken tile resumes."""
    _on(monkeypatch)
    d = _make_decoder(with_codec=True)
    d._tiles_await_idr = {1, 2, 3}

    d._decode_one(_IDR, 0)

    assert d._tiles_await_idr == set()     # all tiles re-rooted
    assert d._dpb_has_idr is True
    assert len(d._codec.packets) == 1      # the IDR itself was decoded


def test_decode_one_cold_start_drops_all_tiles(monkeypatch):
    """Cold start with the flag ON and an empty await set must still drop
    EVERY tile's P-frames (the `_dpb_has_idr` clause), so the flag never
    regresses the pre-first-IDR behavior."""
    _on(monkeypatch)
    d = _make_decoder(with_codec=True)
    d._dpb_has_idr = False
    d._tiles_await_idr = set()

    for ti in range(4):
        d._decode_one(_PFRAME, ti)

    assert d._pre_idr_drops == 4
    assert len(d._codec.packets) == 0      # nothing decoded pre-IDR


def test_decode_one_recovery_then_idr_roundtrip(monkeypatch):
    """End-to-end: break tile 2, recover (per-tile), confirm tile 2's
    P-frame is dropped while tile 0's decodes, then an IDR clears the
    hold and tile 2 resumes."""
    _on(monkeypatch)
    d = _make_decoder(with_codec=True)
    d._gate._states[2].bad_streak = 1
    d._try_recovery()
    assert d._tiles_await_idr == {2}

    d._decode_one(_PFRAME, 2)
    assert len(d._codec.packets) == 0      # tile 2 held
    d._decode_one(_PFRAME, 0)
    assert len(d._codec.packets) == 1      # tile 0 flowing

    d._decode_one(_IDR, 0)                 # re-root
    assert d._tiles_await_idr == set()
    d._decode_one(_PFRAME, 2)              # tile 2 resumes
    assert len(d._codec.packets) == 3      # IDR + tile-2 P now decoded


# ── flag OFF: byte-identical legacy path ─────────────────────────────

def test_flag_off_recovery_uses_global_gate(monkeypatch):
    """With ISS_PERTILE_RECOVERY off, `_try_recovery` is the legacy path:
    global `_dpb_has_idr = False`, await set untouched (stays empty)."""
    monkeypatch.setattr(hevc, "_PERTILE_RECOVERY", False)
    d = _make_decoder()
    d._gate._states[2].bad_streak = 1

    d._try_recovery()

    assert d._dpb_has_idr is False         # legacy global gate
    assert d._tiles_await_idr == set()     # never populated
    for slot in d._tiles:
        assert slot.saw_idr_since_reset is False


def test_flag_off_get_frame_ignores_await_set(monkeypatch):
    """With the flag off, a stray await-set entry has no effect — only
    `_dpb_has_idr` gates `get_frame`."""
    monkeypatch.setattr(hevc, "_PERTILE_RECOVERY", False)
    d = _make_decoder()
    d._reformatter = [None]
    d._seen_fmts = set()
    d._tiles_await_idr = {0}               # would block under the flag
    _publish_into(d, 0)

    assert d.get_frame(0) is not None      # flag off → not held
