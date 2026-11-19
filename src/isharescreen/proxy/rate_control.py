"""Receiver rate-control feedback (RTCP APP "RCTL").

Apple's own Screen Sharing viewer sends, besides RRs and LTR acks, one
RTCP APP packet named "RCTL" every 50 ms on the video leg (UDP port+1).
It is AVConference's receiver-side rate-control feedback, and its last
field is a receiver bandwidth estimate (BWE, kbps) that the host's video
encoder uses as its target. Without it the host holds a flat ~20 Mbps
while the screen is busy; with it the host follows the estimate (up to
the u16 wire limit, 65535 kbps).

Wire format (built by `rtcp.build_rtcp_app_rctl`, protected with the
viewer->server video SRTCP key, the same context as RRs and LTR acks):

    80 cc 00 07  <sender ssrc>  52 43 54 4c ("RCTL")
    85 00 00 04                 info header (type 2, 20 bytes)
    u16 echo                    low 16 bits of the newest video RTP timestamp
    u32 0
    u16 echo_delay_ms           ms since that packet arrived
    u16 now                     receiver clock, 1/1024 s
    u16 owrd                    one-way relative delay (0: not measured)
    u16 burst << 12 | loss      longest loss run this tick (cap 15) |
                                windowed loss, whole percent
    u16 bwe                     bandwidth estimate, kbps

Opt-in: ISS_RCTL=1 turns it on (Session); ISS_RCTL_BWE=<kbps> pins the
estimate instead of the adaptive one below.

This module is transport-agnostic: `RateControl.note_packet` is fed from
the video receive path, `RateControl.tick` returns the next packet body
(unprotected), and the session sends it. The estimator is a small
loss-based AIMD in the spirit of draft-ietf-rmcat-gcc's loss controller:

  * starts at 50 Mbps (the native viewer reaches ~50 Mbps from the first
    seconds on the same link);
  * multiplicative increase (6 %/s, 1 %/s near the last cut point) toward
    65 Mbps while windowed loss is under 2 % and the host fills at least
    60 % of the estimate (an idle screen never decays the estimate);
  * a cut (x0.9, or toward 0.9 x the receive rate when the link is
    saturated, never below 0.6 x per step) only on SUSTAINED loss: two
    consecutive 1 s buckets over 10 %. Wi-Fi loses packets in short
    bursts; one bad second never cuts;
  * receiver backlog: when this client cannot process packets as fast as
    they arrive (the receive queue between the socket thread and the
    decrypt/depacketize thread keeps growing), the receiver -- not the
    link -- is the bottleneck. A growing backlog over `backlog_high`
    packets cuts toward 0.9 x the processed rate (at most once a second,
    and only while the estimate is above that rate); any backlog over
    `backlog_low` blocks increases. Without this a slow
    client asks for more than it can handle, its queue overflows, and the
    resulting loss turns into keyframe storms;
  * 2-10 % loss holds; floor 4 Mbps; 1 s startup grace.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

from .protocol.rtcp import build_rtcp_app_rctl


RCTL_INTERVAL_S = 0.05


@dataclass
class EstimatorConfig:
    initial_kbps: float = 50_000.0
    floor_kbps: float = 4_000.0
    ceiling_kbps: float = 65_000.0       # BWE is a u16 on the wire
    increase_per_s: float = 0.06
    near_cut_increase_per_s: float = 0.01
    near_cut_band: float = 0.10
    decrease_factor: float = 0.9
    max_step_down: float = 0.6
    increase_below_loss: float = 0.02
    decrease_above_loss: float = 0.10
    sustained_loss_buckets: int = 2
    window_s: float = 1.0
    min_packets_for_loss: int = 20
    increase_holdoff_s: float = 2.0
    app_limited_ratio: float = 0.6
    startup_grace_s: float = 1.0
    backlog_low: int = 300
    backlog_high: int = 1500
    backlog_holdoff_s: float = 1.0


class BandwidthEstimator:
    """Loss-based AIMD bandwidth estimate, ticked every RCTL interval.
    Pure: time is the `now` passed to `update` (any monotonic seconds)."""

    def __init__(self, config: Optional[EstimatorConfig] = None) -> None:
        self.config = c = config or EstimatorConfig()
        self.estimate_kbps = min(max(c.initial_kbps, c.floor_kbps), c.ceiling_kbps)
        self.state = "startup"
        self.decreases = 0
        self._start: Optional[float] = None
        self._last_now: Optional[float] = None
        self._last_decrease: Optional[float] = None
        self._cut_point_kbps: Optional[float] = None
        # (t, dt, bytes, expected, received) per tick within `window_s`.
        self._window: list[tuple[float, float, int, int, int]] = []
        # Non-overlapping loss bucket: [start, expected, received].
        self._bucket: Optional[list] = None
        self.lossy_buckets = 0
        # (t, backlog) samples over the last second, to see it grow.
        self._backlog: list[tuple[float, int]] = []

    @property
    def bwe_kbps(self) -> int:
        return max(0, min(0xFFFF, int(round(self.estimate_kbps))))

    @property
    def loss_fraction(self) -> Optional[float]:
        exp = sum(w[3] for w in self._window)
        rec = sum(w[4] for w in self._window)
        if exp < self.config.min_packets_for_loss:
            return None
        return max(0.0, (exp - rec) / exp)

    @property
    def receive_kbps(self) -> Optional[float]:
        span = sum(w[1] for w in self._window)
        if span < self.config.window_s / 2:
            return None
        return sum(w[2] for w in self._window) * 8 / 1000 / span

    def update(self, now: float, rx_bytes: int, expected: int, received: int,
               backlog: int = 0) -> int:
        """One tick. `rx_bytes`/`expected`/`received`: video bytes and
        packets this tick, as processed (after the receive queue);
        `backlog`: packets waiting in that queue now."""
        c = self.config
        dt = max(0.0, now - self._last_now) if self._last_now is not None else RCTL_INTERVAL_S
        self._last_now = now
        if self._start is None:
            self._start = now
        self._window.append((now, dt, rx_bytes, expected, received))
        self._window = [w for w in self._window if w[0] > now - c.window_s]
        bucket_closed = self._advance_bucket(now, dt, expected, received)
        self._backlog.append((now, backlog))
        self._backlog = [b for b in self._backlog if b[0] > now - c.window_s]

        if now - self._start < c.startup_grace_s - 1e-6:
            self.state = "startup"
            return self.bwe_kbps

        # Cut only while the estimate is above what we process: the host does
        # not go below its own content-driven rate, so once the estimate is
        # under the processed rate another cut cannot shrink the backlog.
        rx = self.receive_kbps
        growing = (backlog > c.backlog_high and backlog > self._backlog[0][1]
                   and (rx is None or self.estimate_kbps > rx))
        if growing:
            if (self._last_decrease is None
                    or now - self._last_decrease >= c.backlog_holdoff_s - 1e-6):
                self._decrease(now)
            else:
                self.state = "hold-backlog"
            return self.bwe_kbps
        if backlog > c.backlog_low:
            self.state = "hold-backlog"
            return self.bwe_kbps

        sustained = self.lossy_buckets >= c.sustained_loss_buckets
        if sustained:
            if bucket_closed:
                self._decrease(now)
            else:
                self.state = "hold-off"
            return self.bwe_kbps

        loss = self.loss_fraction
        if loss is None:
            self.state = "hold-idle"
            return self.bwe_kbps
        if loss >= c.increase_below_loss:
            self.state = "hold"
            return self.bwe_kbps
        if (self._last_decrease is not None
                and now - self._last_decrease < c.increase_holdoff_s - 1e-6):
            self.state = "hold-off"
            return self.bwe_kbps
        rx = self.receive_kbps
        if rx is None or rx < c.app_limited_ratio * self.estimate_kbps:
            self.state = "hold-app"
            return self.bwe_kbps
        rate = c.increase_per_s
        cut = self._cut_point_kbps
        if cut is not None and abs(self.estimate_kbps - cut) <= cut * c.near_cut_band:
            rate = c.near_cut_increase_per_s
        self.estimate_kbps = min(c.ceiling_kbps, self.estimate_kbps * (1 + rate) ** dt)
        self.state = "increase"
        return self.bwe_kbps

    def _advance_bucket(self, now: float, dt: float, expected: int, received: int) -> bool:
        c = self.config
        b = self._bucket if self._bucket is not None else [now - dt, 0, 0]
        b[1] += expected
        b[2] += received
        if now - b[0] < c.window_s - 1e-6:
            self._bucket = b
            return False
        self._bucket = [now, 0, 0]
        if b[1] >= c.min_packets_for_loss:
            loss = max(0.0, (b[1] - b[2]) / b[1])
            self.lossy_buckets = self.lossy_buckets + 1 if loss > c.decrease_above_loss else 0
        else:
            self.lossy_buckets = 0
        return True

    def _decrease(self, now: float) -> None:
        c = self.config
        target = self.estimate_kbps * c.decrease_factor
        rx = self.receive_kbps
        if rx is not None and rx < self.estimate_kbps:
            target = min(target, max(rx * c.decrease_factor,
                                     self.estimate_kbps * c.max_step_down))
        self._cut_point_kbps = self.estimate_kbps
        self.estimate_kbps = max(c.floor_kbps, target)
        self._last_decrease = now
        self.decreases += 1
        self.state = "decrease"


class _SeqCounter:
    """Per-SSRC expected/received counts (RFC 3550 A.1 style extended max
    sequence), drained per tick. Late packets show up as received > expected
    in a later tick, so a windowed sum cancels reordering."""

    __slots__ = ("max_ext", "expected", "received", "run")

    def __init__(self, seq: int) -> None:
        self.max_ext = seq
        self.expected = 1
        self.received = 1
        self.run = 0

    def note(self, seq: int) -> None:
        self.received += 1
        delta = (seq - (self.max_ext & 0xFFFF)) & 0xFFFF
        if delta == 0 or delta >= 0x8000:
            return                      # duplicate or late
        if delta > 3000:
            # Restart / big jump: resync instead of counting a huge loss.
            self.max_ext = (self.max_ext & ~0xFFFF) | seq
            self.expected += 1
            return
        self.max_ext += delta
        self.expected += delta
        self.run = max(self.run, delta - 1)


class RateControl:
    """RCTL state for one media session. Thread-safe: `note_packet` runs on
    the video receive thread, `tick` on the sender's timer."""

    def __init__(self, *, pinned_bwe_kbps: Optional[int] = None,
                 config: Optional[EstimatorConfig] = None) -> None:
        self.estimator = BandwidthEstimator(config)
        self.pinned_bwe_kbps = pinned_bwe_kbps
        self.last_bwe = 0
        self.last_loss_pct = 0
        self.last_burst = 0
        self.sent = 0
        self.last_backlog = 0
        self._lock = threading.Lock()
        self._last_rtp: Optional[tuple[int, float]] = None
        self._bytes = 0
        self._seq: dict[int, _SeqCounter] = {}

    def note_packet(self, ssrc: int, seq: int, rtp_ts: int, nbytes: int, now: float) -> None:
        with self._lock:
            self._last_rtp = (rtp_ts, now)
            self._bytes += nbytes
            c = self._seq.get(ssrc)
            if c is None:
                self._seq[ssrc] = _SeqCounter(seq)
            else:
                c.note(seq)

    def tick(self, sender_ssrc: int, now: float, backlog: int = 0) -> Optional[bytes]:
        """One feedback interval. Returns the unprotected RCTL packet, or
        None before the first video packet. `backlog`: packets received
        but not yet processed (see BandwidthEstimator.update)."""
        with self._lock:
            last = self._last_rtp
            rx_bytes, self._bytes = self._bytes, 0
            expected = received = burst = 0
            for c in self._seq.values():
                expected += c.expected
                received += c.received
                burst = max(burst, c.run)
                c.expected = c.received = c.run = 0
        if last is None:
            return None
        bwe = self.estimator.update(now, rx_bytes, expected, received, backlog)
        self.last_backlog = backlog
        if self.pinned_bwe_kbps is not None:
            bwe = self.pinned_bwe_kbps
        loss = self.estimator.loss_fraction or 0.0
        self.last_bwe = bwe
        self.last_loss_pct = min(100, int(round(loss * 100)))
        self.last_burst = min(15, burst)
        rtp_ts, arrival = last
        self.sent += 1
        return build_rtcp_app_rctl(
            sender_ssrc,
            echo=rtp_ts & 0xFFFF,
            echo_delay_ms=min(0xFFFF, max(0, int((now - arrival) * 1000))),
            now_1024=int(now * 1024) & 0xFFFF,
            burst_loss=self.last_burst,
            loss=self.last_loss_pct,
            bwe_kbps=bwe,
        )

    def describe(self) -> str:
        e = self.estimator
        loss = e.loss_fraction
        rx = e.receive_kbps
        mode = "pinned" if self.pinned_bwe_kbps is not None else "adaptive"
        return (
            f"rctl bwe={self.last_bwe} ({mode}; est={e.bwe_kbps}) "
            f"loss%={'-' if loss is None else f'{loss * 100:.1f}'} "
            f"rx_kbps={'-' if rx is None else f'{rx:.0f}'} state={e.state} "
            f"backlog={self.last_backlog} decreases={e.decreases} sent={self.sent}"
        )


__all__ = [
    "BandwidthEstimator",
    "EstimatorConfig",
    "RCTL_INTERVAL_S",
    "RateControl",
]
