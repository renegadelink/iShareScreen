"""RTCP APP "RCTL" rate-control feedback: wire layout, estimator, sender."""
from __future__ import annotations

import queue
import struct
import threading
from unittest.mock import Mock

from isharescreen.proxy.protocol.rtcp import build_rtcp_app_rctl
from isharescreen.proxy.protocol.srtp import SRTCPDecryptor, SRTCPEncryptor
from isharescreen.proxy.rate_control import (
    BandwidthEstimator,
    EstimatorConfig,
    RateControl,
)
from isharescreen.proxy.session import Session, SessionConfig


TICK = 0.05


# ── wire format ──────────────────────────────────────────────────────

def test_rctl_packet_layout():
    pkt = build_rtcp_app_rctl(
        0x11223344, echo=0xABCD1234, echo_delay_ms=7, now_1024=0x12345,
        owrd=0, burst_loss=3, loss=5, bwe_kbps=50000,
    )
    assert len(pkt) == 32
    assert pkt.hex() == (
        "80cc0007" "11223344" "5243544c"
        "85000004" "1234" "00000000" "0007" "2345" "0000" "3005" "c350"
    )


def test_rctl_fields_are_clamped():
    pkt = build_rtcp_app_rctl(
        1, echo=0, echo_delay_ms=10**6, now_1024=0, burst_loss=99, loss=5000,
        bwe_kbps=100_000,
    )
    delay, _, _, loss_word, bwe = struct.unpack(">HHHHH", pkt[22:32])
    assert delay == 0xFFFF
    assert loss_word >> 12 == 15
    assert loss_word & 0xFFF == 0xFFF
    assert bwe == 0xFFFF


# ── estimator ────────────────────────────────────────────────────────

def _run(est, seconds, *, start, kbps, loss=0.0, pps=2000):
    """Feed `seconds` of ticks at `kbps` with a `loss` fraction."""
    t = start
    n = int(round(seconds / TICK))
    for _ in range(n):
        t += TICK
        expected = int(pps * TICK)
        received = int(round(expected * (1 - loss)))
        est.update(t, int(kbps * 1000 / 8 * TICK), expected, received)
    return t


def test_estimator_starts_at_50_mbps_and_climbs_to_ceiling_when_filled():
    est = BandwidthEstimator()
    assert est.bwe_kbps == 50_000
    t = 0.0
    # Host fills the estimate: climb (6 %/s) until the 65 Mbps ceiling.
    for _ in range(12):
        t = _run(est, 1.0, start=t, kbps=est.estimate_kbps)
    assert est.bwe_kbps == 65_000
    assert est.state == "increase"


def test_estimator_holds_while_screen_is_idle():
    est = BandwidthEstimator()
    t = _run(est, 10.0, start=0.0, kbps=2_000, pps=200)
    assert est.bwe_kbps == 50_000
    assert est.state == "hold-app"
    # Static screen: too few packets to judge loss also holds.
    _run(est, 2.0, start=t, kbps=10, pps=2)
    assert est.bwe_kbps == 50_000


def test_single_lossy_second_never_cuts():
    est = BandwidthEstimator()
    t = _run(est, 3.0, start=0.0, kbps=50_000)
    before = est.estimate_kbps
    t = _run(est, 1.0, start=t, kbps=50_000, loss=0.3)
    t = _run(est, 3.0, start=t, kbps=50_000)
    assert est.decreases == 0
    assert est.estimate_kbps >= before


def test_sustained_loss_cuts():
    est = BandwidthEstimator()
    t = _run(est, 3.0, start=0.0, kbps=50_000)
    _run(est, 3.0, start=t, kbps=50_000, loss=0.2)
    assert est.decreases >= 1
    assert est.estimate_kbps < 50_000
    # Never below the per-step floor (0.6x) or the 4 Mbps floor.
    assert est.estimate_kbps >= 4_000


def test_cut_follows_receive_rate_when_saturated():
    est = BandwidthEstimator()
    t = _run(est, 3.0, start=0.0, kbps=40_000)
    _run(est, 2.1, start=t, kbps=40_000, loss=0.2)
    # 0.9 x the ~40 Mbps receive rate rather than 0.9 x 50 Mbps.
    assert est.decreases == 1
    assert 34_000 < est.estimate_kbps < 37_000


def test_moderate_loss_holds():
    est = BandwidthEstimator()
    t = _run(est, 3.0, start=0.0, kbps=50_000)
    t = _run(est, 1.0, start=t, kbps=50_000, loss=0.05)   # window fills
    before = est.estimate_kbps
    _run(est, 5.0, start=t, kbps=50_000, loss=0.05)
    assert est.decreases == 0
    assert est.estimate_kbps == before
    assert est.state == "hold"


def test_growing_receive_backlog_cuts_toward_processed_rate():
    # The client processes ~35 Mbps while the host sends more: its receive
    # queue grows. The estimate must come down to what it can process.
    est = BandwidthEstimator()
    t = 0.0
    backlog = 0
    for _ in range(int(6 / TICK)):
        t += TICK
        backlog += 60 if est.estimate_kbps > 36_000 else -60
        backlog = max(backlog, 0)
        est.update(t, int(35_000 * 1000 / 8 * TICK), 150, 150, backlog)
    assert est.decreases >= 1
    assert 30_000 <= est.estimate_kbps <= 38_000


def test_backlog_never_cuts_below_processed_rate():
    # A CPU-bound client: the backlog stays high whatever the host sends.
    est = BandwidthEstimator()
    t = 0.0
    for i in range(int(20 / TICK)):
        t += TICK
        est.update(t, int(28_000 * 1000 / 8 * TICK), 120, 120, 2000 + i)
    assert est.estimate_kbps >= 28_000 * 0.9 * 0.9


def test_backlog_blocks_increase_but_draining_backlog_does_not_cut():
    est = BandwidthEstimator()
    t = _run(est, 2.0, start=0.0, kbps=50_000)
    for i in range(20):                       # backlog builds: cuts
        t += TICK
        est.update(t, int(50_000 * 1000 / 8 * TICK), 100, 100, 250 * i)
    assert est.decreases >= 1
    backlog = 5000
    for _ in range(int(1.0 / TICK)):         # it peaks and stays put
        t += TICK
        est.update(t, int(50_000 * 1000 / 8 * TICK), 100, 100, backlog)
    cuts, before = est.decreases, est.estimate_kbps
    # A large but draining backlog: hold, neither cut nor increase.
    for _ in range(40):
        t += TICK
        backlog -= 100
        est.update(t, int(50_000 * 1000 / 8 * TICK), 100, 100, backlog)
    assert est.decreases == cuts
    assert est.estimate_kbps == before
    assert est.state == "hold-backlog"


def test_floor_and_ceiling_config():
    est = BandwidthEstimator(EstimatorConfig(initial_kbps=1_000, ceiling_kbps=70_000))
    assert est.bwe_kbps == 4_000          # floor
    est = BandwidthEstimator(EstimatorConfig(initial_kbps=90_000, ceiling_kbps=90_000))
    assert est.bwe_kbps == 0xFFFF         # u16 on the wire


# ── per-tick loss accounting ─────────────────────────────────────────

def _decode(pkt):
    echo = struct.unpack(">H", pkt[16:18])[0]
    delay = struct.unpack(">H", pkt[22:24])[0]
    loss_word, bwe = struct.unpack(">HH", pkt[28:32])
    return echo, delay, loss_word >> 12, loss_word & 0xFFF, bwe


def test_no_packet_before_video():
    rc = RateControl()
    assert rc.tick(1, 0.0) is None


def test_tick_reports_echo_delay_and_loss_run():
    rc = RateControl()
    t = 10.0
    for i in range(100):
        if 40 <= i < 43:            # one 3-packet hole
            continue
        rc.note_packet(0xA, i, 0x10000 + i, 1200, t)
    pkt = rc.tick(0x55667788, t + 0.012)
    assert struct.unpack(">I", pkt[4:8])[0] == 0x55667788
    echo, delay, burst, loss_pct, bwe = _decode(pkt)
    assert echo == (0x10000 + 99) & 0xFFFF
    assert delay == 12
    assert burst == 3
    assert loss_pct == 3
    assert bwe == 50_000
    # Counters drain per tick.
    rc.note_packet(0xA, 100, 0x10064, 1200, t + 0.05)
    _, _, burst, _, _ = _decode(rc.tick(1, t + 0.06))
    assert burst == 0


def test_late_packets_cancel_loss_and_seq_wrap():
    rc = RateControl()
    seqs = [65530, 65531, 65533, 65534, 65535, 0, 1, 65532, 2, 3]
    for s in seqs:
        rc.note_packet(7, s, 0, 100, 0.0)
    for _ in range(3):
        rc.note_packet(7, 4, 0, 100, 0.0)   # duplicates count as received
    e = rc.estimator
    rc.tick(1, 0.05)
    # 10 expected (65530..3), 13 received: no loss reported.
    assert e.loss_fraction is None or e.loss_fraction == 0.0


def test_pinned_bwe():
    rc = RateControl(pinned_bwe_kbps=30_000)
    rc.note_packet(1, 0, 0, 100, 0.0)
    assert _decode(rc.tick(1, 0.05))[4] == 30_000


# ── session integration ──────────────────────────────────────────────

def test_session_sends_protected_rctl_on_video_port():
    key = bytes(range(46))
    s = Session.__new__(Session)
    s._config = SessionConfig(host="192.0.2.19", username="u", password="p")
    s._dest_host = "192.0.2.19"
    s._sock_video = Mock()
    s._srtcp_enc = SRTCPEncryptor.from_blob(key)
    s._our_video_ssrc = 0x55667788
    s._stop_evt = threading.Event()
    s._video_q = queue.Queue()
    s._rctl = RateControl()
    s._rctl.note_packet(0xA, 1, 0x1234, 1200, 0.0)

    def stop_after_send(*_a):
        s._stop_evt.set()
    s._sock_video.sendto.side_effect = stop_after_send
    s._rctl_loop()

    wire, dest = s._sock_video.sendto.call_args.args
    assert dest == ("192.0.2.19", 5901)
    plain = SRTCPDecryptor.from_blob(key).unprotect(wire)
    assert plain[:12] == bytes.fromhex("80cc0007 55667788 5243544c".replace(" ", ""))
    assert len(plain) == 32


def test_rctl_env_toggle(monkeypatch):
    monkeypatch.delenv("ISS_RCTL", raising=False)
    monkeypatch.delenv("ISS_RCTL_BWE", raising=False)
    s = Session(SessionConfig(host="192.0.2.19", username="u", password="p"))
    assert s._rctl_enabled is False       # opt-in
    assert s._rctl_pinned_bwe is None
    monkeypatch.setenv("ISS_RCTL", "1")
    monkeypatch.setenv("ISS_RCTL_BWE", "20000")
    s = Session(SessionConfig(host="192.0.2.19", username="u", password="p"))
    assert s._rctl_enabled is True
    assert s._rctl_pinned_bwe == 20_000
