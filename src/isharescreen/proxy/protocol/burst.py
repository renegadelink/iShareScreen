"""Initial-burst gathering for the session-start HEVC stream.

Right after the 0x1c answer arrives, the server emits a session-start burst
on UDP: a few Aggregation Packets carrying VPS/SPS/PPS, then ~50 fragmented
NAL packets forming the first IDR. This burst lands before our decoder is
running. The naturally-received burst is exactly what the decoder needs to
seed itself — we must NOT clear the UDP buffer, and we must NOT send an FBU
during this window (either would replace the real burst with a tiny
incremental IDR that's insufficient to start decode).
"""
from __future__ import annotations

import logging
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from ..media.bitstream import BitReader, remove_emulation_prevention
from ..media.nalu import IDR_RANGE, NAL_PPS, NAL_SPS, NAL_VPS, reassemble_group
from ..media.avc_nalu import (
    APPLE_CONFIG_MARKER, NAL_SLICE_IDR, parse_avc_config, reassemble_h264)
from .srtp import SRTPDecryptor


log = logging.getLogger(__name__)


class BurstStarved(RuntimeError):
    """The session-start burst arrived empty or without parameter sets.

    Retryable: TCP negotiation succeeded, so the caller can re-poke the host
    (re-send the stream-start FBU request / FIR-PLI) and re-arm the burst
    window before giving up. `packets` is how many UDP packets were seen;
    `reason` is a short tag (`no-video-rtp` = nothing arrived;
    `missing-param-sets` = packets arrived but no VPS/SPS/PPS)."""

    def __init__(self, packets: int, reason: str, deadline_seconds: float):
        self.packets = packets
        self.reason = reason
        self.deadline_seconds = deadline_seconds
        super().__init__(
            f"burst starved: {reason} ({packets} packets in "
            f"{deadline_seconds:.1f}s)"
        )


@dataclass
class InitialBurst:
    """Output of `gather_initial_burst`. The caller hands `tile_nalus` into
    the decoder's `feed_burst` and seeds its packet grouper with
    `burst_pending` so cross-tile ordering is preserved at the burst→stream
    handoff."""
    processed_pkt_idx: int
    ssrc_to_tile: dict[int, int]
    vps: bytes
    sps: bytes
    all_pps: dict[int, bytes]
    last_idr: dict[int, bytes]
    tile_nalus: dict[int, list[bytes]]
    burst_pending: dict[tuple[int, int], list[tuple[int, bool, bytes]]]
    codec: str = "hevc"


def gather_initial_burst(
    udp_video_buf: list[bytes],
    decryptor: SRTPDecryptor,
    *,
    quality_tier: int = 0,
    settle_seconds: float = 0.3,
    deadline_seconds: float = 2.0,
    min_packets: int = 100,
    codec: str = "hevc",
) -> InitialBurst:
    """Drain `udp_video_buf` into NAL units. Returns parameter sets, the per-
    tile NAL cache, and any incomplete groups for the streaming loop."""
    deadline = time.time() + deadline_seconds
    while len(udp_video_buf) < min_packets and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(settle_seconds)
    log.info("initial-burst packets: %d", len(udp_video_buf))

    vps_nalu: Optional[bytes] = None
    sps_nalu: Optional[bytes] = None
    all_pps: dict[int, bytes] = {}
    tile_nalus: defaultdict[int, list[bytes]] = defaultdict(list)
    last_idr: dict[int, bytes] = {}
    ssrc_to_tile: dict[int, int] = {}
    ssrc_ts_groups: defaultdict[tuple[int, int], list[tuple[int, bool, bytes]]] = defaultdict(list)
    completed: set[tuple[int, int]] = set()
    processed = 0

    def _refresh_ssrc_map() -> None:
        primary = decryptor.get_primary_ssrc_group(tier=quality_tier)
        if not primary:
            return
        sorted_ssrcs = sorted(primary)
        new_map = {s: i for i, s in enumerate(sorted_ssrcs)}
        if new_map != ssrc_to_tile:
            ssrc_to_tile.clear()
            ssrc_to_tile.update(new_map)
            log.info(
                "SSRC group (tier %d): %s",
                quality_tier, [f"0x{s:08x}" for s in sorted_ssrcs],
            )

    nonlocal_state = {"vps": vps_nalu, "sps": sps_nalu}

    def _scan_new() -> None:
        nonlocal processed
        snap = list(udp_video_buf)
        for idx in range(processed, len(snap)):
            pair = decryptor.decrypt(snap[idx])
            if not pair:
                continue
            hdr, payload = pair
            ssrc = struct.unpack(">I", hdr[8:12])[0]
            seq = struct.unpack(">H", hdr[2:4])[0]
            ts = struct.unpack(">I", hdr[4:8])[0]
            marker = bool(hdr[1] & 0x80)
            ssrc_ts_groups[(ssrc, ts)].append((seq, marker, payload))
        processed = len(snap)

        _refresh_ssrc_map()
        if not ssrc_to_tile:
            return

        for key in list(ssrc_ts_groups.keys()):
            if key in completed:
                continue
            grp = ssrc_ts_groups[key]
            if not any(m for _, m, _ in grp):
                continue
            ssrc, _ts = key
            ti = ssrc_to_tile.get(ssrc)
            if ti is None:
                completed.add(key)
                continue

            # Sort by seq with wraparound awareness.
            seqs = [s for s, _, _ in grp]
            if seqs and max(seqs) - min(seqs) > 0x8000:
                base = min(seqs)
                packets = sorted(grp, key=lambda x: (x[0] - base) & 0xFFFF)
            else:
                packets = sorted(grp, key=lambda x: x[0])

            for _s, _t, _p in packets:
                if _p:
                    nonlocal_state.setdefault("raw_hdr", set()).add(_p[0])
            if codec == "avc":
                payloads = [p for _, _, p in packets]
                # SPS/PPS arrive in Apple's 0x92 avc1/avcC config packet, not
                # as Annex-B param NALs.
                for _p in payloads:
                    if _p and _p[0] == APPLE_CONFIG_MARKER:
                        _cfg = parse_avc_config(_p)
                        if _cfg:
                            nonlocal_state["sps"] = _cfg[0]
                            all_pps[0] = _cfg[1]
                for nalu in reassemble_h264(payloads):
                    if not nalu:
                        continue
                    nonlocal_state.setdefault("hdr_bytes", set()).add(nalu[0])
                    t = nalu[0] & 0x1F
                    if t == NAL_SLICE_IDR:
                        last_idr[ti] = nalu
                        tile_nalus[ti] = [nalu]
                    elif 1 <= t <= 5:
                        tile_nalus[ti].append(nalu)
                completed.add(key)
                continue
            for nalu in reassemble_group([p for _, _, p in packets]):
                if len(nalu) < 2:
                    continue
                nt = (nalu[0] >> 1) & 0x3F
                nonlocal_state.setdefault("hdr_bytes", set()).add(nalu[0])
                if nt == NAL_VPS:
                    nonlocal_state["vps"] = nalu
                elif nt == NAL_SPS:
                    nonlocal_state["sps"] = nalu
                elif nt == NAL_PPS:
                    rbsp = remove_emulation_prevention(nalu[2:])
                    pps_id = BitReader(rbsp).read_ue()
                    all_pps[pps_id] = nalu
                elif nt in IDR_RANGE:
                    last_idr[ti] = nalu
                    tile_nalus[ti] = [nalu]
                else:
                    # Tiles whose IDR didn't land in the burst window
                    # still have their P-frames buffered: feeding them
                    # to the shared decoder is safe (libavcodec just
                    # discards them until a usable reference exists),
                    # and the session FIRs any tile missing from
                    # `last_idr` so a real IDR follows shortly. Dropping
                    # them outright instead caused 4 s gaps that woke
                    # the SSRC-adoption gate and triggered ping-pong.
                    tile_nalus[ti].append(nalu)
            completed.add(key)

    _scan_new()

    burst_pending = {
        key: list(grp) for key, grp in ssrc_ts_groups.items() if key not in completed
    }

    # CODEC-DETECT: infer codec from observed NALU first bytes.
    # HEVC: nal_unit_type=(b>>1)&0x3f in {VPS=32,SPS=33,PPS=34,AP=48,FU=49}.
    # H.264: forbidden=0 and nal_unit_type=b&0x1f in {1,5,7,8,24,28,29}.
    # H.264 data partition NALUs (types 2-4) byte-collide with HEVC FU/SPS/PPS
    # (e.g. type-2 byte 0x62 = HEVC FU type 49). When codec=avc and only
    # transport/FU bytes appear without HEVC VPS/SPS/PPS parameter-set bytes,
    # treat the stream as H.264 to avoid a false-positive HEVC classification.
    _raw = nonlocal_state.get("hdr_bytes", set()) | nonlocal_state.get("raw_hdr", set())
    _has_hevc_params = any(((b >> 1) & 0x3F) in (NAL_VPS, NAL_SPS, NAL_PPS) for b in _raw)
    # HEVC AP (type 48, byte 0x60/0x61) is unambiguous: H.264 uses STAP-A
    # (type 24, byte 0x78) instead. HEVC FU (type 49, byte 0x62/0x63) is
    # ambiguous — it collides with H.264 data partition type 2/3.
    _has_hevc_ap  = any(((b >> 1) & 0x3F) == 48 for b in _raw)
    _has_hevc_fu  = any(((b >> 1) & 0x3F) == 49 for b in _raw)
    _is_hevc = _has_hevc_params or _has_hevc_ap or _has_hevc_fu
    _is_h264 = any((b & 0x80) == 0 and (b & 0x1F) in (1, 5, 7, 8, 24, 28, 29) for b in _raw) and not _has_hevc_params
    # Tiebreaker for ambiguous FU bytes only: if codec=avc and no AP packets
    # (which would be conclusive HEVC evidence), treat as H.264 data partition.
    if _has_hevc_fu and not _has_hevc_params and not _has_hevc_ap and not _is_h264 and codec == "avc":
        _is_hevc = False
        _is_h264 = True
    log.info(
        "CODEC-DETECT: raw RTP headers=%s -> %s",
        sorted(hex(b) for b in _raw),
        "H.264/AVC" if _is_h264 else "HEVC" if _is_hevc else "unknown",
    )

    vps_out = nonlocal_state["vps"]
    sps_out = nonlocal_state["sps"]
    # H.264 has no VPS; only SPS+PPS are required.
    missing_params = (sps_out is None or not all_pps) if codec == "avc" else (
        vps_out is None or sps_out is None or not all_pps)
    if missing_params:
        reason = "no-video-rtp" if len(udp_video_buf) < 20 else "missing-param-sets"
        raise BurstStarved(len(udp_video_buf), reason, deadline_seconds)

    log.info("PPS pool: %d", len(all_pps))
    for ti in sorted(tile_nalus.keys()):
        nt_counts: defaultdict[int, int] = defaultdict(int)
        for nalu in tile_nalus[ti]:
            nt_counts[(nalu[0] >> 1) & 0x3F] += 1
        log.info("tile %d NALUs: %s", ti, dict(nt_counts))
    log.info("IDRs from burst: tiles %s", sorted(last_idr.keys()))

    return InitialBurst(
        processed_pkt_idx=processed,
        ssrc_to_tile=ssrc_to_tile,
        vps=vps_out or b"",
        sps=sps_out,
        all_pps=all_pps,
        last_idr=last_idr,
        tile_nalus=dict(tile_nalus),
        burst_pending=burst_pending,
        codec=codec,
    )


__all__ = ["BurstStarved", "InitialBurst", "gather_initial_burst"]
