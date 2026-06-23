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
from ..media.nalu import (
    IDR_RANGE, NAL_PPS, NAL_SPS, NAL_VPS, reassemble_group,
    H264_NAL_IDR, H264_NAL_SPS, H264_NAL_PPS,
    is_avc_config, parse_avc_config, reassemble_group_h264,
)
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
    tile NAL cache, and any incomplete groups for the streaming loop.

    `codec` selects the RTP depay + parameter-set harvest: `"hevc"` (Apple
    RFC-7798 + DONL, VPS/SPS/PPS) or `"avc"`/`"h264"` (RFC-6184 + DON, with
    SPS/PPS delivered up front in an Apple `avc1`/`avcC` config wrapper)."""
    is_h264 = codec in ("avc", "h264")
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
            if is_h264 and is_avc_config(payload):
                # Apple delivers H.264 SPS/PPS up front in an avc1/avcC config
                # wrapper, not as Annex-B NALs in a timestamp group. Harvest it
                # here, independent of group/marker completion.
                sps_c, pps_list = parse_avc_config(payload)
                log.info("AVC config packet: %dB -> SPS=%s PPS=%d",
                         len(payload), "yes" if sps_c else "no", len(pps_list))
                if sps_c:
                    nonlocal_state["sps"] = sps_c
                for _j, _pps in enumerate(pps_list):
                    all_pps[_j] = _pps
                continue
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

            grp_payloads = [p for _, _, p in packets]
            if is_h264:
                # H.264: 1-byte NAL header, IDR = type 5. SPS/PPS already
                # harvested from the avcC wrapper above, so slices are all we
                # classify here.
                for nalu in reassemble_group_h264(grp_payloads):
                    if not nalu:
                        continue
                    nt = nalu[0] & 0x1F
                    if nt == H264_NAL_IDR:
                        last_idr[ti] = nalu
                        tile_nalus[ti] = [nalu]
                    elif nt in (H264_NAL_SPS, H264_NAL_PPS):
                        continue   # params come via avcC, not in-band
                    else:
                        tile_nalus[ti].append(nalu)
                completed.add(key)
                continue
            for nalu in reassemble_group(grp_payloads):
                if len(nalu) < 2:
                    continue
                nt = (nalu[0] >> 1) & 0x3F
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

    vps_out = nonlocal_state["vps"]
    sps_out = nonlocal_state["sps"]
    # H.264 has no VPS; only SPS+PPS are required.
    params_ok = (
        (sps_out is not None and all_pps) if is_h264
        else (vps_out is not None and sps_out is not None and all_pps)
    )
    if not params_ok:
        reason = "no-video-rtp" if len(udp_video_buf) < 20 else "missing-param-sets"
        raise BurstStarved(len(udp_video_buf), reason, deadline_seconds)
    if is_h264 and vps_out is None:
        vps_out = b""   # interface parity; the H.264 decoder ignores it

    log.info("PPS pool: %d", len(all_pps))
    for ti in sorted(tile_nalus.keys()):
        nt_counts: defaultdict[int, int] = defaultdict(int)
        for nalu in tile_nalus[ti]:
            nt_counts[(nalu[0] & 0x1F) if is_h264 else ((nalu[0] >> 1) & 0x3F)] += 1
        log.info("tile %d NALUs: %s", ti, dict(nt_counts))
    log.info("IDRs from burst: tiles %s", sorted(last_idr.keys()))

    return InitialBurst(
        processed_pkt_idx=processed,
        ssrc_to_tile=ssrc_to_tile,
        vps=vps_out,
        sps=sps_out,
        all_pps=all_pps,
        last_idr=last_idr,
        tile_nalus=dict(tile_nalus),
        burst_pending=burst_pending,
    )


__all__ = ["BurstStarved", "InitialBurst", "gather_initial_burst"]
