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
from .srtp import SRTPDecryptor


log = logging.getLogger(__name__)


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

            for nalu in reassemble_group([p for _, _, p in packets]):
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
    if vps_out is None or sps_out is None or not all_pps:
        if len(udp_video_buf) < 20:
            raise RuntimeError(
                f"HP_DECLINED_LIKELY_SW_ENCODE: only {len(udp_video_buf)} video packets in "
                f"{deadline_seconds:.1f}s burst. HP was negotiated at the TCP layer but no "
                "HEVC RTP arrived. Most common cause: the host lacks hardware VideoToolbox "
                "HEVC encode (virtualised macOS guests without AVE/M2Scaler paravirt) and "
                "screensharingd silently fell back to classic VNC/ZRLE which this client "
                "does not implement. Test against a physical Mac to confirm."
            )
        raise RuntimeError("initial burst missing VPS/SPS/PPS")

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
        vps=vps_out,
        sps=sps_out,
        all_pps=all_pps,
        last_idr=last_idr,
        tile_nalus=dict(tile_nalus),
        burst_pending=burst_pending,
    )


__all__ = ["InitialBurst", "gather_initial_burst"]
