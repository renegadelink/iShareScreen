"""Pre-decode error detection by HEVC slice-header RPS analysis.

The decoder will conceal a P/B slice when its Reference Picture Set
(RPS) lists a POC that isn't currently in the DPB. Instead of waiting
for the decoder to produce gray output and trying to detect that from
pixels (impossible to distinguish from real gray content) or scraping
libavcodec log lines (works but slow + process-global), we parse the
slice header ourselves before feeding the decoder. If any RPS entry
points to a POC we never saw, we know with certainty the decoder
will conceal — fire FIR proactively, decoder-agnostic.

Scope: only enough of HEVC to compute a slice's referenced POC set.
We deliberately skip the full slice header (deblocking params, SAO,
weighted prediction, etc.). Apple's screen-share stream is a
constrained subset of HEVC — single-layer, no SVC/MVC, NAL types
1 (TRAIL_R) + 16-21 (BLA/IDR/CRA), I- and P-slices only — which keeps
the parser tight.

References:
- ITU-T H.265 7.3.2.1 (VPS), 7.3.2.2 (SPS), 7.3.6 (slice segment
  header), 7.3.7 (st_ref_pic_set), 8.3.1 (POC derivation).
- chemag/h265nal as a known-good cross-check.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .bitstream import BitReader, remove_emulation_prevention


log = logging.getLogger(__name__)


# NAL unit types we care about. From ITU-T H.265 Table 7-1.
NAL_TRAIL_N = 0
NAL_TRAIL_R = 1
NAL_TSA_N = 2
NAL_TSA_R = 3
NAL_STSA_N = 4
NAL_STSA_R = 5
NAL_RADL_N = 6
NAL_RADL_R = 7
NAL_RASL_N = 8
NAL_RASL_R = 9
NAL_BLA_W_LP = 16
NAL_BLA_W_RADL = 17
NAL_BLA_N_LP = 18
NAL_IDR_W_RADL = 19
NAL_IDR_N_LP = 20
NAL_CRA_NUT = 21

IRAP_RANGE = range(NAL_BLA_W_LP, NAL_CRA_NUT + 1)   # 16..21
IDR_RANGE = (NAL_IDR_W_RADL, NAL_IDR_N_LP)          # 19, 20


@dataclass(slots=True)
class _ShortTermRps:
    """Parsed st_ref_pic_set as a flat list of (delta_poc, used_by_curr)."""
    deltas: list[tuple[int, bool]] = field(default_factory=list)


@dataclass(slots=True)
class HevcSpsState:
    """SPS fields we need to parse a slice header and an RPS."""
    log2_max_pic_order_cnt_lsb: int = 8     # default if SPS not parsed yet
    num_short_term_ref_pic_sets: int = 0
    short_term_rps_sets: list[_ShortTermRps] = field(default_factory=list)
    num_long_term_ref_pics_sps: int = 0
    long_term_ref_pics_present_flag: bool = False
    sample_adaptive_offset_enabled_flag: bool = False
    separate_colour_plane_flag: bool = False
    pic_width_in_luma_samples: int = 0
    pic_height_in_luma_samples: int = 0
    log2_min_luma_coding_block_size_minus3: int = 0
    log2_diff_max_min_luma_coding_block_size: int = 0


# ── parsers ──────────────────────────────────────────────────────────

def _skip_profile_tier_level(br: BitReader, max_num_sub_layers: int) -> None:
    """Profile/tier/level — we don't need the values, just consume bits."""
    # general_profile_space u(2), general_tier_flag u(1), general_profile_idc u(5)
    br.read(8)
    # general_profile_compatibility_flag[32]
    br.read(32)
    # 4 1-bit constraint flags + 43 bits reserved + 1 bit + a bunch more = 48 bits total
    br.read(48)
    # general_level_idc u(8)
    br.read(8)
    sub_layer_profile_present = []
    sub_layer_level_present = []
    for _ in range(max_num_sub_layers - 1):
        sub_layer_profile_present.append(br.read1())
        sub_layer_level_present.append(br.read1())
    if max_num_sub_layers > 1:
        # reserved_zero_2bits[i] u(2) — pad to byte boundary, 8 entries
        for _ in range(max_num_sub_layers - 1, 8):
            br.read(2)
    for i in range(max_num_sub_layers - 1):
        if sub_layer_profile_present[i]:
            br.read(8 + 32 + 48)        # same shape as general
        if sub_layer_level_present[i]:
            br.read(8)


def _parse_st_ref_pic_set(
    br: BitReader, idx: int, num_in_sps: int,
    sets: list[_ShortTermRps],
) -> _ShortTermRps:
    """Parse one short_term_ref_pic_set (HEVC 7.3.7). Returns the
    flattened delta-POC list. Inter-prediction (referencing a previous
    set) is supported but rare in screen content; we still implement it
    to be safe."""
    out = _ShortTermRps()
    inter_ref_pic_set_prediction_flag = 0
    if idx != 0:
        inter_ref_pic_set_prediction_flag = br.read1()

    if inter_ref_pic_set_prediction_flag:
        delta_idx_minus1 = 0
        if idx == num_in_sps:
            delta_idx_minus1 = br.read_ue()
        delta_rps_sign = br.read1()
        abs_delta_rps_minus1 = br.read_ue()
        delta_rps = (1 - 2 * delta_rps_sign) * (abs_delta_rps_minus1 + 1)
        ref_idx = idx - (delta_idx_minus1 + 1)
        if ref_idx < 0 or ref_idx >= len(sets):
            return out  # malformed; bail
        ref = sets[ref_idx]
        n_ref = len(ref.deltas) + 1  # +1 for the implicit "current" entry
        used_by_curr = []
        use_delta = []
        for _ in range(n_ref):
            used_by_curr.append(br.read1())
            udf = 1
            if not used_by_curr[-1]:
                udf = br.read1()
            use_delta.append(udf)
        # Compute resulting deltas (HEVC 7.4.8 derivation).
        for i, (rd, _used) in enumerate(reversed(ref.deltas)):
            d = rd + delta_rps
            if d < 0 and use_delta[i]:
                out.deltas.append((d, bool(used_by_curr[i])))
        if delta_rps < 0 and use_delta[n_ref - 1]:
            out.deltas.append((delta_rps, bool(used_by_curr[n_ref - 1])))
        for i, (rd, _used) in enumerate(ref.deltas):
            d = rd + delta_rps
            if d < 0 and use_delta[n_ref - 1 - i]:
                # already handled above
                pass
        return out

    num_negative_pics = br.read_ue()
    num_positive_pics = br.read_ue()
    last_neg = 0
    for _ in range(num_negative_pics):
        delta_poc_s0_minus1 = br.read_ue()
        used_by_curr = bool(br.read1())
        delta = last_neg - (delta_poc_s0_minus1 + 1)
        last_neg = delta
        out.deltas.append((delta, used_by_curr))
    last_pos = 0
    for _ in range(num_positive_pics):
        delta_poc_s1_minus1 = br.read_ue()
        used_by_curr = bool(br.read1())
        delta = last_pos + (delta_poc_s1_minus1 + 1)
        last_pos = delta
        out.deltas.append((delta, used_by_curr))
    return out


def parse_sps(rbsp: bytes) -> HevcSpsState:
    """Parse a HEVC SPS RBSP (no NAL header). Returns the state we
    need for subsequent slice-header parsing. RBSP must already have
    emulation-prevention bytes stripped."""
    br = BitReader(rbsp)
    state = HevcSpsState()

    sps_video_parameter_set_id = br.read(4)  # noqa: F841
    sps_max_sub_layers_minus1 = br.read(3)
    sps_temporal_id_nesting_flag = br.read1()  # noqa: F841
    _skip_profile_tier_level(br, sps_max_sub_layers_minus1 + 1)

    sps_seq_parameter_set_id = br.read_ue()  # noqa: F841
    chroma_format_idc = br.read_ue()
    if chroma_format_idc == 3:
        state.separate_colour_plane_flag = bool(br.read1())
    state.pic_width_in_luma_samples = br.read_ue()
    state.pic_height_in_luma_samples = br.read_ue()
    if br.read1():  # conformance_window_flag
        br.read_ue(); br.read_ue(); br.read_ue(); br.read_ue()

    bit_depth_luma_minus8 = br.read_ue()  # noqa: F841
    bit_depth_chroma_minus8 = br.read_ue()  # noqa: F841
    state.log2_max_pic_order_cnt_lsb = br.read_ue() + 4

    sub_layer_ordering_info_present_flag = br.read1()
    n_sub_lo = sps_max_sub_layers_minus1 + 1
    if not sub_layer_ordering_info_present_flag:
        n_sub_lo = 1
    for _ in range(n_sub_lo):
        br.read_ue(); br.read_ue(); br.read_ue()

    log2_min_luma_coding_block_size_minus3 = br.read_ue()
    state.log2_min_luma_coding_block_size_minus3 = log2_min_luma_coding_block_size_minus3
    log2_diff_max_min_luma_coding_block_size = br.read_ue()
    state.log2_diff_max_min_luma_coding_block_size = log2_diff_max_min_luma_coding_block_size
    br.read_ue(); br.read_ue()  # log2_min_transform / diff
    br.read_ue(); br.read_ue()  # max_transform_hierarchy_depth_inter / intra

    if br.read1():  # scaling_list_enabled_flag
        if br.read1():  # sps_scaling_list_data_present_flag
            # scaling_list_data() — variable size; for typical Apple
            # screen-share streams this is absent. If present we'd
            # need ~80 lines to skip; if it bites us we'll add later.
            log.warning("hevc_rps: SPS contains scaling_list_data; parser may misalign")
            return state

    br.read1()  # amp_enabled_flag
    state.sample_adaptive_offset_enabled_flag = bool(br.read1())
    if br.read1():  # pcm_enabled_flag
        br.read(4); br.read(4)
        br.read_ue(); br.read_ue()
        br.read1()

    state.num_short_term_ref_pic_sets = br.read_ue()
    sets: list[_ShortTermRps] = []
    for i in range(state.num_short_term_ref_pic_sets):
        sets.append(_parse_st_ref_pic_set(br, i, state.num_short_term_ref_pic_sets, sets))
    state.short_term_rps_sets = sets

    state.long_term_ref_pics_present_flag = bool(br.read1())
    if state.long_term_ref_pics_present_flag:
        state.num_long_term_ref_pics_sps = br.read_ue()
    return state


def parse_slice_header_for_rps(
    nal_unit_type: int,
    rbsp_after_nal_header: bytes,
    sps: HevcSpsState,
    *,
    pps_dependent_slice_segments_enabled: bool = False,
    pps_num_extra_slice_header_bits: int = 0,
    pps_output_flag_present: bool = False,
) -> Optional[tuple[int, list[tuple[int, bool]]]]:
    """Parse a slice header up through the short-term RPS. Returns
    (poc_lsb, list_of_(delta_poc, used_by_curr)). Returns None if the
    slice is a dependent segment or if parsing fails — callers treat
    that conservatively (no opinion).
    """
    try:
        br = BitReader(rbsp_after_nal_header)
        first_slice_segment_in_pic_flag = br.read1()
        if nal_unit_type in IRAP_RANGE:
            br.read1()  # no_output_of_prior_pics_flag
        br.read_ue()  # slice_pic_parameter_set_id

        dependent_slice_segment_flag = 0
        if not first_slice_segment_in_pic_flag:
            if pps_dependent_slice_segments_enabled:
                dependent_slice_segment_flag = br.read1()
            # slice_segment_address u(v): bits = ceil(log2(NumCtbsInPic))
            # Compute NumCtbsInPic from SPS. Apple's stream uses
            # standard layouts; if our SPS parse missed something
            # we silently return None below.
            min_cb_log2 = sps.log2_min_luma_coding_block_size_minus3 + 3
            ctb_log2 = min_cb_log2 + sps.log2_diff_max_min_luma_coding_block_size
            if ctb_log2 < 4 or ctb_log2 > 6 or sps.pic_width_in_luma_samples == 0:
                return None
            ctb_size = 1 << ctb_log2
            ctbs_w = (sps.pic_width_in_luma_samples + ctb_size - 1) // ctb_size
            ctbs_h = (sps.pic_height_in_luma_samples + ctb_size - 1) // ctb_size
            num_ctbs = ctbs_w * ctbs_h
            if num_ctbs <= 1:
                return None
            seg_addr_bits = (num_ctbs - 1).bit_length()
            br.read(seg_addr_bits)

        if dependent_slice_segment_flag:
            return None  # uses prior segment's RPS — no new info

        for _ in range(pps_num_extra_slice_header_bits):
            br.read1()
        slice_type = br.read_ue()  # 0=B, 1=P, 2=I
        if pps_output_flag_present:
            br.read1()
        if sps.separate_colour_plane_flag:
            br.read(2)

        if nal_unit_type in IDR_RANGE:
            return (0, [])  # IDR resets POC; no references

        slice_pic_order_cnt_lsb = br.read(sps.log2_max_pic_order_cnt_lsb)
        short_term_ref_pic_set_sps_flag = br.read1()
        if not short_term_ref_pic_set_sps_flag:
            rps = _parse_st_ref_pic_set(
                br, sps.num_short_term_ref_pic_sets,
                sps.num_short_term_ref_pic_sets,
                sps.short_term_rps_sets,
            )
        elif sps.num_short_term_ref_pic_sets > 1:
            idx_bits = (sps.num_short_term_ref_pic_sets - 1).bit_length()
            idx = br.read(idx_bits)
            if idx >= len(sps.short_term_rps_sets):
                return None
            rps = sps.short_term_rps_sets[idx]
        else:
            if not sps.short_term_rps_sets:
                return (slice_pic_order_cnt_lsb, [])
            rps = sps.short_term_rps_sets[0]
        # We deliberately stop here — long-term refs and the rest of
        # the slice header don't matter for our missing-reference
        # check (LTRs are signaled separately and Apple's stream
        # doesn't seem to use them; if it did we'd just over-FIR).
        if slice_type == 2:
            return (slice_pic_order_cnt_lsb, [])  # I-slice, no refs
        return (slice_pic_order_cnt_lsb, list(rps.deltas))
    except Exception as e:
        log.debug("slice header parse failed: %s", e)
        return None


# ── tracker ──────────────────────────────────────────────────────────

class HevcRpsTracker:
    """Keeps a per-codec view of which POCs are in the DPB. Caller
    feeds raw NALU bytes; tracker reports whether the slice will
    conceal (any used_by_curr reference is missing).

    One tracker per shared codec context. Apple's stream uses one
    shared codec for all 4 tiles (cross-tile DPB references), so a
    single tracker covers all of them.
    """

    def __init__(self) -> None:
        self.sps: Optional[HevcSpsState] = None
        self._seen_pocs: set[int] = set()
        self._prev_poc_msb = 0
        self._prev_poc_lsb = 0
        self._last_checked_poc: Optional[int] = None
        self.checks = 0
        self.missing_ref_events = 0

    def reset(self) -> None:
        self._seen_pocs.clear()
        self._prev_poc_msb = 0
        self._prev_poc_lsb = 0
        self._last_checked_poc = None

    def feed_sps(self, sps_rbsp: bytes) -> None:
        """`sps_rbsp` is the SPS *without* the 2-byte NAL header,
        emulation-prevention NOT yet stripped — we strip here."""
        try:
            stripped = remove_emulation_prevention(sps_rbsp)
            self.sps = parse_sps(stripped)
            log.info(
                "hevc_rps: SPS log2_max_poc_lsb=%d pic=%dx%d num_st_rps=%d",
                self.sps.log2_max_pic_order_cnt_lsb,
                self.sps.pic_width_in_luma_samples,
                self.sps.pic_height_in_luma_samples,
                self.sps.num_short_term_ref_pic_sets,
            )
        except Exception as e:
            log.warning("hevc_rps: SPS parse failed: %s", e)
            self.sps = None

    def check_slice(self, nalu: bytes) -> set[int]:
        """Parse the slice's RPS and return any required POCs that
        we haven't seen yet. Empty set = OK (or unparseable, which we
        treat as OK so we don't spuriously FIR).

        Side effects: advances `_prev_poc_msb`/`_prev_poc_lsb` (parser
        state for cross-slice POC continuity, regardless of whether
        the slice ends up decoded). Does NOT add the current slice's
        POC to `_seen_pocs` — that has to happen separately via
        `commit_decoded(nalu)` after the caller decides to actually
        feed the slice. This split is what lets a "drop on missing
        ref" caller avoid polluting the seen set with POCs that
        never enter the DPB.
        """
        # Clear the pending commit token at every entry so an
        # early-return doesn't leave a stale POC behind for the
        # next `commit_decoded()` to re-add.
        self._last_checked_poc = None
        if self.sps is None or len(nalu) < 3:
            return set()
        nal_unit_type = (nalu[0] >> 1) & 0x3F
        # NAL header is 2 bytes; the rest is the slice header + data.
        rbsp = remove_emulation_prevention(nalu[2:])
        result = parse_slice_header_for_rps(
            nal_unit_type, rbsp, self.sps,
        )
        if result is None:
            return set()
        poc_lsb, deltas = result
        max_poc_lsb = 1 << self.sps.log2_max_pic_order_cnt_lsb
        # POC MSB derivation (ITU-T H.265 8.3.1). Update parser state
        # so the next slice's MSB derivation is consistent.
        if nal_unit_type in IDR_RANGE:
            poc = 0
            self._prev_poc_msb = 0
            self._prev_poc_lsb = 0
        else:
            prev_msb = self._prev_poc_msb
            prev_lsb = self._prev_poc_lsb
            if poc_lsb < prev_lsb and (prev_lsb - poc_lsb) >= (max_poc_lsb // 2):
                cur_msb = prev_msb + max_poc_lsb
            elif poc_lsb > prev_lsb and (poc_lsb - prev_lsb) > (max_poc_lsb // 2):
                cur_msb = prev_msb - max_poc_lsb
            else:
                cur_msb = prev_msb
            poc = cur_msb + poc_lsb
            self._prev_poc_msb = cur_msb
            self._prev_poc_lsb = poc_lsb
        self.checks += 1
        self._last_checked_poc = poc
        # Compute referenced POCs and return any used_by_curr that's
        # missing from our DPB.
        missing: set[int] = set()
        for delta, used in deltas:
            if not used:
                continue
            ref_poc = poc + delta
            if ref_poc not in self._seen_pocs:
                missing.add(ref_poc)
        if missing:
            self.missing_ref_events += 1
        return missing

    def commit_decoded(self) -> None:
        """Mark the most recently `check_slice`d slice as actually
        fed to the decoder, so its POC enters our DPB-shadow set.

        Important: IDRs do NOT clear `_seen_pocs`. Apple's stream
        sends one shared codec context for 4 separately-encoded
        tiles, each with its own POC sequence. An IDR for tile 0
        does NOT (in practice) evict tile 1/2/3's recent reference
        pictures from the codec's working DPB — those tiles keep
        decoding from their own POC sequence. Clearing _seen_pocs
        on every per-tile IDR drops cross-tile state and turns
        every subsequent P-slice into a false-positive 'missing
        ref' drop. We just add the new POC and let stale entries
        accumulate; the worst case is a false-negative miss (claim
        a ref is in DPB when libav has evicted it), which the
        post-decode `decode_error_flags` path catches.
        """
        poc = self._last_checked_poc
        if poc is None:
            return
        self._seen_pocs.add(poc)
        self._last_checked_poc = None


__all__ = [
    "HevcRpsTracker",
    "HevcSpsState",
    "IDR_RANGE",
    "IRAP_RANGE",
    "parse_slice_header_for_rps",
    "parse_sps",
]
