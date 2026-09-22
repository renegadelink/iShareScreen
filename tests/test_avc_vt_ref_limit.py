"""macOS VideoToolbox H.264 reference-frame limit."""
from __future__ import annotations

from isharescreen.proxy.media import avc
from isharescreen.proxy.media.avc import AvcDecoder

# Apple SPS captured live: 3072x1920 declares 8 refs, 2608x1480 declares 12.
SPS_8_REFS = bytes.fromhex("27640034ac13142403000f1a6e181a0204")
SPS_12_REFS = bytes.fromhex("27640033ac131434028c0bbf966e181a0204")
PPS = bytes.fromhex("28ee3cb0")


def _max_num_ref_frames(sps: bytes) -> int:
    """Read max_num_ref_frames from a High-profile, POC type 0 SPS."""
    bits = [(b >> i) & 1 for b in sps[1:] for i in range(7, -1, -1)]
    pos = 24  # profile_idc, constraint flags, level_idc

    def ue() -> int:
        nonlocal pos
        zeros = 0
        while bits[pos] == 0:
            zeros += 1
            pos += 1
        pos += 1
        val = 0
        for _ in range(zeros):
            val = (val << 1) | bits[pos]
            pos += 1
        return (1 << zeros) - 1 + val

    ue()                 # seq_parameter_set_id
    ue(); ue(); ue()     # chroma_format_idc, bit_depth_luma/chroma
    pos += 2             # qpprime_y_zero_transform_bypass, scaling_matrix_present
    ue()                 # log2_max_frame_num_minus4
    assert ue() == 0     # pic_order_cnt_type
    ue()                 # log2_max_pic_order_cnt_lsb_minus4
    return ue()


def test_macos_raises_small_ref_count_only_to_10(monkeypatch):
    monkeypatch.setattr(avc.sys, "platform", "darwin")

    patched = avc._patch_avc_sps_dpb(SPS_8_REFS)

    assert _max_num_ref_frames(patched) == 10
    assert patched[3] == 60


def test_macos_keeps_16_when_apple_declares_more(monkeypatch):
    monkeypatch.setattr(avc.sys, "platform", "darwin")

    assert _max_num_ref_frames(avc._patch_avc_sps_dpb(SPS_12_REFS)) == 16


def test_other_platforms_still_use_16(monkeypatch):
    monkeypatch.setattr(avc.sys, "platform", "linux")

    assert _max_num_ref_frames(avc._patch_avc_sps_dpb(SPS_8_REFS)) == 16


def test_vt_refusal_uses_software_without_disabling_hardware(monkeypatch):
    monkeypatch.setattr(avc.sys, "platform", "darwin")
    decoder = AvcDecoder(1)
    decoder._sps = SPS_12_REFS
    decoder._pps = PPS
    attempted: list[str] = []
    software, hardware = object(), object()
    monkeypatch.setattr(avc, "_vt_accepts_h264", lambda _sps, _pps: False)
    monkeypatch.setattr(
        decoder,
        "_try_hwaccel_locked",
        lambda name, _extra: attempted.append(name) or hardware,
    )
    monkeypatch.setattr(decoder, "_make_sw_context", lambda _extra: software)

    with decoder._codec_lock:
        codec = decoder._ensure_codec_locked()

    assert codec is software
    assert attempted == []
    assert decoder._hw_failed is False

    # A later resize to a canvas VideoToolbox accepts gets hardware again.
    monkeypatch.setattr(avc, "_vt_accepts_h264", lambda _sps, _pps: True)
    decoder._sps = SPS_8_REFS
    decoder._codec = None

    with decoder._codec_lock:
        codec = decoder._ensure_codec_locked()

    assert codec is hardware
    assert attempted == ["videotoolbox"]
    assert decoder.hw_accel == "videotoolbox"
