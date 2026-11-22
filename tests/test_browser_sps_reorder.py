"""The SPS sent to browsers says the stream never reorders frames."""
from __future__ import annotations

from isharescreen.frontend.wt import server
from isharescreen.proxy.media.avc import _sps_signal_no_reorder

HIGH = bytes([0x67, 100, 0x00, 52, 0xAC, 0x2B])


def test_high_profile_gets_constraint_set3():
    assert _sps_signal_no_reorder(HIGH) == bytes([0x67, 100, 0x10, 52, 0xAC, 0x2B])


def test_other_profiles_and_nalus_untouched():
    hi422 = bytes([0x67, 122, 0x00, 52, 0xAC])      # set3 means "intra only" there
    assert _sps_signal_no_reorder(hi422) == hi422
    pps = bytes([0x68, 0xEE, 0x3C, 0x80])
    assert _sps_signal_no_reorder(pps) == pps
    assert _sps_signal_no_reorder(b"") == b""


def test_browser_sps_and_codec_string_carry_it(monkeypatch):
    monkeypatch.setattr(server, "_BROWSER_SPS_PATCH", False)
    assert server._browser_sps(HIGH)[2] & 0x10
    monkeypatch.setattr(server, "_BROWSER_NO_REORDER", False)
    assert server._browser_sps(HIGH) == HIGH
