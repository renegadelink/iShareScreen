"""Unit tests for the pluggable decoder registry (media/registry.py).

These exercise the selection LOGIC without a GPU: platform filtering, the
hardware-then-priority ordering, override aliasing, and codec resolution. The
availability probes are monkeypatched at their module-global seams
(`_hevc444_method`, `_vt_available`) — the lambdas in the specs resolve those
names at call time, so patching the module attribute steers selection.
"""
from __future__ import annotations

from isharescreen.proxy.media import registry as R


def test_normalize_override_aliases():
    assert R.normalize_override("auto", "hevc") is None
    assert R.normalize_override("", "hevc") is None
    assert R.normalize_override(None, "hevc") is None
    assert R.normalize_override("vt", "hevc") == "vt-hevc444"
    assert R.normalize_override("videotoolbox", "hevc") == "vt-hevc444"
    assert R.normalize_override("qsv", "hevc") == "qsv-hevc444"
    # "libav" is codec-specific
    assert R.normalize_override("libav", "hevc") == "libav-hevc444"
    assert R.normalize_override("libav", "avc") == "libav-avc420"
    # already a spec name → passthrough
    assert R.normalize_override("qsv-hevc444", "hevc") == "qsv-hevc444"


def test_candidates_platform_filtered_and_sorted(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    names = [s.name for s in R.candidates("hevc")]
    assert "vt-hevc444" not in names                    # darwin-only filtered
    assert names == ["libav-hevc444", "qsv-hevc444", "libav-hevc444-sw"]  # hw prio 60>50, then sw

    monkeypatch.setattr(R.sys, "platform", "darwin")
    names = [s.name for s in R.candidates("hevc")]
    assert names[0] == "vt-hevc444"                     # prio 100 first
    assert "qsv-hevc444" not in names                   # win/linux only


def test_windows_prefers_generic_then_qsv(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")

    monkeypatch.setattr(R, "_hevc444_method", lambda: "libav")
    assert R.select("hevc").name == "libav-hevc444"     # generic wins when it works
    assert R.can_decode("hevc", "444") is True
    assert R.resolve_codec("auto") == "hevc"

    monkeypatch.setattr(R, "_hevc444_method", lambda: "qsv")
    assert R.select("hevc").name == "qsv-hevc444"        # QSV is the fallback

    monkeypatch.setattr(R, "_hevc444_method", lambda: None)
    # libav-hevc444-sw (software) is always available, so select() never returns None
    assert R.select("hevc").name == "libav-hevc444-sw"
    assert R.can_decode("hevc", "444") is True           # SW decoder available
    assert R.can_decode("hevc", "444", hardware_only=True) is False
    assert R.resolve_codec("auto") == "avc"              # auto ignores SW → falls back to AVC


def test_macos_prefers_vt_then_libav(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R, "_hevc444_method", lambda: "libav")

    monkeypatch.setattr(R, "_vt_available", lambda: True)
    assert R.select("hevc").name == "vt-hevc444"         # prio 100 over libav 60

    monkeypatch.setattr(R, "_vt_available", lambda: False)
    assert R.select("hevc").name == "libav-hevc444"      # VT down → libav fallback


def test_override_honored_unsupported_or_unknown_falls_to_auto(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setattr(R, "_hevc444_method", lambda: "libav")
    assert R.select("hevc", override="qsv-hevc444").name == "qsv-hevc444"
    # vt-hevc444 is darwin-only → on win32 it's unsupported → auto (libav)
    assert R.select("hevc", override="vt-hevc444").name == "libav-hevc444"
    # unknown name → auto
    assert R.select("hevc", override="nope").name == "libav-hevc444"


def test_avc_always_available_everywhere(monkeypatch):
    for plat in ("darwin", "win32", "linux"):
        monkeypatch.setattr(R.sys, "platform", plat)
        assert R.can_decode("avc", "420") is True
        assert R.select("avc").name == "libav-avc420"
        assert R.resolve_codec("avc") == "avc"           # explicit passthrough
