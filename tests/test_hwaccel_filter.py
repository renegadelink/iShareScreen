"""Unit tests for the hwaccel candidate filter and the is_hwaccel reader
(media/decode_common.py). No GPU: `hwdevices_available` is monkeypatched at
its PyAV module seam, which `available_hwaccels` imports at call time."""
from __future__ import annotations

import av.codec.hwaccel as hwaccel_mod

from isharescreen.proxy.media.decode_common import available_hwaccels, context_is_hwaccel


def test_keeps_only_supported_types_in_candidate_order(monkeypatch):
    monkeypatch.setattr(hwaccel_mod, "hwdevices_available", lambda: ["drm", "cuda"], raising=False)
    assert available_hwaccels(("cuda", "vaapi", "drm")) == ("cuda", "drm")


def test_build_reporting_none_leaves_no_candidate(monkeypatch):
    monkeypatch.setattr(hwaccel_mod, "hwdevices_available", lambda: [], raising=False)
    assert available_hwaccels(("cuda", "vaapi")) == ()


def test_enumeration_error_keeps_candidates(monkeypatch):
    def boom():
        raise RuntimeError("no enumeration")

    monkeypatch.setattr(hwaccel_mod, "hwdevices_available", boom, raising=False)
    assert available_hwaccels(("d3d11va", "dxva2")) == ("d3d11va", "dxva2")


def test_pyav_without_enumeration_keeps_candidates(monkeypatch):
    monkeypatch.delattr(hwaccel_mod, "hwdevices_available", raising=False)
    assert available_hwaccels(("videotoolbox",)) == ("videotoolbox",)


class _Ctx:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def test_context_is_hwaccel_reads_bool_or_unknown():
    assert context_is_hwaccel(_Ctx(is_hwaccel=True)) is True
    assert context_is_hwaccel(_Ctx(is_hwaccel=False)) is False
    assert context_is_hwaccel(_Ctx()) is None
    assert context_is_hwaccel(_Ctx(is_hwaccel=1)) is None
