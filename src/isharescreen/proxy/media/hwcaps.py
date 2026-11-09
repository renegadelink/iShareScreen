"""Cached HEVC hardware decode probes for codec and chroma auto-selection.

Decode tiny packaged 4:4:4 and 4:2:0 IDRs through the production libav
hardware backends with software fallback disabled. Successful output
confirms the basic decode path, including NV12 frames downloaded by PyAV;
it does not establish throughput or support at every resolution.
"""
from __future__ import annotations

import logging
from importlib.resources import files
import os
import sys

log = logging.getLogger(__name__)

# Tiny self-contained IDRs; see samples/README.md for their purpose and format.
_HEVC444_SAMPLE = files(__package__).joinpath("samples/hevc444-idr.hevc").read_bytes()
_HEVC420_SAMPLE = files(__package__).joinpath("samples/hevc420-idr.hevc").read_bytes()

# Per-platform hwaccel candidates to probe, in priority order — mirrors the
# decoder's own list in hevc.py so the probe tests what production would use.
_PROBE_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "d3d12va"),
    "*": ("vaapi", "cuda"),
}

_cache: dict[str, bool] = {}


def _probe_one(hwaccel_type: str, *, chroma: str = "444") -> bool:
    """True iff `hwaccel_type` hardware-decodes the requested HEVC sample.

    Uses `allow_software_fallback=False` so libav will NOT silently decode in
    software when the GPU lacks the profile: a produced frame then means the
    GPU did it, while an unsupported profile yields no frame / an exception.
    (Inspecting `frame.format` is unreliable — VideoToolbox hands back a normal
    `nv24`/`nv12` frame even on a true hardware decode.)"""
    try:
        import av
        from av.codec.hwaccel import HWAccel

        try:
            hw = HWAccel(device_type=hwaccel_type, allow_software_fallback=False)
        except TypeError:
            # Older PyAV without the kwarg can't distinguish HW from a SW
            # fallback here; be conservative and report unsupported (→ AVC).
            log.info("hevc%s probe (%s): PyAV lacks allow_software_fallback; "
                     "assuming no HW support", chroma, hwaccel_type)
            return False
        ctx = av.CodecContext.create("hevc", "r", hwaccel=hw)
        sample = _HEVC420_SAMPLE if chroma == "420" else _HEVC444_SAMPLE
        frames = list(ctx.decode(av.Packet(sample)))
        frames += list(ctx.decode(None))  # flush
        ok = bool(frames)
        log.info("hevc%s probe (%s): %s%s", chroma, hwaccel_type,
                 "HW-decoded" if ok else "no HW frame",
                 (" (fmt=%s)" % frames[0].format.name) if frames else "")
        return ok
    except Exception as e:
        log.info("hevc%s probe (%s): unavailable (%s)", chroma, hwaccel_type, e)
        return False


def supports_hevc420_hwdecode() -> bool:
    """Cached real hardware decode probe; software fallback never counts."""
    if "420" not in _cache:
        hwaccels = _PROBE_HWACCELS.get(sys.platform, _PROBE_HWACCELS["*"])
        _cache["420"] = any(_probe_one(h, chroma="420") for h in hwaccels)
    return _cache["420"]


_method_cache: dict[str, str] = {}


def _qsv_hevc444() -> bool:
    """Whether Intel Quick Sync (`hevc_qsv`) hardware-decodes HEVC 4:4:4.
    On Intel Gen11+/Xe iGPUs this succeeds where the generic libav hwaccel
    can't (libav's d3d11va HEVC path lacks the 4:4:4 profile). Lazy import so
    the QSV decoder module is only pulled in when probing."""
    try:
        from .qsvhevc import qsv_hevc444_available
        return qsv_hevc444_available()
    except Exception as e:
        log.info("hevc444 qsv probe: unavailable (%s)", e)
        return False


def hevc444_decode_method() -> "str | None":
    """How HEVC 4:4:4 can be hardware-decoded here: ``"qsv"`` (Intel Quick
    Sync), ``"libav"`` (generic libav hwaccel / native VideoToolbox), or
    ``None`` (no HW 4:4:4 → AVC fallback). Cached for the process lifetime."""
    if "method" in _method_cache:
        return _method_cache["method"] or None
    override = os.environ.get("ISS_HEVC444")
    if override == "0":
        _method_cache["method"] = ""
        return None
    method = ""
    if sys.platform == "darwin":
        method = "libav"          # native VideoToolbox path (vtdecode.py)
    else:
        hwaccels = _PROBE_HWACCELS.get(sys.platform, _PROBE_HWACCELS["*"])
        if any(_probe_one(h) for h in hwaccels):
            method = "libav"
        elif _qsv_hevc444():
            method = "qsv"
    if override == "1" and not method:
        method = "libav"          # forced on; trust the libav path
    _method_cache["method"] = method
    return method or None


def supports_hevc444_hwdecode() -> bool:
    """Whether this platform's GPU can hardware-decode HEVC 4:4:4 by any path.
    Cached for the process lifetime (the answer can't change without new
    hardware).

    `ISS_HEVC444=0`/`1` disables / forces the 4:4:4 probe result —
    an escape hatch for a misbehaving probe and for testing the fallback path
    on 4:4:4-capable hardware."""
    return hevc444_decode_method() is not None


# NOTE: codec resolution (`--codec auto` -> hevc/avc) moved to
# media/registry.py::resolve_video_format. This module supplies hardware probes;
# the registry chooses the codec and chroma, and frontends apply that choice.

__all__ = ["supports_hevc444_hwdecode", "supports_hevc420_hwdecode", "hevc444_decode_method"]
