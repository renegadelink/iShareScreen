"""GPU capability probe for HEVC 4:4:4 hardware decode + codec auto-selection.

Apple's screen-share default video codec is HEVC RExt 4:4:4, which only recent
GPUs hardware-decode (AMD RDNA3+, Intel Arc / 12th-gen+, NVIDIA Ada+); on the
rest, libav silently falls back to CPU decode, which is far too slow for a live
4-tile stream. H.264 4:2:0 (the AVC codec bank, see offers.py) hardware-decodes
on virtually every GPU (D3D11VA on Windows, VAAPI on Linux, VideoToolbox on
macOS). `--codec auto` (the default) resolves to HEVC 4:4:4 when the GPU can
hardware-decode it and to H.264 4:2:0 otherwise.

Detection is a *real* hardware decode of a tiny embedded HEVC 4:4:4 IDR through
the same libav hwaccel path production uses. If the decoded frame comes back in
a hardware pixel format the GPU supports it; if libav fell back to software
(frame format `yuv444p`) or the hwaccel failed to initialise, it does not. This
tests the actual decode path rather than trusting a static profile table, so it
can't be fooled by a profile GUID that's advertised but broken.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)

# A 64x64 HEVC Main 4:4:4 8-bit IDR access unit (VPS + SPS + PPS + IDR slice,
# Annex-B), produced by libx265 from a yuv444p frame. Self-contained: it
# decodes with a fresh decoder context, no external parameter sets. ~97 bytes.
_HEVC444_SAMPLE = bytes.fromhex(
    "0000000140010c01ffff0408000003009e280000030000baba0240000000014201"
    "010408000003009e280000030000ba90041020b2dd25261734040000030004003d"
    "090020000000014401c070306011200000012801ade0d117ffd39173238b80"
)

# Per-platform hwaccel candidates to probe, in priority order — mirrors the
# decoder's own list in hevc.py so the probe tests what production would use.
_PROBE_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "d3d12va"),
    "*": ("vaapi", "cuda"),
}

_cache: dict[str, bool] = {}


def _probe_one(hwaccel_type: str) -> bool:
    """True iff `hwaccel_type` hardware-decodes the embedded HEVC 4:4:4 sample.

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
            log.info("hevc444 probe (%s): PyAV lacks allow_software_fallback; "
                     "assuming no HW 4:4:4", hwaccel_type)
            return False
        ctx = av.CodecContext.create("hevc", "r", hwaccel=hw)
        frames = list(ctx.decode(av.Packet(_HEVC444_SAMPLE)))
        frames += list(ctx.decode(None))  # flush
        ok = bool(frames)
        log.info("hevc444 probe (%s): %s%s", hwaccel_type,
                 "HW-decoded" if ok else "no HW frame",
                 (" (fmt=%s)" % frames[0].format.name) if frames else "")
        return ok
    except Exception as e:
        log.info("hevc444 probe (%s): unavailable (%s)", hwaccel_type, e)
        return False


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

    `ISS_HEVC444=0`/`1` overrides the probe (force AVC fallback / force HEVC) —
    an escape hatch for a misbehaving probe and for testing the fallback path
    on 4:4:4-capable hardware."""
    return hevc444_decode_method() is not None


# NOTE: codec resolution (`--codec auto` -> hevc/avc) moved to
# media/registry.py::resolve_codec, which asks the decoder registry whether any
# 4:4:4 decoder is available rather than probing here directly. hwcaps remains
# the probe provider (the registry's availability callbacks delegate to it).

__all__ = ["supports_hevc444_hwdecode", "hevc444_decode_method"]
