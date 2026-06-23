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


_libav_cache: dict[str, bool] = {}
_qsv_cache: dict[str, bool] = {}


def _qsv_hevc444() -> bool:
    """Raw probe: whether Intel Quick Sync (`hevc_qsv`) hardware-decodes HEVC
    4:4:4. Succeeds on Intel Gen11+/Xe where older generic d3d11va couldn't.
    Lazy import so the QSV module is only pulled in when probing."""
    try:
        from .qsvhevc import qsv_hevc444_available
        return qsv_hevc444_available()
    except Exception as e:
        log.info("hevc444 qsv probe: unavailable (%s)", e)
        return False


# The generic-libav and Intel-QSV HEVC 4:4:4 paths are probed INDEPENDENTLY
# (not first-wins) so the decoder registry can prefer the vendor-specific
# decoder when both work, per the configured priorities. Each is cached for
# the process lifetime (hardware capability can't change without new hardware)
# and honors the `ISS_HEVC444=0/1` escape hatch.

def libav_hevc444_hwdecode() -> bool:
    """Generic libav hwaccel path: d3d11va RExt / vaapi / cuda on Win/Linux, or
    native VideoToolbox on macOS. `ISS_HEVC444=0/1` forces off/on."""
    if "v" in _libav_cache:
        return _libav_cache["v"]
    override = os.environ.get("ISS_HEVC444")
    if override == "0":
        r = False
    elif sys.platform == "darwin":
        r = True                       # native VideoToolbox path (vtdecode.py)
    elif override == "1":
        r = True                       # forced on; trust the libav path
    else:
        hwaccels = _PROBE_HWACCELS.get(sys.platform, _PROBE_HWACCELS["*"])
        r = any(_probe_one(h) for h in hwaccels)
    _libav_cache["v"] = r
    return r


def qsv_hevc444_hwdecode() -> bool:
    """Intel Quick Sync (`hevc_qsv`) path. `ISS_HEVC444=0` forces off; macOS has
    no QSV."""
    if "v" in _qsv_cache:
        return _qsv_cache["v"]
    if os.environ.get("ISS_HEVC444") == "0" or sys.platform == "darwin":
        r = False
    else:
        r = _qsv_hevc444()
    _qsv_cache["v"] = r
    return r


def hevc444_decode_method() -> "str | None":
    """Back-compat summary of the independent probes: 'libav' (generic,
    reported first) / 'qsv' / None. The registry uses the probes directly so it
    can prefer QSV by priority; this helper just preserves the old API."""
    if libav_hevc444_hwdecode():
        return "libav"
    if qsv_hevc444_hwdecode():
        return "qsv"
    return None


def supports_hevc444_hwdecode() -> bool:
    """Whether ANY hardware path decodes HEVC 4:4:4 here. Drives codec
    negotiation (no HW 4:4:4 → AVC 4:2:0). `ISS_HEVC444=0/1` overrides the
    probe — an escape hatch for a misbehaving probe or for testing the fallback
    on capable hardware."""
    return libav_hevc444_hwdecode() or qsv_hevc444_hwdecode()


# NOTE: codec resolution (`--codec auto` -> hevc/avc) moved to
# media/registry.py::resolve_codec, which asks the decoder registry whether any
# 4:4:4 decoder is available rather than probing here directly. hwcaps remains
# the probe provider (the registry's availability callbacks delegate to it).

__all__ = ["supports_hevc444_hwdecode", "hevc444_decode_method",
           "libav_hevc444_hwdecode", "qsv_hevc444_hwdecode"]
