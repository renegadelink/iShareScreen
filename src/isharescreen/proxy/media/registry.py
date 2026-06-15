"""Pluggable video-decoder registry: declarative capability specs + automatic
best-fit selection + manual override.

This replaces the hand-written `_use_qsv`/`_use_vt`/else branching that used to
live in `session.py` and the codec-resolution probe in `hwcaps.py`. Each
`DecoderSpec` declares *what* a decoder handles (codec, chroma, platform,
hardware/software, priority) plus *how* to probe availability and build it. The
selector filters to the negotiated codec, sorts hardware-before-software then by
priority, and returns the first spec whose `available()` probe passes. Two knobs
sit on top:

  * codec negotiation — `resolve_codec("auto")` offers HEVC when any 4:4:4
    decoder is available here, else AVC 4:2:0 (`can_decode`).
  * manual override — `ISS_DECODER=<name>` / `--decoder <name>` forces a spec;
    `--list-decoders` prints the live matrix (run it on an Intel box to see
    whether `libav-hevc444` covers 4:4:4 via the generic DXVA-RExt path, making
    `qsv-hevc444` redundant — see the windows-hevc444 notes).

The decoder CLASSES are untouched — this is selection plumbing only. Specs are
centralized here with *lazy* `build`/`available` callbacks (the decoder class +
`hwcaps` probes are imported on use), so this module has no import-time
dependency on the decoders and there are no import cycles.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecoderSpec:
    name: str                       # stable id, e.g. "vt-hevc444"
    codec: str                      # "hevc" | "avc"
    chroma: str                     # "444" | "420"
    platforms: tuple[str, ...]      # sys.platform values, or "*" for any
    kind: str                       # "hardware" | "software"
    priority: int                   # higher = preferred among same codec
    available: Callable[[], bool]   # runtime probe (cheap; hwcaps caches)
    build: Callable[..., object]    # factory(num_tiles, **opts) -> decoder
    note: str = ""

    def supported_here(self) -> bool:
        return "*" in self.platforms or sys.platform in self.platforms


# ── availability probes (delegate to hwcaps; lazy import) ────────────────

def _hevc444_method() -> Optional[str]:
    """'qsv' | 'libav' | None — how HEVC 4:4:4 can be HW-decoded here."""
    from .hwcaps import hevc444_decode_method
    return hevc444_decode_method()


def _vt_available() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        from . import vtdecode
        return vtdecode.available()
    except Exception as e:  # pragma: no cover
        log.debug("vt availability probe failed: %s", e)
        return False


# ── build factories (lazy import of the decoder class) ───────────────────
# Each accepts the common kwargs and ignores the rest via **_, so the
# registry can call every spec uniformly even though VT/QSV decoders don't
# take `prefer_hwaccel`.

def _build_vt(num_tiles, *, enable_quality_gate=True, on_frame_published=None, **_):
    from .vtdecode import VTHevcDecoder
    return VTHevcDecoder(num_tiles=num_tiles, enable_quality_gate=enable_quality_gate,
                         on_frame_published=on_frame_published)


def _build_qsv(num_tiles, *, enable_quality_gate=True, on_frame_published=None, **_):
    from .qsvhevc import QsvHevcDecoder
    return QsvHevcDecoder(num_tiles=num_tiles, enable_quality_gate=enable_quality_gate,
                          on_frame_published=on_frame_published)


def _build_libav_hevc(num_tiles, *, enable_quality_gate=True, on_frame_published=None,
                      prefer_hwaccel=True, **_):
    from .hevc import HevcDecoder
    return HevcDecoder(num_tiles=num_tiles, enable_quality_gate=enable_quality_gate,
                       on_frame_published=on_frame_published, prefer_hwaccel=prefer_hwaccel)


def _build_avc(num_tiles, *, enable_quality_gate=True, on_frame_published=None,
               prefer_hwaccel=True, **_):
    from .avc import AvcDecoder
    return AvcDecoder(num_tiles=num_tiles, enable_quality_gate=enable_quality_gate,
                      on_frame_published=on_frame_published, prefer_hwaccel=prefer_hwaccel)


# ── the registry ─────────────────────────────────────────────────────────
# Priorities encode the current selection order:
#   HEVC 4:4:4: VT (macOS, 100) > libav generic (60) > Intel QSV (50).
#   libav-hevc444 and qsv-hevc444 are mutually exclusive in practice
#   (`_hevc444_method()` returns exactly one of 'libav'/'qsv'/None), so the
#   gap just encodes "prefer the generic path when both could apply".
#   AVC 4:2:0: one adaptive libav decoder (always available; H.264 4:2:0 HW-
#   decodes on every platform, with an internal SW fallback + AMD workaround).
_REGISTRY: list[DecoderSpec] = [
    DecoderSpec("vt-hevc444", "hevc", "444", ("darwin",), "hardware", 100,
                available=lambda: _vt_available(), build=_build_vt,
                note="direct VideoToolbox (VTDecompressionSession, no libav RPS layer)"),
    DecoderSpec("libav-hevc444", "hevc", "444", ("*",), "hardware", 60,
                available=lambda: _hevc444_method() == "libav", build=_build_libav_hevc,
                note="libav generic hwaccel — d3d11va RExt / vaapi / cuda (VT-hwaccel fallback on mac)"),
    DecoderSpec("qsv-hevc444", "hevc", "444", ("win32", "linux"), "hardware", 50,
                available=lambda: _hevc444_method() == "qsv", build=_build_qsv,
                note="Intel Quick Sync (hevc_qsv) — fallback where generic DXVA lacks 4:4:4"),
    DecoderSpec("libav-avc420", "avc", "420", ("*",), "hardware", 50,
                available=lambda: True, build=_build_avc,
                note="libav H.264 + platform hwaccel, SW fallback (incl. AMD VCN POC-wrap workaround)"),
]


# ── override aliasing (back-compat with the old ISS_DECODER values) ──────
_ALIASES = {
    "vt": "vt-hevc444",
    "videotoolbox": "vt-hevc444",
    "qsv": "qsv-hevc444",
    # "libav" is codec-specific (HEVC vs AVC) → resolved in normalize_override
}


def normalize_override(raw: Optional[str], codec: str) -> Optional[str]:
    """Map a user override (--decoder / ISS_DECODER) to a spec name. Accepts
    new spec names directly and the legacy short aliases; 'auto'/empty → None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    if raw in ("auto", ""):
        return None
    if raw == "libav":
        return "libav-hevc444" if codec == "hevc" else "libav-avc420"
    return _ALIASES.get(raw, raw)


# ── queries + selection ──────────────────────────────────────────────────

def all_specs() -> list[DecoderSpec]:
    return list(_REGISTRY)


def candidates(codec: str, chroma: Optional[str] = None) -> list[DecoderSpec]:
    """Specs for `codec` (optionally `chroma`) supported on this platform,
    best-first (hardware before software, then priority)."""
    out = [s for s in _REGISTRY
           if s.codec == codec and s.supported_here()
           and (chroma is None or s.chroma == chroma)]
    out.sort(key=lambda s: (s.kind == "hardware", s.priority), reverse=True)
    return out


def can_decode(codec: str, chroma: str) -> bool:
    """True if any available decoder here handles (codec, chroma). Drives codec
    negotiation."""
    return any(s.available() for s in candidates(codec, chroma))


def select(codec: str, *, override: Optional[str] = None) -> Optional[DecoderSpec]:
    """Best available spec for `codec`, or `override` by name when supported.
    Returns None if nothing is available (caller falls back)."""
    if override:
        for s in _REGISTRY:
            if s.name == override:
                if not s.supported_here():
                    log.warning("decoder override %r not supported on %s — using auto",
                                override, sys.platform)
                    break
                return s
        else:
            log.warning("unknown decoder override %r — using auto", override)
    for s in candidates(codec):
        if s.available():
            return s
    return None


def build_best(codec: str, *, override: Optional[str], num_tiles: int, **opts):
    """Select + construct a decoder for `codec`, falling through to the next
    available candidate if construction fails (preserves the old HW→libav
    fallback). Returns (spec, decoder) or (None, None)."""
    chosen = select(codec, override=override)
    ordered = ([chosen] if chosen else [])
    ordered += [s for s in candidates(codec) if s is not chosen and s.available()]
    for spec in ordered:
        try:
            dec = spec.build(num_tiles, **opts)
            log.info("decoder: %s — %s", spec.name, spec.note)
            return spec, dec
        except Exception as e:
            log.warning("decoder %s failed to start (%s) — trying next candidate",
                        spec.name, e)
    return None, None


# ── codec negotiation (moved here from hwcaps to break the import cycle) ──

def resolve_codec(choice: str) -> str:
    """Resolve a `--codec` value to a concrete 'hevc' / 'avc'. 'auto' offers
    HEVC when any 4:4:4 decoder is available here, else H.264 4:2:0."""
    if choice != "auto":
        return choice
    if can_decode("hevc", "444"):
        log.info("codec=auto: a HEVC 4:4:4 decoder is available -> hevc")
        return "hevc"
    log.info("codec=auto: no HEVC 4:4:4 decoder -> avc (H.264 4:2:0)")
    return "avc"


def describe() -> str:
    """Human-readable matrix with live availability — for --list-decoders."""
    rows = ["decoder         codec chroma  platform   kind      prio  available  note",
            "-" * 100]
    for s in _REGISTRY:
        avail = "  -  "
        if s.supported_here():
            try:
                avail = " yes " if s.available() else " no  "
            except Exception as e:  # pragma: no cover
                avail = f"err:{e}"[:5]
        plats = ",".join(s.platforms)
        rows.append(f"{s.name:<15} {s.codec:<5} {s.chroma:<6}  {plats:<10} "
                    f"{s.kind:<9} {s.priority:<4}  {avail:<9}  {s.note}")
    return "\n".join(rows)


__all__ = ["DecoderSpec", "all_specs", "candidates", "can_decode", "select",
           "build_best", "normalize_override", "resolve_codec", "describe"]
