"""Requested HEVC chroma capabilities shared by negotiation and decoding."""
import os


def hevc_chroma() -> str:
    value = os.environ.get("ISS_HEVC_CHROMA", "444").strip()
    if value not in ("420", "444"):
        raise ValueError("ISS_HEVC_CHROMA must be 420 or 444")
    return value


def video_pixel_formats() -> int:
    # AVConference video settings field 8: bits 0/1 advertise 8-bit
    # full/video-range 4:2:0. Keep the original mask unless explicitly opted in.
    return 3 if hevc_chroma() == "420" else 63
