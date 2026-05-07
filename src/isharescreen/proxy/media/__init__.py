"""HEVC + AAC-ELD decoders, NAL reassembly, frame types.

Modules:

    tiles         `TileFrame` — decoder→consumer protocol type
    hevc          shared-context HEVC decoder (HW + SW fallback)
    aac_eld       AAC-ELD decoder (multiple backend implementations)
    nalu          NAL-unit reassembly + helpers (IDR_RANGE, etc.)
    quality_gate  per-tile gray/black/decode-error gating with FIR escalation
"""
