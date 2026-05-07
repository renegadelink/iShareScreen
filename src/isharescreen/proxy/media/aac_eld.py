"""AAC-ELD-SBR 48 kHz stereo decoder for Apple's PT=101 audio.

Two backends, picked by platform:

  - **AudioToolbox** (`_AudioToolboxBackend`) — macOS, via direct ctypes.
    Ships with the OS, lowest latency, zero extra install.
  - **libfdk-aac** (`_LibFdkBackend`) — Linux and Windows, via direct
    ctypes against the system shared library. One package install:
    `apt install libfdk-aac2` (Debian/Ubuntu),
    `dnf install fdk-aac` (Fedora), or
    `libfdk-aac-2.dll` on Windows.

We deliberately don't fall back to PyAV. libavcodec 62 has **no native
AAC-ELD decoder** (`aac` / `aac_fixed` / `aac_latm` all return
`PatchWelcomeError` when AOT-39 extradata is supplied), and standard
PyAV PyPI wheels don't ship `libfdk_aac` for licensing reasons. The only
PyAV codec that handles AAC-ELD-SBR is `aac_at`, which is just
AudioToolbox in a wrapper — strictly worse than calling AudioToolbox
ourselves.

Apple's PT=101 wire format is one raw AAC-ELD-SBR access unit per RTP
packet. No RFC 3640 AU-headers, no RED framing. Silence AUs are 4
bytes of zeros. 480 frames per packet, which under SBR maps to 20 ms
of perceived audio.

Both backends return `(N, 2)` float32 stereo at 48 kHz, or `None` on
decode failure / first-call priming where the decoder has nothing to
emit yet.
"""
from __future__ import annotations

import ctypes
import logging
import os
import struct
import sys
from collections import deque
from typing import Optional, Protocol

import numpy as np


log = logging.getLogger(__name__)


# ── codec spec constants ──────────────────────────────────────────────

OUTPUT_SAMPLE_RATE = 48_000
OUTPUT_CHANNELS = 2
SAMPLES_PER_FRAME = 480              # AAC-ELD frame size at 48 kHz / 20 ms

# AudioSpecificConfig per ISO/IEC 14496-3 §1.6.2.1.
# Bit layout of the 48-bit value 0xf8e65132e000:
#   audioObjectType        5 bits = 0b11111  (=31, "escape")
#   audioObjectTypeExt     6 bits = 0b000111 (+32 → AOT 39 = AAC-ELD)
#   samplingFrequencyIndex 4 bits = 0b0011   (=3, 48 kHz)
#   channelConfiguration   4 bits = 0b0010   (=2, stereo)
#   GASpecificConfig:
#     frameLengthFlag        1 bit = 0b1     (480-sample short frame)
#     dependsOnCoreCoder     1 bit = 0b0
#     extensionFlag          1 bit = 0b0
#   ELDSpecificConfig:
#     ldSbrPresentFlag       1 bit = 0b1     (SBR enabled)
#     ldSbrSamplingRate      1 bit = 0b0
#     ldSbrCrcFlag           1 bit = 0b0
#     eldExtType             4 bits = 0b0000 (ELDEXT_TERM)
#   trailing zero pad to byte boundary
AUDIO_SPECIFIC_CONFIG = bytes.fromhex("f8e65132e000")

# AudioToolbox `kAudioFormatProperty_MagicCookie` value Apple's
# screensharingd emits — the full 43-byte MPEG-4 ES_Descriptor wrapping
# the 6-byte AudioSpecificConfig above. Extracted via VCAudioPayload
# `getMagicCookie:withLength:`. Static across sessions; depends only on
# (codecType, sampleRate, channels) which are fixed for PT=101.
MAGIC_COOKIE = bytes.fromhex(
    "03808080260000000480808018401400180000013880000000000580808006"
    "f8e65132e000068080800102"
)


# ── public protocol ──────────────────────────────────────────────────

class AacEldDecoder(Protocol):
    """Common shape of every backend. Decoders are stateful — each
    `decode()` call may consume one input AU and return zero or more
    output frames as a stereo float32 ndarray."""

    def decode(self, payload: bytes) -> Optional[np.ndarray]:
        """Decode one RTP payload. Returns `(N, 2)` float32 stereo or
        `None` if the decoder has no output yet (priming) or the AU was
        unusable. N ≤ SAMPLES_PER_FRAME."""
        ...

    def close(self) -> None:
        """Release native resources. Safe to call multiple times."""
        ...


# ── small helpers ────────────────────────────────────────────────────

def _fourcc(s: str) -> int:
    """ASCII FourCharCode → big-endian uint32."""
    return struct.unpack(">I", s.encode("ascii"))[0]


# ── backend 1: AudioToolbox (macOS native) ────────────────────────────

class _ASBD(ctypes.Structure):
    """`AudioStreamBasicDescription` — CoreAudio format descriptor."""
    _fields_ = [
        ("mSampleRate", ctypes.c_double),
        ("mFormatID", ctypes.c_uint32),
        ("mFormatFlags", ctypes.c_uint32),
        ("mBytesPerPacket", ctypes.c_uint32),
        ("mFramesPerPacket", ctypes.c_uint32),
        ("mBytesPerFrame", ctypes.c_uint32),
        ("mChannelsPerFrame", ctypes.c_uint32),
        ("mBitsPerChannel", ctypes.c_uint32),
        ("mReserved", ctypes.c_uint32),
    ]


class _AudioBuffer(ctypes.Structure):
    _fields_ = [
        ("mNumberChannels", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
        ("mData", ctypes.c_void_p),
    ]


class _AudioBufferList1(ctypes.Structure):
    _fields_ = [
        ("mNumberBuffers", ctypes.c_uint32),
        ("mBuffers", _AudioBuffer * 1),
    ]


class _StreamPacketDesc(ctypes.Structure):
    _fields_ = [
        ("mStartOffset", ctypes.c_int64),
        ("mVariableFramesInPacket", ctypes.c_uint32),
        ("mDataByteSize", ctypes.c_uint32),
    ]


# AudioToolbox callback type. AudioConverter calls this to pull more input.
_FillCB = ctypes.CFUNCTYPE(
    ctypes.c_int32,                                   # OSStatus return
    ctypes.c_void_p,                                  # AudioConverterRef
    ctypes.POINTER(ctypes.c_uint32),                  # ioNumberDataPackets (in/out)
    ctypes.POINTER(_AudioBufferList1),                # ioData
    ctypes.POINTER(ctypes.POINTER(_StreamPacketDesc)),# outDataPacketDescription
    ctypes.c_void_p,                                  # inUserData
)


# Custom non-zero, non-EOS status returned from our callback when the
# input queue has drained. AudioToolbox treats 0 as EOS (which would tear
# the converter down for streaming use); any other value is treated as
# "the data source can't provide more right now, return what you have".
_CB_STATUS_QUEUE_EMPTY = 1

_LPCM_FLAGS_F32_PACKED = 0x09  # kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked


class _AudioToolboxBackend:
    """AudioToolbox AudioConverter driving the kAudioFormatMPEG4AAC_ELD_SBR
    decoder. Output goes through a packet-pull callback that drains a
    short FIFO of input AUs."""

    def __init__(self) -> None:
        self._at = ctypes.CDLL(
            "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
        )
        self._configure_signatures()
        self._converter = ctypes.c_void_p()
        self._make_converter()
        self._set_magic_cookie()

        self._input_queue: deque[bytes] = deque()
        self._input_buf = ctypes.create_string_buffer(4096)
        self._output_buf = ctypes.create_string_buffer(SAMPLES_PER_FRAME * OUTPUT_CHANNELS * 4)
        self._packet_desc = _StreamPacketDesc()
        self._packet_desc_ptr = ctypes.pointer(self._packet_desc)

        # Hold a strong ref to the trampoline so it doesn't get GC'd while
        # AudioToolbox holds a function pointer to it.
        self._cb_trampoline = _FillCB(self._fill_callback)

    # -- public API ----------------------------------------------------

    def decode(self, payload: bytes) -> Optional[np.ndarray]:
        if not payload:
            return None
        self._input_queue.append(bytes(payload))

        abl = _AudioBufferList1()
        abl.mNumberBuffers = 1
        abl.mBuffers[0].mNumberChannels = OUTPUT_CHANNELS
        abl.mBuffers[0].mDataByteSize = len(self._output_buf)
        abl.mBuffers[0].mData = ctypes.cast(self._output_buf, ctypes.c_void_p).value

        n_packets = ctypes.c_uint32(SAMPLES_PER_FRAME)
        status = self._at.AudioConverterFillComplexBuffer(
            self._converter, self._cb_trampoline, None,
            ctypes.byref(n_packets), ctypes.byref(abl), None,
        )
        # Our callback's queue-empty sentinel propagates as the function's
        # status. Treat both 0 (real success) and the sentinel as success.
        if (status not in (0, _CB_STATUS_QUEUE_EMPTY)) or n_packets.value == 0:
            return None
        raw = ctypes.string_at(self._output_buf, abl.mBuffers[0].mDataByteSize)
        # Copy off the output buffer so downstream filters can mutate.
        return np.frombuffer(raw, dtype=np.float32).reshape(-1, OUTPUT_CHANNELS).copy()

    def close(self) -> None:
        if self._converter and self._converter.value:
            self._at.AudioConverterDispose(self._converter)
            self._converter = ctypes.c_void_p()

    # -- setup helpers -------------------------------------------------

    def _configure_signatures(self) -> None:
        c, OS = ctypes.c_void_p, ctypes.c_int32
        self._at.AudioConverterNew.restype = OS
        self._at.AudioConverterNew.argtypes = [
            ctypes.POINTER(_ASBD), ctypes.POINTER(_ASBD), ctypes.POINTER(c),
        ]
        self._at.AudioConverterSetProperty.restype = OS
        self._at.AudioConverterSetProperty.argtypes = [c, ctypes.c_uint32, ctypes.c_uint32, c]
        self._at.AudioConverterFillComplexBuffer.restype = OS
        self._at.AudioConverterFillComplexBuffer.argtypes = [
            c, _FillCB, c,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(_AudioBufferList1),
            ctypes.POINTER(_StreamPacketDesc),
        ]
        self._at.AudioConverterDispose.restype = OS
        self._at.AudioConverterDispose.argtypes = [c]

    def _make_converter(self) -> None:
        in_fmt = _ASBD(
            mSampleRate=float(OUTPUT_SAMPLE_RATE),
            mFormatID=_fourcc("aacf"),  # kAudioFormatMPEG4AAC_ELD_SBR
            mFormatFlags=0,
            mBytesPerPacket=0,
            mFramesPerPacket=SAMPLES_PER_FRAME,
            mBytesPerFrame=0,
            mChannelsPerFrame=OUTPUT_CHANNELS,
            mBitsPerChannel=0,
            mReserved=0,
        )
        out_fmt = _ASBD(
            mSampleRate=float(OUTPUT_SAMPLE_RATE),
            mFormatID=_fourcc("lpcm"),
            mFormatFlags=_LPCM_FLAGS_F32_PACKED,
            mBytesPerPacket=OUTPUT_CHANNELS * 4,
            mFramesPerPacket=1,
            mBytesPerFrame=OUTPUT_CHANNELS * 4,
            mChannelsPerFrame=OUTPUT_CHANNELS,
            mBitsPerChannel=32,
            mReserved=0,
        )
        status = self._at.AudioConverterNew(
            ctypes.byref(in_fmt), ctypes.byref(out_fmt), ctypes.byref(self._converter),
        )
        if status != 0:
            raise RuntimeError(f"AudioConverterNew failed: 0x{status & 0xffffffff:08x}")

    def _set_magic_cookie(self) -> None:
        cookie_buf = ctypes.create_string_buffer(MAGIC_COOKIE, len(MAGIC_COOKIE))
        status = self._at.AudioConverterSetProperty(
            self._converter, _fourcc("dmgc"), len(MAGIC_COOKIE), cookie_buf,
        )
        if status != 0:
            raise RuntimeError(
                f"AudioConverterSetProperty(MagicCookie) failed: 0x{status & 0xffffffff:08x}"
            )

    def _fill_callback(self, _conv, io_n_packets, io_data, out_pkt_desc, _user):
        """Pull one queued AU into AudioConverter's input buffer slot."""
        if not self._input_queue:
            io_n_packets[0] = 0
            return _CB_STATUS_QUEUE_EMPTY

        pkt = self._input_queue.popleft()
        ctypes.memmove(self._input_buf, pkt, len(pkt))
        io_data[0].mBuffers[0].mData = ctypes.cast(self._input_buf, ctypes.c_void_p).value
        io_data[0].mBuffers[0].mDataByteSize = len(pkt)
        io_data[0].mBuffers[0].mNumberChannels = OUTPUT_CHANNELS

        self._packet_desc.mStartOffset = 0
        self._packet_desc.mVariableFramesInPacket = 0
        self._packet_desc.mDataByteSize = len(pkt)
        if out_pkt_desc:
            out_pkt_desc[0] = self._packet_desc_ptr
        io_n_packets[0] = 1
        return 0


# ── backend 2: libfdk-aac ─────────────────────────────────────────────

# Transport types (from fdk-aac/libAACdec/include/aacdecoder_lib.h).
_FDK_TT_MP4_RAW = 0
_FDK_OK = 0
# Output buffer is sized for the worst case: 1024 samples × 8 channels.
_FDK_OUT_MAX_SAMPLES = 8192


class _LibFdkBackend:
    """libfdk-aac via ctypes — works wherever the shared library is on the
    loader path. Decodes one AU per `decode()` call into int16, then
    converts to float32 stereo."""

    _SO_CANDIDATES = (
        "libfdk-aac.so.2",
        "libfdk-aac.so.1",
        "libfdk-aac.so",
        "/usr/lib/x86_64-linux-gnu/libfdk-aac.so.2",
        "/opt/homebrew/lib/libfdk-aac.2.dylib",
        "/usr/local/lib/libfdk-aac.2.dylib",
        "libfdk-aac.2.dylib",
        "libfdk-aac-2.dll",
        "fdk-aac.dll",
    )

    def __init__(self) -> None:
        self._lib = self._load_library()
        self._configure_signatures()
        self._handle = self._lib.aacDecoder_Open(_FDK_TT_MP4_RAW, 1)
        if not self._handle:
            raise RuntimeError("aacDecoder_Open returned NULL")

        # Configure with the AAC-ELD-SBR AudioSpecificConfig.
        asc_buf = (ctypes.c_ubyte * len(AUDIO_SPECIFIC_CONFIG)).from_buffer_copy(AUDIO_SPECIFIC_CONFIG)
        asc_ptr = (ctypes.POINTER(ctypes.c_ubyte) * 1)(
            ctypes.cast(asc_buf, ctypes.POINTER(ctypes.c_ubyte))
        )
        asc_len = (ctypes.c_uint * 1)(len(AUDIO_SPECIFIC_CONFIG))
        err = self._lib.aacDecoder_ConfigRaw(self._handle, asc_ptr, asc_len)
        if err != _FDK_OK:
            self._lib.aacDecoder_Close(self._handle)
            self._handle = None
            raise RuntimeError(f"aacDecoder_ConfigRaw failed: 0x{err:04x}")

        self._asc_keep = asc_buf  # keep alive for decoder's lifetime
        self._output_buf = (ctypes.c_int16 * _FDK_OUT_MAX_SAMPLES)()

    # -- public API ----------------------------------------------------

    def decode(self, payload: bytes) -> Optional[np.ndarray]:
        if not payload:
            return None
        pkt = bytes(payload)
        in_buf = (ctypes.c_ubyte * len(pkt)).from_buffer_copy(pkt)
        in_bufs = (ctypes.POINTER(ctypes.c_ubyte) * 1)(
            ctypes.cast(in_buf, ctypes.POINTER(ctypes.c_ubyte))
        )
        in_sizes = (ctypes.c_uint * 1)(len(pkt))
        valid = ctypes.c_uint(len(pkt))

        if self._lib.aacDecoder_Fill(self._handle, in_bufs, in_sizes, ctypes.byref(valid)) != _FDK_OK:
            return None
        if self._lib.aacDecoder_DecodeFrame(
            self._handle, self._output_buf, _FDK_OUT_MAX_SAMPLES, 0,
        ) != _FDK_OK:
            return None

        info = self._lib.aacDecoder_GetStreamInfo(self._handle)
        if not info:
            return None

        # CStreamInfo first 12 bytes (the only fields we need): three INT
        # at offsets 0/4/8 = sampleRate, frameSize, numChannels.
        sample_rate = ctypes.c_int.from_address(info + 0).value
        frame_size = ctypes.c_int.from_address(info + 4).value
        n_channels = ctypes.c_int.from_address(info + 8).value
        if frame_size <= 0 or n_channels <= 0:
            return None

        n_samples = frame_size * n_channels
        raw = np.frombuffer(self._output_buf, dtype=np.int16, count=n_samples)
        stereo = raw.reshape(-1, n_channels).astype(np.float32) / 32768.0
        if n_channels == 1:
            stereo = np.repeat(stereo, OUTPUT_CHANNELS, axis=1)

        # Some libfdk builds report the core (non-SBR) sample rate. Linear-
        # interp upsample to 48 kHz so downstream stays at one fixed rate.
        if 0 < sample_rate < OUTPUT_SAMPLE_RATE:
            factor = OUTPUT_SAMPLE_RATE // sample_rate
            if factor >= 2:
                stereo = np.repeat(stereo, factor, axis=0)

        return np.ascontiguousarray(stereo)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.aacDecoder_Close(self._handle)
            self._handle = None

    # -- setup helpers -------------------------------------------------

    @classmethod
    def _load_library(cls) -> ctypes.CDLL:
        # Windows: Python ≥ 3.8 ignores PATH for ctypes by design
        # (security hardening), so we have to register known DLL
        # search directories explicitly. Cover the common
        # libfdk-aac install locations:
        #   - MSYS2 (`pacman -S mingw-w64-x86_64-fdk-aac`)
        #   - vcpkg (`vcpkg install fdk-aac:x64-windows`)
        #   - Anaconda / conda-forge
        # Also pull in the venv's own Scripts/ dir so a DLL the user
        # dropped next to `iss.exe` works without copying its
        # transitive dependencies. `add_dll_directory` returns a
        # cookie we don't need to keep — the entry stays for the
        # lifetime of the process.
        if sys.platform == "win32":
            for d in cls._win_dll_search_dirs():
                if os.path.isdir(d):
                    try:
                        os.add_dll_directory(d)
                    except (OSError, AttributeError):
                        pass
        last_err: Optional[OSError] = None
        for name in cls._SO_CANDIDATES:
            try:
                return ctypes.CDLL(name)
            except OSError as e:
                last_err = e
        raise RuntimeError(f"libfdk-aac shared library not found: {last_err}")

    @staticmethod
    def _win_dll_search_dirs() -> list[str]:
        """Common Windows install locations for libfdk-aac (and its
        own transitive deps like libgcc_s_seh-1.dll). Order matters
        only for which version wins if multiple are installed."""
        import sys as _sys
        dirs: list[str] = [
            r"C:\tools\msys64\mingw64\bin",
            r"C:\msys64\mingw64\bin",
            r"C:\vcpkg\installed\x64-windows\bin",
        ]
        # Scoop's MSYS2 install. `scoop install msys2` puts everything
        # under %USERPROFILE%\scoop\apps\msys2\current\.
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            dirs.append(
                os.path.join(userprofile, "scoop", "apps", "msys2",
                             "current", "mingw64", "bin")
            )
        # Same dir as the running python.exe (covers "user drops the
        # DLL into the venv's Scripts dir").
        exe_dir = os.path.dirname(_sys.executable)
        if exe_dir:
            dirs.append(exe_dir)
        # Anaconda / conda-forge
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            dirs.append(os.path.join(conda_prefix, "Library", "bin"))
        return dirs

    def _configure_signatures(self) -> None:
        L = self._lib
        L.aacDecoder_Open.argtypes = [ctypes.c_int, ctypes.c_uint]
        L.aacDecoder_Open.restype = ctypes.c_void_p

        L.aacDecoder_ConfigRaw.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_uint),
        ]
        L.aacDecoder_ConfigRaw.restype = ctypes.c_uint

        L.aacDecoder_Fill.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        L.aacDecoder_Fill.restype = ctypes.c_uint

        L.aacDecoder_DecodeFrame.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int, ctypes.c_uint,
        ]
        L.aacDecoder_DecodeFrame.restype = ctypes.c_uint

        L.aacDecoder_GetStreamInfo.argtypes = [ctypes.c_void_p]
        L.aacDecoder_GetStreamInfo.restype = ctypes.c_void_p

        L.aacDecoder_Close.argtypes = [ctypes.c_void_p]
        L.aacDecoder_Close.restype = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ── factory ──────────────────────────────────────────────────────────

# Backend identifiers accepted by `make_aac_eld_decoder(prefer=...)` and
# the `AAC_BACKEND` env var. CLI flag `--aac-backend` should be wired to
# pass `prefer=` directly.
BACKEND_AUDIOTOOLBOX = "audiotoolbox"
BACKEND_FDK = "fdk"
_VALID_BACKENDS = frozenset({BACKEND_AUDIOTOOLBOX, BACKEND_FDK})

# Per-platform install instructions for the libfdk-aac runtime, included
# in the runtime error if it can't be loaded. Keep these accurate to what
# users actually have to do.
_LIBFDK_INSTALL_HELP = """\
libfdk-aac is required for audio on Linux and Windows.
Install one of:
  Debian/Ubuntu:  sudo apt install libfdk-aac2
  Fedora/RHEL:    sudo dnf install fdk-aac
  Arch:           sudo pacman -S libfdk-aac
  macOS (rare):   brew install fdk-aac
  Windows:        install MSYS2 (https://www.msys2.org), then run:
                    pacman -Sy --noconfirm mingw-w64-x86_64-fdk-aac
                  This drops libfdk-aac-2.dll at
                    C:\\msys64\\mingw64\\bin\\
                  which iss searches automatically. (PATH is *not*
                  used — Python ≥ 3.8 hardens DLL search; iss calls
                  os.add_dll_directory() on the known install paths
                  for you.)
"""


def _make_libfdk() -> AacEldDecoder:
    """Build the libfdk-aac backend with a friendly install hint on failure."""
    try:
        return _LibFdkBackend()
    except Exception as e:
        raise RuntimeError(
            f"AAC-ELD-SBR decoder unavailable: {e}\n\n{_LIBFDK_INSTALL_HELP}"
        ) from e


def make_aac_eld_decoder(prefer: Optional[str] = None) -> Optional[AacEldDecoder]:
    """Return the best available AAC-ELD-SBR decoder, or `None` when no
    backend is reachable. Callers treat `None` as "audio disabled" — the
    session continues with video-only.

    Resolution order:
      1. Explicit `prefer=` argument (typically from a `--aac-backend` CLI flag).
         If forced and the backend can't be built, raises so the user
         sees their explicit choice failed.
      2. `AAC_BACKEND` env var (back-compat).
      3. Platform default — AudioToolbox on Darwin, libfdk-aac elsewhere.
         On Linux/Windows, libfdk-aac is *optional*: AudioToolbox is
         macOS-only and FFmpeg's native AAC bails on AOT 39 with
         `PatchWelcomeError`, so libfdk-aac is the only viable decoder
         for those platforms today. It's GPL-licensed and not
         redistributable in our wheel; logging a warning and returning
         None lets the rest of the stream survive instead of
         hard-failing the whole session.
    """
    forced = prefer or os.environ.get("AAC_BACKEND", "")
    forced = forced.lower().strip() or None
    if forced is not None and forced not in _VALID_BACKENDS:
        raise ValueError(
            f"unknown AAC backend {forced!r}; expected one of {sorted(_VALID_BACKENDS)}"
        )

    if forced == BACKEND_AUDIOTOOLBOX:
        return _AudioToolboxBackend()
    if forced == BACKEND_FDK:
        return _make_libfdk()

    # Platform default.
    if sys.platform == "darwin":
        try:
            return _AudioToolboxBackend()
        except Exception as e:
            log.warning(
                "AudioToolbox unavailable (%s); falling through to libfdk-aac",
                e,
            )

    # Linux / Windows / non-Darwin fall-through: libfdk-aac is optional.
    try:
        return _LibFdkBackend()
    except Exception as e:
        log.warning(
            "AAC-ELD decoder unavailable, audio disabled: %s\n\n%s",
            e, _LIBFDK_INSTALL_HELP,
        )
        return None


__all__ = [
    "AUDIO_SPECIFIC_CONFIG",
    "AacEldDecoder",
    "BACKEND_AUDIOTOOLBOX",
    "BACKEND_FDK",
    "MAGIC_COOKIE",
    "OUTPUT_CHANNELS",
    "OUTPUT_SAMPLE_RATE",
    "SAMPLES_PER_FRAME",
    "make_aac_eld_decoder",
]
