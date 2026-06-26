"""High-level Session orchestrating the full Apple HP screen-share pipeline.

A `Session` wraps:
  - the TCP control channel + enc1103 cipher
  - the three UDP receivers (control, video, audio)
  - the HEVC tile decoder
  - the AAC-ELD audio decoder
  - the RTCP / heartbeat / FIR / NACK sender
  - the InputController

A consumer (the native viewer or any library user) constructs
`Session(SessionConfig(...))`, calls `connect()`, polls
`get_frame(tile_idx)` / waits on `wait_for_fresh_tile()`, hands an
audio callback in via `set_audio_callback()`, drives input via
`session.input.pointer_event(...)`, and finally calls `close()`.

Reconnection model: `connect()` and `close()` are cycle-able on the
same `Session`. After `close()`, calling `connect()` again restarts
the handshake against the same host. If the TCP dies mid-session the
rx-loops set `is_connected` False; the consumer detects that and
calls `connect()` again.
"""
from __future__ import annotations

import logging
import os
import queue
import socket
import struct
import zlib
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .media.tiles import TileFrame
from .input import InputController
from .media.aac_eld import AacEldDecoder, make_aac_eld_decoder
from .media.hevc import HevcDecoder
from .media.avc import AvcDecoder
from .media.nalu import first_donl, reassemble_group
from .media.avc_nalu import reassemble_h264
from .media.registry import resolve_codec
from .media.quality_gate import TileVisState
from .. import __version__ as _iss_version
from .control import ControlServer
from .protocol.burst import BurstStarved, InitialBurst, gather_initial_burst
from .protocol.negotiation import (
    AdvertiseDims,
    NegotiationResult,
    connect_and_negotiate,
)
from .protocol.offers import extract_offer_ssrc, create_offers
from .protocol.rfb import warmup_tcp
from .protocol.rtcp import (
    build_empty_sr,
    build_fir,
    build_fir_legacy,
    build_nack,
    build_pli,
    build_rr,
    build_rtcp_app_ltrp,
    compound_with_rr,
    parse_sr_arrivals,
)
from .protocol.srtp import (
    SRTCPDecryptor,
    SRTCPEncryptor,
    SRTPDecryptor,
    SRTPEncryptor,
)


log = logging.getLogger(__name__)


# ── tunable constants ─────────────────────────────────────────────────

# Inter-tx-pulse interval. Each tick: send PT=101 audio heartbeat, send
# RTCP RR (and SR every Nth tick), drain pending FIR/PLI/NACK from the
# decoder's gate, run stall watchdog. 500 ms keeps daemon happy without
# wasting CPU on idle ticks.
_TX_INTERVAL_S = 0.5
_RTCP_SR_EVERY_N_TICKS = 10              # 10 × 0.5 s = 5 s SR cadence
# How often to emit the per-tile profile snapshot. 4 ticks × 0.5 s = 2 s
# cadence — enough granularity to spot a stuck tile within a few seconds
# without flooding the log.
_TX_PROFILE_EVERY_N_TICKS = 4

# How long we wait on a UDP socket recv before re-checking the stop flag.
_UDP_RECV_TIMEOUT_S = 1.0
_TCP_RECV_TIMEOUT_S = 1.0

# Initial-burst re-arm: occasionally TCP negotiation fully succeeds and the
# host accepts the canvas, but screensharingd never starts sending RTP -- a
# host-side encoder/agent stall. Rather than bail fatally we tear the TCP
# down and re-handshake from scratch a couple of times; a fresh agent
# usually streams. (The far more common "0 packets" cause -- an IPv6-only
# hostname vs IPv4-only UDP sockets -- is handled up front by
# _resolve_host_ipv4, not here.)
_BURST_RETRY_ATTEMPTS = 3
_BURST_RETRY_SLEEP_S = 0.8

# Cap on the per-socket drain queue. The drain thread does only recvfrom +
# put — its only job is to empty the kernel UDP buffer fast enough that
# packets aren't dropped by the kernel (RcvbufErrors). The process thread
# does decrypt + dispatch from the queue. If the process thread can't keep
# up the queue fills and we drop packets at the app layer instead — same
# end result as a kernel drop, but moved to a place we can measure.
#
# 16384 (~24 MB worst-case at ~1500 B) absorbs the bursts a HiDPI/2x stream
# produces: at ~300 Mbps the packet rate is ~28k pps, and the old 4096 cap
# was only ~0.15 s of buffer — measured overflowing in bursts (snapshot
# showed depth≈1 but drop=15197, i.e. the process thread keeps up on average
# but spikes blew past the shallow queue, breaking the ref chain). The deeper
# queue is ~0.6 s, enough for the encoder's per-frame/IDR bursts; the process
# thread drains it back to ~empty between spikes so the added latency only
# appears during a genuine overload, not steady state.
_UDP_DRAIN_QUEUE_MAX = 16384

# Stall threshold — if no decoded video frame in this long, mark the
# session as soft-dead. Consumer reconnects by calling close() + connect().
_STALL_THRESHOLD_S = 5.0

# Dynamic SSRC adoption: when this many packets each from a fresh group
# of 4 unknown SSRCs have arrived, rebuild ssrc→tile and request fresh IDRs.
_DYNAMIC_SSRC_PACKET_THRESHOLD = 5

# Minimum gap between successive SSRC adoptions. Without this, two
# concurrent SSRC groups (e.g. during an agent transition where Apple
# briefly emits both) cause us to flip-flop every few packets, restarting
# the HEVC decoder each time and producing severe artifacts. 1.5 s is
# wide enough that the losing group's count tops out and the survivor
# wins, narrow enough that genuine post-transition adoption isn't
# delayed beyond what one IDR cycle masks.
_SSRC_ADOPT_COOLDOWN_S = 1.5

# Frame-publishing silence required before SSRC adoption is allowed.
# Apple's daemon often emits the burst on a different SSRC group
# than the live stream — the only way to recover is by adopting the
# new group, so this needs to fire fast. The blacklist (in
# `_note_unknown_ssrc`) prevents re-adopting a previously-failed
# group, eliminating the historical ping-pong concern.
_SSRC_ADOPT_STALL_S = 2.0

# How long an incomplete RTP group can live in `pending_groups` before we
# evict it as a permanent hole (the marker never arrived). Avoids the
# pending-groups dict growing unbounded on lossy links.
_PENDING_GROUP_TTL_S = 0.2

# How long a completed (marker-flushed) (SSRC, ts) key stays in the dedup
# set so we drop late retransmits of already-processed packets without
# re-feeding duplicate NALUs into the decoder. 2 s is comfortably wider
# than any reasonable RTP retransmit window.
_FLUSHED_DEDUP_TTL_S = 2.0

# PT=101 keepalive payload — opaque-but-required handshake bytes.
# Empirically: any 4-byte payload works as long as it's sent at the
# negotiated SSRC. Apple's own client sends the same constant.
_HEARTBEAT_PAYLOAD = bytes.fromhex("00683400")

# Apple HP audio stream format on the wire.
_AUDIO_PT = 101


# ── public config + result types ─────────────────────────────────────

@dataclass(frozen=True)
class _CursorImage:
    """A decoded cursor pixmap from RFB enc 1104.

    `rgba` is row-major RGBA8888 with `width * height * 4` bytes.
    Hotspot is the click point in cursor-local pixel coords.
    Frontend hands this to `glfw.create_cursor` + `glfw.set_cursor`.
    """
    width: int
    height: int
    hotspot_x: int
    hotspot_y: int
    rgba: bytes


@dataclass
class SessionConfig:
    """Connection parameters. Defaults are conservative; the CLI
    overrides whatever the user customised."""

    host: str
    username: str
    password: str
    port: int = 5900
    auth_mode: Literal["srp", "nonsrp"] = "srp"
    advertise: Optional[AdvertiseDims] = None
    hdr: bool = False
    audio: bool = True
    # HiDPI mode for the host's virtual display, resolved to a backing:point
    # ratio by the frontend (which knows the window size):
    #   "on"   → always 2× (Retina): crisp, but ~4× the pixels = more
    #            bandwidth, and UI renders half-size on a non-Retina client.
    #   "off"  → always 1× (flat, backing == logical): far less bandwidth,
    #            correctly-sized UI on non-Retina (Linux/Windows) clients.
    #   "auto" → match the LOCAL display: 2× on a Retina client, 1× on a
    #            non-Retina one (so the stream maps 1:1 to the client's pixels);
    #            2× is downgraded to 1× when it wouldn't fit the host's
    #            3840×2160 cap (window logical > 1920×1080).
    # The frontend passes the resolved scale into send_dynamic_resolution and
    # the initial AdvertiseDims; the proxy itself only carries the mode.
    hidpi: str = "auto"

    # When auth user differs from the console user, ask the console user
    # to share their existing session (Apple's "Ask to share" choice in
    # the Screen Sharing.app prompt). On accept, the viewer joins the
    # console session in observe-only mode rather than starting an
    # alt-user virtual display. False (default) = no SessionSelect exchange.
    share_console: bool = False

    # When auth user differs from the console user, log them in to a
    # fresh virtual display via Apple's cmd=2 SessionSelect path
    # (encrypted creds at body+0x48 / +0x88; daemon spawns user2's
    # vdisplay; iss replays Apple Screen Sharing's canned 308-byte SDC so the
    # encoder targets the new vdisplay rather than the console user's
    # screen). Mutually exclusive with `share_console`.
    alt_session: bool = False

    # Curtain mode: send the SetDisplayConfiguration message during
    # negotiation so the daemon spins up a SkyLight virtual display
    # and blanks the host's physical screen while we view. When False,
    # we skip that message entirely — the daemon then encodes whatever
    # is on the physical display, no curtain. Default on for privacy
    # (the host's screen isn't broadcast to anyone walking past). The
    # cmd=2 alt-session path ignores this flag and always sends Apple's
    # canned SDC instead.
    curtain: bool = True

    # Which 4-SSRC quality tier to subscribe to (0 = highest, ascending = lower).
    quality_tier: int = 0

    # AAC-ELD backend override (None → platform default).
    aac_backend: Optional[str] = None

    # Whether to perform Apple's two-TCP warmup before the real session.
    # Required for surviving lock-screen → login → desktop transitions
    # without the daemon closing our TCP. Defaults to True.
    warmup_tcp: bool = True

    # UDP port overrides — None ⇒ port + 0/1 (Apple's layout: audio
    # rtcp-muxes onto ctrl_port; video rtcp-muxes onto video_port).
    # Custom values let a NAT relay route the streams.
    udp_ctrl_port: Optional[int] = None
    udp_video_port: Optional[int] = None
    udp_bind_host: str = ""  # "" = INADDR_ANY

    # Local control socket: when set, the Session opens a ControlServer at
    # this path (UDS on POSIX, sidecar `.port` file on Windows) so an
    # external TUI / monitor can subscribe to snapshots and (later) issue
    # commands. None = no control server.
    control_socket: Optional[str] = None

    # Packet capture: when set, every byte that crosses the TCP control
    # socket and the two UDP media sockets is written — exactly as it goes
    # on the wire (still enc1103/SRTP encrypted) — to a classic `.pcap` at
    # this path, with synthetic Ethernet/IPv4/TCP|UDP framing. The result is
    # byte-identical to a packet-capture dump and feeds straight into the
    # bundled Python dissector (which derives the keys from the cleartext
    # handshake it sees) or Wireshark. None ⇒ no capture; zero overhead.
    record_pcap: Optional[str] = None

    # Dynamic resolution: the frontend issues a mid-session resize (via
    # Session.send_dynamic_resolution) whenever the viewer window changes
    # size, so the host's virtual display re-renders sharp at the new size
    # instead of stretching a fixed canvas. This is an IN-BAND change on
    # the existing connection — 0x1d → server 0x451 → 0x1c re-offer →
    # encoder restart — no reconnect. Purely a frontend concern; the
    # protocol layer doesn't read this flag, it rides along with the rest
    # of the config to the desktop viewer. Off ⇒ classic fixed canvas
    # (the window just stretches the stream on resize).
    dynamic_resolution: bool = False

# ── internal RTP packet group ────────────────────────────────────────

# A single RTP packet's metadata after SRTP decrypt. Kept as a tuple to
# keep `pending_groups` cheap — these accumulate by the thousands per
# second on busy streams.
_PktTuple = tuple[int, bool, bytes]  # (seq, marker, payload)


# ── Session ──────────────────────────────────────────────────────────

class Session:
    """Connection + decode pipeline for one Apple HP screen-share session."""

    def __init__(self, config: SessionConfig) -> None:
        self._config = config

        # Connection state — None when disconnected.
        self._negotiation: Optional[NegotiationResult] = None
        # Video codec: 'avc' (H.264 4:2:0) when ISS_VIDEO_CODEC=avc is offered,
        # else 'hevc' (the byte-identical default the server answers with HEVC).
        self._video_codec = (
            "avc" if os.environ.get("ISS_VIDEO_CODEC", "").lower() == "avc"
            else "hevc"
        )
        # AVC hardware-decode periodic IDR re-anchor (d3d11va POC-wrap
        # workaround). The full machinery (_maybe_reanchor + _tx_loop driver)
        # lives on the AVC feature branch and is NOT ported here; pin to False
        # so the AVC decoder-selection branch uses the normal path. Re-enabling
        # requires porting _maybe_reanchor — do not just flip this.
        self._avc_reanchor_enabled: bool = False
        # Runtime-updated canvas dimensions from AppleDisplayLayout (0x451).
        # `_runtime_canvas_*` = backing/pixel size (decoded frame dimensions).
        # `_runtime_scaled_*` = logical/scaled size (window/client coordinate space).
        self._runtime_canvas_w: int = 0
        self._runtime_canvas_h: int = 0
        self._runtime_scaled_w: int = 0
        self._runtime_scaled_h: int = 0
        self._needs_post_layout_fir: bool = False
        self._needs_param_harvest: bool = False
        # Cross-RTP-group accumulators for the post-resize param harvest.
        self._harvest_vps: Optional[bytes] = None
        self._harvest_sps: Optional[bytes] = None
        self._harvest_pps: dict[int, bytes] = {}
        self._decoder: Optional[object] = None
        self._aac: Optional[AacEldDecoder] = None
        self._input: Optional[InputController] = None
        # Reassembler for multi-cipher-frame msg 0x1f (clipboard) sends.
        # Stateful across rx-loop iterations.
        from .protocol.clipboard import ClipboardReassembler
        self._clipboard_reassembler = ClipboardReassembler()
        self._ssrc_to_tile: dict[int, int] = {}
        # Per-tile received coded bytes — to see which screen band (tile)
        # eats the bandwidth (tile 0 = top/menu-bar strip, 1-3 down the
        # screen). Surfaced as KB/s in the profile log.
        self._tile_bytes: dict[int, int] = {}
        self._last_tile_bytes: dict[int, int] = {}
        self._last_ssrc_adopt_ts: float = 0.0
        self._ssrc_blacklist: set[int] = set()
        self._last_profile_good: list[int] = []
        self._last_profile_clean: list[int] = []

        # Cipher state for the TX channel.
        self._video_decryptor: Optional[SRTPDecryptor] = None
        self._audio_decryptor: Optional[SRTPDecryptor] = None
        self._audio_encryptor: Optional[SRTPEncryptor] = None
        self._srtcp_dec: Optional[SRTCPDecryptor] = None
        self._srtcp_enc: Optional[SRTCPEncryptor] = None
        self._our_video_ssrc: Optional[int] = None
        self._our_audio_ssrc: Optional[int] = None

        # UDP sockets (bound on connect, closed on close).
        # _sock_ctrl carries audio RTP + audio/control RTCP (rtcp-muxed
        # on UDP 5900); the ctrl process loop demuxes by RTP payload-type byte.
        self._sock_ctrl: Optional[socket.socket] = None
        self._sock_video: Optional[socket.socket] = None
        # IPv4 address the host resolves to. The HP UDP transport (RTP/RTCP)
        # is IPv4-only, so TCP must land on the same IPv4 — resolved once in
        # _connect_internal. Defaults to the configured host until then.
        self._dest_host: str = self._config.host
        # Optional control server (TUI subscribers); see `control.py`.
        self._control: Optional[ControlServer] = None
        # Optional packet capture; see `util/pcap_recorder.py`. Created on
        # connect when cfg.record_pcap is set, finalised on teardown.
        # `_record_cycle` counts connect() cycles so a reconnect writes a
        # fresh file instead of truncating the previous session's capture.
        self._recorder = None  # Optional[PcapRecorder]
        self._record_cycle = 0
        # Wall-clock instant the current session went LIVE (post-burst).
        # Used to compute uptime for control-snapshot consumers.
        self._connect_wall_ts: float = 0.0
        # Runtime audio mute, toggled by control-socket commands. When
        # True, decoded PCM frames are still received + decrypted (so
        # the stream stays in sync) but not forwarded to the sink.
        self._audio_user_mute: bool = False

        # Drain queues — see _video_drain_loop / _ctrl_drain_loop. The
        # drain threads do nothing but recvfrom + put_nowait; the
        # process threads do all decrypt + dispatch work.
        self._video_q: queue.Queue[bytes] = queue.Queue(maxsize=_UDP_DRAIN_QUEUE_MAX)
        self._ctrl_q: queue.Queue[bytes] = queue.Queue(maxsize=_UDP_DRAIN_QUEUE_MAX)
        self._video_q_dropped = 0
        self._ctrl_q_dropped = 0
        # RX/TX byte+packet counters. _rx_msg_type_counts[type] tracks
        # inbound RFB msg-type histogram on the TCP control channel —
        # critical for diagnosing the "cursor freeze after idle" bug
        # (we want to know whether daemon's msg type=0x00 sends stop,
        # OR whether iss is failing to process them).
        self._rx_pkts_video = 0
        self._rx_pkts_ctrl = 0
        self._rx_pkts_tcp = 0
        self._rx_bytes_video = 0
        self._rx_bytes_ctrl = 0
        self._tx_pkts = 0
        self._cursor_msgs_processed = 0
        self._cursor_last_t: float = 0.0
        self._rx_msg_type_counts: dict[int, int] = {}
        # Snapshot baselines for delta computation in _log_profile_snapshot.
        self._last_rx_pkts: tuple[int, int, int] = (0, 0, 0)
        self._last_rx_bytes: tuple[int, int] = (0, 0)
        self._last_tx_pkts: int = 0
        self._last_profile_nalu: list[dict[int, int]] = []

        # Per-(SSRC, ts) accumulating groups + their first-arrival time
        # for TTL eviction. `_recently_flushed` dedupes late retransmits of
        # already-processed groups; entries TTL out after _FLUSHED_DEDUP_TTL_S.
        self._pending_groups: dict[tuple[int, int], list[_PktTuple]] = {}
        self._group_arrival: dict[tuple[int, int], float] = {}
        self._recently_flushed: dict[tuple[int, int], float] = {}

        # Per-SSRC sequence tracking for NACK detection + receiver reports.
        self._max_seq: dict[int, int] = {}
        self._roc: dict[int, int] = {}
        self._nack_pending: dict[int, set[int]] = defaultdict(set)
        # Cumulative packet-loss tracking. Lets consumers report a
        # real loss rate instead of guessing concealment is loss-driven.
        self._received_pkts: int = 0
        self._lost_pkts: int = 0
        # Monotonic timestamp of the most recent video packet received.
        # Used by the stall watchdog to distinguish "decoder is wedged
        # despite incoming packets" (real stall — fire FIR) from "Apple
        # paused sending video because the screen is idle" (no problem
        # — silently wait). Without this, the watchdog incorrectly
        # fires FIR storms during long static-screen periods where
        # Apple's encoder rate-controls down to zero.
        self._last_video_pkt_t: float = 0.0
        # Per-tile loss counter (cumulative since session start). Lets
        # the profile log distinguish "tile went bad because we lost
        # packets" from "tile went bad with zero observed loss" — i.e.
        # network failure vs. decoder/code bug.
        self._lost_pkts_per_tile: list[int] = []
        self._last_profile_lost_per_tile: list[int] = []

        # Server SR arrivals for RR's lsr/dlsr fields.
        self._server_sr: dict[int, tuple[int, float]] = {}

        # Threads + lifecycle.
        self._stop_evt = threading.Event()
        self._threads: list[threading.Thread] = []
        self._fresh_evt = threading.Event()
        self._connected = False
        self._closing = False

        # Stall detection.
        self._last_publish_t = 0.0
        self._tx_tick = 0

        # LTRP fast-recovery: ack each cleanly-decoded base-tile frame by its
        # DONL (decoding-order number) so the host's encoder can use it as a
        # long-term reference. The DONL acks resolve to real encoder reference
        # tokens, keeping the encoder's references near (within our decoder's
        # picture buffer) instead of reaching far back and forcing a full IDR.
        # ON by default; set ISS_LTRP=0 to disable.
        self._ltr_enabled = os.environ.get("ISS_LTRP", "1") != "0"
        self._ltr_last_acked: int = 0
        self._ltr_acks_sent: int = 0
        # The per-ack LTR log fires at frame rate and drowns the debug log
        # (a `tail` then only covers a few seconds, hiding real events like
        # gray-outs). Off by default; ISS_LOG_LTR=1 re-enables it for LTRP
        # debugging. Acks-flowing is otherwise visible via the profile line.
        self._log_ltr_acks = os.environ.get("ISS_LOG_LTR") == "1"
        # Dedup state for the misc-status (0x14) push log — only emit when
        # the cmd value changes, not on every repeat.
        self._last_misc_status_cmd: object = None

        # DPB-break detection state. `_dpb_error_window` is a sliding
        # ring of monotonic timestamps for libav "Could not find ref"
        # events; threshold breach inside the window fires fast-path
        # FIR (see `_on_libav_concealment`). `_last_decoder_restart_t`
        # is set to monotonic() whenever any code path tears down +
        # recreates the decoder, used to grace-suppress the fast path
        # during the burst tail.
        self._dpb_error_window: deque[float] = deque()
        self._last_decoder_restart_t: float = 0.0
        # Escalation state: when a "Could not find ref" storm persists
        # through repeated targeted FIRs, the tile-0 IDR fan-out doesn't
        # clear it (tiles 1-3 are P-only; the host won't send them their own
        # IDR). Count consecutive storm FIRs; past the threshold, escalate to
        # a force-IDR of ALL tiles (the manual force-IDR action), which does
        # clear the gray.
        self._dpb_fir_count: int = 0
        self._last_dpb_error_t: float = 0.0
        # Gray-out event aggregator. Multiple paths (per-tile gate FIR,
        # batched DPB-break FIR) can emit FIRs within a few ms. Buffer
        # the tile indices and the most recent libav concealment message
        # so a single INFO summary line lands per event instead of 4
        # DEBUG lines the user can't see at default verbosity.
        self._grayout_window_t: float = 0.0
        self._grayout_window_tiles: set[int] = set()
        self._last_concealment_msg: str = ""
        # Throttle for the per-tile stuck-tile FIR backstop (`_check_stall`
        # Path B). Separate from `_last_decoder_restart_t` so it never
        # blocks the 15 s total-freeze restart (Path A).
        self._last_stuck_tile_fir_t: float = 0.0

        # Per-tile FIR rate limit, applied at the wire layer in
        # `_send_fir_for_tile` so it coalesces requests from every
        # caller (SSRC adoption, quality_gate, libav concealment fast
        # path, soft concealment, force-IDR, stall watchdog) into one FIR
        # per tile per `_FIR_MIN_INTERVAL_S` window. Without this,
        # multiple recovery paths firing within ~500 ms during an
        # SSRC restart caused Apple to send several IDRs per tile,
        # the decoder rejected all but the first as duplicate POCs,
        # and subsequent P-frames silently produced gray output.
        self._last_fir_per_tile: dict[int, float] = {}

        # Audio sink. Set by consumer; called whenever a PCM chunk decodes.
        self._audio_callback: Optional[Callable[[np.ndarray], None]] = None

        # Per-tile H.264 access-unit tap (AVC only). When set, each tile's
        # reassembled Annex-B access unit is handed to the callback as
        # (tile_idx, rtp_timestamp, au_bytes) from the video-process thread —
        # this is the pass-through feed for the browser frontend, which decodes
        # the H.264 itself (no decode/encode in iss for that path). The native
        # decoder keeps running in parallel so its FIR/keyframe-recovery still
        # drives the stream's health (the browser benefits from the same
        # recovered keyframes via this forward).
        self._video_au_callback: Optional[Callable[[int, int, bytes], None]] = None

        # Clipboard text from host. Optional callback; called whenever a
        # full msg 0x1f arrives and parses to a text item. The session
        # also pushes to the local OS clipboard directly (see
        # `local_clipboard.push_text`) so headless and frontend-less
        # callers still get clipboard sync.
        self._clipboard_text_callback: Optional[Callable[[str], None]] = None
        # Most-recent text we pushed to the local clipboard. Used as an
        # echo guard by the frontend's outbound poll thread so a value
        # that came FROM the host doesn't get sent BACK as a fresh
        # outbound msg 0x06 ClientCutText.
        self._last_received_clipboard_text: Optional[str] = None

        # Cursor pseudo-encoding (RFB enc 1104) state.
        # Server sends per-cursor pixmaps with a `cache_id` and may
        # later send just the cache_id to repeat a previously-sent
        # cursor. Cache holds (width, height, hotspot_x, hotspot_y,
        # rgba_bytes) keyed by cache_id. Frontend receives a callback
        # whenever a cursor frame lands so it can update the local
        # OS cursor (no rendering involved — we hand the pixmap to
        # GLFW which sets the system cursor).
        self._cursor_cache: dict[int, "_CursorImage"] = {}
        self._cursor_callback: Optional[
            Callable[[Optional["_CursorImage"]], None]
        ] = None
        # The daemon sends the initial cursor pixmap at the instant of connect
        # — often a few ms before the desktop frontend installs its callback
        # (set_cursor_callback) — and never re-sends it. Stash the last real
        # cursor so a late-registering callback can be replayed it immediately;
        # without this the cursor overlay never receives a texture.
        self._last_cursor_notified: Optional["_CursorImage"] = None
        # Safety counter for the cursor-keepalive: any video/raw rect arriving
        # over the RFB channel means an FBU request pulled video (the old
        # full-screen-request bug). Stays 0 with the 1x1 keepalive; surfaced
        # in the profile log so a regression is visible.
        self._fbu_video_rects = 0

    # ── public lifecycle ─────────────────────────────────────────────

    def connect(self) -> None:
        """Run the full handshake + media setup. Idempotent: a no-op if
        already connected. Raises on failure; partial state is torn down."""
        if self._connected:
            return
        try:
            self._connect_internal()
            self._connected = True
        except Exception:
            self._teardown()
            raise

    def close(self) -> None:
        """Stop all threads, release sockets, drop decoders. Safe to call
        multiple times. After close(), `connect()` can be called again to
        start a fresh session against the same host."""
        if self._closing:
            return
        self._closing = True
        try:
            self._teardown()
        finally:
            self._closing = False
            self._connected = False

    def __enter__(self) -> "Session":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ── public consumer API ──────────────────────────────────────────

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        """Latest decoded frame for `tile_idx`, or None if no new frame
        since the last call (or the gate blocked the latest)."""
        if self._decoder is None:
            return None
        if not 0 <= tile_idx < self.num_tiles:
            raise ValueError(
                f"tile_idx {tile_idx} out of range [0, {self.num_tiles})"
            )
        return self._decoder.get_frame(tile_idx)

    def wait_for_fresh_tile(self, timeout: float = 0.033) -> bool:
        """Block until any tile publishes a new frame, or `timeout` elapses.
        Consumers loop: wait → iterate `get_frame(ti)` for each tile →
        repeat. Returns False on timeout (no fresh frames)."""
        fired = self._fresh_evt.wait(timeout)
        if fired:
            self._fresh_evt.clear()
        return fired

    def set_audio_callback(
        self, cb: Optional[Callable[[np.ndarray], None]],
    ) -> None:
        """Install a callback invoked from the audio RX thread with each
        decoded PCM chunk: `(N, 2) float32` at 48 kHz. Pass None to remove."""
        self._audio_callback = cb

    def set_video_au_callback(
        self, cb: Optional[Callable[[int, int, bytes], None]],
    ) -> None:
        """Install a callback invoked from the video-process thread with each
        tile's reassembled H.264 access unit: `(tile_idx, rtp_timestamp,
        annexb_au_bytes)`. AVC codec only. Pass None to remove. The browser
        frontend uses this to forward H.264 to a WebCodecs decoder without iss
        decoding/encoding."""
        self._video_au_callback = cb

    def video_params(self) -> tuple[bytes, dict]:
        """(sps, all_pps) harvested from the AVC config — what a downstream
        WebCodecs decoder needs to configure(). Empty until the burst lands."""
        dec = self._decoder
        sps = getattr(dec, "_sps", b"") if dec is not None else b""
        pps = getattr(dec, "_pps", b"") if dec is not None else b""
        return sps, ({0: pps} if pps else {})

    def set_clipboard_text_callback(
        self, cb: Optional[Callable[[str], None]],
    ) -> None:
        """Install a callback invoked from the TCP control RX thread when
        a remote clipboard text update arrives. Useful for the frontend
        to suppress an outbound echo when the local clipboard updates to
        text it just received."""
        self._clipboard_text_callback = cb

    def set_cursor_callback(
        self, cb: Optional[Callable[[_CursorImage], None]],
    ) -> None:
        """Install a callback invoked from the TCP control RX thread when
        a new cursor pixmap arrives via RFB enc 1104. The frontend should
        hand the image to GLFW (`glfw.create_cursor` + `glfw.set_cursor`)
        so the local OS cursor changes to match what the host's mouse
        looks like (I-beam over text, resize cursors over window edges,
        busy spinner, etc.). Set to None to remove."""
        self._cursor_callback = cb
        # The daemon's first cursor pixmap can arrive before this callback is
        # installed (and isn't re-sent); replay the most recent one so a
        # late-registering frontend still gets the cursor shape.
        if cb is not None and self._last_cursor_notified is not None:
            try:
                cb(self._last_cursor_notified)
            except Exception as e:
                log.debug("cursor replay on register failed: %s", e)

    @property
    def last_received_clipboard_text(self) -> Optional[str]:
        return self._last_received_clipboard_text

    def request_fir(self, tile_idx: Optional[int] = None) -> None:
        """Externally trigger an FIR (forces a fresh IDR). Without args,
        targets every tile. Used by the TUI's force-IDR ('f') action, the
        SSRC-adoption restart path, and tests.

        Funnels through the gate's `keyframe_required` set so the
        sticky re-arm logic in `consume_fir_request` will keep
        retrying until each tile actually recovers — guards against
        the case where the IDR response to this FIR is lost in the
        same packet-loss event that motivated the call.
        """
        if self._decoder is None or self._negotiation is None:
            return
        gate = self._decoder._gate
        if tile_idx is None:
            gate.force_keyframe_all()
            # Bound by the gate's real tile count: a mid-session SSRC adoption
            # (notably under dynamic-resolution) can bump num_tiles while the
            # decoder is restart()ed, not recreated, so its gate keeps the
            # construction-time size. Avoids an IndexError that would kill the
            # caller's thread.
            for ti in range(min(self.num_tiles, len(gate._states))):
                self._send_fir_for_tile(ti)
        else:
            gate.mark_decode_error(tile_idx)
            self._send_fir_for_tile(tile_idx)

    def send_dynamic_resolution(
        self, width: int, height: int, hidpi_scale: int = 2,
    ) -> None:
        """Request a mid-session display resize without reconnecting.

        Sends a runtime SetDisplayConfiguration (0x1d) on the existing
        encrypted TCP channel with display_flags=DYNAMIC_RESOLUTION.
        The server responds with AppleDisplayLayout (0x451) and
        restarts the HEVC encoder at the new resolution — new SSRCs,
        new SPS/PPS, new IDR burst. The 0x451 handler updates canvas
        dims; the frontend polls `canvas_dims` to detect the change.
        No reconnect, no re-authentication.
        """
        from .protocol.rfb import build_virtual_display
        from .protocol.negotiation import build_fbu_request

        neg = self._negotiation
        if neg is None:
            raise RuntimeError("Not connected")
        log.info("send_dynamic_resolution: requesting %dx%d", width, height)
        msg = build_virtual_display(
            width=width, height=height, hidpi_scale=hidpi_scale, hdr=False,
        )
        neg.cipher.encrypt_and_send(neg.sock, msg)
        # Re-arm TCP and nudge encoder with FIRs for fresh IDRs.
        try:
            neg.cipher.encrypt_and_send(
                neg.sock, build_fbu_request(incremental=False)
            )
        except OSError:
            pass
        self.request_fir()

    # ── public state inspection ──────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def server_dims(self) -> tuple[int, int]:
        n = self._negotiation
        return (n.server_width, n.server_height) if n else (0, 0)

    @property
    def canvas_dims(self) -> tuple[int, int]:
        """Backing/pixel dimensions of the HEVC decoder output — the size
        the GPU textures must be allocated at. May be larger than
        `scaled_dims` on HiDPI virtual displays."""
        rw, rh = self._runtime_canvas_w, self._runtime_canvas_h
        if rw and rh:
            return (rw, rh)
        n = self._negotiation
        return (n.canvas_width, n.canvas_height) if n else (0, 0)

    @property
    def scaled_dims(self) -> tuple[int, int]:
        """Logical/scaled display dimensions — the coordinate space the
        client uses for window sizing and input mapping."""
        sw, sh = self._runtime_scaled_w, self._runtime_scaled_h
        if sw and sh:
            return (sw, sh)
        n = self._negotiation
        return (n.canvas_width, n.canvas_height) if n else (0, 0)

    @property
    def num_tiles(self) -> int:
        # Burst-observed count is authoritative — see comment near where
        # _observed_tile_count is set, post-burst, in _negotiate_and_burst.
        observed = getattr(self, "_observed_tile_count", 0)
        if observed:
            return observed
        n = self._negotiation
        from .protocol.offers import tiles_per_frame
        return n.canvas_tiles if n and n.canvas_tiles else tiles_per_frame()

    @property
    def hw_accel(self) -> Optional[str]:
        return self._decoder.hw_accel if self._decoder else None

    @property
    def input(self) -> InputController:
        if self._input is None:
            raise RuntimeError("Session is not connected; call connect() first")
        return self._input

    def tile_state(self, tile_idx: int) -> Optional[TileVisState]:
        return self._decoder.tile_state(tile_idx) if self._decoder else None

    @property
    def packet_stats(self) -> tuple[int, int]:
        """Cumulative ``(received, lost)`` packet counts since connect.
        Loss is detected at the SRTP RX layer via SSRC sequence-number
        gaps. A non-zero loss rate is the ground-truth signal that the
        encoder was forced to conceal — separate from heuristic
        post-decode concealment detection."""
        return self._received_pkts, self._lost_pkts

    @property
    def lost_pkts_per_tile(self) -> list[int]:
        """Per-tile cumulative packet loss (matches the order of
        `tile_state(i)`). Empty list before the decoder is initialised."""
        return list(self._lost_pkts_per_tile)

    @property
    def last_publish_age_s(self) -> float:
        """Seconds since the decoder last produced a fresh tile frame.
        0.0 if a frame just landed; very large if we're stuck. Negative
        if the session hasn't published anything yet."""
        if self._last_publish_t == 0.0:
            return -1.0
        return time.monotonic() - self._last_publish_t

    # ── connect / teardown internals ─────────────────────────────────

    def _connect_internal(self) -> None:
        cfg = self._config
        log.info("connecting to %s:%d (%s)", cfg.host, cfg.port, cfg.auth_mode)

        # Resolve to IPv4 up front and use it for BOTH the TCP control
        # connection and the UDP RTP/RTCP sockets. iss's HP transport binds
        # IPv4-only UDP sockets, so if the host is a name that resolves to
        # IPv6 first (e.g. a Bonjour `.local` name on Windows, where
        # getaddrinfo returns only IPv6), TCP negotiation succeeds over IPv6
        # but no video RTP ever lands on the IPv4 socket — the session looks
        # like it connected but gets 0 burst packets. Pinning IPv4 keeps the
        # control and media paths on the same family.
        self._dest_host = self._resolve_host_ipv4(cfg.host)
        if self._dest_host != cfg.host:
            log.info("resolved %s -> %s (IPv4 for HP UDP transport)",
                     cfg.host, self._dest_host)

        # Packet capture: open the recorder now that the server IPv4 is
        # known, so the TCP/UDP socket taps below can stamp real endpoints.
        # On a reconnect cycle (_recorder kept across handshake retries) we
        # keep the existing one; a full close() finalises it.
        if cfg.record_pcap and self._recorder is None:
            from pathlib import Path
            from ..util.pcap_recorder import PcapRecorder, local_ip_towards
            # First connect uses the path verbatim (matching the CLI's
            # expectation, truncating any stale file like tcpdump -w). Each
            # subsequent connect() in this Session — i.e. a reconnect — adds
            # a "-N" suffix so the prior session's capture (often the one
            # holding the failure that triggered the reconnect) is preserved.
            path = cfg.record_pcap
            if self._record_cycle > 0:
                p = Path(path)
                path = str(p.with_name(
                    f"{p.stem}-{self._record_cycle}{p.suffix}"))
            try:
                client_ip = local_ip_towards(self._dest_host)
                self._recorder = PcapRecorder(path, client_ip, self._dest_host)
            except Exception as e:
                log.warning("pcap recording disabled (setup failed: %s)", e)
                self._recorder = None
            self._record_cycle += 1

        # 1) Bind UDP sockets BEFORE the handshake — the session-start
        # burst lands within ~100 ms of the 0x1c answer, and the kernel
        # would drop early packets if the sockets aren't ready. Apple
        # rtcp-muxes audio + audio-RTCP on UDP ctrl_port (5900), and
        # video + video-RTCP on UDP video_port (5901). The ctrl process
        # loop demuxes audio RTP from RTCP by RTP payload-type byte.
        ctrl_port = cfg.udp_ctrl_port or cfg.port
        video_port = cfg.udp_video_port or (cfg.port + 1)
        # Bind host defaults to INADDR_ANY but can be pinned to one interface
        # (config field or ISS_UDP_BIND_HOST env) so a client can run on a host
        # already serving screen-share on 5900-5902 on a DIFFERENT interface IP
        # without a clash — the ports stay default (they're symmetric with the
        # server's), only the local interface narrows.
        bind_host = cfg.udp_bind_host or os.environ.get("ISS_UDP_BIND_HOST", "")
        self._sock_ctrl = self._bind_udp(bind_host, ctrl_port)
        self._sock_video = self._bind_udp(bind_host, video_port)
        # Tap the media sockets so SRTP RTP/RTCP (and our heartbeats /
        # firewall punches) land in the capture alongside the TCP control
        # stream. The tap is a transparent proxy — every other socket call
        # delegates to the real socket.
        if self._recorder is not None:
            self._sock_ctrl = self._recorder.wrap_udp(self._sock_ctrl)
            self._sock_video = self._recorder.wrap_udp(self._sock_video)
        # NAT diagnostic: log the bound local address pair so a user behind
        # NAT can confirm iss is listening where they expect, and we can
        # compare with the src-address-of-first-packet log later.
        try:
            log.info(
                "UDP bound: ctrl=%s:%d video=%s:%d -> sending to %s:%d/%s:%d",
                *self._sock_ctrl.getsockname()[:2],
                *self._sock_video.getsockname()[:2],
                self._dest_host, ctrl_port, self._dest_host, video_port,
            )
        except Exception:
            pass
        # On Linux the kernel silently caps SO_RCVBUF at
        # `net.core.rmem_max` (default ~208 KB on Ubuntu/Debian/Fedora),
        # which is too small for Apple's HP burst and produces gray-tile
        # artifacts via packet loss the per-tile RTP loss counter can't
        # see (drops happen below the socket layer). Only warn when
        # curtain is OFF — with curtain on, the host encodes only the
        # SkyLight virtual display sized to the advertised resolution,
        # and the resulting bursts fit comfortably in the default
        # buffer. The trouble is specifically the no-curtain path
        # where the host downscales its full physical panel.
        if not cfg.curtain:
            self._check_rmem_cap(self._sock_video)

        # 2) Apple's two-TCP warmup (TCP #1: register session with daemon).
        if cfg.warmup_tcp:
            try:
                warmup_tcp(self._dest_host, cfg.port)
            except (TimeoutError, socket.timeout, ConnectionRefusedError,
                    ConnectionResetError, OSError) as e:
                # Host-unreachable signals — don't burn another 15 s on
                # the main TCP just to land on the same error. Bail
                # immediately with a friendly message. The caller's
                # `iss.cli.main` translates this to a clean exit.
                if isinstance(e, (TimeoutError, socket.timeout)):
                    raise ConnectionError(
                        f"{cfg.host}:{cfg.port} did not respond. The "
                        f"host may be off, on a different network, "
                        f"behind a firewall, or its IP may have "
                        f"changed."
                    ) from e
                if isinstance(e, ConnectionRefusedError):
                    raise ConnectionError(
                        f"{cfg.host}:{cfg.port} actively rejected the "
                        f"connection. The host is reachable but nothing "
                        f"is listening on this port — check that Screen "
                        f"Sharing is enabled in System Settings → "
                        f"General → Sharing, or that the port number "
                        f"is right."
                    ) from e
                # Other OSErrors (e.g. "no route to host", "host down")
                # fall through with their native message.
                raise ConnectionError(
                    f"can't connect to {cfg.host}:{cfg.port}: {e}"
                ) from e
            except Exception as e:
                # Non-network warmup failure (decoded protocol mismatch,
                # etc.) — keep the legacy "best-effort warmup" behaviour.
                log.warning("warmup TCP failed (%s); continuing without it", e)

        # 3-6) Handshake → key derivation → start-burst, with full-reconnect
        # retry on a starved burst (the host accepted the canvas but never
        # started streaming). See _handshake_with_reconnect.
        #
        # Across the whole handshake we spray low-rate firewall punches (see
        # _firewall_punch_loop) so the host's media is treated as established
        # return traffic and a non-admin viewer needs no inbound rule. It
        # runs the entire time because the host can start streaming mid-
        # negotiation (during the degenerate-canvas re-query), well before
        # we'd otherwise regain control to punch.
        punch_stop = threading.Event()
        punch_thr = threading.Thread(
            target=self._firewall_punch_loop, args=(punch_stop,),
            name="iss-fw-punch", daemon=True,
        )
        punch_thr.start()
        try:
            burst = self._handshake_with_reconnect(cfg)
        finally:
            punch_stop.set()
            punch_thr.join(timeout=1.0)
        self._ssrc_to_tile = dict(burst.ssrc_to_tile)
        self._pending_groups = burst.burst_pending
        self._group_arrival = {key: time.monotonic() for key in burst.burst_pending}

        # Observed tile count from the burst SSRC group. The negotiation
        # answer's canvas_tiles field is sometimes missing/zero for smaller
        # canvases that the host encodes as 2 tiles instead of 4. The burst
        # is authoritative — len(ssrc_to_tile) is the actual stream count.
        observed_n = len(self._ssrc_to_tile)
        declared_n = self._negotiation.canvas_tiles if self._negotiation else 0
        self._observed_tile_count = observed_n
        if observed_n > 0 and declared_n and observed_n != declared_n:
            log.warning(
                "tile count MISMATCH: negotiation declared %d, burst observed %d. "
                "Frontend will use observed count to avoid bottom-half-green render.",
                declared_n, observed_n,
            )
        elif observed_n > 0 and not declared_n:
            log.info(
                "tile count: negotiation didn't report canvas_tiles; "
                "using observed=%d from burst SSRC group",
                observed_n,
            )

        # 7) Decoder init.
        from .protocol.offers import tiles_per_frame
        num_tiles = self._negotiation.canvas_tiles or tiles_per_frame()
        # Size per-tile loss counters now that num_tiles is known.
        self._lost_pkts_per_tile = [0] * num_tiles
        self._last_profile_lost_per_tile = [0] * num_tiles
        # Recovery is driven by two trusted signals — RTP sequence
        # gaps (NACK retransmits) and libavcodec's own concealment
        # log messages (handled below). The quality_gate is just a
        # FIR-pending bookkeeping object now; pixel-content heuristics
        # have been removed because they cannot distinguish real gray
        # screen content from concealment fill.
        # `ISS_FORCE_SW_HEVC=1` forces the libavcodec software HEVC
        # decoder regardless of HW availability — useful for diagnosing
        # whether a stream issue is HW-decoder-specific (vaapi/d3d11va/
        # videotoolbox quirks) and for verifying the SW fallback path
        # actually works on each platform.
        import os as _os
        import sys as _sys
        from .media import registry
        prefer_hwaccel = _os.environ.get("ISS_FORCE_SW_HEVC", "0") == "0"
        # Decoder selection. On macOS the native VideoToolbox path
        # (VTDecompressionSession, vtdecode.py) decodes Apple's stream the way
        # its own viewer does — VideoToolbox manages the DPB and conceals the
        # odd missing-ref frame instead of BLOCKING, which is what libav's
        # hwaccel glue does (errno=35 → our drain/drop recovery → cascade
        # freeze). It's the default on macOS; `ISS_DECODER=libav` forces the
        # cross-platform libav path, and `ISS_FORCE_SW_HEVC=1` forces libav
        # software (and therefore libav).
        _decoder_choice = _os.environ.get("ISS_DECODER", "auto").lower()
        # Per-codec hardware-accel preference, then registry-driven selection
        # (media/registry.py owns the capability matrix + override handling).
        if self._video_codec == "avc":
            # AVC path: the host streams H.264 4:2:0. d3d11va's H.264 decoder
            # corrupts Apple's stream at the frame_num/POC wrap (~34 s in): the
            # 4 tiles are one interleaved low-delay stream anchored on per-tile
            # long-term references, and the D3D11 decoder's reference resolution
            # collapses when poc_lsb wraps (log2_max_poc_lsb=13 → every ~34 s) —
            # all four tiles start decoding off the same picture, so the canvas
            # dissolves into four identical bands. Software libav decodes the
            # identical bitstream correctly indefinitely (verified clean past
            # the wrap), so force software for AVC on Windows. Override with
            # ISS_AVC_HWACCEL=1, or ISS_AVC_HW_REANCHOR for HW + periodic IDR.
            avc_prefer_hwaccel = prefer_hwaccel
            if self._avc_reanchor_enabled:
                # Keep hardware decode; avoid the d3d11va POC-wrap bug by
                # periodically re-anchoring with a fresh IDR (see _maybe_
                # reanchor, driven from _tx_loop). Lower latency/CPU than the
                # software fallback at the cost of a keyframe burst every N s.
                log.info("AVC: hardware decode + automatic IDR re-anchor "
                         "(every ~%d frames since IDR, d3d11va POC-wrap "
                         "workaround)", self._AVC_REANCHOR_FRAMES)
            elif (_sys.platform == "win32"
                    and _os.environ.get("ISS_AVC_HWACCEL", "0") == "0"):
                avc_prefer_hwaccel = False
                log.info("AVC: forcing software decode "
                         "(d3d11va H.264 POC-wrap corruption workaround)")
            _pf = avc_prefer_hwaccel
            _override = registry.normalize_override(_decoder_choice, "avc")
        else:
            # HEVC 4:4:4: the registry prefers native VideoToolbox on macOS and
            # the generic libav d3d11va-RExt / Intel QSV paths on Windows/Linux.
            # ISS_FORCE_SW_HEVC=1 pins the libav software path.
            _pf = prefer_hwaccel
            _override = ("libav-hevc444" if not prefer_hwaccel
                         else registry.normalize_override(_decoder_choice, "hevc"))

        _spec, self._decoder = registry.build_best(
            self._video_codec, override=_override, num_tiles=num_tiles,
            enable_quality_gate=True, on_frame_published=self._on_frame_published,
            prefer_hwaccel=_pf,
        )
        if self._decoder is None:
            raise RuntimeError(
                f"no usable decoder for codec {self._video_codec!r} "
                f"(override={_override!r})")
        if not prefer_hwaccel:
            log.info("ISS_FORCE_SW_HEVC=1: HW decoders disabled")
        self._decoder.set_params(burst.vps, burst.sps, burst.all_pps)
        self._decoder.start()
        # Mark the start of the burst-feed window so the DPB-break
        # fast-path stays graced through it; libav fires "Could not
        # find ref" transiently as the cold context catches up.
        self._last_decoder_restart_t = time.monotonic()
        self._dpb_error_window.clear()
        self._decoder.feed_burst(burst.tile_nalus)
        # Install a libavcodec log callback that turns concealment
        # log messages into FIR requests. Idempotent — subsequent
        # connects keep the same callback hooked.
        self._install_libav_log_callback()

        # 8) Audio decoder.
        if cfg.audio:
            self._aac = make_aac_eld_decoder(prefer=cfg.aac_backend)

        # 9) Input controller bound to TCP control + cipher.
        #
        # We clamp pointer events to *canvas* dims, not ServerInit's
        # reported dims. ServerInit can report a composite that's
        # smaller than the canvas (e.g. dual-display 2940×1912 when
        # user2's freshly-spawned vdisplay is encoded at 3840×2160) —
        # clamping pointer-y to 1911 then makes the bottom ~12% of
        # the canvas unreachable. The desktop frontend's `to_canvas`
        # already produces coords in canvas space, so we just pass
        # canvas dims through.
        #
        # When alt-session is active, the controller wraps mouse
        # events in msg 0x10 (HandleEncryptedEventMessage) so the
        # daemon's uid-gated msg 0x05 PointerEvent path doesn't
        # silently drop them — see input.py for the gate analysis.
        self._input = InputController(
            self._negotiation.sock,
            self._negotiation.cipher,
            server_width=self._negotiation.canvas_width,
            server_height=self._negotiation.canvas_height,
            alt_session=cfg.alt_session,
        )

        # 9b) Inbound clipboard sync (host → local OS). Enable the daemon's
        # change-notification subscription (msg 0x15) and prime the local
        # clipboard with the current host contents (msg 0x0b). Opt out by
        # setting ISS_CLIP=0.
        if os.environ.get("ISS_CLIP", "1") != "0":
            try:
                self._input.send_clipboard_enable()
                self._input.send_clipboard_fetch()
            except Exception as e:  # pragma: no cover
                log.debug("clipboard enable/prime failed: %s", e)
        self._clipboard_reassembler = self._clipboard_reassembler.__class__()

        # 10) Spawn rx + tx threads.
        self._stop_evt.clear()
        self._fresh_evt.clear()
        self._last_publish_t = time.monotonic()
        self._spawn_threads()
        self._spawn_outbound_clipboard_poller()

        # 10b) Open the local control socket (TUI / external monitor) if
        # the caller asked for one. Set the cached `hello` payload now so
        # any client connecting later gets the static session header
        # immediately, before the first periodic snapshot.
        if cfg.control_socket:
            try:
                self._connect_wall_ts = time.time()
                self._control = ControlServer(
                    cfg.control_socket, on_command=self._handle_ctrl_command,
                )
                self._control.start()
                self._control.set_hello(
                    host=cfg.host,
                    dest_host=self._dest_host,
                    port=cfg.port,
                    server={"w": self._negotiation.server_width,
                            "h": self._negotiation.server_height},
                    canvas={"w": self._negotiation.canvas_width,
                            "h": self._negotiation.canvas_height,
                            "tiles": num_tiles},
                    decoder=self._decoder._hw_name or "software",
                    audio_enabled=cfg.audio,
                    curtain=cfg.curtain,
                    share_console=cfg.share_console,
                    alt_session=cfg.alt_session,
                    advertise=(
                        {"w": cfg.advertise.width, "h": cfg.advertise.height,
                         "hidpi": cfg.advertise.hidpi_scale}
                        if cfg.advertise else None
                    ),
                    started_at=self._connect_wall_ts,
                    iss_version=_iss_version,
                )
                self._control.publish_event(
                    "connected", canvas={"w": self._negotiation.canvas_width,
                                         "h": self._negotiation.canvas_height},
                )
            except OSError as e:
                log.warning("control socket setup failed (%s); continuing without it", e)
                self._control = None

        # 11) FIR any tile that didn't pick up a real IDR in the burst.
        # Apple's HP encoder sometimes only emits an IDR for tile 0 in
        # the opening packets and expects the client to ask for IDRs for
        # the others; without that, tiles 1-3 never decode and show as
        # gray fill. The cost is one extra round-trip per missing tile.
        missing_idr = [ti for ti in range(num_tiles) if ti not in burst.last_idr]
        if missing_idr:
            log.debug("burst missed IDRs for tiles %s; sending FIR", missing_idr)
            for ti in missing_idr:
                self._send_fir_for_tile(ti)

    def _handshake_with_reconnect(self, cfg) -> InitialBurst:
        """Negotiate + gather the start burst, re-handshaking from scratch on
        a starved burst.

        By here the control path is pinned to IPv4 (see _resolve_host_ipv4),
        so a starved burst is genuinely host-side: screensharingd accepted
        the canvas over TCP but its agent never started sending RTP. An
        in-session re-poke can't unstick that -- the wedge lives in the
        host-side ScreensharingAgent. The reliable recovery is a fresh
        handshake: closing the TCP drops the wedged agent so a clean one
        spawns (the manual "just run it again" fix, automated). Only after
        `_BURST_RETRY_ATTEMPTS` do we surface an error."""
        last_starved: Optional[BurstStarved] = None
        for attempt in range(_BURST_RETRY_ATTEMPTS):
            try:
                return self._negotiate_and_burst(cfg)
            except BurstStarved as e:
                last_starved = e
                if attempt + 1 >= _BURST_RETRY_ATTEMPTS:
                    break
                log.warning(
                    "burst starved (%s, %d pkts); full reconnect %d/%d",
                    e.reason, e.packets, attempt + 1, _BURST_RETRY_ATTEMPTS,
                )
                self._teardown_negotiation_tcp()
                time.sleep(_BURST_RETRY_SLEEP_S)

        cw = self._negotiation.canvas_width if self._negotiation else 0
        ch = self._negotiation.canvas_height if self._negotiation else 0
        pkts = last_starved.packets if last_starved else 0
        raise RuntimeError(
            f"HP stream never started: {pkts} HEVC video packets arrived across "
            f"{_BURST_RETRY_ATTEMPTS} fresh handshakes. The host accepted a "
            f"{cw}x{ch} encoder canvas over IPv4, but screensharingd never sent "
            f"any RTP -- a host-side stall. On the Mac, toggle Screen Sharing "
            f"off/on in System Settings (clears a wedged agent), then reconnect; "
            f"--no-curtain (mirror the physical screen) can also help."
        )

    def _negotiate_and_burst(self, cfg) -> InitialBurst:
        """One full handshake attempt: build fresh offers, negotiate, derive
        SRTP keys, and drain the session-start burst. Raises `BurstStarved`
        if the host accepted the canvas but never started sending RTP -- the
        caller tears down the TCP and re-handshakes. Fresh offers/SSRCs each
        attempt so the host treats us as a brand-new viewer instead of
        matching the wedged prior session.

        Apple's daemon depends on the audio section being present in the 0x1c
        (without it the encoder-canvas reply degenerates and no burst
        arrives), so we always send a valid audio offer; `cfg.audio=False`
        only skips local decode + playback."""
        video_offer, audio_offer = create_offers()
        # Stash for mid-session 0x1c re-offers (dynamic resolution).
        self._video_offer = video_offer
        self._audio_offer = audio_offer
        self._our_video_ssrc = extract_offer_ssrc(video_offer, is_video=True)
        self._our_audio_ssrc = extract_offer_ssrc(audio_offer, is_video=False)
        log.info(
            "our SSRCs: video=0x%08x audio=0x%08x (decode=%s)",
            self._our_video_ssrc or 0, self._our_audio_ssrc or 0,
            "on" if cfg.audio else "off",
        )

        self._negotiation = connect_and_negotiate(
            self._dest_host, cfg.port, cfg.username, cfg.password,
            auth_mode=cfg.auth_mode,
            advertise=cfg.advertise,
            hdr=cfg.hdr,
            curtain=cfg.curtain,
            audio_offer=audio_offer,
            video_offer=video_offer,
            share_console=cfg.share_console,
            alt_session=cfg.alt_session,
            recorder=self._recorder,
        )

        # Persist the post-auth transport key next to the pcap (sibling
        # `.key` file) so the recorded session decodes without a wrapper.
        if self._recorder is not None:
            self._recorder.write_key(self._negotiation.ecb_key.hex())

        keys = self._negotiation.keys
        self._video_decryptor = self._negotiation.video_decryptor
        self._audio_decryptor = SRTPDecryptor.from_blob(keys.audio_key_s)
        self._srtcp_dec = SRTCPDecryptor.from_blob(keys.video_key_s)
        self._srtcp_enc = SRTCPEncryptor.from_blob(keys.video_key_v)
        if self._our_audio_ssrc is not None:
            self._audio_encryptor = SRTPEncryptor.from_blob(
                keys.audio_key_v, self._our_audio_ssrc,
            )

        # Drain the start burst. On starvation we deliberately do NOT poke
        # the host with an FBU here -- a FramebufferUpdateRequest during the
        # burst window makes the daemon answer with a tiny incremental update
        # that lacks VPS/SPS/PPS (see burst.py), which just looks like a
        # different starvation. Let BurstStarved propagate so the caller does
        # a clean full reconnect, which is what actually restarts the stream.
        burst_buf: list[bytes] = []
        # H.264 needs a longer drain + higher packet floor than HEVC: its tiles
        # are independent slice chains and the burst must gather enough to seed
        # the shared decoder context before the streaming loop starts.
        _is_avc = self._video_codec == "avc"
        self._drain_socket_into(
            self._sock_video, burst_buf, max_seconds=4.0 if _is_avc else 2.0)
        return gather_initial_burst(
            burst_buf, self._video_decryptor, quality_tier=cfg.quality_tier,
            codec=self._video_codec,
            min_packets=400 if _is_avc else 100,
        )

    def _teardown_negotiation_tcp(self) -> None:
        """Drop the current negotiation TCP so the host releases the wedged
        ScreensharingAgent before we re-handshake. RST (SO_LINGER 0) rather
        than a lingering FIN so the daemon tears down fast -- mirrors how
        Apple's own client closes (see project notes on no-goodbye)."""
        neg = self._negotiation
        if neg is None:
            return
        try:
            neg.sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0),
            )
        except OSError:
            pass
        try:
            neg.sock.close()
        except OSError:
            pass
        self._negotiation = None

    def _handle_ctrl_command(self, cmd: dict) -> None:
        """Dispatch a JSON command received from a control-socket client
        (TUI, monitor). Runs on the ControlServer's per-client read thread
        so do NOT block here -- if a future command needs heavy work,
        offload it. Best effort: unknown actions are logged and ignored."""
        action = cmd.get("action")
        if action == "fir":
            # Force an IDR refresh on every tile. Same effect as the
            # TUI's force-IDR ('f') action.
            try:
                self.request_fir(None)
            except Exception:
                log.exception("ctrl fir failed")
            return
        if action == "set":
            key = cmd.get("key")
            value = cmd.get("value")
            # Audio toggle: unhook (or rehook) the sink callback. The
            # network audio still arrives; we just stop feeding it to
            # PortAudio. Cheap, reversible.
            if key == "audio":
                self._audio_user_mute = not bool(value)
                log.info("ctrl: audio %s",
                         "muted" if self._audio_user_mute else "unmuted")
                return
            log.info("ctrl: unsupported set key %r", key)
            return
        if action == "quit":
            log.info("ctrl: quit requested by client")
            # close() is safe from any thread; it just sets _stop_evt
            # and joins. Don't block here -- spawn a tiny shutdown
            # thread so the read loop can return cleanly first.
            threading.Thread(target=self.close, name="iss-ctrl-shutdown",
                             daemon=True).start()
            return
        log.info("ctrl: unknown action %r", action)

    def _spawn_outbound_clipboard_poller(self) -> None:
        """Background thread that polls the local OS clipboard and ships
        new text to the host as msg 0x1f. Frontend-agnostic — works in
        headless, desktop, browser, anything.

        Echo guard: skips sending text we just received from the host
        (Session._last_received_clipboard_text), preventing a feedback
        loop with the inbound 0x1f path.

        Off-switch: ISS_CLIP=0 disables the entire clipboard subsystem
        (also prevents the inbound prime fetch above). Also a no-op when
        the platform clipboard isn't reachable (e.g. Linux without
        xclip/wl-clipboard installed) — `local_clipboard.available()`
        already logged an actionable hint at session startup."""
        if os.environ.get("ISS_CLIP", "1") == "0":
            return
        if self._input is None:
            return
        from . import local_clipboard
        if not local_clipboard.available():
            return

        def _norm(s: Optional[str]) -> str:
            """Normalize for echo-guard comparison: collapse CRLF→LF and
            strip trailing whitespace. Windows clipboard tends to keep a
            trailing CRLF; macOS doesn't; Linux varies. Without this
            normalization a cross-platform round-trip would ping-pong
            forever between iss and the daemon."""
            return (s or "").replace("\r\n", "\n").rstrip()

        last_seen = _norm(local_clipboard.read_text())

        def _loop() -> None:
            nonlocal last_seen
            while not self._stop_evt.is_set():
                self._stop_evt.wait(1.0)
                if self._stop_evt.is_set():
                    return
                cur = _norm(local_clipboard.read_text())
                if not cur or cur == last_seen:
                    continue
                last_seen = cur
                # Don't echo back text the host just pushed to us.
                if cur == _norm(self._last_received_clipboard_text):
                    continue
                # Send LF-normalized text — Apple's NSPasteboard expects
                # Unix line endings, and Mac downgrades to latin-1 if the
                # bytes look non-utf8-clean.
                try:
                    self._input.cut_text(cur)
                    log.info("clipboard send: %d chars (preview=%r)",
                             len(cur), cur[:40].replace("\n", "\\n"))
                except Exception as e:
                    log.debug("outbound cut_text send failed: %s", e)

        t = threading.Thread(target=_loop, name="iss-clip-out", daemon=True)
        t.start()
        self._threads.append(t)
        log.debug("outbound clipboard poller started")

    def _teardown(self) -> None:
        # Drop ourselves from the libav-handler's active-sessions set
        # so concealment events stop routing here and the Session can
        # be garbage-collected on reconnect cycles.
        active = getattr(Session, "_active_sessions", None)
        if active is not None:
            active.discard(self)

        # Tell any subscribed TUI clients we're going down, then close
        # the control socket. Send the event first so the disconnect
        # reason has somewhere to land before sockets shut.
        if self._control is not None:
            try:
                self._control.publish_event("disconnected")
            except OSError:
                pass
            self._control.close()
            self._control = None

        self._stop_evt.set()
        for t in self._threads:
            if t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2.0)
        self._threads = []

        if self._input is not None:
            self._input.close()
            self._input = None
        if self._decoder is not None:
            self._decoder.close()
            self._decoder = None
        if self._aac is not None:
            self._aac.close()
            self._aac = None

        for sock_attr in ("_sock_ctrl", "_sock_video"):
            sock = getattr(self, sock_attr, None)
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                setattr(self, sock_attr, None)

        if self._negotiation is not None:
            try:
                # SO_LINGER {1, 0} → RST instead of FIN, so the daemon's
                # encoder shuts down promptly (~ms vs 75 s FIN timeout).
                self._negotiation.sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
                self._negotiation.sock.close()
            except OSError:
                pass
            self._negotiation = None

        # Finalise the packet capture (flush + close the file) and print a
        # ready-to-run dissector command. Done after the sockets are shut so
        # any last bytes (RST teardown excluded — that's below the socket
        # API) are already recorded.
        if self._recorder is not None:
            path = self._recorder.path
            hint = self._recorder.dissector_hint()
            pkts, nbytes = self._recorder.close()  # drains queue, returns counts
            if path:
                log.info(
                    "pcap saved: %s (%d packets, %d bytes)\ndecode with:\n%s",
                    path, pkts, nbytes, hint,
                )
            self._recorder = None

        self._video_decryptor = None
        self._audio_decryptor = None
        self._audio_encryptor = None
        self._srtcp_dec = None
        self._srtcp_enc = None
        self._ssrc_to_tile = {}
        self._pending_groups = {}
        self._group_arrival = {}
        self._recently_flushed = {}
        self._max_seq = {}
        self._roc = {}
        self._nack_pending = defaultdict(set)
        self._server_sr = {}
        self._runtime_canvas_w = 0
        self._runtime_canvas_h = 0
        self._runtime_scaled_w = 0
        self._runtime_scaled_h = 0
        self._needs_post_layout_fir = False
        self._needs_param_harvest = False
        self._harvest_vps = None
        self._harvest_sps = None
        self._harvest_pps = {}

    # ── thread spawning ──────────────────────────────────────────────

    def _spawn_threads(self) -> None:
        targets: list[tuple[str, Callable]] = [
            ("iss-video-drain", self._video_drain_loop),
            ("iss-video-process", self._video_process_loop),
            ("iss-ctrl-drain", self._ctrl_drain_loop),
            ("iss-ctrl-process", self._ctrl_process_loop),
            ("iss-tcp-rx", self._tcp_rx_loop),
            ("iss-tx", self._tx_loop),
        ]
        # Note: there's no separate iss-audio-rx thread because Apple
        # rtcp-muxes audio onto _sock_ctrl (UDP 5900); the ctrl process
        # loop routes audio RTP to _handle_audio_rtp.
        for name, target in targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    # ── decoder publish hook ─────────────────────────────────────────

    def _on_frame_published(self, tile_idx: int) -> None:
        """Called from a HevcDecoder worker thread after a frame is
        published. Wakes anyone in `wait_for_fresh_tile`."""
        self._last_publish_t = time.monotonic()
        self._fresh_evt.set()
        self._send_ltr_ack(tile_idx)

    def _send_ltr_ack(self, tile_idx: int) -> None:
        """Acknowledge the last cleanly-decoded base-tile frame so Apple's
        encoder can use it as a long-term reference (single-RTT recovery
        instead of a full IDR on loss).

        The ack must carry the frame's **DONL** (HEVC Decoding Order Number).
        The host reads a big-endian u32 at the APP packet's offset 12 and
        scans a fixed-size ring for an *exact* match, mapping it to an encoder
        reference token. That ring is keyed on the per-frame DONL the encoder
        stamps — which is exactly the DONL carried in every payload header
        (same value space, +1 per frame). A value from any other counter
        (e.g. the HEVC POC) misses the ring entirely and is a silent no-op,
        leaving the encoder to fall back to full-IDR recovery (the persistent
        gray). See nalu.first_donl.

        We ack tile 0 only: it's the anchor SSRC that carries IDRs, and one
        stream's ack rate matches the host's own observed rate.
        """
        if not self._ltr_enabled or tile_idx != 0:
            return
        dec = self._decoder
        if dec is None:
            return
        donl = dec.last_clean_donl[0] if dec.last_clean_donl else None
        if donl is None:
            return
        # DONL is monotonic per tile but 16-bit (wraps ~every 18 min at
        # 60 fps); ack on any change rather than strict ">" so acks resume
        # after a wrap instead of stalling until the counter climbs back.
        if donl == self._ltr_last_acked:
            return
        self._ltr_last_acked = donl
        enc = self._srtcp_enc
        ssrc = self._our_video_ssrc
        # PT=204 LTR-acks go on the video RTCP-mux port, not the ctrl port.
        sock = self._sock_video
        if enc is None or ssrc is None or sock is None:
            return
        try:
            pkt = build_rtcp_app_ltrp(ssrc, donl)
            sock.sendto(enc.protect(pkt), (self._dest_host, self._video_dest_port))
            self._ltr_acks_sent += 1
            if self._log_ltr_acks:
                log.debug("LTR ack sent: DONL=%d (tile 0)", donl)
        except OSError as e:
            log.debug("LTR ack send failed: %s", e)

    def _dpb_trace(self, msg: str) -> None:
        """ISS_DPB_TRACE=1 diagnostic. Classify a libav 'Could not find
        ref with POC N' as EVICTED (we fed POC N but libav dropped it
        from its DPB → the decoder retains too few frames) vs GAP (we
        never fed N → transport loss / pre-IDR drop / reorder). The RPS
        tracker's `_seen_pocs` never evicts, so membership == 'we fed
        this exact POC at least once'. `lsb_hit` guards against an
        MSB-derivation mismatch between libav and our tracker: a POC we
        fed under a different MSB still shows the same poc_lsb."""
        dec = self._decoder
        if dec is None:
            return
        low = msg.lower()
        i = low.rfind("poc")
        if i < 0:
            return
        digits = ""
        for ch in msg[i + 3:]:
            if ch.isdigit() or (ch == "-" and not digits):
                digits += ch
            elif digits:
                break
        try:
            poc = int(digits)
        except ValueError:
            return
        try:
            seen = frozenset(dec._rps_tracker._seen_pocs)
        except Exception:
            return
        in_seen = poc in seen
        lsb = poc % 2048
        lsb_hit = in_seen or any((p % 2048) == lsb for p in seen)
        mx = max(seen) if seen else -1
        mn = min(seen) if seen else -1
        log.warning(
            "DPBTRACE poc=%d %s dist_from_max=%d lsb_hit=%s "
            "seen_count=%d seen_range=[%d..%d]",
            poc,
            "EVICTED(fed-then-dropped)" if in_seen else "GAP(never-fed)",
            (mx - poc) if mx >= 0 else -1,
            lsb_hit, len(seen), mn, mx,
        )

    # ── libav log → decoder-concealment hook ──────────────────────────
    # PyAV forwards libav log messages into the Python `logging`
    # module under loggers named like `libav.h265`, `libav.hevc`,
    # `libav.h264`, etc. We attach a Handler to the `libav` parent
    # logger and filter for concealment-related messages, then call
    # mark_decode_error on every tile (the message has no tile
    # attribution since the codec context is shared). This is a
    # stopgap; the cleaner solution is HEVC slice-header RPS parsing
    # (we know which POCs a slice references and whether we have
    # them, before we even feed the slice — pre-decode error
    # detection, decoder-agnostic, no string matching).
    def _install_libav_log_callback(self) -> None:
        # The handler itself is process-global (installed at most
        # once), but every Session has to register itself in the
        # active set so libav events from its decoder route back to
        # it. Otherwise reconnect cycles silently lose concealment
        # signals — the second Session never gets added if registration
        # was nested under the install gate.
        if not hasattr(Session, "_active_sessions"):
            Session._active_sessions: set = set()
        Session._active_sessions.add(self)

        if getattr(Session, "_libav_log_installed", False):
            return  # already installed; just registering this Session is enough
        try:
            import av.logging as _avlog  # type: ignore
        except ImportError:
            log.debug("av.logging not available; no decoder-error hook")
            Session._libav_log_installed = True
            return
        # Raise PyAV's libav verbosity so concealment WARNINGs reach
        # the Python logger.
        # Surface EVERY libav message reaching the handler (before the
        # keyword whitelist below drops it), so a HW decoder's own error
        # reports aren't silently discarded — this is how we find out what
        # d3d11va actually emits when it goes gray. Default ON for now (set
        # ISS_LOG_LIBAV=0 to silence). Kept at WARNING level: the messages
        # that matter (decode / hwaccel errors) are ERROR-class, so it stays
        # quiet during normal playback and only fires on real trouble.
        _log_all_libav = os.environ.get("ISS_LOG_LIBAV", "1") != "0"
        try:
            _avlog.set_libav_level(_avlog.WARNING)
            _avlog.set_level(_avlog.WARNING)
        except Exception:
            pass

        concealment_keywords = (
            "could not find ref",
            "concealing",
            "non-existing pps",
            "no frame!",
            "missing reference",
            "decode_slice_header error",
            "skipping bitstream",
        )

        class _LibavConcealmentHandler(logging.Handler):
            def emit(self_h, record):  # noqa: N805
                try:
                    raw = record.getMessage().strip()
                except Exception:
                    return
                msg = raw.lower()
                if _log_all_libav:
                    # Diagnostic: log it ALL, before the whitelist drops it.
                    log.info("LIBAV_RAW[%s] %s", record.levelname, raw)
                if not any(kw in msg for kw in concealment_keywords):
                    return
                for sess in tuple(getattr(Session, "_active_sessions", ())):
                    try:
                        sess._on_libav_concealment(record.getMessage().strip())
                    except Exception:
                        pass

        handler = _LibavConcealmentHandler(level=logging.WARNING)
        logging.getLogger("libav").addHandler(handler)

        # libav emits one ERROR per concealment event ("Could not find ref
        # with POC N") for every routine packet-loss recovery. iss already
        # handles those via the concealment handler above and surfaces a
        # single WARN ("DPB break: N events"), so the per-event ERRORs are
        # pure noise to the user. Drop them from the root handlers
        # (stderr / log file / TUI panel) without touching the libav
        # logger itself, so our concealment-detection handler still fires.
        _conceal_patterns = (
            "could not find ref",
            "non-existing pps",
            "concealing",
        )

        class _SuppressLibavConcealment(logging.Filter):
            def filter(self_f, record):  # noqa: N805
                if not record.name.startswith("libav"):
                    return True
                try:
                    msg = record.getMessage().lower()
                except Exception:
                    return True
                return not any(p in msg for p in _conceal_patterns)

        _suppress = _SuppressLibavConcealment()
        for h in logging.getLogger().handlers:
            h.addFilter(_suppress)

        Session._libav_log_installed = True
        log.info("libav concealment-log handler installed (root handlers filter the noise)")

    # Sliding-window thresholds for "Could not find ref" detection.
    # Trigger the fast FIR path on the FIRST "Could not find ref" event
    # after the post-restart grace expires. Apple's HEVC encoder uses
    # ~7 short-term refs per P-slice so a single bad slice produces ~7
    # events; the older threshold of 12 required >=2 bad slices in the
    # same second and skipped recovery for single-slice corruption,
    # leaving the user with persistent gray artifacts that only a manual
    # force-IDR cleared. The post-restart grace below + the fast cooldown +
    # mark_decode_error's own per-tile cooldown all cap the FIR rate,
    # so a single bad slice no longer produces a FIR storm.
    _DPB_ERR_WINDOW_S: float = 1.0
    _DPB_ERR_THRESHOLD: int = 1
    # Don't re-fire the fast path within this cooldown. Apple needs
    # ~100-300 ms to deliver the IDR our previous FIR triggered;
    # 1.0 s leaves comfortable headroom for the IDR to land + the
    # decoder to digest + a couple of clean frames to flow before
    # we'd consider the session "still broken" enough to FIR again.
    # During sustained packet loss this means up to ~1 s of gray
    # before the next FIR, but avoids the FIR storm that 0.5 s
    # produced (3+ FIRs per second of loss generating wasteful IDRs
    # the daemon also can't deliver cleanly through the same loss).
    _DPB_FAST_COOLDOWN_S: float = 1.0
    # Suppress the fast path for this long after every decoder reset.
    # Set to 0 because the cooldown (`_DPB_FAST_COOLDOWN_S`) already
    # bounds FIR rate to ~1/s, and a too-generous grace blocked
    # recovery during the most common gray-screen failure mode: an
    # initial-burst FIR that didn't yield a usable IDR for tiles 1-3,
    # producing a "Could not find ref" flood inside the first second
    # of the session. The original 3 s grace existed to ignore Apple's
    # P-frame tail after an SSRC-adoption restart — those events were
    # cosmetic, but the FIR they triggered was harmless too (Apple
    # answers with a fresh IDR which the decoder uses).
    _DPB_RESTART_GRACE_S: float = 0.0
    # Minimum interval between FIRs targeting the same tile, regardless
    # of which code path requests one. Apple responds to each FIR with
    # a fresh IDR; firing several in this window yields multiple IDRs
    # with the same POC, the decoder rejects all but the first as
    # duplicates, and the resulting state break produces persistent
    # gray on the affected tile. 250 ms is shorter than Apple's IDR
    # round-trip (~100-300 ms) plus a safety margin, so legitimate
    # back-to-back retries still wait long enough to see whether the
    # first FIR worked.
    _FIR_MIN_INTERVAL_S: float = 0.25
    # Persistent-DPB-break escalation. When a "Could not find ref" storm
    # survives this many targeted fast-path FIRs (each ~1 s apart), escalate
    # to force-IDR ALL tiles (request_fir(None) = the TUI 'f' action), which
    # users confirm reliably clears the persistent gray. ~3 s of continuous
    # FIR-resistant corruption before escalating — fast enough to beat a
    # manual click, slow enough not to blanket-FIR on transient single-frame
    # loss (which the targeted FIR + natural IDR already handles).
    _DPB_FORCEALL_AFTER_FIRS: int = 3
    # A gap this long with no "Could not find ref" means the previous storm
    # cleared (real recovery); reset the escalation counter.
    _DPB_STORM_RESET_S: float = 3.0
    # How long the link must be loss-FREE before a decoder wedge is treated
    # as a genuine (lossless) saturation wedge eligible for a codec restart.
    # A loss event takes up to ~4 s to surface as a `gap > threshold` wedge,
    # so a short window misclassifies loss-rooted broken-ref wedges as
    # "lossless" and restarts them — which wipes the shared DPB and turns
    # one wedge into a multi-second restart cascade. 8 s keeps loss-rooted
    # wedges on the no-flush FIR path; genuine lossless wedges still restart.
    _SATURATION_LOSS_FREE_WINDOW_S: float = 8.0
    # Minimum publish-gap before a lossless wedge earns a codec restart. The
    # no-flush + FIR path recovers the common periodic no-loss wedge in ~3-6 s
    # via Apple's FIR->IDR. 8 s lets the native-aligned FIR path finish first;
    # restart only fires for a decoder that is GENUINELY stuck (no recovery).
    _SATURATION_RESTART_GAP_S: float = 8.0

    def _on_libav_concealment(self, msg: str) -> None:
        """Two response modes for libav's "Could not find ref" log:

          1. **Sustained DPB break** — when ≥`_DPB_ERR_THRESHOLD`
             events accumulate inside `_DPB_ERR_WINDOW_S`, fire FIR
             immediately bypassing the steady-state guard. The guard
             below was suppressing recovery here because frames ARE
             flowing during a "could not find ref" cascade — they're
             just corrupt. Apple responds with fresh IDRs ~100-300 ms
             later and the decoder recovers naturally; we deliberately
             do NOT call `flush_buffers()` (that wedges vaapi and
             d3d11va, turning recovery into a permanent freeze).

          2. **Soft concealment** — anything else (unrelated libav
             warnings during healthy decode) takes the original
             rate-limited path with the steady-state guard, so a
             cosmetic warning during clean playback doesn't trigger
             a FIR cascade.
        """
        if self._decoder is None:
            return
        now = time.monotonic()

        # Track sustained "Could not find ref" events. Threshold breach
        # means real corruption that won't self-heal before the next
        # natural IDR cycle.
        if "could not find ref" in msg.lower():
            self._last_concealment_msg = msg
            # Storm tracking for reconnect escalation: a long gap since the
            # last ref-miss means the prior storm cleared → reset the count.
            if now - self._last_dpb_error_t > self._DPB_STORM_RESET_S:
                self._dpb_fir_count = 0
            self._last_dpb_error_t = now
            if os.environ.get("ISS_DPB_TRACE") == "1":
                self._dpb_trace(msg)
            events = self._dpb_error_window
            events.append(now)
            cutoff = now - self._DPB_ERR_WINDOW_S
            while events and events[0] < cutoff:
                events.popleft()
            # Suppress fast path during post-restart grace — burst
            # tail produces transient ref-misses that aren't a real
            # DPB break.
            in_grace = (now - self._last_decoder_restart_t
                        < self._DPB_RESTART_GRACE_S)
            last_fast = getattr(self, "_last_dpb_fast_recovery_t", 0.0)
            if (not in_grace
                    and len(events) >= self._DPB_ERR_THRESHOLD
                    and now - last_fast >= self._DPB_FAST_COOLDOWN_S):
                self._last_dpb_fast_recovery_t = now
                events.clear()
                self._dpb_fir_count += 1
                if self._dpb_fir_count >= self._DPB_FORCEALL_AFTER_FIRS:
                    # The per-tile FIR above isn't clearing the storm. Escalate
                    # to the SAME action the TUI 'f' key does — request_fir(None)
                    # → gate.force_keyframe_all() + FIR every tile. Users report
                    # this manual Force-IDR ALWAYS clears the persistent gray,
                    # where the targeted auto-FIR (tile 0 / bad tiles only)
                    # doesn't. We hold it back to the persistent case so we
                    # don't blanket-FIR all tiles on every transient loss.
                    log.warning(
                        "DPB break persisted through %d FIRs — escalating to "
                        "force-IDR ALL tiles (= TUI 'f' action)",
                        self._dpb_fir_count,
                    )
                    self._dpb_fir_count = 0
                    self.request_fir(None)
                else:
                    log.warning(
                        "DPB break: %d 'Could not find ref' events in %.1fs "
                        "— FIR for fresh IDR (attempt %d/%d before force-all)",
                        self._DPB_ERR_THRESHOLD, self._DPB_ERR_WINDOW_S,
                        self._dpb_fir_count, self._DPB_FORCEALL_AFTER_FIRS,
                    )
                # Don't blanket-FIR all tiles. The libav log line
                # doesn't tell us which tile produced the ref-miss,
                # but the per-tile post-decode `had_decode_error`
                # path that fires on every bad frame already flags
                # the actual erroring tiles in the gate. Triggering
                # an all-tiles FIR here adds tiles that aren't
                # actually broken to `keyframe_required`, where they
                # then sit waiting for IDRs that Apple has no reason
                # to send (chronic spam). Only escalate tiles that
                # the gate already recognizes as bad — nudge them
                # by re-marking, which resets their re-arm timer.
                if self._decoder is not None:
                    # Clamp to the gate's real size (see request_fir): num_tiles
                    # can outrun the restart()ed decoder's gate after an SSRC
                    # adoption, and indexing past _states would IndexError here.
                    states = self._decoder._gate._states
                    bad_tiles = [
                        ti for ti in range(min(self.num_tiles, len(states)))
                        if states[ti].bad_streak > 0
                    ]
                    for ti in bad_tiles:
                        # Sourced from the libav "Could not find ref" log →
                        # reliable=False: this is the silent-gray-capable
                        # signal that must hold the recovery quiet-window open.
                        self._decoder._gate.mark_decode_error(ti, reliable=False)
                return

        # Soft-concealment path with steady-state + rate-limit guards.
        if (self._last_publish_t > 0.0
                and now - self._last_publish_t < 0.5):
            return
        last = getattr(self, "_last_libav_fir_t", 0.0)
        if now - last < 2.0:
            return
        self._last_libav_fir_t = now
        log.warning("libav decoder error: %s", msg[:120])
        # Soft-concealment, also sourced from the libav log → reliable=False.
        self._decoder._gate.mark_decode_error(0, reliable=False)

    # ── video RX: drain + process split ──────────────────────────────
    #
    # Linux's default UDP socket buffer is small (208 KB on Ubuntu/Fedora
    # with no rmem_max bump) and bursts of HP video at 60 fps overflow
    # it before a single thread can recvfrom + decrypt + reassemble each
    # packet — kernel reports the drops as `RcvbufErrors` in
    # /proc/net/snmp. Splitting recvfrom (drain) from decrypt+dispatch
    # (process) lets the drain thread empty the kernel buffer quickly
    # while the process thread does the heavier work in parallel.

    def _video_drain_loop(self) -> None:
        """Tight inner loop: recvfrom → enqueue. No processing here so
        the kernel UDP buffer empties as fast as the GIL allows. If the
        process thread falls behind and the queue fills, we drop at the
        app layer (same end result as kernel drop, but observable).

        `ISS_DROP_PCT=N` simulates N% inbound packet loss for regression
        testing of the recovery paths; defaults to 0 (no synthetic drop)."""
        import random
        drop_pct = float(os.environ.get("ISS_DROP_PCT", "0"))
        # Burst-loss injector (0 = off): drop one frame's worth of packets
        # every N seconds — see the burst block below.
        self._burst_every_s = float(os.environ.get("ISS_DROP_BURST_EVERY_S", "0"))
        # Hold off synthetic loss until the stream has established, so the
        # initial burst + first IDR aren't dropped (which just stalls
        # startup and tells us nothing about steady-state recovery).
        drop_after_s = float(os.environ.get("ISS_DROP_AFTER_S", "8"))
        loop_start = time.monotonic()
        sock = self._sock_video
        if sock is None:
            return
        sock.settimeout(_UDP_RECV_TIMEOUT_S)
        q = self._video_q
        loss_count = 0
        # NAT diagnostic: track unique source addresses seen on this socket.
        # When iss runs behind a NAT, the host may send streams to the wrong
        # external port; iss receives nothing on the bound port. Logging the
        # source addresses we DO see (if any) reveals whether traffic is
        # arriving from an unexpected port (NAT translation evidence).
        src_seen: dict[tuple, int] = {}
        while not self._stop_evt.is_set():
            try:
                pkt, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            self._rx_pkts_video += 1
            self._rx_bytes_video += len(pkt)
            prev = src_seen.get(_addr, 0)
            src_seen[_addr] = prev + 1
            if prev == 0:
                log.info("video UDP: first packet from src=%s:%d", _addr[0], _addr[1])
            if (drop_pct > 0
                    and time.monotonic() - loop_start > drop_after_s
                    and random.random() * 100 < drop_pct):
                loss_count += 1
                if loss_count % 100 == 1:
                    log.info("ISS_DROP_PCT=%.1f%% — synthetic loss count=%d", drop_pct, loss_count)
                continue
            # Burst-loss model: every `_burst_every_s` seconds, drop ALL video
            # packets for a short window (~one frame). This is the loss
            # pattern LTR-anchored recovery is meant to fix — a clean single
            # missing frame — unlike continuous random loss which shreds the
            # whole reference chain. Recovery quality = does the stream resume
            # without a fresh IDR (LTR-P) or stall until FIR→IDR?
            if self._burst_every_s > 0:
                now_b = time.monotonic()
                since = now_b - loop_start
                if since > drop_after_s:
                    phase = since % self._burst_every_s
                    if phase < 0.020:  # ~20 ms ≈ one frame at 60 fps
                        loss_count += 1
                        continue
            try:
                q.put_nowait(pkt)
            except queue.Full:
                self._video_q_dropped += 1

    def _video_process_loop(self) -> None:
        decryptor = self._video_decryptor
        if decryptor is None:
            return
        q = self._video_q
        while not self._stop_evt.is_set():
            try:
                pkt = q.get(timeout=_UDP_RECV_TIMEOUT_S)
            except queue.Empty:
                self._evict_stale_groups()
                continue

            res = decryptor.decrypt(pkt)
            if res is None:
                continue
            hdr, payload = res
            ssrc = struct.unpack(">I", hdr[8:12])[0]
            seq = struct.unpack(">H", hdr[2:4])[0]
            ts = struct.unpack(">I", hdr[4:8])[0]
            marker = bool(hdr[1] & 0x80)

            self._track_seq(ssrc, seq)
            self._note_unknown_ssrc(ssrc)

            key = (ssrc, ts)
            if key in self._recently_flushed:
                continue
            grp = self._pending_groups.get(key)
            if grp is None:
                grp = []
                self._pending_groups[key] = grp
                self._group_arrival[key] = time.monotonic()
            grp.append((seq, marker, payload))

            if marker:
                self._flush_group(key)

    def _flush_group(self, key: tuple[int, int]) -> None:
        if self._decoder is None:
            return
        grp = self._pending_groups.pop(key, None)
        self._group_arrival.pop(key, None)
        if grp is None:
            return
        self._recently_flushed[key] = time.monotonic()
        ssrc, _ts = key
        ti = self._ssrc_to_tile.get(ssrc)
        if ti is None:
            return  # SSRC not part of the subscribed tier

        # Sort by seq, wraparound-aware.
        seqs = [s for s, _, _ in grp]
        if seqs and max(seqs) - min(seqs) > 0x8000:
            base = min(seqs)
            packets = sorted(grp, key=lambda x: (x[0] - base) & 0xFFFF)
        else:
            packets = sorted(grp, key=lambda x: x[0])

        ordered = [p for _, _, p in packets]
        self._tile_bytes[ti] = self._tile_bytes.get(ti, 0) + sum(len(p) for p in ordered)

        # Frame-completeness check (native jitter-buffer behaviour). An
        # access unit is fragmented across consecutive RTP sequence
        # numbers; a gap between this group's sorted packets means at
        # least one slice / FU fragment never arrived, so the reassembled
        # NALUs are truncated. Apple's own viewer does NOT feed an
        # incomplete frame to the decoder — it skips it and requests
        # recovery — because feeding a partial access unit produces
        # malformed-bitstream errors that can wedge the decoder. We mirror
        # that below: harvest param sets first (those NALUs may be intact
        # and small), then drop the frame if it's incomplete. `&0xFFFF`
        # makes the 65535->0 wrap a diff of 1 and a duplicate seq a diff of
        # 0; only diff>1 is a gap.
        oseq = [p[0] for p in packets]
        incomplete = any(
            ((oseq[i + 1] - oseq[i]) & 0xFFFF) > 1
            for i in range(len(oseq) - 1)
        )

        # The access unit's DONL (decoding-order number) — the LTR ring key
        # the LTRP ack must echo. Same for every NALU in the group.
        donl = first_donl(ordered) if self._ltr_enabled else None
        _reassemble = reassemble_h264 if self._video_codec == "avc" else reassemble_group
        au_cb = self._video_au_callback if self._video_codec == "avc" else None

        nalus = list(_reassemble(ordered))

        # If a dynamic resolution change is in progress, harvest fresh
        # VPS/SPS/PPS from the new encoder's burst so the restarted
        # decoder has the correct dimensions. (HEVC only — the AVC path
        # doesn't drive the virtual-display re-render.)
        if self._needs_param_harvest and nalus and self._video_codec != "avc":
            self._harvest_param_sets(nalus)
            if self._needs_param_harvest:
                # Harvest still incomplete: the shared decoder context is
                # still sized for the OLD canvas. Feeding these new-canvas
                # NALUs into it breaks the DPB on every resize and can
                # wedge the decoder. Drop the group — the encoder re-sends
                # the IDR burst (we FIR on harvest completion below, and
                # `_check_stall` FIR-storms if the param sets are ever
                # lost), so we resume cleanly once the new SPS/PPS land and
                # the decoder is rebuilt.
                return
            # Harvest just completed inside the call above: `set_params` +
            # `restart` rebuilt the context at the new dimensions, so the
            # remaining NALUs in THIS group (the new-canvas IDR slice) are
            # safe to feed below.

        if incomplete:
            # Don't feed a partial access unit. The broken reference chain
            # is re-rooted by the FIR the decoder's missing-ref / wedge
            # path requests once a later complete frame references this
            # dropped one.
            return

        au_parts: list[bytes] = []
        for nalu in nalus:
            self._decoder.feed_nalu(nalu, ti, donl=donl)
            if au_cb is not None:
                au_parts.append(b"\x00\x00\x00\x01" + bytes(nalu))
        if au_cb is not None and au_parts:
            try:
                au_cb(ti, key[1], b"".join(au_parts))
            except Exception:
                log.debug("video_au_callback raised", exc_info=True)

    def _harvest_param_sets(self, nalus: list[bytes]) -> None:
        """Accumulate fresh VPS/SPS/PPS from the new encoder's burst, then
        install them and restart the decoder once all three are seen.

        Apple's HEVC encoder emits parameter sets only with an IDR burst,
        and they can be split across multiple RTP timestamp groups (VPS in
        one Aggregation Packet, SPS/PPS in another, or a lost one re-sent
        on its own). So we accumulate into `self._harvest_*` across calls
        rather than requiring all three in a single flushed group — matching
        how `gather_initial_burst` harvests them. PPS is keyed by its
        ue(v)-decoded pic_parameter_set_id (same as burst.py), not the raw
        byte. No nal-ref-idc filter: that is an H.264 concept; in HEVC
        those bits are part of nal_unit_type."""
        from .media.nalu import NAL_VPS, NAL_SPS, NAL_PPS
        from .media.bitstream import BitReader, remove_emulation_prevention

        for nalu in nalus:
            if len(nalu) < 2:
                continue
            nt = (nalu[0] >> 1) & 0x3F
            if nt == NAL_VPS:
                self._harvest_vps = nalu
            elif nt == NAL_SPS:
                self._harvest_sps = nalu
            elif nt == NAL_PPS and len(nalu) > 2:
                try:
                    pps_id = BitReader(
                        remove_emulation_prevention(nalu[2:])
                    ).read_ue()
                except Exception:
                    continue
                self._harvest_pps[pps_id] = nalu

        if self._harvest_vps and self._harvest_sps and self._harvest_pps:
            log.info(
                "harvested param sets from new stream: VPS=%dB SPS=%dB PPS=%d",
                len(self._harvest_vps), len(self._harvest_sps),
                len(self._harvest_pps),
            )
            self._decoder.set_params(
                self._harvest_vps, self._harvest_sps, dict(self._harvest_pps),
            )
            self._decoder.restart()
            self._needs_param_harvest = False
            self.request_fir()

    def _evict_stale_groups(self) -> None:
        """Drop incomplete groups whose marker never arrived, and expire
        old entries from the late-arrival dedup set. Both bounds are
        proportional to expected RTP timing, so the dicts stay small."""
        now = time.monotonic()
        if self._pending_groups:
            for k in [
                k for k, t in self._group_arrival.items()
                if now - t > _PENDING_GROUP_TTL_S
            ]:
                self._pending_groups.pop(k, None)
                self._group_arrival.pop(k, None)
                self._recently_flushed[k] = now
        if self._recently_flushed:
            for k in [
                k for k, t in self._recently_flushed.items()
                if now - t > _FLUSHED_DEDUP_TTL_S
            ]:
                self._recently_flushed.pop(k, None)

    # ── sequence tracking + dynamic SSRC adoption ────────────────────

    def _track_seq(self, ssrc: int, seq: int) -> None:
        prev = self._max_seq.get(ssrc)
        # Packet-loss accounting (cumulative since session start).
        # `_received_pkts` is incremented unconditionally; `_lost_pkts`
        # is incremented when we observe a forward gap in the SSRC's
        # sequence numbers.
        self._received_pkts += 1
        self._last_video_pkt_t = time.monotonic()
        if prev is None:
            self._max_seq[ssrc] = seq
            self._roc[ssrc] = 0
            return
        diff = (seq - prev) & 0xFFFF
        if diff == 0 or diff > 0x8000:
            return  # duplicate or reorder
        # Forward jump → record any skipped seqs as NACK candidates.
        # Transport-layer error signal: when we observe a sequence-
        # number gap, we KNOW packets are missing and the next P/B
        # slice for that tile will reference data we may not have.
        # Mark the tile's decoder state suspect so a FIR will fire
        # if NACK retransmits don't recover in time — the transport
        # detects loss before the decoder ever sees the bad slice.
        tile_idx = self._ssrc_to_tile.get(ssrc)
        had_loss = False
        for missed in range(1, min(diff, 32)):
            lost = (prev + missed) & 0xFFFF
            self._nack_pending[ssrc].add(lost)
            self._lost_pkts += 1
            had_loss = True
            if tile_idx is not None and tile_idx < len(self._lost_pkts_per_tile):
                self._lost_pkts_per_tile[tile_idx] += 1
        if had_loss and tile_idx is not None and self._decoder is not None:
            try:
                self._decoder._gate.mark_decode_error(tile_idx)
            except Exception:
                pass
        if seq < prev:
            self._roc[ssrc] += 1
        self._max_seq[ssrc] = seq

    def _note_unknown_ssrc(self, ssrc: int) -> None:
        if ssrc in self._ssrc_to_tile or self._video_decryptor is None:
            return
        # Adopt only when we have no working group OR the current
        # group has been silent for `_SSRC_ADOPT_STALL_S`. The earlier
        # "first_adoption skips stall" check was wrong: it gated on
        # whether we'd ever abandoned a group (`_ssrc_blacklist`
        # non-empty), but the burst sets `_ssrc_to_tile` directly
        # WITHOUT going through adoption, so `_ssrc_blacklist` stays
        # empty even when we have a working burst group → the stall
        # guard was effectively disabled forever, allowing adoption
        # to ping-pong every time Apple emits a duplicate group.
        # The right invariant: if our current group is publishing,
        # don't switch.
        now = time.monotonic()
        have_active_group = bool(self._ssrc_to_tile)
        recently_published = (
            self._last_publish_t > 0.0
            and now - self._last_publish_t < _SSRC_ADOPT_STALL_S
        )
        if have_active_group and recently_published:
            return
        # Once at least 4 unknown SSRCs each have ≥N packets, swap maps
        # and request fresh IDRs. Skip any SSRC that's been part of a
        # previously-adopted group — that prevents the ping-pong loop
        # where Apple dual-broadcasts and we keep flipping back to a
        # group we already abandoned.
        counts = self._video_decryptor.ssrc_counts
        candidates = sorted(
            s for s in counts
            if s not in self._ssrc_to_tile
            and s not in self._ssrc_blacklist
            and counts[s] >= _DYNAMIC_SSRC_PACKET_THRESHOLD
        )
        from .protocol.offers import tiles_per_frame
        want = tiles_per_frame()  # SSRCs per group = tilesPerFrame we offered
        if len(candidates) < want:
            return
        # Apple emits `want` CONSECUTIVE SSRCs per tile group (one SSRC per
        # tile). Picking the first `want` sorted candidates can grab SSRCs
        # from TWO different broadcast groups when Apple is double-
        # publishing — the resulting Frankenstein map decodes some tiles
        # correctly and silently drops the others. Build runs of
        # consecutive SSRCs and adopt the first complete run. (want=1 → a
        # single-SSRC single-picture stream; want=4 → the tiled stream.)
        new_group: list[int] | None = None
        run = [candidates[0]]
        if want == 1:
            new_group = run
        else:
            for s in candidates[1:]:
                if s - run[-1] <= 1 and len(run) < want:
                    run.append(s)
                    if len(run) == want:
                        new_group = run
                        break
                else:
                    run = [s]
        if new_group is None:
            return  # no consecutive run of `want` yet — wait for more data
        new_map = {s: i for i, s in enumerate(new_group)}
        if new_map == self._ssrc_to_tile:
            return
        log.info(
            "adopting fresh SSRC group: %s",
            [f"0x{s:08x}" for s in new_group],
        )
        # Blacklist the previously-adopted group so we never go back.
        self._ssrc_blacklist.update(self._ssrc_to_tile.keys())
        self._ssrc_to_tile = new_map
        # Keep the authoritative tile count in step with the adopted
        # group so num_tiles (and the frontend's per-tile loops) track a
        # mid-session tile-count change rather than the connect-time value.
        self._observed_tile_count = len(new_map)
        self._last_ssrc_adopt_ts = now
        # Grace window: pretend frames just flowed so the frame-flow gate
        # gives the new mapping ~0.5 s to start producing before we
        # consider another adoption.
        self._last_publish_t = now
        if self._decoder is not None:
            # If a dynamic resolution change is in flight, defer the
            # decoder restart until the video process loop harvests
            # fresh VPS/SPS/PPS from the new encoder's burst. The old
            # param sets are stale (old resolution) and would make
            # the restarted decoder reject all frames.
            if self._needs_param_harvest:
                log.info("SSRC adoption deferred — waiting for param harvest")
                self._dpb_error_window.clear()
                self.request_fir()
            else:
                # Restart the decoder + FIR for the new tiles. We tried
                # skipping the restart to avoid a 1.5–6 s outage, but
                # without it the SW fallback path can't re-feed burst
                # NALUs and the new context starves until an unprompted
                # IDR shows up (often never).
                self._dpb_error_window.clear()
                # Guard against rapid back-to-back SSRC rotations (e.g.
                # Apple emitting two new groups within 2 s at curtain
                # start): the second full restart discards IDR frames
                # in-flight for the first group's FIR, making recovery
                # nearly impossible. If we just restarted within 3 s
                # the decoder is already fresh — skip the redundant
                # teardown and let the FIR re-anchor the new group.
                # NOTE: read the prior restart time BEFORE stamping `now`,
                # otherwise the gap is always 0 and the restart never fires.
                last_restart = self._last_decoder_restart_t
                if now - last_restart >= 3.0:
                    self._last_decoder_restart_t = now
                    self._decoder.restart()
                self.request_fir()

    # ── RTCP RX loop (server SR for jitter / dlsr) ──────────────────

    # ── ctrl/audio RX: drain + process split ─────────────────────────
    #
    # Same drain/process split as the video path. Apple sends BOTH RTCP
    # (PT 200-207) AND audio RTP on the same port (5900 by default).
    # We classify by checking the RTP version + PT byte: PT 200-207 →
    # SRTCP; otherwise → audio RTP. Without this classification, audio
    # packets sit unprocessed in the kernel buffer and the host-side
    # audio appears silent.

    def _ctrl_drain_loop(self) -> None:
        """Tight inner loop: recvfrom → enqueue. See _video_drain_loop
        for the rationale on splitting drain from process."""
        sock = self._sock_ctrl
        if sock is None:
            return
        sock.settimeout(_UDP_RECV_TIMEOUT_S)
        q = self._ctrl_q
        while not self._stop_evt.is_set():
            try:
                pkt, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            self._rx_pkts_ctrl += 1
            self._rx_bytes_ctrl += len(pkt)
            try:
                q.put_nowait(pkt)
            except queue.Full:
                self._ctrl_q_dropped += 1

    def _ctrl_process_loop(self) -> None:
        srtcp_dec = self._srtcp_dec
        if srtcp_dec is None:
            return
        q = self._ctrl_q
        while not self._stop_evt.is_set():
            try:
                pkt = q.get(timeout=_UDP_RECV_TIMEOUT_S)
            except queue.Empty:
                continue
            # Classify: RTP version-2 + non-RTCP PT = audio RTP packet.
            if (
                len(pkt) >= 2
                and (pkt[0] & 0xC0) == 0x80
                and not (200 <= (pkt[1] & 0x7F) <= 207)
            ):
                self._handle_audio_rtp(pkt)
                continue
            # Otherwise SRTCP.
            decrypted = srtcp_dec.unprotect(pkt)
            if decrypted is None:
                continue
            for ssrc, ntp_mid32, arrival in parse_sr_arrivals(decrypted):
                self._server_sr[ssrc] = (ntp_mid32, arrival)

    def _handle_audio_rtp(self, pkt: bytes) -> None:
        """Decrypt an SRTP audio packet from the muxed CTRL port,
        feed it to the AAC-ELD decoder, dispatch PCM to the callback."""
        decryptor = self._audio_decryptor
        decoder = self._aac
        if decryptor is None or decoder is None:
            return
        res = decryptor.decrypt(pkt)
        if res is None:
            return
        _hdr, payload = res
        try:
            pcm = decoder.decode(payload)
        except Exception as e:
            log.warning("AAC-ELD decode (ctrl-mux) error: %s", e)
            return
        cb = self._audio_callback
        if pcm is not None and cb is not None and not self._audio_user_mute:
            try:
                cb(pcm)
            except Exception as e:
                log.warning("audio callback (ctrl-mux) raised: %s", e)

    # ── TCP control RX loop (drain + handle 0x14) ────────────────────

    def _tcp_rx_loop(self) -> None:
        if self._negotiation is None:
            return
        sock = self._negotiation.sock
        cipher = self._negotiation.cipher
        sock.settimeout(_TCP_RECV_TIMEOUT_S)
        buf = bytearray()

        # Replay any msgs the negotiation phase decrypted but couldn't
        # dispatch (typically the daemon's initial cursor pixmap that
        # arrived in the same chunk as the 0x1c answer). Without this
        # replay, the cursor pseudo-encoding cache starts empty and
        # subsequent cache-hit refs from the daemon all miss.
        for msg in self._negotiation.leftover_msgs:
            log.debug("dispatching leftover negotiation msg type=0x%02x len=%d",
                      msg[0] if msg else 0, len(msg))
            self._handle_tcp_msg(msg)
        self._negotiation.leftover_msgs.clear()

        while not self._stop_evt.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except (OSError, ConnectionError):
                log.info("TCP control read failed; marking session dead")
                self._connected = False
                self._fresh_evt.set()  # wake any waiter so they see is_connected=False
                return
            if not chunk:
                log.info("TCP control closed by peer; marking session dead")
                self._connected = False
                self._fresh_evt.set()
                return
            buf += chunk
            msgs, consumed = cipher.decrypt_stream(bytes(buf))
            for msg in msgs:
                self._handle_tcp_msg(msg)
            if consumed:
                del buf[:consumed]

    def _handle_tcp_msg(self, msg: bytes) -> None:
        if not msg:
            return
        # Lightweight always-on counter — per-RFB-type inbound histogram.
        # Cheap (one dict bump) and gives us a smoking-gun signal for
        # bugs like "cursor freezes after idle" — if msg type=0x00 stops
        # incrementing during the freeze, the daemon stopped sending.
        self._rx_pkts_tcp += 1
        t = msg[0]
        self._rx_msg_type_counts[t] = self._rx_msg_type_counts.get(t, 0) + 1
        # Optional RE hook: dump every inbound RFB msg type + body.
        if os.environ.get("ISS_LOG_RFB_IN") == "1":
            log.info("RX type=0x%02x len=%d body=%s", msg[0], len(msg), msg[:512].hex())

        # Multi-cipher-frame 0x1f reassembly: continuation frames don't
        # carry a type byte, so any inbound msg while reassembly is
        # in-progress belongs to the in-flight clipboard send.
        if self._clipboard_reassembler.in_progress:
            full = self._clipboard_reassembler.feed(msg)
            if full is not None:
                self._handle_clipboard_send(full)
            return

        msg_type = msg[0]

        # 0x14: misc-status push. Wire layout (8 bytes):
        #   [0]    type     = 0x14
        #   [1]    pad      = 0x00
        #   [2..3] u16 BE   = body length (= 4 for the heartbeat shape)
        #   [4..5] u16 BE   = client/flags (observed = 1)
        #   [6..7] u16 BE   = cmd          (THIS is the dispatch byte)
        # cmd values observed:
        #   12 = MiscStatus heartbeat (every 2.1 s) — fires when the
        #        viewer's command-mask grants the bit
        #   2  = remote clipboard changed (autopasteboard generation
        #        bumped) — viewer responds with msg 0x0b fetch
        #   11 = UserSessionChanged
        if msg_type == 0x14:
            cmd = struct.unpack(">H", msg[6:8])[0] if len(msg) >= 8 else None
            # Only log on change — the host re-pushes the same status
            # (typically cmd=4) every couple seconds, which is pure noise.
            if cmd != self._last_misc_status_cmd:
                log.debug("server sent 0x14 misc-status (cmd=%s): %s", cmd, msg.hex())
                self._last_misc_status_cmd = cmd
            if cmd == 2:
                log.info("remote clipboard changed; sending fetch (0x0b)")
                try:
                    self._input.send_clipboard_fetch()
                except Exception as e:  # pragma: no cover
                    log.debug("clipboard fetch send failed: %s", e)
            return

        # 0x1f: HandleViewerClipboardSend — first frame of a multi-frame
        # remote-pasteboard payload. Feed it to the reassembler.
        if msg_type == 0x1f:
            full = self._clipboard_reassembler.feed(msg)
            if full is not None:
                self._handle_clipboard_send(full)
            return

        # NOTE: the mid-session 0x1c re-offer's answer is delivered framed
        # as a 0x00 FBU (an embedded bplist), NOT as a top-level 0x1c — so
        # there is no `msg_type == 0x1c` branch here. The authoritative new
        # geometry comes from the 0x451 AppleDisplayLayout rect handled in
        # _handle_fbu, and the decoder picks up the new dimensions from the
        # harvested SPS. (An earlier 0x1c branch that called
        # extract_canvas_dims was a dead no-op — that helper rejects any
        # message whose first byte isn't 0x00.)

        # 0x00: FrameBufferUpdate. On the TCP control channel this only
        # ever carries pseudo-encoding rects (cursor enc 1104, etc.) —
        # actual pixel data flows over UDP/SRTP, not here.
        if msg_type == 0x00:
            self._handle_fbu(msg)
            return

    def _handle_fbu(self, msg: bytes) -> None:
        """Walk a FrameBufferUpdate received on the TCP control channel.
        Each rect has a 12-byte header (x, y, w, h, encoding) followed
        by encoding-specific payload. We only care about pseudo-
        encodings here — the only one Apple sends on this path is
        enc 1104 (cursor).

        After processing, send a fresh incremental FBU request so the
        daemon keeps streaming updates (request-response pattern, like
        Apple SS.app)."""
        if len(msg) < 4:
            return
        # Layout:
        #   [0]    type = 0x00
        #   [1]    pad
        #   [2..3] u16 BE n_rects
        #   then n_rects × { rect_header (12B) + encoding_payload }
        n_rects = struct.unpack(">H", msg[2:4])[0]
        offset = 4
        saw_cursor = False
        _dbg_cursor = os.environ.get("ISS_CURSOR_DEBUG") == "1"
        _dbg_encs: list[str] = []
        for _ in range(n_rects):
            if offset + 12 > len(msg):
                log.debug("FBU truncated at rect header (offset=%d)", offset)
                if _dbg_cursor:
                    log.info("FBU n_rects=%d encs=[%s] TRUNCATED len=%d",
                             n_rects, ",".join(_dbg_encs), len(msg))
                return
            f0, f1, f2, f3, encoding = struct.unpack(
                ">HHHHi", msg[offset:offset + 12],
            )
            offset += 12
            if _dbg_cursor:
                _dbg_encs.append("0x%x@(%d,%d,%d,%d)" % (
                    encoding & 0xffffffff, f0, f1, f2, f3))
            if encoding == 1104:
                saw_cursor = True
                consumed = self._handle_cursor_rect(msg, offset, f0, f1, f2, f3)
                if consumed < 0:
                    return
                offset += consumed
            elif encoding in (1010, 1011, 1105, 1107, 1109, 1110,
                              0x3f2, 0x3f3, 0x3ea, 0x453, 0x455, 0x456):
                # Apple-private pseudo-encodings that all share one wire
                # format: u16 BE payload_len + payload. We don't decode
                # them but MUST skip past them rather than abort the walk —
                # otherwise a 1104/0x450 cursor rect that follows in the
                # SAME FBU is lost (which froze the cursor shape right after
                # login). Two families share this skip path:
                #   Config blobs (known content, none required to render):
                #     1010 — media-stream negotiation blob (bplist)
                #     1011 — sibling of 1010 in some sessions
                #     1105 — display info struct (dims, "main" sentinel, scale)
                #     1107 — vendor keysyms
                #     1109 — keyboard input source string
                #     1110 — device info (host model identifier string)
                #   Media-stream reconfig rects (0x453/0x455/0x456 and the
                #     0x3ea/0x3f2/0x3f3 group) the daemon emits at a
                #     login/session switch.
                if offset + 2 > len(msg):
                    return
                sz = struct.unpack(">H", msg[offset:offset + 2])[0]
                if offset + 2 + sz > len(msg):
                    return
                offset += 2 + sz
            elif encoding == 0x451:
                # AppleDisplayLayout: server confirms/announces display
                # geometry. Payload is a u16 prefix_length followed by
                # that many bytes of layout data (scaled_w/h, backing
                # w/h at +4..+11). The remainder is opaque trailing
                # fields (revision gap). Consume the rect and update
                # the runtime canvas dimensions so the frontend can
                # resize its framebuffer.
                #
                # A layout event marks a display/session transition — the
                # point at which the daemon's cursor sender otherwise goes
                # silent (the post-login/lock freeze). We re-arm the cursor
                # (enc 1104) sender with a single non-incremental FBU request
                # at the END of this branch. We deliberately do NOT send
                # AutoFrameBufferUpdate (msg 0x09) — it free-runs full video
                # frames over TCP (~800 pps idle) and isn't needed for the
                # cursor. Done on EVERY 0x451 so cursor SELECTs keep flowing
                # across a login/lock switch.
                if offset + 2 > len(msg):
                    return
                prefix_len = struct.unpack(">H", msg[offset:offset + 2])[0]
                # Validate the declared payload fits before reading it, so
                # a truncated/short rect can't raise struct.error and kill
                # the RX thread (mirrors the 1010/1011 branch).
                if offset + 2 + prefix_len > len(msg):
                    return
                needs_post_layout_arm = False
                if prefix_len >= 12:
                    sw = struct.unpack(">H", msg[offset + 4:offset + 6])[0]
                    sh = struct.unpack(">H", msg[offset + 6:offset + 8])[0]
                    bw = struct.unpack(">H", msg[offset + 8:offset + 10])[0]
                    bh = struct.unpack(">H", msg[offset + 10:offset + 12])[0]
                    # Backing (pixel) dims drive GPU texture sizing; only
                    # trust them when present. Never substitute the scaled
                    # (logical) size for backing — on HiDPI that would
                    # under-size the textures and overrun on upload.
                    new_bw = bw if (bw and bh) else self._runtime_canvas_w
                    new_bh = bh if (bw and bh) else self._runtime_canvas_h
                    if sw and sh:
                        # Only consider it a runtime change if we already
                        # had a canvas — the first 0x451 is the initial
                        # layout confirmation, not a resize.
                        had_canvas = (
                            self._runtime_canvas_w > 0
                            and self._runtime_canvas_h > 0
                        )
                        changed = had_canvas and (
                            new_bw != self._runtime_canvas_w
                            or new_bh != self._runtime_canvas_h
                        )
                        if new_bw and new_bh:
                            self._runtime_canvas_w = new_bw
                            self._runtime_canvas_h = new_bh
                            # Keep the pointer clamp in step with the canvas
                            # the frontend maps clicks into (canvas_dims);
                            # otherwise the host mis-places the cursor after
                            # a mid-session resize.
                            if self._input is not None:
                                self._input.set_server_dims(new_bw, new_bh)
                        self._runtime_scaled_w = sw
                        self._runtime_scaled_h = sh
                        log.info(
                            "AppleDisplayLayout: scaled=%dx%d backing=%dx%d"
                            " changed=%s",
                            sw, sh, bw, bh, changed,
                        )
                        if changed:
                            self._needs_param_harvest = True
                            # Start a clean harvest — discard any stale
                            # accumulators from a prior (old-resolution)
                            # harvest so they can't complete this one early.
                            self._harvest_vps = None
                            self._harvest_sps = None
                            self._harvest_pps = {}
                            log.info("AppleDisplayLayout: flagged param harvest for new canvas")
                            # The server is waiting for a full-refresh request
                            # at the new geometry before it resumes the HEVC
                            # encoder; do the media re-offer below, after the
                            # cursor re-arm has sent the non-incremental FBUR.
                            needs_post_layout_arm = True
                # Re-arm the cursor (enc 1104) sender now that this layout is
                # parsed (a single non-incremental FBUR — no 0x09). Always, so
                # the cursor keeps flowing even on no-geometry-change layouts.
                self._send_cursor_rearm()
                if needs_post_layout_arm:
                    # Geometry actually changed — additionally re-offer the
                    # media session (0x1c) so the encoder restarts at the new
                    # canvas. The FBUR it needs was just sent by the re-arm.
                    self._schedule_post_layout_arm()
                offset += 2 + prefix_len
            else:
                # A non-cursor, non-config rect on the control channel means
                # the daemon answered with video/raw pixel data over RFB, or
                # an unknown pseudo-encoding whose payload size we can't tell.
                # Either way we give up parsing the rest of this FBU (dropping
                # any cursor rect that follows). The next msg starts with a
                # fresh type byte so this isn't fatal. Count it as the safety
                # signal that a request pulled video (the failure mode of the
                # old full-screen request that destabilised RTP).
                self._fbu_video_rects += 1
                if _dbg_cursor:
                    log.info("FBU walk ABORT at unknown encoding=0x%x (%d); "
                             "encs so far=[%s] — rects after this are dropped "
                             "(video_rects=%d)",
                             encoding & 0xffffffff, encoding,
                             ",".join(_dbg_encs), self._fbu_video_rects)
                else:
                    log.debug("FBU rect with video/unknown encoding=%d; aborting "
                              "walk (video_rects=%d)", encoding,
                              self._fbu_video_rects)
                return
        if _dbg_cursor and (n_rects != 1 or any("0x450" in e for e in _dbg_encs)):
            # Log multi-rect FBUs and any FBU carrying a cursor (0x450/1104)
            # rect, so we can see whether a cursor command is present but
            # mis-framed vs genuinely absent during a post-login freeze.
            log.info("FBU n_rects=%d encs=[%s]", n_rects, ",".join(_dbg_encs))
        # Re-arm the cursor pipeline. The daemon re-arms its sender after each
        # rect and needs a fresh request to keep cursor (enc 1104) shape
        # updates flowing; re-request a MINIMAL 1x1 region after each cursor
        # update so shape changes keep flowing, without pulling video. (A
        # non-incremental FBUR re-arm is also sent from the 0x451 handler on a
        # display/session transition.)
        if saw_cursor and self._input is not None:
            try:
                self._input.request_framebuffer_update()
            except Exception:
                pass

    def _schedule_post_layout_arm(self) -> None:
        """Called from the 0x451 handler on a geometry CHANGE. Re-offers the
        media session (0x1c) with the same SRTP keys so the server restarts
        the HEVC encoder at the new canvas dimensions. Without the 0x1c
        re-offer the server stops encoding — it won't resize the media canvas
        without a fresh media-stream configuration.

        The non-incremental FBU request the encoder restart needs is already
        sent by `_send_cursor_rearm()` immediately before this call (see the
        0x451 handler), so this method no longer sends its own FBUR — that
        avoided a duplicate full-refresh request per resize.
        """
        neg = self._negotiation
        if neg is None:
            return
        from .protocol.negotiation import build_0x1c

        # Re-offer the media session. We reuse the same SRTP keys —
        #    the server accepts this and restarts the encoder at the new
        #    canvas size. (The native app generates new keys for each
        #    round; we may revisit this once the full round-trip key
        #    lifecycle is understood.)
        try:
            vo = getattr(self, '_video_offer', None)
            ao = getattr(self, '_audio_offer', None)
            if vo is None or ao is None:
                vo, ao = create_offers()
            msg_1c = build_0x1c(
                ao, vo, neg.keys,
                alt_session=self._config.alt_session,
            )
            neg.cipher.encrypt_and_send(neg.sock, msg_1c)
            log.info("post-layout: sent 0x1c re-offer (%dB)", len(msg_1c))
        except OSError as e:
            log.warning("post-layout 0x1c send failed: %s", e)

        # 3) Defer FIR to the TX thread.
        self._needs_post_layout_fir = True

    def _send_cursor_rearm(self) -> None:
        """Re-arm the daemon's TCP cursor pseudo-encoding (enc 1104) sender so
        the cursor keeps flowing across a display/session transition.

        Sends a single non-incremental FramebufferUpdateRequest (msg 0x03).
        We deliberately do NOT send AutoFrameBufferUpdate (msg 0x09) here: it
        arms the daemon to free-run full video frames over TCP (~800 pps on a
        static screen) and our RE notes confirm it is not required for the
        cursor — the periodic 1x1 incremental FBU poll (_maybe_poll_cursor)
        already re-arms the enc-1104 sender. This call is lightweight — unlike
        _schedule_post_layout_arm it does NOT re-offer the media session — so
        it's safe to call at every 0x451, including the no-geometry-change
        layout events emitted at a login/lock/agent switch (the case that
        froze the cursor before).
        """
        neg = self._negotiation
        if neg is None:
            return
        from .protocol.negotiation import build_fbu_request
        try:
            neg.cipher.encrypt_and_send(
                neg.sock, build_fbu_request(incremental=False))
        except OSError as e:
            log.debug("cursor re-arm (FBUR) failed: %s", e)

    def _handle_cursor_rect(
        self, msg: bytes, offset: int,
        f0: int, f1: int, f2: int, f3: int,
    ) -> int:
        """Parse one enc 1104 cursor rect starting at `msg[offset]`.

        Returns bytes consumed past the 12-byte rect header (i.e. the
        size of enc 1104's own payload), or -1 if the buffer was
        truncated.

        Field order: the four rect-header u16s are
        {hotspot_x, hotspot_y, cursor_w, cursor_h}. Body follows as
        {cache_id (u32 BE), comp_size (u32 BE), zlib(BGRA pixmap +
        alpha mask)}.
        """
        hotspot_x, hotspot_y, cursor_w, cursor_h = f0, f1, f2, f3
        if offset + 8 > len(msg):
            return -1
        cache_id, comp_size = struct.unpack(">II", msg[offset:offset + 8])
        payload_off = offset + 8
        if payload_off + comp_size > len(msg):
            return -1
        if os.environ.get("ISS_CURSOR_DEBUG") == "1":
            log.info("cursor-rect: dims=%dx%d hot=(%d,%d) cache_id=%d comp_size=%d",
                     cursor_w, cursor_h, hotspot_x, hotspot_y, cache_id, comp_size)
        if comp_size == 0:
            # Cache hit: server sent just the cache_id. Pull our cached
            # cursor for this id and re-apply. Cache IDs are arbitrary
            # per-viewer identifiers — none have special semantics
            # (early code special-cased cache_id=1000 as "OS default
            # arrow" but the daemon assigns 1000 like any other id).
            cached = self._cursor_cache.get(cache_id)
            if cached is None:
                # The daemon sometimes references a cache_id we never
                # received a NEW pixmap for (intermittent — likely
                # due to a per-viewer "already sent" tracker on the
                # daemon side that doesn't always match reality). Fall
                # back to the local OS default cursor so transitions
                # to the system arrow render correctly. Trade-off: at
                # boundaries where the host cursor rapidly toggles
                # between a known shape and an unknown cache_id, the
                # cursor will visibly flicker between the cached shape
                # and the local OS arrow (different hotspot — hence
                # the apparent visual jump).
                log.debug("cursor cache miss for cache_id=%d", cache_id)
                self._notify_cursor(None)
                return 8
            log.debug(
                "cursor: cache_id=%d hit (%dx%d)",
                cache_id, cached.width, cached.height,
            )
            self._notify_cursor(cached)
            return 8
        # Cache miss: full cursor follows, zlib-deflated. Apple's
        # encoder ends the deflate stream with Z_SYNC_FLUSH (0000ffff)
        # rather than Z_FINISH, so `zlib.decompress` rejects it as
        # truncated — `decompressobj` accepts it.
        compressed = msg[payload_off:payload_off + comp_size]
        try:
            d = zlib.decompressobj()
            raw = d.decompress(compressed) + d.flush()
        except zlib.error as e:
            log.warning("cursor zlib decompress failed: %s", e)
            return 8 + comp_size
        # Format (verified empirically): w*h*4 BGRA pixmap, then w*h*1
        # alpha mask. We combine the mask into the pixmap's A channel
        # since GLFW wants single RGBA buffer.
        pixmap_size = cursor_w * cursor_h * 4
        mask_size = cursor_w * cursor_h
        expected = pixmap_size + mask_size
        if len(raw) != expected:
            log.debug(
                "cursor size mismatch (got %d, expected %d = %dx%d * 5)",
                len(raw), expected, cursor_w, cursor_h,
            )
            return 8 + comp_size
        # Build RGBA8888: swap BGR→RGB and overwrite A with the mask.
        ba = bytearray(pixmap_size)
        for px in range(cursor_w * cursor_h):
            o = px * 4
            ba[o]     = raw[o + 2]   # R = src.R (was at offset 2 in BGRA)
            ba[o + 1] = raw[o + 1]   # G
            ba[o + 2] = raw[o]       # B = src.B (was at offset 0 in BGRA)
            ba[o + 3] = raw[pixmap_size + px]  # A from mask
        img = _CursorImage(
            width=cursor_w, height=cursor_h,
            hotspot_x=hotspot_x, hotspot_y=hotspot_y,
            rgba=bytes(ba),
        )
        self._cursor_cache[cache_id] = img
        # Bound cache size — Apple uses ~16, give us a generous ceiling.
        if len(self._cursor_cache) > 64:
            # Drop oldest entry (insertion-ordered dict).
            oldest = next(iter(self._cursor_cache))
            self._cursor_cache.pop(oldest, None)
        log.debug(
            "cursor: cache_id=%d %dx%d hotspot=(%d,%d) compressed=%dB",
            cache_id, cursor_w, cursor_h, hotspot_x, hotspot_y, comp_size,
        )
        self._notify_cursor(img)
        return 8 + comp_size

    def _notify_cursor(self, img: Optional[_CursorImage]) -> None:
        self._cursor_msgs_processed += 1
        self._cursor_last_t = time.monotonic()
        if img is not None:
            self._last_cursor_notified = img
        if os.environ.get("ISS_CURSOR_DEBUG") == "1":
            if img is None:
                log.info("cursor-notify: None (cache-miss / revert-to-default)")
            else:
                log.info("cursor-notify: %dx%d hotspot=(%d,%d)",
                         img.width, img.height, img.hotspot_x, img.hotspot_y)
        cb = self._cursor_callback
        if cb is None:
            return
        try:
            cb(img)
        except Exception as e:
            log.warning("cursor callback raised: %s", e)

    def _handle_clipboard_send(self, full_msg: bytes) -> None:
        """A complete 0x1f payload arrived — decompress, parse, push the
        first text flavor onto the local clipboard so the user can paste."""
        from .protocol import clipboard as clip
        from . import local_clipboard
        hdr = clip.parse_clipboard_send_header(full_msg)
        if hdr is None:
            return
        _promise, _reserved, uncompressed, compressed = hdr
        try:
            decompressed = clip.decompress_clipboard_payload(
                full_msg[16:16 + compressed]
            )
        except Exception as e:
            log.warning("clipboard decompress failed: %s", e)
            return
        if len(decompressed) != uncompressed:
            log.debug(
                "clipboard inner size mismatch (got %d, expected %d) — continuing",
                len(decompressed), uncompressed,
            )
        try:
            items = clip.parse_clipboard_items(decompressed)
        except Exception as e:
            log.warning("clipboard parse failed: %s", e)
            return
        utis = [it.primary_uti for it in items]
        text = clip.text_from_items(items)
        if text is None:
            log.info("clipboard recv: no text flavour (utis=%s)", utis)
            return
        preview = text[:40].replace("\n", "\\n")
        log.info(
            "clipboard recv: %d items, text=%d chars (preview=%r)",
            len(items), len(text), preview,
        )
        self._last_received_clipboard_text = text
        local_clipboard.push_text(text)
        cb = self._clipboard_text_callback
        if cb is not None:
            try:
                cb(text)
            except Exception as e:  # pragma: no cover
                log.debug("clipboard text callback raised: %s", e)

    # ── TX loop (heartbeat + RTCP + on-demand FIR/PLI/NACK + watchdog) ─

    def _maybe_poll_cursor(self) -> None:
        """Periodic minimal (1x1) FramebufferUpdateRequest, every tx tick
        (~2/s), to keep the daemon's cursor (enc 1104) sender armed so cursor-
        shape changes keep flowing after the screen goes idle. The daemon
        re-arms its sender after each rect and goes quiet without ongoing
        requests (the 'cursor freezes after idle' bug — RE'd: the daemon's
        threads are all alive, it's request-starvation, not a dead pthread).
        The 1x1 region keeps it from pulling video (see
        InputController.request_framebuffer_update)."""
        if self._input is None:
            return
        try:
            self._input.request_framebuffer_update()
        except Exception:
            pass

    def _tx_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._tx_tick += 1
            try:
                self._send_heartbeat()
                self._send_rr_and_maybe_sr()
                if self._needs_post_layout_fir:
                    self._needs_post_layout_fir = False
                    self.request_fir()
                self._drain_pending_fir()
                self._drain_pending_nack()
                self._check_stall()
                self._maybe_poll_cursor()
                if self._tx_tick % _TX_PROFILE_EVERY_N_TICKS == 0:
                    self._log_profile_snapshot()
            except Exception as e:
                log.debug("tx loop tick error: %s", e)
            self._stop_evt.wait(_TX_INTERVAL_S)

    def _log_profile_snapshot(self) -> None:
        """Per-tile decoded-frame counts + decoder path + adoption state.
        Logged at INFO so it's visible without -v; cadence is the loop
        tick multiplier so the volume stays manageable on long sessions."""
        if self._decoder is None:
            return
        good = self._decoder.good_counts
        baseline = self._last_profile_good if len(self._last_profile_good) == len(good) else [0] * len(good)
        delta = [good[i] - baseline[i] for i in range(len(good))]
        # `restart()` resets `good_counts` to zero; if delta is negative,
        # the decoder was rebuilt since our last snapshot — treat the new
        # values as the new baseline rather than reporting nonsense rates.
        if any(d < 0 for d in delta):
            delta = list(good)
        self._last_profile_good = list(good)
        # Per-tile loss delta over the same interval. Lets the operator
        # tell at a glance whether a stuck tile correlates with packet
        # loss (network) or has zero observed loss (code bug).
        lost = self._lost_pkts_per_tile if self._lost_pkts_per_tile else [0] * len(good)
        last_lost = (
            self._last_profile_lost_per_tile
            if len(self._last_profile_lost_per_tile) == len(lost)
            else [0] * len(lost)
        )
        loss_delta = [max(0, lost[i] - last_lost[i]) for i in range(len(lost))]
        self._last_profile_lost_per_tile = list(lost)
        elapsed = _TX_INTERVAL_S * _TX_PROFILE_EVERY_N_TICKS
        rates = [round(d / elapsed, 1) for d in delta]
        # Clean (non-concealed) per-tile rate — the honest "real video"
        # signal. A tile with a high `rates` but ~0 `clean_rates` is GRAY:
        # it's churning concealed frames at full throughput. (good_count
        # counts concealed frames; clean_count does not.)
        clean = self._decoder.clean_counts
        cbase = (self._last_profile_clean
                 if len(self._last_profile_clean) == len(clean)
                 else [0] * len(clean))
        cdelta = [clean[i] - cbase[i] for i in range(len(clean))]
        if any(d < 0 for d in cdelta):
            cdelta = list(clean)
        self._last_profile_clean = list(clean)
        clean_rates = [round(d / elapsed, 1) for d in cdelta]
        # Per-tile KB/s — which screen band is eating the bandwidth.
        n_t = len(clean)
        tile_kbs = []
        for ti in range(n_t):
            db = self._tile_bytes.get(ti, 0) - self._last_tile_bytes.get(ti, 0)
            tile_kbs.append(round(max(0, db) / elapsed / 1024, 1))
        self._last_tile_bytes = dict(self._tile_bytes)
        # RX/TX rate deltas over the same interval.
        rx_v = self._rx_pkts_video; rx_c = self._rx_pkts_ctrl; rx_t = self._rx_pkts_tcp
        bv = self._rx_bytes_video; bc = self._rx_bytes_ctrl
        tx = self._input.tx_pkts if self._input is not None else 0
        last_rx_v, last_rx_c, last_rx_t = self._last_rx_pkts
        last_bv, last_bc = self._last_rx_bytes
        last_tx = self._last_tx_pkts
        rx_v_pps = round((rx_v - last_rx_v) / elapsed, 1)
        rx_c_pps = round((rx_c - last_rx_c) / elapsed, 1)
        rx_t_pps = round((rx_t - last_rx_t) / elapsed, 2)
        tx_pps = round((tx - last_tx) / elapsed, 2)
        rx_v_mbps = round((bv - last_bv) * 8 / elapsed / 1_000_000, 2)
        rx_c_kbps = round((bc - last_bc) * 8 / elapsed / 1_000, 1)
        self._last_rx_pkts = (rx_v, rx_c, rx_t)
        self._last_rx_bytes = (bv, bc)
        self._last_tx_pkts = tx
        cursor_age = (
            time.monotonic() - self._cursor_last_t
            if self._cursor_last_t > 0 else -1.0
        )
        # Aggregate loss across ALL SSRCs (including unmapped ones not
        # in _ssrc_to_tile). Subtract per-tile mapped loss to see how
        # much went to unmapped — non-zero unmapped loss during a gray
        # event means Apple is publishing extra SSRC groups we ignore.
        loss_total = self._lost_pkts
        loss_unmapped = loss_total - sum(self._lost_pkts_per_tile)
        # Per-tile NALU-type histogram delta. Surfaces e.g. "tile 2
        # stopped getting nt=20 IDRs from minute X" or "burst of nt=39
        # SEI right before the gray".
        nalu_now = (
            [dict(d) for d in self._decoder.nalu_counts_per_tile]
            if self._decoder is not None else [{} for _ in good]
        )
        last_nalu = (
            self._last_profile_nalu
            if len(self._last_profile_nalu) == len(nalu_now)
            else [{} for _ in nalu_now]
        )
        nalu_delta = []
        for i, cur in enumerate(nalu_now):
            d = {t: cur[t] - last_nalu[i].get(t, 0)
                 for t in cur if cur[t] - last_nalu[i].get(t, 0) > 0}
            nalu_delta.append(d)
        self._last_profile_nalu = nalu_now
        nalu_str = " ".join(
            f"t{i}:{{{','.join(f'{t}:{n}' for t,n in sorted(d.items()))}}}"
            for i, d in enumerate(nalu_delta)
        )
        # Compact msg-type histogram, sorted by type — easy to scan
        # for "msg type 0x00 stopped incrementing" during a freeze.
        types_str = " ".join(
            f"{t:02x}:{n}" for t, n in sorted(self._rx_msg_type_counts.items())
        )
        decoder_name = self._decoder._hw_name or "software"
        ssrc_groups = (
            len(self._video_decryptor.ssrc_counts) // 4
            if self._video_decryptor else 0
        )
        last_publish_age = (
            time.monotonic() - self._last_publish_t
            if self._last_publish_t > 0 else -1.0
        )
        cursor_age_ms = int(cursor_age * 1000) if cursor_age >= 0 else -1
        # Periodic profile log is debug-level: the same data flows out the
        # control socket as a structured snapshot for the TUI / monitors,
        # so a normal user doesn't need it spamming the log every 2 s.
        # LTRP recovery telemetry: ack count proves iss is acking; far_ref
        # (a P-slice referencing a POC >=16 frames back, vs the normal ~7)
        # is the wire signature that the host honoured an ack and recovered
        # via a long-term-style reference instead of a full IDR. far>0 with
        # acks climbing == LTRP working end-to-end. far==0 while gray recurs
        # == acks still no-op. See hevc_rps far_ref_events / _send_ltr_ack.
        rps = getattr(self._decoder, "_rps_tracker", None)
        ltr_far = getattr(rps, "far_ref_events", 0)
        ltr_dist = getattr(rps, "max_ref_distance", 0)
        ltr_miss = getattr(rps, "missing_ref_events", 0)
        log.debug(
            "profile: decoder=%s tiles=%s rates=%s clean_rates=%s gray_tiles=%s fps tile_KBs=%s loss/tile=%s "
            "ltr=ack%d/far%d/dist%d/miss%d "
            "loss_total=%d unmapped=%d "
            "ssrc_groups=%d last_publish=%.1fs ago "
            "udp_q=video[%d/%d drop=%d] ctrl[%d/%d drop=%d] "
            "rx_pps=video[%.1f/%dMbps] ctrl[%.1f/%.1fkbps] tcp[%.2f] "
            "tx_pps=%.2f cursor=%dms_ago/n=%d tcp_types={%s} "
            "nalu={%s}",
            decoder_name,
            good, rates, clean_rates, sorted(self._decoder.bad_tiles), tile_kbs, loss_delta,
            self._ltr_acks_sent, ltr_far, ltr_dist, ltr_miss,
            loss_total, loss_unmapped,
            ssrc_groups,
            last_publish_age,
            self._video_q.qsize(), _UDP_DRAIN_QUEUE_MAX, self._video_q_dropped,
            self._ctrl_q.qsize(), _UDP_DRAIN_QUEUE_MAX, self._ctrl_q_dropped,
            rx_v_pps, int(rx_v_mbps),
            rx_c_pps, rx_c_kbps,
            rx_t_pps,
            tx_pps,
            cursor_age_ms,
            self._cursor_msgs_processed,
            types_str,
            nalu_str,
        )

        # Mirror the same data, structured, onto the control socket for
        # any subscribed TUI / monitor clients. The shape is documented in
        # control.py.
        if self._control is not None:
            self._control.publish_snapshot({
                "uptime_s": (
                    time.time() - self._connect_wall_ts
                    if self._connect_wall_ts > 0 else 0.0
                ),
                "decoder": decoder_name,
                "tiles": [
                    {"frames": g, "fps": r, "loss_s": l}
                    for g, r, l in zip(good, rates, loss_delta)
                ],
                "loss_total": loss_total,
                "loss_unmapped": loss_unmapped,
                "ssrc_groups": ssrc_groups,
                "last_publish_age_s": round(last_publish_age, 2),
                "udp_q": {
                    "video": {"depth": self._video_q.qsize(),
                              "cap": _UDP_DRAIN_QUEUE_MAX,
                              "drop": self._video_q_dropped},
                    "ctrl":  {"depth": self._ctrl_q.qsize(),
                              "cap": _UDP_DRAIN_QUEUE_MAX,
                              "drop": self._ctrl_q_dropped},
                },
                "rx": {"video_pps": rx_v_pps, "video_mbps": rx_v_mbps,
                       "ctrl_pps":  rx_c_pps, "ctrl_kbps":  rx_c_kbps,
                       "tcp_pps":   rx_t_pps},
                "tx": {"pps": tx_pps},
                "cursor": {"age_ms": cursor_age_ms,
                           "count": self._cursor_msgs_processed},
                "tcp_msg_types": {
                    f"{t:02x}": n
                    for t, n in sorted(self._rx_msg_type_counts.items())
                },
                "nalu_delta_per_tile": [
                    {str(t): n for t, n in d.items()}
                    for d in nalu_delta
                ],
                "decode_latency_ms": round(
                    getattr(self._decoder, "decode_latency_ms", 0.0), 1
                ) if self._decoder is not None else None,
                "decode_q": {
                    "depth": getattr(self._decoder, "decode_queue_depth", 0),
                    "cap": getattr(self._decoder, "decode_queue_cap", 512),
                    "drop": getattr(self._decoder, "decode_queue_drops", 0),
                } if self._decoder is not None else None,
            })

    def _send_heartbeat(self) -> None:
        # Apple rtcp-muxes audio onto the same UDP port as control RTCP
        # (UDP 5900). Send our PT=101 keepalive there, not to a
        # separate audio port — Apple doesn't listen anywhere else.
        sock = self._sock_ctrl
        enc = self._audio_encryptor
        if sock is None or enc is None:
            return
        try:
            pkt = enc.encrypt(_HEARTBEAT_PAYLOAD, pt=_AUDIO_PT)
            sock.sendto(pkt, (self._dest_host, self._ctrl_dest_port))
        except OSError as e:
            log.debug("heartbeat send failed: %s", e)

    def _send_rr_and_maybe_sr(self) -> None:
        sock = self._sock_ctrl
        enc = self._srtcp_enc
        sender_ssrc = self._our_video_ssrc
        if sock is None or enc is None or sender_ssrc is None:
            return

        sources = list(self._ssrc_to_tile.keys())
        ssrc_stats = {
            s: {"max_seq": self._max_seq.get(s, 0), "roc": self._roc.get(s, 0)}
            for s in sources
        }
        rr = build_rr(
            sender_ssrc, source_ssrcs=sources,
            ssrc_stats=ssrc_stats, sr_data=self._server_sr,
        )
        if self._tx_tick % _RTCP_SR_EVERY_N_TICKS == 0:
            rr = build_empty_sr(sender_ssrc) + rr

        try:
            sock.sendto(enc.protect(rr), (self._dest_host, self._ctrl_dest_port))
        except OSError as e:
            log.debug("RR/SR send failed: %s", e)

    def _drain_pending_fir(self) -> None:
        if self._decoder is None:
            return
        # Apple-idle suppression: if no video packet has arrived in the
        # last 1.5 s, Apple's encoder rate-controlled to silence on a
        # static screen — there's nothing to FIR productively (Apple
        # ISN'T encoding). Wait for packets to resume; the gate's
        # keyframe_required stays populated so the next packet that
        # does arrive triggers a fresh FIR cycle naturally.
        if self._last_video_pkt_t > 0.0:
            quiet_for = time.monotonic() - self._last_video_pkt_t
            if quiet_for >= 1.5:
                return
        sent: list[int] = []
        for ti in self._decoder.consume_fir_request():
            if self._send_fir_for_tile(ti, log_per_tile=False):
                sent.append(ti)
        if sent:
            log.debug("FIR/PLI sent for tiles %s", sorted(sent))
        # Flush any pending gray-out aggregation that's older than the
        # debounce window so a single-tile event still surfaces if no
        # follow-up FIR fires.
        if (self._grayout_window_tiles
                and time.monotonic() - self._grayout_window_t
                >= self._GRAYOUT_WINDOW_S):
            self._flush_grayout_event()

    def _send_fir_for_tile(self, tile_idx: int, log_per_tile: bool = True) -> bool:
        """Returns True if a FIR was actually emitted (False if rate-
        limited or missing wire state). `log_per_tile=False` suppresses
        the per-tile DEBUG line so the bulk-drain caller can log a
        single aggregated line for the whole set."""
        target_ssrc = next(
            (s for s, t in self._ssrc_to_tile.items() if t == tile_idx),
            None,
        )
        if target_ssrc is None or self._our_video_ssrc is None:
            return False
        # Coalesce multi-source FIR storms — see _FIR_MIN_INTERVAL_S.
        now = time.monotonic()
        last = self._last_fir_per_tile.get(tile_idx, 0.0)
        if now - last < self._FIR_MIN_INTERVAL_S:
            return False
        self._last_fir_per_tile[tile_idx] = now
        sock = self._sock_ctrl
        enc = self._srtcp_enc
        if sock is None or enc is None:
            return False
        # Combine the AVPF FIR (PT=206) + PLI with the LEGACY FIR (PT=192,
        # RFC 2032) the native viewer uses — screensharingd often ignores the
        # AVPF FIR but answers the legacy one with a fresh IDR. The server
        # processes whichever it honors first; recovery latency is gated on it.
        seq = (self._tx_tick & 0xFF)
        compound = compound_with_rr(
            self._our_video_ssrc,
            build_fir(self._our_video_ssrc, target_ssrc, seq)
            + build_pli(self._our_video_ssrc, target_ssrc)
            + build_fir_legacy(target_ssrc),
        )
        try:
            sock.sendto(enc.protect(compound), (self._dest_host, self._ctrl_dest_port))
            if log_per_tile:
                log.debug("FIR/PLI sent for tile %d (ssrc=0x%08x)", tile_idx, target_ssrc)
            self._record_grayout_tile(tile_idx, now)
            return True
        except OSError as e:
            log.debug("FIR send failed: %s", e)
            return False

    # Gray-out aggregator -------------------------------------------------
    # Coalesce per-tile FIR emissions inside a short window into one
    # session-level INFO line. The libav concealment message captured at
    # detection time is appended so we can tell LTRP-ack poisoning (same
    # ref/POC across tiles) from natural encoder DPB blips (different
    # refs per tile) without enabling -vv.
    _GRAYOUT_WINDOW_S: float = 0.25

    def _record_grayout_tile(self, tile_idx: int, now: float) -> None:
        if (not self._grayout_window_tiles
                or now - self._grayout_window_t >= self._GRAYOUT_WINDOW_S):
            if self._grayout_window_tiles:
                self._flush_grayout_event()
            self._grayout_window_t = now
        self._grayout_window_tiles.add(tile_idx)

    def _flush_grayout_event(self) -> None:
        if not self._grayout_window_tiles:
            return
        tiles = sorted(self._grayout_window_tiles)
        msg = self._last_concealment_msg or "no libav concealment captured"
        log.info("gray-out: tiles %s recovering via FIR (libav: %s)", tiles, msg)
        self._grayout_window_tiles.clear()
        self._last_concealment_msg = ""

    def _drain_pending_nack(self) -> None:
        sock = self._sock_ctrl
        enc = self._srtcp_enc
        sender = self._our_video_ssrc
        if sock is None or enc is None or sender is None:
            return
        for ssrc in list(self._nack_pending.keys()):
            seqs = self._nack_pending.pop(ssrc, set())
            if not seqs:
                continue
            nack = build_nack(sender, ssrc, seqs)
            if not nack:
                continue
            compound = compound_with_rr(sender, nack)
            try:
                sock.sendto(enc.protect(compound), (self._dest_host, self._ctrl_dest_port))
            except OSError as e:
                log.debug("NACK send failed: %s", e)

    def _check_stall(self) -> None:
        """Decoder-stall recovery. Two failure modes:

          A. *All* tiles frozen — `_last_publish_t` keeps advancing
             through staleness. Handled below: 3 s → FIR storm,
             6 s → decoder restart.
          B. *One tile* frozen but the others publish (Apple sometimes
             ignores per-tile FIRs after a SkyLight transition). The
             session-wide last_publish_t looks healthy because the
             other tiles update it. A per-tile watchdog keeps a backstop
             FIR storm on tiles whose `bad_streak` is high — but it does
             NOT restart/flush the shared decoder (that wipes the healthy
             tiles' DPB → all-tile freeze cascade; see hevc._try_recovery).
             The stuck tile rides its last frame until its IDR re-roots it.

        Only Path A's 15 s *total* freeze (no tile publishing at all) still
        falls back to a decoder restart — at that point there are no
        healthy tiles left to orphan.

        Never marks the session dead from this path — consumers
        handle reconnect on stall via the TCP read-loop separately."""
        if not self._connected or self._last_publish_t == 0.0:
            return
        gap = time.monotonic() - self._last_publish_t
        now = time.monotonic()

        # Track whether packet loss is actively growing — the discriminator
        # for Path B's two stuck-tile failure modes. A broken-ref stall from
        # real loss must NOT flush (cascade); a *lossless* stall is a genuine
        # VideoToolbox saturation wedge that only a restart can clear.
        cur_loss = self._lost_pkts
        if cur_loss > getattr(self, "_loss_at_prev_stall_check", 0):
            self._last_loss_growth_t = now
        self._loss_at_prev_stall_check = cur_loss

        # --- A: session-wide stall ---
        # Apple usually responds to a FIR within ~1 RTT, so keep
        # FIR-storming for a generous window before we burn a full
        # decoder restart. Restarts add ~1 s of dead time and clear
        # the DPB; if Apple's stream is just briefly silent (no new
        # IDR yet) the restart can't help — the next IDR-arrival
        # path is the same either way. Only restart after a really
        # long unrecovered silence (≥ 15 s) where the decoder may
        # genuinely be stuck on internal state.
        #
        # Saturation wedge (fast path): video packets ARE flowing but the
        # decoder has produced nothing for a few seconds AND loss is not
        # growing → a genuine VideoToolbox internal wedge (errno=35 on every
        # send, even the recovery IDR — incompressible content above the HW
        # decoder's real-time throughput). EAGAIN no longer bumps bad_streak
        # (see hevc._handle_decode_error) so Path B can't catch this, and the
        # no-flush path can't clear an internal VT wedge — only a codec
        # rebuild does. Gated on NO recent loss so a broken-ref (loss) stall
        # never restarts here (it stays no-flush). Catches it at ~4 s instead
        # of the 15 s last resort.
        recent_loss = (now - getattr(self, "_last_loss_growth_t", 0.0)
                       < self._SATURATION_LOSS_FREE_WINDOW_S)
        pkts_flowing = (self._last_video_pkt_t > 0.0
                        and now - self._last_video_pkt_t < 0.5)
        if (gap > self._SATURATION_RESTART_GAP_S and self._decoder is not None
                and not recent_loss and pkts_flowing):
            last_restart = getattr(self, "_last_decoder_restart_t", 0.0)
            if now - last_restart >= 3.0:
                self._last_decoder_restart_t = now
                log.warning(
                    "decoder stuck %.1fs, packets flowing, no loss for %.0fs — "
                    "no-flush FIR didn't recover; restart decoder + FIR",
                    gap, self._SATURATION_LOSS_FREE_WINDOW_S,
                )
                try:
                    self._decoder.restart()
                except Exception as e:
                    log.debug("saturation restart failed: %s", e)
                self.request_fir()
                return
        if gap > 15.0 and self._decoder is not None:
            last_restart = getattr(self, "_last_decoder_restart_t", 0.0)
            if now - last_restart >= 8.0:
                # Same Apple-idle suppression as the 3 s path: if no
                # packets are arriving, Apple isn't encoding, restart
                # + FIR can't help. Wait for packets to resume.
                quiet_for = (now - self._last_video_pkt_t
                             if self._last_video_pkt_t > 0.0 else 0.0)
                if quiet_for >= 1.5:
                    return
                self._last_decoder_restart_t = now
                log.warning(
                    "decoder stuck %.1fs (long); restart decoder + FIR storm",
                    gap,
                )
                try:
                    self._decoder.restart()
                except Exception as e:
                    log.debug("decoder.restart() failed: %s", e)
                self.request_fir()
                return
        if gap > 3.0:
            last_fir = getattr(self, "_last_stall_fir_t", 0.0)
            if now - last_fir >= 1.5:
                # Apple-idle suppression: if no video packet has
                # arrived in the last 1.5 s, Apple's encoder has rate-
                # controlled to silence on a static screen, not
                # genuinely stuck. We don't have anything to FIR
                # productively (Apple's encoder ISN'T encoding) — wait
                # for it to wake on its own when content changes.
                quiet_for = now - self._last_video_pkt_t
                if (self._last_video_pkt_t > 0.0
                        and quiet_for >= 1.5):
                    return
                self._last_stall_fir_t = now
                # Defer to the gate if it's already actively trying to
                # recover all tiles — piling on more FIRs only causes
                # overlapping IDR responses (duplicate-POC rejection
                # in the decoder). Just log so the operator still sees
                # the stall, and let the gate's sticky retry handle it.
                if self._decoder is not None and len(
                        self._decoder._gate._keyframe_required
                        ) >= self.num_tiles:
                    log.warning(
                        "decoder stuck %.1fs (gate already recovering "
                        "all tiles, deferring)", gap,
                    )
                else:
                    log.warning("decoder stuck %.1fs; FIR storm", gap)
                    self.request_fir()

        # --- B: persistent decoder concealment ---
        # `bad_streak` counts libav-reported decode errors + RTP
        # sequence-gap events (the libav handler is rate-limited to
        # 250 ms, so max ~4 events/s). 30 events ≈ 7-8 s of sustained
        # concealment — long enough that a normal post-FIR recovery
        # has time to land an IDR + a couple of clean frames before
        # the watchdog forces another restart, but short enough that
        # a genuinely-stuck decoder gets recycled within 10 s.
        if self._decoder is None:
            return
        STUCK_TILE_ERRORS = 30
        try:
            states = [self._decoder._gate._states[i] for i in range(self.num_tiles)]
        except Exception:
            return
        worst = max((s.bad_streak for s in states), default=0)
        if worst >= STUCK_TILE_ERRORS:
            recent_loss = (now - getattr(self, "_last_loss_growth_t", 0.0)
                           < self._SATURATION_LOSS_FREE_WINDOW_S)
            stuck = [i for i in range(self.num_tiles)
                     if states[i].bad_streak >= STUCK_TILE_ERRORS]
            if recent_loss:
                # Broken reference chain from real packet loss. A stuck TILE
                # must NOT restart/flush the SHARED decoder: `restart()`
                # wipes all four tiles' DPB, orphaning the 3 healthy tiles
                # (no per-tile IDR re-roots them) → an all-tile freeze +
                # errno=35 restart-loop (the exact cascade hevc._try_recovery
                # avoids). Keep a backstop FIR storm; the healthy tiles run on
                # and hevc.py drops the stuck tile's P-frames until its IDR.
                last_fir = getattr(self, "_last_stuck_tile_fir_t", 0.0)
                if now - last_fir >= 2.0:
                    self._last_stuck_tile_fir_t = now
                    log.warning(
                        "tile stuck (worst bad_streak=%d, tiles=%s, loss "
                        "active); FIR storm, no flush (native-aligned)",
                        worst, stuck,
                    )
                    self.request_fir()
            else:
                # Lossless stall = genuine VideoToolbox saturation wedge
                # (errno=35 on every send, even the recovery IDR — content
                # above the HW decoder's real-time throughput). The no-flush
                # path can't clear an internal VT wedge; only a codec rebuild
                # does. Safe to restart here: with no loss there are no
                # missing refs to orphan, so no cascade. Throttled so the
                # decoder can re-bootstrap before another restart.
                last_restart = getattr(self, "_last_decoder_restart_t", 0.0)
                if now - last_restart >= 4.0:
                    self._last_decoder_restart_t = now
                    log.warning(
                        "tile stuck (worst bad_streak=%d, tiles=%s, no loss) "
                        "— VT saturation wedge; restart decoder + FIR",
                        worst, stuck,
                    )
                    try:
                        self._decoder.restart()
                    except Exception as e:
                        log.debug("saturation-wedge restart failed: %s", e)
                    self.request_fir()

    # ── tx-side port helpers ─────────────────────────────────────────

    @property
    def _ctrl_dest_port(self) -> int:
        return self._config.udp_ctrl_port or self._config.port

    @property
    def _video_dest_port(self) -> int:
        return self._config.udp_video_port or (self._config.port + 1)

    # ── socket helpers ───────────────────────────────────────────────

    @staticmethod
    def _resolve_host_ipv4(host: str) -> str:
        """Resolve `host` to an IPv4 literal so the TCP control connection
        lands on the same family as the IPv4-only UDP RTP/RTCP sockets.

        Bonjour `.local` names (and some dual-stack DNS) resolve to IPv6
        first on Windows — `socket.getaddrinfo` returns only AAAA records —
        so a default `create_connection` negotiates HP over IPv6 while the
        UDP sockets sit on IPv4 and never receive video. Numeric IPv4 hosts
        pass straight through. Raises ConnectionError if the host has no
        IPv4 address at all (use the Mac's IPv4 directly in that case)."""
        try:
            infos = socket.getaddrinfo(
                host, None, socket.AF_INET, socket.SOCK_STREAM,
            )
        except socket.gaierror as e:
            raise ConnectionError(
                f"{host} did not resolve to an IPv4 address ({e}). iss's "
                f"screen-share UDP transport is IPv4-only; connect using the "
                f"Mac's IPv4 address instead of a hostname."
            ) from e
        if not infos:
            raise ConnectionError(
                f"{host} has no IPv4 address (it resolves only to IPv6). "
                f"iss's screen-share UDP transport is IPv4-only; connect "
                f"using the Mac's IPv4 address instead."
            )
        return infos[0][4][0]

    def _punch_firewall(self) -> None:
        """Send one 1-byte datagram out of each media socket to the host's
        matching port, opening the local stateful firewall's inbound path
        for that port. The host streams symmetrically (its 5901 -> our 5901,
        its 5900 -> our 5900), so once we've sent outbound to those endpoints
        the host's inbound media counts as established return traffic and a
        non-admin viewer needs no inbound firewall rule.

        Driven repeatedly by `_firewall_punch_loop` for the whole handshake:
        a punch only sticks once the host is listening on the port (before
        that the ICMP unreachable tears the nascent pinhole down), and the
        host can start streaming at any point during negotiation, so we keep
        punching until the burst lands. The 1-byte payload is too short to
        be RTP, so the host's media depacketiser just drops it (UDP, so no
        RST). Best effort: on failure we fall back to needing an inbound
        rule."""
        host = self._dest_host
        for sock, port in ((self._sock_ctrl, self._ctrl_dest_port),
                           (self._sock_video, self._video_dest_port)):
            if sock is None:
                continue
            try:
                sock.sendto(b"\x00", (host, port))
            except OSError as e:
                log.debug("firewall punch on UDP %d failed: %s", port, e)

    def _firewall_punch_loop(self, stop: threading.Event) -> None:
        """Re-punch the media-port pinholes every 100 ms until `stop` is set.

        Run for the whole handshake: the host can begin streaming at any
        point during negotiation (notably mid degenerate-canvas re-query),
        and the pinhole has to already exist when its first media packet
        arrives. Punches sent while the host port is still closed are simply
        dropped (ICMP unreachable, harmless); the one that lands once the
        host is listening establishes the flow."""
        while not stop.is_set():
            self._punch_firewall()
            stop.wait(0.1)

    @staticmethod
    def _bind_udp(host: str, port: int) -> socket.socket:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Big rmem so the burst doesn't get dropped before we drain it.
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        s.bind((host, port))
        return s

    @staticmethod
    def _check_rmem_cap(sock: socket.socket) -> None:
        """Linux silently caps `SO_RCVBUF` at `net.core.rmem_max`. The
        default is ~208 KB on Ubuntu/Debian/Fedora — too small for
        Apple's HP burst — and the resulting kernel-level UDP drops
        produce gray-tile artifacts that don't show up in iss's
        per-tile RTP loss counter (drops happen *below* the socket
        layer, so we never see the gap).
        Read back what the kernel actually gave us and warn loudly with
        the one-liner fix if it's too small. No-op on macOS/Windows
        where the OS default is generous enough.
        """
        if sys.platform != "linux":
            return
        # Kernel doubles the requested SO_RCVBUF for bookkeeping
        # (socket(7)). We asked for 4 MiB, so a healthy readback is
        # ~8 MiB. Anything under ~2 MiB readback means the cap clamped
        # us hard.
        try:
            actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        except OSError:
            return
        if actual >= 2 * 1024 * 1024:
            return
        rmem_max = "<unknown>"
        try:
            with open("/proc/sys/net/core/rmem_max") as f:
                rmem_max = f.read().strip()
        except OSError:
            pass
        log.warning(
            "Linux UDP receive buffer is small (SO_RCVBUF readback=%d B, "
            "net.core.rmem_max=%s). With --no-curtain on, Apple's HEVC "
            "burst will overflow this buffer and the kernel will silently "
            "drop packets — expect gray-tile artifacts. To fix:\n"
            "  • drop --no-curtain (the default --curtain encodes a small "
            "virtual display whose bitrate fits the kernel default), or\n"
            "  • bump the kernel ceiling for this boot only:\n"
            "        sudo sysctl -w net.core.rmem_max=33554432\n"
            "  • or persist it across reboots:\n"
            "        echo 'net.core.rmem_max=33554432' | "
            "sudo tee /etc/sysctl.d/99-isharescreen.conf\n"
            "        sudo sysctl --system",
            actual, rmem_max,
        )

    @staticmethod
    def _drain_socket_into(
        sock: socket.socket, into: list[bytes], *, max_seconds: float,
    ) -> None:
        """Pull packets off `sock` for up to `max_seconds`. Used to drain
        the burst before the rx thread starts."""
        deadline = time.monotonic() + max_seconds
        sock.settimeout(0.05)
        while time.monotonic() < deadline:
            try:
                pkt, _ = sock.recvfrom(65536)
            except socket.timeout:
                # Burst typically lands within 200-400 ms; once we see
                # no packets for 50 ms after some have arrived, we have
                # enough.
                if into:
                    return
                continue
            except OSError:
                return
            into.append(pkt)
        sock.settimeout(_UDP_RECV_TIMEOUT_S)


__all__ = ["Session", "SessionConfig"]
