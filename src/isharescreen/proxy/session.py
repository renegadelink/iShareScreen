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
import socket
import struct
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from .media.tiles import TileFrame
from .input import InputController
from .media.aac_eld import AacEldDecoder, make_aac_eld_decoder
from .media.hevc import HevcDecoder
from .media.nalu import reassemble_group
from .media.quality_gate import TileVisState
from .protocol.burst import gather_initial_burst
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
    build_nack,
    build_pli,
    build_rr,
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

    # UDP port overrides — None ⇒ port + 0/1/2 (Apple's default convention).
    # Custom values let a NAT relay route the streams.
    udp_ctrl_port: Optional[int] = None
    udp_video_port: Optional[int] = None
    udp_audio_port: Optional[int] = None
    udp_bind_host: str = ""  # "" = INADDR_ANY


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
        self._decoder: Optional[HevcDecoder] = None
        self._aac: Optional[AacEldDecoder] = None
        self._input: Optional[InputController] = None
        self._ssrc_to_tile: dict[int, int] = {}
        self._last_ssrc_adopt_ts: float = 0.0
        self._ssrc_blacklist: set[int] = set()
        self._last_profile_good: list[int] = []

        # Cipher state for the TX channel.
        self._video_decryptor: Optional[SRTPDecryptor] = None
        self._audio_decryptor: Optional[SRTPDecryptor] = None
        self._audio_encryptor: Optional[SRTPEncryptor] = None
        self._srtcp_dec: Optional[SRTCPDecryptor] = None
        self._srtcp_enc: Optional[SRTCPEncryptor] = None
        self._our_video_ssrc: Optional[int] = None
        self._our_audio_ssrc: Optional[int] = None

        # UDP sockets (bound on connect, closed on close).
        self._sock_ctrl: Optional[socket.socket] = None
        self._sock_video: Optional[socket.socket] = None
        self._sock_audio: Optional[socket.socket] = None

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

        # Audio sink. Set by consumer; called whenever a PCM chunk decodes.
        self._audio_callback: Optional[Callable[[np.ndarray], None]] = None

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

    def request_fir(self, tile_idx: Optional[int] = None) -> None:
        """Externally trigger an FIR (forces a fresh IDR). Without args,
        targets every tile. Mainly useful for tests / debug; the gate
        normally drives FIR autonomously."""
        if self._decoder is None or self._negotiation is None:
            return
        if tile_idx is None:
            for ti in range(self.num_tiles):
                self._send_fir_for_tile(ti)
        else:
            self._send_fir_for_tile(tile_idx)

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
        n = self._negotiation
        return (n.canvas_width, n.canvas_height) if n else (0, 0)

    @property
    def num_tiles(self) -> int:
        n = self._negotiation
        return n.canvas_tiles if n and n.canvas_tiles else 4

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

        # 1) Bind UDP sockets BEFORE the handshake — the session-start
        # burst lands within ~100 ms of the 0x1c answer, and the kernel
        # would drop early packets if the sockets aren't ready.
        ctrl_port = cfg.udp_ctrl_port or cfg.port
        video_port = cfg.udp_video_port or (cfg.port + 1)
        audio_port = cfg.udp_audio_port or (cfg.port + 2)
        self._sock_ctrl = self._bind_udp(cfg.udp_bind_host, ctrl_port)
        self._sock_video = self._bind_udp(cfg.udp_bind_host, video_port)
        self._sock_audio = self._bind_udp(cfg.udp_bind_host, audio_port)

        # 2) Apple's two-TCP warmup (TCP #1: register session with daemon).
        if cfg.warmup_tcp:
            try:
                warmup_tcp(cfg.host, cfg.port)
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

        # 3) Build offers; remember our SSRCs (server only sends to these).
        # We tested omitting the audio offer entirely when `cfg.audio`
        # is False to save the server from allocating audio at all,
        # but Apple's daemon depends on the audio section being present
        # in the 0x1c — without it, the encoder-canvas reply degenerates
        # and the burst never arrives. So we always send a valid audio
        # offer; `cfg.audio=False` only skips local decode + playback.
        # Wire bandwidth cost: zero on a silent host (Apple doesn't
        # send audio RTP unless sound is actually playing); a few KB/s
        # of dropped-on-floor traffic when audio is playing.
        video_offer, audio_offer = create_offers()
        self._our_video_ssrc = extract_offer_ssrc(video_offer, is_video=True)
        self._our_audio_ssrc = extract_offer_ssrc(audio_offer, is_video=False)
        log.info(
            "our SSRCs: video=0x%08x audio=0x%08x (decode=%s)",
            self._our_video_ssrc or 0, self._our_audio_ssrc or 0,
            "on" if cfg.audio else "off",
        )

        # 4) TCP #2: full handshake.
        self._negotiation = connect_and_negotiate(
            cfg.host, cfg.port, cfg.username, cfg.password,
            auth_mode=cfg.auth_mode,
            advertise=cfg.advertise,
            hdr=cfg.hdr,
            curtain=cfg.curtain,
            audio_offer=audio_offer,
            video_offer=video_offer,
            share_console=cfg.share_console,
            alt_session=cfg.alt_session,
        )

        # 5) SRTP / SRTCP key derivation (both directions).
        keys = self._negotiation.keys
        self._video_decryptor = self._negotiation.video_decryptor
        self._audio_decryptor = SRTPDecryptor.from_blob(keys.audio_key_s)
        self._srtcp_dec = SRTCPDecryptor.from_blob(keys.video_key_s)
        self._srtcp_enc = SRTCPEncryptor.from_blob(keys.video_key_v)
        if self._our_audio_ssrc is not None:
            self._audio_encryptor = SRTPEncryptor.from_blob(
                keys.audio_key_v, self._our_audio_ssrc,
            )

        # 6) Initial burst — VPS/SPS/PPS + first IDR.
        # The video RX thread isn't running yet, so the burst sits in the
        # kernel UDP buffer; gather_initial_burst drains it.
        burst_buf: list[bytes] = []
        self._drain_socket_into(self._sock_video, burst_buf, max_seconds=2.0)
        burst = gather_initial_burst(
            burst_buf, self._video_decryptor,
            quality_tier=cfg.quality_tier,
        )
        self._ssrc_to_tile = dict(burst.ssrc_to_tile)
        self._pending_groups = burst.burst_pending
        self._group_arrival = {key: time.monotonic() for key in burst.burst_pending}

        # 7) Decoder init.
        num_tiles = self._negotiation.canvas_tiles or 4
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
        prefer_hwaccel = _os.environ.get("ISS_FORCE_SW_HEVC", "0") == "0"
        self._decoder = HevcDecoder(
            num_tiles=num_tiles,
            enable_quality_gate=True,
            on_frame_published=self._on_frame_published,
            prefer_hwaccel=prefer_hwaccel,
        )
        if not prefer_hwaccel:
            log.info("ISS_FORCE_SW_HEVC=1: HW decoders disabled")
        self._decoder.set_params(burst.vps, burst.sps, burst.all_pps)
        self._decoder.start()
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

        # 10) Spawn rx + tx threads.
        self._stop_evt.clear()
        self._fresh_evt.clear()
        self._last_publish_t = time.monotonic()
        self._spawn_threads()

        # 11) FIR any tile that didn't pick up a real IDR in the burst.
        # Apple's HP encoder sometimes only emits an IDR for tile 0 in
        # the opening packets and expects the client to ask for IDRs for
        # the others; without that, tiles 1-3 never decode and show as
        # gray fill. The cost is one extra round-trip per missing tile.
        missing_idr = [ti for ti in range(num_tiles) if ti not in burst.last_idr]
        if missing_idr:
            log.info("burst missed IDRs for tiles %s; sending FIR", missing_idr)
            for ti in missing_idr:
                self._send_fir_for_tile(ti)

    def _teardown(self) -> None:
        # Drop ourselves from the libav-handler's active-sessions set
        # so concealment events stop routing here and the Session can
        # be garbage-collected on reconnect cycles.
        active = getattr(Session, "_active_sessions", None)
        if active is not None:
            active.discard(self)

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

        for sock_attr in ("_sock_ctrl", "_sock_video", "_sock_audio"):
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

    # ── thread spawning ──────────────────────────────────────────────

    def _spawn_threads(self) -> None:
        targets: list[tuple[str, Callable]] = [
            ("iss-video-rx", self._video_rx_loop),
            ("iss-ctrl-rx", self._ctrl_rx_loop),
            ("iss-tcp-rx", self._tcp_rx_loop),
            ("iss-tx", self._tx_loop),
        ]
        if self._config.audio:
            targets.append(("iss-audio-rx", self._audio_rx_loop))
        for name, target in targets:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    # ── decoder publish hook ─────────────────────────────────────────

    def _on_frame_published(self, _tile_idx: int) -> None:
        """Called from a HevcDecoder worker thread after a frame is
        published. Wakes anyone in `wait_for_fresh_tile`."""
        self._last_publish_t = time.monotonic()
        self._fresh_evt.set()

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
                    msg = record.getMessage().lower()
                except Exception:
                    return
                if not any(kw in msg for kw in concealment_keywords):
                    return
                for sess in tuple(getattr(Session, "_active_sessions", ())):
                    try:
                        sess._on_libav_concealment(record.getMessage().strip())
                    except Exception:
                        pass

        handler = _LibavConcealmentHandler(level=logging.WARNING)
        logging.getLogger("libav").addHandler(handler)
        Session._libav_log_installed = True
        log.info("libav concealment-log handler installed")

    def _on_libav_concealment(self, msg: str) -> None:
        """Soft-concealment path: rate-limited FIR on tile 0 with a
        steady-state guard. We deliberately do NOT trigger a full
        decoder flush here — flushing libavcodec via flush_buffers()
        wedges vaapi (Linux) and d3d11va in some configurations,
        producing a permanent freeze instead of recovery."""
        if self._decoder is None:
            return
        now = time.monotonic()
        if (self._last_publish_t > 0.0
                and now - self._last_publish_t < 0.5):
            return
        last = getattr(self, "_last_libav_fir_t", 0.0)
        if now - last < 2.0:
            return
        self._last_libav_fir_t = now
        log.warning("libav decoder error: %s", msg[:120])
        self._decoder._gate.mark_decode_error(0)

    # ── video RX loop ────────────────────────────────────────────────

    def _video_rx_loop(self) -> None:
        sock = self._sock_video
        decryptor = self._video_decryptor
        if sock is None or decryptor is None:
            return
        sock.settimeout(_UDP_RECV_TIMEOUT_S)

        while not self._stop_evt.is_set():
            try:
                pkt, _addr = sock.recvfrom(65536)
            except socket.timeout:
                self._evict_stale_groups()
                continue
            except OSError:
                return

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

        for nalu in reassemble_group([p for _, _, p in packets]):
            self._decoder.feed_nalu(nalu, ti)

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
        if len(candidates) < 4:
            return
        # Apple emits 4 CONSECUTIVE SSRCs per tile group (one SSRC per
        # tile). Picking the first 4 sorted candidates can grab SSRCs
        # from TWO different broadcast groups when Apple is double-
        # publishing — the resulting Frankenstein map decodes 2 of 4
        # tiles correctly and silently drops the others. Build runs of
        # consecutive SSRCs and adopt the first complete-4 run.
        new_group: list[int] | None = None
        run = [candidates[0]]
        for s in candidates[1:]:
            if s - run[-1] <= 1 and len(run) < 4:
                run.append(s)
                if len(run) == 4:
                    new_group = run
                    break
            else:
                run = [s]
        if new_group is None:
            return  # no consecutive 4-run yet — wait for more data
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
        self._last_ssrc_adopt_ts = now
        # Grace window: pretend frames just flowed so the frame-flow gate
        # gives the new mapping ~0.5 s to start producing before we
        # consider another adoption.
        self._last_publish_t = now
        if self._decoder is not None:
            # Restart the decoder + FIR for the new tiles. We tried
            # skipping the restart to avoid a 1.5–6 s outage, but
            # without it the SW fallback path can't re-feed burst
            # NALUs and the new context starves until an unprompted
            # IDR shows up (often never).
            self._decoder.restart()
            self.request_fir()

    # ── audio RX loop ────────────────────────────────────────────────

    def _audio_rx_loop(self) -> None:
        sock = self._sock_audio
        decryptor = self._audio_decryptor
        decoder = self._aac
        if sock is None or decryptor is None or decoder is None:
            log.warning(
                "audio rx loop NOT starting: sock=%s decryptor=%s decoder=%s",
                sock, decryptor, decoder,
            )
            return
        sock.settimeout(_UDP_RECV_TIMEOUT_S)
        log.info("audio rx loop started")
        pkt_count = 0
        decoded_count = 0
        last_log = time.monotonic()

        while not self._stop_evt.is_set():
            now = time.monotonic()
            if now - last_log >= 5.0:
                log.info(
                    "audio rx: %d packets, %d decoded in last %.1fs "
                    "(callback=%s)",
                    pkt_count, decoded_count, now - last_log,
                    "set" if self._audio_callback else "unset",
                )
                pkt_count = decoded_count = 0
                last_log = now
            try:
                pkt, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            pkt_count += 1
            res = decryptor.decrypt(pkt)
            if res is None:
                continue
            _hdr, payload = res
            try:
                pcm = decoder.decode(payload)
            except Exception as e:
                log.warning("AAC-ELD decode error: %s", e)
                continue
            if pcm is not None:
                decoded_count += 1
            cb = self._audio_callback
            if pcm is not None and cb is not None:
                try:
                    cb(pcm)
                except Exception as e:
                    log.warning("audio callback raised: %s", e)

    # ── RTCP RX loop (server SR for jitter / dlsr) ──────────────────

    def _ctrl_rx_loop(self) -> None:
        """Read packets from the muxed CTRL/audio UDP port.

        Apple sends BOTH RTCP (PT 200-207) AND audio RTP on the same
        port (5900 by default). We classify by checking the RTP
        version + PT byte: PT 200-207 → SRTCP; otherwise → audio RTP.
        Without this classification, audio packets sit unprocessed in
        the kernel buffer and the host-side audio appears silent."""
        sock = self._sock_ctrl
        srtcp_dec = self._srtcp_dec
        if sock is None or srtcp_dec is None:
            return
        sock.settimeout(_UDP_RECV_TIMEOUT_S)

        while not self._stop_evt.is_set():
            try:
                pkt, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                return
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
        feed it to the AAC-ELD decoder, dispatch PCM to the callback.
        Same logic as `_audio_rx_loop` body — extracted so both the
        muxed and dedicated audio sockets feed one pipeline."""
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
        if pcm is not None and cb is not None:
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
        msg_type = msg[0]
        # 0x14 UserSessionChanged: agent transition (lock/login/desktop).
        # Logging is enough for v0.2 — the dynamic SSRC adoption already
        # handles the visible side (decoder restart on new SSRCs).
        if msg_type == 0x14:
            log.info("server sent UserSessionChanged (0x14): %s", msg.hex())

    # ── TX loop (heartbeat + RTCP + on-demand FIR/PLI/NACK + watchdog) ─

    def _tx_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._tx_tick += 1
            try:
                self._send_heartbeat()
                self._send_rr_and_maybe_sr()
                self._drain_pending_fir()
                self._drain_pending_nack()
                self._check_stall()
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
        log.info(
            "profile: decoder=%s tiles=%s rates=%s fps loss/tile=%s "
            "ssrc_groups=%d last_publish=%.1fs ago",
            self._decoder._hw_name or "software",
            good, rates, loss_delta,
            len(self._video_decryptor.ssrc_counts) // 4 if self._video_decryptor else 0,
            time.monotonic() - self._last_publish_t if self._last_publish_t > 0 else -1.0,
        )

    def _send_heartbeat(self) -> None:
        sock = self._sock_audio
        enc = self._audio_encryptor
        if sock is None or enc is None:
            return
        try:
            pkt = enc.encrypt(_HEARTBEAT_PAYLOAD, pt=_AUDIO_PT)
            sock.sendto(pkt, (self._config.host, self._audio_dest_port))
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
            sock.sendto(enc.protect(rr), (self._config.host, self._ctrl_dest_port))
        except OSError as e:
            log.debug("RR/SR send failed: %s", e)

    def _drain_pending_fir(self) -> None:
        if self._decoder is None:
            return
        for ti in self._decoder.consume_fir_request():
            self._send_fir_for_tile(ti)

    def _send_fir_for_tile(self, tile_idx: int) -> None:
        target_ssrc = next(
            (s for s, t in self._ssrc_to_tile.items() if t == tile_idx),
            None,
        )
        if target_ssrc is None or self._our_video_ssrc is None:
            return
        sock = self._sock_ctrl
        enc = self._srtcp_enc
        if sock is None or enc is None:
            return
        # Combine PLI (lighter, lower-priority) with FIR via the compound
        # builder — server processes whichever it honors first.
        seq = (self._tx_tick & 0xFF)
        compound = compound_with_rr(
            self._our_video_ssrc,
            build_fir(self._our_video_ssrc, target_ssrc, seq)
            + build_pli(self._our_video_ssrc, target_ssrc),
        )
        try:
            sock.sendto(enc.protect(compound), (self._config.host, self._ctrl_dest_port))
            log.debug("FIR/PLI sent for tile %d (ssrc=0x%08x)", tile_idx, target_ssrc)
        except OSError as e:
            log.debug("FIR send failed: %s", e)

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
                sock.sendto(enc.protect(compound), (self._config.host, self._ctrl_dest_port))
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
             other tiles update it. We need a per-tile watchdog: if
             any tile's quality-gate has been firing bad-state for
             >3 s, force a decoder restart so all tiles re-bootstrap
             from a fresh burst.

        Never marks the session dead from this path — consumers
        handle reconnect on stall via the TCP read-loop separately."""
        if not self._connected or self._last_publish_t == 0.0:
            return
        gap = time.monotonic() - self._last_publish_t
        now = time.monotonic()

        # --- A: session-wide stall ---
        # Apple usually responds to a FIR within ~1 RTT, so keep
        # FIR-storming for a generous window before we burn a full
        # decoder restart. Restarts add ~1 s of dead time and clear
        # the DPB; if Apple's stream is just briefly silent (no new
        # IDR yet) the restart can't help — the next IDR-arrival
        # path is the same either way. Only restart after a really
        # long unrecovered silence (≥ 15 s) where the decoder may
        # genuinely be stuck on internal state.
        if gap > 15.0 and self._decoder is not None:
            last_restart = getattr(self, "_last_decoder_restart_t", 0.0)
            if now - last_restart >= 8.0:
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
                self._last_stall_fir_t = now
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
            last_restart = getattr(self, "_last_decoder_restart_t", 0.0)
            if now - last_restart >= 4.0:
                self._last_decoder_restart_t = now
                log.warning(
                    "tile stuck (worst bad_streak=%d errors); "
                    "restart decoder + FIR storm",
                    worst,
                )
                try:
                    self._decoder.restart()
                except Exception as e:
                    log.debug("per-tile-stall restart failed: %s", e)
                self.request_fir()

    # ── tx-side port helpers ─────────────────────────────────────────

    @property
    def _ctrl_dest_port(self) -> int:
        return self._config.udp_ctrl_port or self._config.port

    @property
    def _audio_dest_port(self) -> int:
        return self._config.udp_audio_port or (self._config.port + 2)

    # ── socket helpers ───────────────────────────────────────────────

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
