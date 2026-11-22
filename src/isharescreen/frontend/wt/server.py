"""WebTransport bridge server.

aioquic provides QUIC + HTTP/3 + WebTransport. We accept HTTP CONNECT
with `:protocol = webtransport` on a single endpoint (`/wt`), serve the
viewer page on `GET /`, and per WT session:

  - Open one unidirectional stream PER ENCODED FRAME from server to
    client. (Per-frame stream isolates head-of-line blocking — a lost
    or slow stream affects only its frame.)
  - Accept one bidirectional stream from client carrying JSON input
    events (mouse, key, scroll).

Multi-viewer fan-out: one iss `Session` + one libx264 encoder produce
encoded NALU bytes once per frame; the bytes are written to every
active WebTransport session in parallel.

The handshake puts the cert SHA-256 fingerprint in the URL so the
browser passes it to `serverCertificateHashes` and skips CA validation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from aiohttp import web
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.asyncio.server import serve as quic_serve
from aioquic.h3.connection import H3_ALPN, H3Connection, SettingsError
from aioquic.h3.events import (
    DatagramReceived,
    DataReceived,
    H3Event,
    HeadersReceived,
    WebTransportStreamDataReceived,
)
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import ProtocolNegotiated, QuicEvent

from ...proxy.session import Session, SessionConfig
from ...proxy.media.avc_nalu import au_is_keyframe, au_has_sps
from ...proxy.media.avc import _patch_avc_sps_dpb
from .audio import OpusEncoder
from .capsule import CapsuleType, H3Capsule, H3CapsuleDecoder
from .cert import write_cert, trust_cert_in_keychain, is_cert_trusted


log = logging.getLogger(__name__)

# Diagnostic flag: dump each browser's/our HTTP/3 SETTINGS on connect (reveals
# which WebTransport draft the client speaks). Off by default (not spammy).
_DEBUG_SETTINGS = os.environ.get("ISS_WT_DEBUG_SETTINGS", "0") == "1"

# Apple's H.264 SPS declares max_num_ref_frames=8 / level 5.2, but under motion
# + packet loss its encoder references past 8 pictures (short-term + long-term
# refs) for loss robustness. A *strict* WebCodecs decoder — verified on Chrome
# for Windows (d3d11va) — honors the declared 8, can't hold the extra
# references, and the decoder freezes to a stale/black frame until the next
# keyframe. Chrome-on-macOS uses VideoToolbox, which ignores the limit and
# decodes fine — which is why this only bites on Windows. iss already patches
# the SPS DPB ceiling (→16, level 6.0) for its *own* libav decoder
# (media/avc.py) but forwarded Apple's ORIGINAL SPS to the browser. Patch the
# forwarded SPS too so the browser's decoder sizes its DPB to hold Apple's full
# reference set. ISS_BROWSER_SPS_PATCH=0 restores the old (raw) behavior for A/B.
_BROWSER_SPS_PATCH = os.environ.get("ISS_BROWSER_SPS_PATCH", "1") != "0"

# --- HTTP/3 SETTINGS codepoints for WebTransport ----------------------
# aioquic 1.3 only knows the draft-02 ENABLE_WEBTRANSPORT (0x2b603742) and the
# RFC 9297 H3_DATAGRAM (0x33). Chrome speaks draft-02, so it works out of the
# box. Safari/WebKit speaks the standardized draft (draft-ietf-webtrans-http3),
# which gates sessions on SETTINGS_WT_MAX_SESSIONS (0xc671706a) and runs the
# Capsule Protocol on the CONNECT stream (see _WebTransportH3 and the session-
# stream routing in _BridgeProtocol). We advertise BOTH the draft-02 and the
# standardized codepoints so a single endpoint serves Chrome and Safari — this
# mirrors the web-platform-tests h3 WebTransport server, which is interop-tested
# against both engines.
_H3_DATAGRAM_RFC = 0x33          # RFC 9297
_H3_DATAGRAM_DRAFT04 = 0xFFD277  # draft-ietf-masque-h3-datagram-04
_ENABLE_CONNECT_PROTOCOL = 0x08  # RFC 9220
_WT_MAX_SESSIONS = 0xC671706A    # draft-ietf-webtrans-http3 (Safari keys on this)


class _WebTransportH3(H3Connection):
    """aioquic's H3Connection extended to advertise the standardized-draft
    WebTransport settings and to track which HTTP-datagram format the peer
    negotiated (so we know datagrams are usable and in which encoding)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # None until SETTINGS arrive; then _H3_DATAGRAM_RFC / _DRAFT04.
        self.datagram_setting: Optional[int] = None

    def _get_local_settings(self):  # type: ignore[no-untyped-def]
        # Base already adds ENABLE_WEBTRANSPORT (0x2b603742) + H3_DATAGRAM
        # (0x33) + ENABLE_CONNECT_PROTOCOL when enable_webtransport=True; keep
        # those (Chrome requires the draft-02 codepoint) and add the
        # standardized ones Safari needs.
        settings = super()._get_local_settings()
        settings[_H3_DATAGRAM_RFC] = 1
        settings[_H3_DATAGRAM_DRAFT04] = 1
        settings[_ENABLE_CONNECT_PROTOCOL] = 1
        settings[_WT_MAX_SESSIONS] = 1
        if _DEBUG_SETTINGS:
            log.info("server SENT H3 SETTINGS: %s",
                     {hex(k): v for k, v in settings.items()})
        return settings

    def _validate_settings(self, settings) -> None:  # type: ignore[no-untyped-def]
        if _DEBUG_SETTINGS:
            log.info("browser H3 SETTINGS: %s",
                     {hex(k): v for k, v in settings.items()})
        # Record the datagram format the peer offered (gates send_datagram so we
        # never blast audio at a peer that can't receive it).
        if settings.get(_H3_DATAGRAM_RFC) == 1:
            self.datagram_setting = _H3_DATAGRAM_RFC
        elif settings.get(_H3_DATAGRAM_DRAFT04) == 1:
            self.datagram_setting = _H3_DATAGRAM_DRAFT04
        else:
            self.datagram_setting = None
            log.warning("peer negotiated no HTTP-datagram support; "
                        "audio-over-datagram disabled for this session")
        # Run aioquic's real validation (the 0/1-value checks run first, so
        # malformed settings are still rejected) but tolerate ONLY its datagram/
        # WebTransport strictness: it predates the standardized WebTransport +
        # RFC 9297 drafts Safari speaks and raises SettingsError on Safari's
        # otherwise-valid SETTINGS. Video still flows over streams; audio is
        # already gated above, so leniency here is safe.
        try:
            super()._validate_settings(settings)
        except SettingsError as e:
            log.debug("tolerating H3 SETTINGS strictness for standardized-draft "
                      "peer: %s", e)

_STATIC_DIR = Path(__file__).parent / "static"

# Experiment flag (feature/single-stream-video-transport): send all video
# frames over ONE long-lived unidirectional stream (length-prefixed) instead
# of a fresh stream per frame. Set ISS_WT_VIDEO_SINGLE_STREAM=1 to enable.
_VIDEO_SINGLE_STREAM = os.environ.get("ISS_WT_VIDEO_SINGLE_STREAM", "0") == "1"
# Single-stream drop-when-behind threshold: when the one video stream's unacked
# backlog exceeds this, DELTA frames stop being written (and a keyframe is
# requested) until the stream drains, bounding end-to-end latency. Roughly this
# many bytes of video is the worst-case added latency before dropping kicks in
# (~100ms at 20 Mbit/s for 256 KB). Lower = tighter latency but more frequent
# keyframe resyncs; tune via ISS_WT_SINGLE_STREAM_HIGH_WATER. The per-frame
# transport bounds latency for free via QUIC's stream-count limit; the single
# stream needs this explicit.
_SINGLE_STREAM_HIGH_WATER = int(
    os.environ.get("ISS_WT_SINGLE_STREAM_HIGH_WATER", str(256 * 1024)))
# Profiling: log iss-side forward gaps (>80ms between AUs) to localize stutter.
_PROFILE = os.environ.get("ISS_WT_PROFILE", "0") == "1"

# Send a fresh keyframe whenever a new viewer joins, so they don't
# wait for the next periodic intra-refresh column to complete.
_FORCE_KEYFRAME_ON_JOIN = True


class _ViewerSession:
    """Per-browser-tab state — the WT session, the bidi input stream,
    and bookkeeping the bridge needs to fan-out encoded bytes."""

    __slots__ = (
        "session_id", "stream_id", "h3", "loop", "transmit",
        "needs_keyframe", "input_stream_id", "input_buffer",
        "input_callback", "dgrams_sent", "last_seen_t",
        "video_stream_id", "frames_dropped", "video_discontinuity",
        "capsule_decoder", "use_single_stream", "request_keyframe",
    )

    def __init__(
        self,
        session_id: int,
        stream_id: int,
        h3: H3Connection,
        loop: asyncio.AbstractEventLoop,
        transmit,
        input_callback,
        use_single_stream: bool = False,
        request_keyframe=None,
    ) -> None:
        self.session_id = session_id
        # The stream_id of the CONNECT request — closing this closes
        # the WT session.
        self.stream_id = stream_id
        self.h3 = h3
        self.loop = loop
        # `transmit` is the QuicConnectionProtocol's `transmit()`
        # method. aioquic queues writes in send_data but only flushes
        # them to UDP when transmit() is called. Auto-flush happens at
        # the END of `quic_event_received` — but our send happens from
        # an async callback (call_soon_threadsafe) outside that path,
        # so we must invoke it manually.
        self.transmit = transmit
        self.needs_keyframe = True
        self.input_stream_id: Optional[int] = None
        self.input_buffer = bytearray()
        self.input_callback = input_callback
        self.dgrams_sent = 0
        self.last_seen_t = time.monotonic()
        # Single-stream video mode: the one long-lived uni stream all
        # frames are length-prefixed onto (None until first frame).
        self.video_stream_id: Optional[int] = None
        self.frames_dropped = 0
        # True after a drop, until the next keyframe resyncs the client —
        # while set we drop ALL deltas (sending one would reference a frame
        # the client never received).
        self.video_discontinuity = False
        # Standardized-draft (Safari) sessions run the Capsule Protocol on the
        # CONNECT stream. Decode bytes that arrive there as capsules rather than
        # HTTP DATA frames. draft-02 (Chrome) sends nothing here, so this stays
        # idle for those sessions.
        self.capsule_decoder = H3CapsuleDecoder()
        # Per-viewer transport: single long-lived uni stream (Safari, to stay
        # under its WT uni-stream credit) vs one stream per frame (default).
        self.use_single_stream = use_single_stream
        # Callback to ask the source for a fresh keyframe (FIR) — used by the
        # single-stream reset-on-backlog path to resync after discarding a
        # stale backlog. None in contexts without a bridge (e.g. tests).
        self.request_keyframe = request_keyframe

    def send_frame(self, payload: bytes) -> None:
        """Deliver one encoded-frame envelope to the browser.

        Two transports, selected by ISS_WT_VIDEO_SINGLE_STREAM:

        - default (per-frame): a fresh unidirectional stream PER FRAME. A
          slow/behind client backs up at the QUIC stream-count limit, so iss
          stops being able to open new streams and effectively DROPS the newest
          frames rather than buffering them forever — bounding latency. Cost:
          no ordering between streams, so the client reorders by seq (jitter),
          and the WT stream-credit limit breaks Safari.

        - single-stream: ONE long-lived uni stream, each envelope length-
          prefixed ([u32 len][envelope]). Ordered (no client reorder), 1 stream
          credit (Safari-friendly). Cost: head-of-line blocking on loss, and it
          can buffer unboundedly if the client falls behind (no drop yet).
        """
        if _VIDEO_SINGLE_STREAM or self.use_single_stream:
            self._send_frame_single(payload)
            return
        try:
            quic = self.h3._quic
            new_id = self.h3.create_webtransport_stream(
                session_id=self.session_id, is_unidirectional=True,
            )
            # aioquic bug-workaround: QuicStreamReceiver.__init__ ignores its
            # `readable` flag, so for outbound unidirectional streams
            # receiver.is_finished never flips True → the stream is never pruned
            # after its FIN is acked, making the per-pass rebuild O(N²) at 60fps.
            # Force it; the sender still tracks FIN-ack so it's pruned on deliver.
            stream = quic._streams.get(new_id)
            if stream is not None:
                stream.receiver.is_finished = True
            quic.send_stream_data(stream_id=new_id, data=payload, end_stream=True)
            self.transmit()
        except Exception as e:
            log.debug("frame send failed (viewer %d): %s", self.session_id, e)

    def _send_frame_single(self, payload: bytes) -> None:
        """Single-stream path: length-prefix each envelope onto ONE long-lived
        unidirectional stream. Used only for viewers whose WebTransport stack
        can't sustain the per-frame default (Safari, which grants ~100 uni-stream
        credits total and replenishes them slowly, so a stream-per-frame runs it
        out of credit in a second or two).

        Drop-when-behind (bounds latency): a reliable ordered stream can't drop
        stale data, so if the consumer lags the producer, frames pile into the
        QUIC send buffer under flow-control backpressure and latency grows without
        bound (the known failure mode of a single reliable stream). We keep the
        buffer bounded by NOT feeding it: once the unacked backlog crosses the
        high-water mark we stop writing DELTA frames and ask the source for a
        fresh keyframe, so the decoder resyncs quickly instead of waiting for a
        periodic intra-refresh. Keyframes and config/cursor always pass; a
        keyframe ends the drop window.

        We deliberately do NOT RESET_STREAM to shed the backlog: each reset would
        consume one of Safari's finite cumulative uni-stream credits, so a lossy
        link that reset a few times a second would exhaust credit and freeze the
        session permanently — trading bounded stutter for a dead stream. One
        stream, opened once, held for the whole session."""
        try:
            quic = self.h3._quic
            t = payload[0] if payload else -1

            # Lazy-open the one persistent stream (never reset — see docstring).
            if self.video_stream_id is None:
                self.video_stream_id = self.h3.create_webtransport_stream(
                    session_id=self.session_id, is_unidirectional=True,
                )
                fresh = quic._streams.get(self.video_stream_id)
                if fresh is not None:
                    fresh.receiver.is_finished = True

            # Deltas are droppable; keyframes/config/cursor are not.
            if t == _TYPE_VIDEO_DELTA:
                if self.video_discontinuity:
                    return                      # waiting for the resync keyframe
                stream = quic._streams.get(self.video_stream_id)
                backlog = 0
                if stream is not None:
                    backlog = stream.sender._buffer_stop - stream.sender._buffer_start
                if backlog > _SINGLE_STREAM_HIGH_WATER:
                    # Stop feeding the stream (bounds the backlog) and resync
                    # from a keyframe rather than resetting the stream.
                    self.video_discontinuity = True
                    self.frames_dropped += 1
                    if self.request_keyframe is not None:
                        self.request_keyframe()
                    if self.frames_dropped in (1, 10, 100) or \
                            self.frames_dropped % 500 == 0:
                        log.info("single-stream: backlog %dKB over high-water "
                                 "(viewer %d, drop #%d) — dropping deltas, "
                                 "requesting keyframe", backlog // 1024,
                                 self.session_id, self.frames_dropped)
                    return
            elif t == _TYPE_VIDEO_KEY:
                self.video_discontinuity = False   # clean resync point

            framed = len(payload).to_bytes(4, "big") + payload
            quic.send_stream_data(
                stream_id=self.video_stream_id, data=framed, end_stream=False,
            )
            self.transmit()
        except Exception as e:
            log.debug("single-stream send failed (viewer %d): %s", self.session_id, e)

    def send_datagram(self, payload: bytes) -> None:
        """Send a WebTransport datagram (HTTP/3 DATAGRAM, RFC 9297).
        Used for audio packets — one UDP datagram per Opus frame, no
        per-frame stream-creation cost (which choked QUIC's stream
        credit at ~160 streams/sec)."""
        # Skip when the peer negotiated no HTTP-datagram support (Safari's
        # standardized-draft stack exposes no datagram writer and disables its
        # datagram reader): send_datagram would either raise per packet — an
        # audio-rate log flood — or drop silently into a channel nothing reads.
        if getattr(self.h3, "datagram_setting", None) is None:
            return
        try:
            self.h3.send_datagram(stream_id=self.session_id, data=payload)
            self.transmit()
            self.dgrams_sent += 1
            if self.dgrams_sent in (1, 50, 200, 1000):
                pending = len(self.h3._quic._datagrams_pending)
                log.info(
                    "viewer %d send_datagram #%d (queue=%d, payload=%dB)",
                    self.session_id, self.dgrams_sent, pending, len(payload),
                )
        except Exception as e:
            log.warning("datagram send failed (viewer %d): %s", self.session_id, e)

    def feed_input_bytes(self, data: bytes) -> None:
        """Append client input bytes; dispatch newline-delimited JSON
        records as they arrive."""
        self.last_seen_t = time.monotonic()
        self.input_buffer.extend(data)
        while True:
            nl = self.input_buffer.find(b"\n")
            if nl < 0:
                break
            line = bytes(self.input_buffer[:nl])
            del self.input_buffer[: nl + 1]
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception as e:
                log.warning("bad input json: %s — %s", e, line[:80])
                continue
            try:
                self.input_callback(msg)
            except Exception as e:
                log.warning("input callback raised: %s on %s", e, msg.get("type"))


class _BridgeProtocol(QuicConnectionProtocol):
    """One per QUIC connection. Owns a single H3Connection and any
    open WebTransport sessions. Routes events to the bridge."""

    def __init__(self, *args, bridge: "WebTransportBridge", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._bridge = bridge
        self._h3: Optional[H3Connection] = None
        # session_id (= CONNECT stream_id) → _ViewerSession
        self._viewers: dict[int, _ViewerSession] = {}

    def quic_event_received(self, event: QuicEvent) -> None:
        if isinstance(event, ProtocolNegotiated):
            log.info("QUIC ALPN negotiated: %s", event.alpn_protocol)
            if event.alpn_protocol in H3_ALPN:
                self._h3 = _WebTransportH3(self._quic, enable_webtransport=True)
        if self._h3 is not None:
            for h3_evt in self._h3.handle_event(event):
                self._h3_event(h3_evt)

    def _h3_event(self, event: H3Event) -> None:
        if isinstance(event, HeadersReceived):
            headers = dict((k, v) for k, v in event.headers)
            method = headers.get(b":method", b"").decode()
            path = headers.get(b":path", b"").decode()
            protocol = headers.get(b":protocol", b"").decode()
            log.info(
                "H3 HEADERS stream=%d method=%s path=%s protocol=%s",
                event.stream_id, method, path, protocol,
            )
            base_path, _, query = path.partition("?")
            if (method == "CONNECT"
                    and protocol in ("webtransport", "webtransport-h3")
                    and base_path == "/wt"):
                # draft-02 uses :protocol=webtransport; draft-08+ (Safari) uses
                # webtransport-h3. Accept both. The client requests the single-
                # long-lived-stream transport via ?transport=single (Safari does,
                # to stay under its WebTransport uni-stream credit); the global
                # env flag forces it for everyone.
                single = _VIDEO_SINGLE_STREAM or "transport=single" in query
                self._accept_webtransport(event.stream_id, single)
            elif method == "GET":
                self._serve_get(event.stream_id, path)
            else:
                self._send_status(event.stream_id, 404, b"")
        elif isinstance(event, WebTransportStreamDataReceived):
            viewer = self._viewers.get(event.session_id)
            if viewer is None:
                return
            # Treat ANY first incoming bidi/uni stream from the client
            # as the input channel. Buffer until newlines arrive.
            if viewer.input_stream_id is None:
                viewer.input_stream_id = event.stream_id
            if viewer.input_stream_id == event.stream_id:
                viewer.feed_input_bytes(event.data)
        elif isinstance(event, DataReceived):
            viewer = self._viewers.get(event.stream_id)
            if viewer is not None:
                # Bytes on the CONNECT stream of an established WebTransport
                # session are Capsule Protocol records (Safari's draft), not
                # HTTP body. Decode them; the one we act on is
                # CLOSE_WEBTRANSPORT_SESSION (orderly teardown).
                self._recv_session_capsules(viewer, event.data,
                                            event.stream_ended)
            # else: GET/POST body bytes — not needed for /wt.
        elif isinstance(event, DatagramReceived):
            viewer = self._viewers.get(event.stream_id)
            if viewer is not None:
                self._bridge.handle_input_datagram(viewer, event.data)

    def _recv_session_capsules(
        self, viewer: "_ViewerSession", data: bytes, stream_ended: bool,
    ) -> None:
        """Decode Capsule Protocol records arriving on a session's CONNECT
        stream. We only need CLOSE_WEBTRANSPORT_SESSION for lifecycle; other
        capsule types (e.g. flow-control) are skipped, matching the WPT
        reference server which relies on SETTINGS-based initial credit."""
        try:
            if data:
                viewer.capsule_decoder.append(data)
            if stream_ended:
                viewer.capsule_decoder.final()
            for capsule in viewer.capsule_decoder:
                if capsule.type == CapsuleType.CLOSE_WEBTRANSPORT_SESSION:
                    log.info("WT viewer %d sent CLOSE_WEBTRANSPORT_SESSION",
                             viewer.session_id)
                    self._close_viewer(viewer)
                    return
        except Exception as e:  # noqa: BLE001
            # A malformed / oversized capsule (or a decoder left half-consumed)
            # is a protocol error on the session stream — don't keep feeding a
            # wedged decoder; tear the viewer down.
            log.warning("capsule decode error (viewer %d): %s — closing viewer",
                        viewer.session_id, e)
            self._close_viewer(viewer)
            return
        if stream_ended:
            # Peer half-closed the CONNECT stream without a close capsule; the
            # session is over either way.
            self._close_viewer(viewer)

    def _close_viewer(self, viewer: "_ViewerSession") -> None:
        if self._viewers.pop(viewer.session_id, None) is not None:
            self._bridge.remove_viewer(viewer)
            log.info("WT viewer %d disconnected (total=%d)",
                     viewer.session_id, len(self._bridge.viewers))

    # -- request handlers ---------------------------------------------

    def _accept_webtransport(
        self, stream_id: int, use_single_stream: bool = False,
    ) -> None:
        assert self._h3 is not None
        self._h3.send_headers(
            stream_id=stream_id,
            headers=[
                (b":status", b"200"),
                (b"sec-webtransport-http3-draft", b"draft02"),
            ],
        )
        # Bump the per-connection unidirectional-stream credit. aioquic
        # hardcodes the initial cap to 128 and only doubles it when
        # `used > value/2` — too slow for 60+ streams/sec workloads.
        # Without this, sustained per-frame stream creation chokes the
        # connection and stalls video. There is no public API for this;
        # production aioquic-based MoQ/WebTransport stacks all patch the
        # private field directly.
        try:
            self._quic._local_max_streams_uni.value = 1 << 16
            self._quic._local_max_streams_uni.sent = 1 << 16
        except Exception as e:
            log.debug("stream credit bump failed: %s", e)
        viewer = _ViewerSession(
            session_id=stream_id,
            stream_id=stream_id,
            h3=self._h3,
            loop=asyncio.get_event_loop(),
            transmit=self.transmit,
            input_callback=self._bridge.handle_input_event,
            use_single_stream=use_single_stream,
            request_keyframe=self._bridge.request_keyframe,
        )
        self._viewers[stream_id] = viewer
        self._bridge.add_viewer(viewer)
        log.info("WT viewer %d connected (total=%d, transport=%s)",
                 stream_id, len(self._bridge.viewers),
                 "single-stream" if use_single_stream else "per-frame")

    def _serve_get(self, stream_id: int, path: str) -> None:
        assert self._h3 is not None
        if path == "/" or path == "":
            html = (_STATIC_DIR / "index.html").read_bytes()
            self._send_response(stream_id, 200, b"text/html; charset=utf-8", html)
        elif path == "/cert-hash":
            payload = self._bridge.cert_sha256.encode()
            self._send_response(stream_id, 200, b"text/plain", payload)
        else:
            self._send_status(stream_id, 404, b"")

    def _send_response(
        self, stream_id: int, status: int,
        content_type: bytes, body: bytes,
    ) -> None:
        assert self._h3 is not None
        self._h3.send_headers(
            stream_id=stream_id,
            headers=[
                (b":status", str(status).encode()),
                (b"content-type", content_type),
                (b"content-length", str(len(body)).encode()),
            ],
        )
        self._h3.send_data(stream_id=stream_id, data=body, end_stream=True)

    def _send_status(self, stream_id: int, status: int, body: bytes) -> None:
        self._send_response(stream_id, status, b"text/plain", body)

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        for viewer in list(self._viewers.values()):
            self._bridge.remove_viewer(viewer)
        super().connection_lost(exc)


class WebTransportBridge:
    """One bridge per process. Lazily creates the iss `Session` on
    first viewer; tears it down when the last viewer leaves."""

    def __init__(
        self,
        config: SessionConfig,
        *,
        port: int = 4433,
        bitrate_kbps: int = 5000,
        framerate: int = 60,
        tile3_crop_rows: int = 0,
        trust_cert_for_safari: bool = False,
    ) -> None:
        self._config = config
        self._port = port
        # Opt-in: install our self-signed cert into the macOS login
        # keychain so Safari (which ignores serverCertificateHashes) can
        # do WebTransport. Set True once trusted — the page then connects
        # without hashes for Safari.
        self._trust_cert_for_safari = trust_cert_for_safari
        self._safari_trusted = False
        self._bitrate_kbps = bitrate_kbps
        self._fps = framerate
        # 0 = auto-derive from canvas height. Non-zero forces a
        # specific count for tuning.
        self._tile3_crop_override = tile3_crop_rows
        self.viewers: list[_ViewerSession] = []
        self._viewers_lock = threading.Lock()
        self._session: Optional[Session] = None
        # Pass-through: no encoder/compose pump. AUs forwarded via _on_video_au.
        self._config_sent: bool = False
        self._last_config_env: Optional[bytes] = None
        self._au_seq: int = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.cert_sha256: str = ""
        self._http_runner: Optional[web.AppRunner] = None
        # Guards against multiple concurrent session-start threads if
        # several viewers connect during the slow connect dance.
        self._session_starting = False
        # Encoded canvas height — initial guess from canvas_h, refined
        # once we see tile 3's actual coded height.
        self._encode_h: int = 0
        self._tile3_crop_resolved: bool = False
        # Opus encoder for audio path.
        self._audio_enc: Optional[OpusEncoder] = None

    # -- lifecycle ----------------------------------------------------

    async def run_async(self) -> None:
        cache = Path.home() / ".cache" / "iss"
        cert_path, key_path, sha256 = write_cert(cache)
        self.cert_sha256 = sha256
        # Install into the login keychain only if asked AND not already trusted
        # (avoids a redundant prompt). The cert is reused across runs, so a
        # prior --trust-cert-for-safari (or the standalone trust step) makes it
        # trusted for good.
        if self._trust_cert_for_safari and not is_cert_trusted(cert_path):
            ok, detail = trust_cert_in_keychain(cert_path)
            if not ok:
                log.warning(
                    "Safari trust: could NOT install cert (%s). Safari will "
                    "still fail — approve the keychain prompt, or use "
                    "Chrome/Edge/Firefox.", detail)
        # Report ACTUAL keychain trust, not just whether the flag was passed —
        # so Safari connects whenever the cert is trusted, even on a plain `iss`
        # run after it was trusted once.
        self._safari_trusted = is_cert_trusted(cert_path)
        if self._safari_trusted:
            log.info("Safari trust: cert is keychain-trusted (Safari can connect)")
        # Match aioquic's reference http3_server.py exactly. The
        # additions I had (`max_data`, `max_stream_data`,
        # `max_datagram_size=1500`) caused TLS handshake completion to
        # stall — the reference uses defaults which are larger and
        # negotiate successfully with Chrome 147.
        # `max_datagram_frame_size` is required for HTTP/3 DATAGRAM
        # which WebTransport sits on top of.
        config = QuicConfiguration(
            is_client=False,
            alpn_protocols=H3_ALPN,
            max_datagram_frame_size=65536,
        )
        config.load_cert_chain(str(cert_path), str(key_path))
        self._loop = asyncio.get_running_loop()
        # Watchdog: evict viewers that have gone silent. Browsers send
        # stats every ~1s and input datagrams whenever the user moves;
        # 5+ seconds of zero traffic = tab almost certainly closed
        # without QUIC having delivered a clean CONNECTION_CLOSE yet.
        # Without this, idle_timeout (~30s) holds the SRP/SRTP session
        # open and Apple keeps encoding HEVC for nobody.
        async def _viewer_watchdog():
            while True:
                await asyncio.sleep(2.0)
                now = time.monotonic()
                with self._viewers_lock:
                    stale = [v for v in self.viewers if now - v.last_seen_t > 5.0]
                for v in stale:
                    log.info("evicting stale viewer %d (last_seen=%.1fs ago)",
                             v.session_id, now - v.last_seen_t)
                    self.remove_viewer(v)
        asyncio.create_task(_viewer_watchdog())

        # Two listeners:
        #   port (UDP)  — WebTransport / HTTP/3 over QUIC. Browsers use
        #     `serverCertificateHashes` here to trust the self-signed
        #     cert without CA validation.
        #   port (TCP)  — plain HTTP for the bootstrap page. NO TLS —
        #     navigating to `https://host:port` in the URL bar would
        #     hit Chrome's CA wall and never load. Plain HTTP avoids
        #     the cert-warning dance.
        # Listen on BOTH UDPv4 and UDPv6. Chrome resolves `localhost`
        # to both ::1 and 127.0.0.1 and prefers ::1 — without an IPv6
        # listener it sees ICMP port-unreachable and reports the WT
        # connect as ERR_CONNECTION_REFUSED without trying IPv4.
        await quic_serve(
            host="::",
            port=self._port,
            configuration=config,
            create_protocol=lambda *a, **kw: _BridgeProtocol(
                *a, bridge=self, **kw,
            ),
        )
        # Also bind a separate IPv4 listener — `::` dual-stack is the
        # default on Linux/macOS but aioquic creates a single socket;
        # if dual-stack is disabled (Windows, hardened Linux) we still
        # need an explicit v4 socket.
        try:
            await quic_serve(
                host="0.0.0.0",
                port=self._port,
                configuration=config,
                create_protocol=lambda *a, **kw: _BridgeProtocol(
                    *a, bridge=self, **kw,
                ),
            )
        except OSError as e:
            log.debug("v4 bind skipped (dual-stack already covers it): %s", e)
        await self._start_http_bootstrap()
        # Use localhost, not the LAN IP: Chrome/Edge only expose WebTransport in
        # a secure context, and http://localhost qualifies while http://<LAN-IP>
        # does not. localhost also matches the self-signed cert (cert.py issues
        # it for localhost / 127.0.0.1, not the LAN address), and the dual-stack
        # listeners above exist precisely so localhost (::1 / 127.0.0.1) works.
        url = f"http://localhost:{self._port}/"
        log.info("WebTransport bridge ready:")
        log.info("  open this URL in Chrome/Edge/Firefox: %s", url)
        log.info("  (HTML on tcp/%d, WebTransport on udp/%d)",
                 self._port, self._port)
        await asyncio.Event().wait()

    async def _start_http_bootstrap(self) -> None:
        """Tiny aiohttp server that delivers the viewer page + cert
        hash over plain HTTP on the same port number (different
        protocol — TCP vs UDP, so no conflict). Eliminates the
        Chrome self-signed-cert warning page."""
        nocache = {
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        }

        async def index(_req: web.Request) -> web.Response:
            html = (_STATIC_DIR / "index.html").read_bytes()
            return web.Response(
                body=html, content_type="text/html", headers=nocache,
            )

        async def cert_hash(_req: web.Request) -> web.Response:
            return web.Response(
                text=self.cert_sha256, headers=nocache,
            )

        async def wt_config(_req: web.Request) -> web.Response:
            """Client bootstrap config. `safariTrusted` tells the page
            whether our cert is installed in the keychain — if so, Safari
            connects WITHOUT serverCertificateHashes (which WebKit doesn't
            support) and validates via the system trust store instead."""
            return web.json_response(
                {"safariTrusted": self._safari_trusted,
                 "videoSingleStream": _VIDEO_SINGLE_STREAM,
                 # In dynamic ("Auto") mode the page reports its window size so
                 # the host re-encodes to match; a fixed --advertise is left
                 # untouched (respect the user's explicit resolution choice).
                 "dynamic": bool(self._config.dynamic_resolution)}, headers=nocache,
            )

        async def audio_worklet(_req: web.Request) -> web.Response:
            js = (_STATIC_DIR / "audio-worklet.js").read_bytes()
            return web.Response(
                body=js, content_type="application/javascript",
                headers=nocache,
            )

        async def log_event(req: web.Request) -> web.Response:
            """Browser-side error/log relay so console messages are
            visible to the iss operator."""
            try:
                payload = await req.json()
                level = payload.get("level", "info")
                msg = payload.get("msg", "")
                logf = log.error if level == "error" else log.info
                logf("browser-log [%s]: %s", level, msg)
            except Exception as e:
                log.debug("log_event error: %s", e)
            return web.Response(text="ok", headers=nocache)

        async def stats_endpoint(req: web.Request) -> web.Response:
            """Server-side debug stats for the browser's overlay bar.
            Polled ~1×/s; intentionally cheap to compute. Includes the
            signals the operator needs to diagnose stuck/gray streams
            (last_publish_age, per-tile loss, hw decoder, viewers).
            Browser-side metrics like `decode_fps` and `gray_frac`
            are owned by the page and not duplicated here."""
            sess = self._session
            decoder = "?"
            rcv = lost = 0
            last_age = -1.0
            lost_per_tile: list[int] = []
            if sess is not None:
                try:
                    decoder = sess.hw_accel or "software"
                    rcv, lost = sess.packet_stats
                    last_age = sess.last_publish_age_s
                    lost_per_tile = sess.lost_pkts_per_tile
                except Exception as e:
                    log.debug("stats_endpoint sess read failed: %s", e)
            payload = {
                "decoder": decoder,
                "loss_pct": round((lost / rcv * 100.0) if rcv > 0 else 0.0, 2),
                "loss_per_tile": lost_per_tile,
                "stuck_s": round(max(0.0, last_age), 1),
                "viewers": len(self.viewers),
                "encoder": "passthrough-h264",
            }
            return web.json_response(payload, headers=nocache)

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/cert-hash", cert_hash)
        app.router.add_get("/wt-config", wt_config)
        app.router.add_get("/audio-worklet.js", audio_worklet)
        app.router.add_post("/log", log_event)
        app.router.add_get("/stats", stats_endpoint)
        runner = web.AppRunner(app, access_log=None)  # don't log every /stats poll
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()
        self._http_runner = runner
        log.info("server cert sha256 = %s", self.cert_sha256)

    def run(self) -> int:
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass
        finally:
            self._teardown_session()
        return 0

    # -- viewer management --------------------------------------------

    def add_viewer(self, viewer: _ViewerSession) -> None:
        with self._viewers_lock:
            self.viewers.append(viewer)
            if self._session is None and not self._session_starting:
                self._session_starting = True
                # Run session.connect() in a thread — it does sync
                # SRP+RFB+burst gather (5-10s). On the asyncio loop
                # this would freeze QUIC event processing and Chrome
                # times out the WT keepalive.
                threading.Thread(
                    target=self._start_session_thread,
                    name="wt-session-start", daemon=True,
                ).start()
            elif self._session is not None:
                # Late joiner: resend the geometry/codec config, and FIR all
                # tiles so a fresh keyframe (with SPS/PPS prepended in
                # _on_video_au) follows for this viewer to start decoding.
                if self._last_config_env is not None:
                    self._broadcast_frame(self._last_config_env)
                # Replay the most-recent cached keyframe so the joiner can paint
                # right away. On an idle host (where the FIR below goes
                # unanswered) this is the ONLY way it gets a decodable frame;
                # on an active host the fresh IDR re-syncs immediately after.
                # Existing viewers drop it (older seq than what they've seen),
                # so a broadcast only reaches the fresh joiner.
                for kf_env in list(getattr(self, "_last_keyframe_env", {}).values()):
                    self._broadcast_frame(kf_env)
                if _FORCE_KEYFRAME_ON_JOIN:
                    try:
                        self._session.request_fir(None)
                    except Exception:
                        pass

    def remove_viewer(self, viewer: _ViewerSession) -> None:
        with self._viewers_lock:
            try:
                self.viewers.remove(viewer)
            except ValueError:
                return
            log.info("WT viewer %d disconnected (remaining=%d)",
                     viewer.session_id, len(self.viewers))
            if not self.viewers:
                self._teardown_session()

    def request_keyframe(self) -> None:
        """Rate-limited FIR request, shared by recovery paths (decoder reset,
        single-stream reset-on-backlog). macOS answers a FIR with an IDR in
        ~75ms; suppress a redundant request while one is still in flight
        (_kf_req_fir_t is cleared in _on_video_au when the IDR arrives)."""
        now = time.monotonic()
        pending = getattr(self, "_kf_req_fir_t", 0.0)
        if pending and (now - pending) <= 0.3:
            return
        try:
            if self._session is not None:
                self._session.request_fir(None)
                self._kf_req_fir_t = now
        except Exception:  # noqa: BLE001
            pass

    # -- session + encoder lifecycle ----------------------------------

    def _start_session_thread(self) -> None:
        """Off-loop: do the slow SRP+RFB+burst dance, then install
        encoder + start the pump thread. Runs once per bridge
        lifecycle (cleared in _teardown_session)."""
        try:
            log.info("starting iss session (%s)", self._config.host)
            session = Session(self._config)
            session.connect()
            # Pass-through: forward Apple's H.264 access units straight to
            # viewers — NO compose, NO re-encode. The native decoder still runs
            # inside Session (it drives FIR/keyframe recovery, which the browser
            # rides via the same forwarded keyframes); we just tap the
            # reassembled per-tile AUs in _on_video_au.
            self._config_sent = False
            self._last_config_env = None
            # Most-recent keyframe envelope per tile — replayed to a late-joining
            # viewer so it can paint immediately even when the host is idle and
            # won't answer a FIR with a fresh IDR (see add_viewer).
            self._last_keyframe_env = {}
            self._au_seq = 0
            # Reset dynamic-resolution dedup state so a reconnect at the SAME
            # window size still re-advertises (the new session starts at the
            # host's default canvas, not the previous re-advertised size).
            self._last_resize_wh = None
            self._last_config_wh = None
            session.set_video_au_callback(self._on_video_au)
            # Cursor (RFB enc 1104): forward pixmaps so the browser paints the
            # host cursor shape as the canvas CSS cursor — and, crucially, the
            # cursor is NOT baked into the framebuffer, so moving the mouse
            # doesn't dirty the frame / wake the encoder.
            session.set_cursor_callback(self._on_cursor)
            # Audio: opus encoder + callback hook. Audio runs on its
            # own thread (the session's audio RX thread). Each Opus
            # packet is sent as a WebTransport DATAGRAM (RFC 9297) —
            # one UDP datagram per packet, no stream-credit cost.
            # Datagrams are unreliable but Opus tolerates single-
            # packet loss gracefully (PLC fills ~5 ms gaps).
            audio_enc = None
            try:
                audio_enc = OpusEncoder()
                session.set_audio_callback(self._on_audio)
            except Exception as e:
                log.warning("audio disabled: opus init failed (%s)", e)
            with self._viewers_lock:
                self._session = session
                self._audio_enc = audio_enc
                # Gate pump: the browser path forwards AUs via _on_video_au and
                # never calls get_frame(), but get_frame() is where the quality
                # gate marks clean/concealed and drives FIR/keyframe recovery —
                # without it the stream wedges after ~15s (decode loses refs, no
                # FIR, Apple stops sending). Pump get_frame() on a thread and
                # discard the frames purely to keep recovery alive.
                self._pump_stop = threading.Event()
                self._gate_thread = threading.Thread(
                    target=self._gate_pump_loop, name="wt-gate-pump", daemon=True)
                self._gate_thread.start()
        except Exception as e:
            self._session_starting = False
            # A refused login or a session the Mac doesn't offer won't change
            # on a retry, and every retry of a wrong password is one more
            # failed login on the Mac (enough of them lock the account).
            from ...proxy.protocol.auth import AuthError
            from ...proxy.protocol.negotiation import SessionSelectError
            if isinstance(e, (AuthError, SessionSelectError)):
                log.error("session start failed, not retrying: %s", e)
                return
            log.error("session start failed: %s — retrying in 1.5 s", e)
            # Auto-retry: schedule another start attempt while we still
            # have at least one viewer connected. Apple's burst is
            # transient — VPS/SPS/PPS sometimes don't survive the first
            # window, and a fresh session gather usually succeeds.
            def _retry():
                with self._viewers_lock:
                    needs_retry = (
                        self._session is None
                        and not self._session_starting
                        and bool(self.viewers)
                    )
                    if not needs_retry:
                        return
                    self._session_starting = True
                threading.Thread(
                    target=self._start_session_thread,
                    name="wt-session-retry", daemon=True,
                ).start()
            threading.Timer(1.5, _retry).start()

    def _on_cursor(self, img) -> None:
        """Cursor RX thread → fan-out one enc-1104 pixmap (RGBA8888 + hotspot)
        to every viewer. The browser turns it into the canvas CSS cursor."""
        loop = self._loop
        if loop is None or img is None or not getattr(img, "rgba", None):
            return
        try:
            env = _cursor_envelope(img.width, img.height,
                                   img.hotspot_x, img.hotspot_y, img.rgba)
        except Exception:
            return
        self._broadcast_frame(env)

    def _on_audio(self, pcm) -> None:
        """Audio RX thread → opus encode → fan-out. PCM is float32
        stereo at 48 kHz."""
        enc = self._audio_enc
        loop = self._loop
        if enc is None or loop is None:
            return
        # Skip Opus encode entirely if no viewer's AudioContext is in
        # the `running` state (browser autoplay-policy default = no
        # listener). Saves ~5% of one CPU core on a busy box. Re-armed
        # via the stats handler when any viewer reports audio_ctx=
        # `running` in the last 2 s.
        if (time.monotonic() - getattr(self, "_audio_listener_last_t", 0.0)) > 2.0:
            return
        # Log every-Nth-callback so we can verify audio is flowing.
        self._audio_calls = getattr(self, "_audio_calls", 0) + 1
        if self._audio_calls in (1, 100, 1000) or self._audio_calls % 5000 == 0:
            log.info("audio callback #%d: pcm shape=%s",
                     self._audio_calls, getattr(pcm, "shape", "?"))
        try:
            packets = enc.encode(pcm)
        except Exception as e:
            log.warning("audio encode error: %s", e)
            return
        if not packets:
            return
        self._audio_packets_sent = getattr(self, "_audio_packets_sent", 0) + len(packets)
        for opus_bytes, ts_us in packets:
            envelope = _audio_envelope(opus_bytes, ts_us)
            with self._viewers_lock:
                viewers = list(self.viewers)
            for viewer in viewers:
                loop.call_soon_threadsafe(viewer.send_datagram, envelope)

    @staticmethod
    def _codec_string(session) -> str:
        """WebCodecs codec id 'avc1.PPCCLL' from the SPS (profile/constraint/
        level bytes). Falls back to High@5.0 if the SPS isn't available yet."""
        sps, _ = session.video_params()
        if sps and _BROWSER_SPS_PATCH:
            sps = _patch_avc_sps_dpb(sps)
        if len(sps) >= 4:
            return "avc1." + sps[1:4].hex()
        return "avc1.640032"

    def _on_video_au(self, tile_idx: int, ts: int, au_bytes: bytes) -> None:
        """Video-process thread → forward one tile's H.264 access unit to every
        viewer as a per-tile unidirectional stream. Keyframes get SPS/PPS
        prepended (Annex-B) so a WebCodecs decoder can start without a separate
        description. The first AU also pushes a one-time geometry/codec config."""
        loop = self._loop
        session = self._session
        if loop is None or session is None:
            return
        is_key = au_is_keyframe(au_bytes)
        if _PROFILE:
            now = time.monotonic()
            last = getattr(self, "_prof_last_au", 0.0)
            if last:
                gap = (now - last) * 1000
                if gap > 80:
                    log.info("iss FORWARD GAP %.0fms (key=%s, tile=%d, bytes=%d)",
                             gap, is_key, tile_idx, len(au_bytes))
            self._prof_last_au = now
            if is_key:
                log.info("iss KEYFRAME (tile=%d, bytes=%d)", tile_idx, len(au_bytes))
        if is_key:
            # An IDR arrived — clear any in-flight FIR request (see the
            # keyframe-request gating in handle_input_event).
            if getattr(self, "_kf_req_fir_t", 0.0):
                self._kf_req_fir_t = 0.0
        if is_key and not au_has_sps(au_bytes):
            # Only inject SPS/PPS when the keyframe doesn't already carry them
            # in-band — otherwise duplicate parameter sets make WebCodecs throw
            # 'Decoding error' on every keyframe.
            sps, all_pps = session.video_params()
            if sps:
                if _BROWSER_SPS_PATCH:
                    sps = _patch_avc_sps_dpb(sps)
                pps = next(iter(all_pps.values()), b"")
                au_bytes = (b"\x00\x00\x00\x01" + sps
                            + b"\x00\x00\x00\x01" + pps + au_bytes)
        cw, ch = session.canvas_dims
        # Per-host-display rects within the combined canvas (0x451 layout). Lets
        # a viewer crop to a single host monitor (?display=N) so a multi-monitor
        # host isn't shown as one squished/black canvas, and drives the Displays
        # picker. The layout can arrive/change AFTER the first keyframe (the
        # 0x451 is async on the control channel), so re-send config when it
        # changes even if the canvas size didn't.
        drects = [
            {"id": r.display_id, "x": r.x, "y": r.y, "w": r.w, "h": r.h}
            for r in session.display_rects
        ]
        _content = session.display_content_rect
        content_rect = (
            {"x": _content.x, "y": _content.y, "w": _content.w, "h": _content.h}
            if _content is not None else None
        )
        # (Re)send config on the first frame, and again when the canvas size
        # changes mid-session (dynamic resolution) — but only on a keyframe, so
        # the browser re-inits its decoder at the new size on a clean IDR that
        # already carries the new SPS (a delta at the new size would fail).
        if (not self._config_sent
                or (is_key and (cw, ch) != getattr(self, "_last_config_wh", None))
                or (is_key and drects != getattr(self, "_last_config_drects", None))
                or (is_key and content_rect
                    != getattr(self, "_last_config_content_rect", None))):
            nt = max(1, session.num_tiles)
            self._last_config_wh = (cw, ch)
            self._last_config_drects = drects
            self._last_config_content_rect = content_rect
            cfg = json.dumps({
                "num_tiles": nt, "canvas_w": cw, "canvas_h": ch,
                "tile_h": ch // nt, "codec": self._codec_string(session),
                "display_rects": drects,
                "content_rect": content_rect,
                "hidpi_scale": self._display_scale(),
            }).encode()
            self._last_config_env = _config_envelope(cfg)
            self._config_sent = True
            self._broadcast_frame(self._last_config_env)
        # Monotonic per-AU sequence in forward order (= the native decode
        # order). The browser reorders by this before decoding.
        seq = self._au_seq
        self._au_seq = (self._au_seq + 1) & 0xFFFFFFFF
        # Stamp iss's own monotonic send time in µs (the browser decodes with a
        # synthetic timestamp and ignores this field, so it's free to carry a
        # real-time send clock for end-to-end latency measurement — the source
        # RTP ts here ran at ~90kHz, which read as µs looked like 0.9s/s of fake
        # "latency growth").
        send_us = int(time.monotonic() * 1_000_000) & 0xFFFFFFFFFFFFFFFF
        env = _frame_envelope(au_bytes, is_key, send_us, tile_idx, seq)
        if is_key:
            # Cache the latest keyframe (SPS/PPS already prepended above) to
            # bootstrap late joiners on an idle host — see add_viewer.
            self._last_keyframe_env[tile_idx] = env
        self._broadcast_frame(env)

    def _gate_pump_loop(self) -> None:
        """Drain get_frame() for every tile so the quality gate + FIR recovery
        run (the browser path consumes AUs directly, never get_frame). Frames
        are discarded — this exists only to keep the native recovery alive."""
        session = self._session
        if session is None:
            return
        nt = session.num_tiles
        stop = self._pump_stop
        while not stop.is_set():
            try:
                for ti in range(nt):
                    session.get_frame(ti)
            except Exception:
                pass
            stop.wait(0.016)  # ~60 Hz

    def _broadcast_frame(self, envelope: bytes) -> None:
        loop = self._loop
        if loop is None:
            return
        with self._viewers_lock:
            viewers = list(self.viewers)
        for viewer in viewers:
            loop.call_soon_threadsafe(viewer.send_frame, envelope)

    def _teardown_session(self) -> None:
        pump_stop = getattr(self, "_pump_stop", None)
        if pump_stop is not None:
            pump_stop.set()
        gt = getattr(self, "_gate_thread", None)
        if gt is not None:
            gt.join(timeout=1.0)
            self._gate_thread = None
        if self._session is not None:
            try:
                self._session.set_video_au_callback(None)
            except Exception:
                pass
            try:
                self._session.set_cursor_callback(None)
            except Exception:
                pass
        if self._audio_enc is not None:
            self._audio_enc.close()
            self._audio_enc = None
        if self._session is not None:
            self._session.set_audio_callback(None)
            self._session.close()
            self._session = None
        self._session_starting = False
        log.info("iss session torn down (no viewers)")

    # -- input handling -----------------------------------------------

    def handle_input_datagram(self, viewer, data: bytes) -> None:
        """Mouse-move datagrams from the browser. Fixed 13-byte format:
        [type:1=0x10][buttons:u32 BE][x:i32 BE][y:i32 BE]"""
        viewer.last_seen_t = time.monotonic()
        if self._session is None:
            return
        if len(data) != 13 or data[0] != 0x10:
            log.debug("input datagram: bad len=%d type=%d", len(data), data[0] if data else -1)
            return
        buttons = int.from_bytes(data[1:5], "big")
        x = int.from_bytes(data[5:9], "big", signed=True)
        y = int.from_bytes(data[9:13], "big", signed=True)
        self._input_dgrams_rx = getattr(self, "_input_dgrams_rx", 0) + 1
        if self._input_dgrams_rx in (1, 50, 200, 1000):
            log.info(
                "input datagram #%d: buttons=%d xy=(%d,%d)",
                self._input_dgrams_rx, buttons, x, y,
            )
        try:
            self._session.input.pointer_event(buttons, x, y)
        except Exception as e:
            log.debug("mouse datagram dispatch error: %s", e)

    def handle_input_event(self, msg: dict) -> None:
        """Dispatch one decoded JSON input event to the iss session."""
        kind = msg.get("type")
        if kind == "keyframe":
            # A browser decoder reset and needs a fresh keyframe to restart.
            # macOS answers a FIR with an IDR in ~75ms, so send one per reset.
            # The ONLY thing we suppress is a redundant FIR while a previous
            # one's IDR is still in flight (gated by _kf_req_fir_t, which
            # _on_video_au clears when the IDR arrives). A stale in-flight FIR
            # (>300ms, IDR likely lost) is retried. The old fixed 0.5s rate
            # limit dropped the recovery FIR during rapid resets → the browser
            # then waited ~920ms for the next *natural* keyframe = the freeze.
            now = time.monotonic()
            pending_fir = getattr(self, "_kf_req_fir_t", 0.0)
            if not pending_fir or (now - pending_fir) > 0.3:
                try:
                    if self._session is not None:
                        self._session.request_fir(None)
                        self._kf_req_fir_t = now
                except Exception:
                    pass
                # Also replay the cached keyframe. The browser just reset its
                # decoder (nextSeq=-1) so it accepts this even though the seq is
                # older than what it had seen — letting it repaint immediately
                # instead of waiting on the FIR, which an IDLE host won't answer.
                # On an active host the fresh IDR follows and supersedes it.
                for kf_env in list(getattr(self, "_last_keyframe_env", {}).values()):
                    self._broadcast_frame(kf_env)
            return
        if kind == "stats":
            # Track whether any viewer is actively listening to audio
            # so the audio thread can skip Opus encode otherwise.
            if msg.get("audio_ctx") == "running":
                self._audio_listener_last_t = time.monotonic()
            # The reliable "broken picture" signal is the CANVAS, not the frame
            # rate: a static/idle screen reads 0 fps but its last frame is real
            # and on-screen (high pixel variance, low grey). Only a genuinely
            # corrupt/concealed picture goes flat or grey. So classify by pixels
            # FIRST, and treat 0 fps with a healthy canvas as NO_FRAMES — purely
            # informational, not a freeze to storm IDRs at.
            verdict = "ok"
            if msg.get("decoder_errored"):
                verdict = "DECODER_ERROR"
            elif msg.get("pixel_std", 0) < 5 and msg.get("decode_fps", 0) < 1:
                # FLAT canvas (low variance) AND frames stopped = a real freeze.
                # A flat canvas WITH frames still flowing is dark/low-contrast
                # CONTENT decoding fine — FIR-ing that every few seconds was the
                # stutter, so require frames to have stalled. (Concealment is
                # caught by MOSTLY_GREY below, not here — it's mid-gray, not flat.)
                verdict = "FROZEN/GREY"
            elif msg.get("gray_frac", 0) > 0.5:
                # >50% mid-gray-128 + desaturated = decoder concealment fill from
                # a broken reference chain — NOT dark content (near-black, not
                # mid-gray). Recover regardless of fps: WebCodecs can emit
                # concealed gray frames at full rate WITHOUT raising an error, so
                # this is the only signal for a frames-flowing gray corruption.
                verdict = "MOSTLY_GREY"
            elif msg.get("decode_fps", 0) < 1:
                verdict = "NO_FRAMES"
            elif msg.get("decoder_queue", 0) > 10:
                verdict = "DECODE_LAG"
            # The browser's own decoder is the authoritative picture. When it's
            # actively decoding a GOOD frame ("ok"), tell the session so it
            # suppresses the session-side libav gate's false-positive FIRs (the
            # "recovering via FIR" spam on a healthy stream). Keyed STRICTLY on
            # "ok" — never on gray/frozen, which may be a silent d3d11va wedge
            # that genuinely needs the gate's FIR to recover.
            if verdict == "ok" and self._session is not None:
                self._session.note_browser_healthy()
            # Throttle the routine "ok" heartbeat to once per 30s — a per-second
            # stats line is log spam. Anything NOT "ok" (a real problem) logs
            # immediately so issues stay visible.
            now = time.monotonic()
            if verdict == "ok" and (now - getattr(self, "_last_stats_log_t", 0.0)) < 30.0:
                pass
            else:
                self._last_stats_log_t = now
                log.info(
                    "browser[%s]: decode=%.1f fps streams=%.1f/s %.1f Mbit/s "
                    "queue=%d canvas=%dx%d pixel(mean=%.1f std=%.1f gray=%.0f%%) "
                    "audio[%s/ctx=%s pkts=%d frames=%d derr=%d serr=%d] "
                    "total=%.1f MB",
                    verdict,
                    msg.get("decode_fps", 0), msg.get("stream_fps", 0),
                    msg.get("mbps", 0), msg.get("decoder_queue", 0),
                    msg.get("canvas_w", 0), msg.get("canvas_h", 0),
                    msg.get("pixel_mean", 0), msg.get("pixel_std", 0),
                    msg.get("gray_frac", 0) * 100,
                    msg.get("audio_state", "?"),
                    msg.get("audio_ctx", "?"),
                    msg.get("audio_packets", 0),
                    msg.get("audio_frames", 0),
                    msg.get("audio_decode_err", 0),
                    msg.get("audio_submit_err", 0),
                    msg.get("total_mb", 0),
                )
            err = msg.get("decoder_error")
            if err:
                log.warning("browser decoder error: %s", err)
            # FIR-storm ONLY when the canvas is actually broken (flat/grey = lost
            # references) — not on 0 fps alone. A healthy static frame reads 0 fps
            # and is perfectly fine; storming IDRs there is futile and noisy.
            # Partial-tile concealment that leaves whole-tile std looking healthy
            # is caught by the grey-fraction check. Rate-limited to ≥3 s.
            now = time.monotonic()
            # A gray/frozen CANVAS and gray/frozen CONTENT are identical pixels,
            # so the pixel verdict alone can never be trusted to FIR — a gray
            # desktop would loop forever. Concealment gray only happens for a
            # REASON: a broken reference chain, which comes from packet loss, or a
            # real browser decode error. So require a reliable signal — the
            # source->iss loss counter grew recently, or the browser flagged a
            # decode error — before treating the verdict as recoverable. On a
            # clean stream (no loss, no error) gray is just content: never FIR.
            cur_loss = getattr(self._session, "_lost_pkts", 0) if self._session else 0
            if cur_loss > getattr(self, "_last_seen_loss", cur_loss):
                self._last_browser_loss_t = now
            self._last_seen_loss = cur_loss
            recent_loss = (now - getattr(self, "_last_browser_loss_t", 0.0)) < 5.0
            reliable = recent_loss or bool(msg.get("decoder_errored"))
            stuck = reliable and verdict in ("MOSTLY_GREY", "FROZEN/GREY")
            if stuck:
                self._stuck_streak = getattr(self, "_stuck_streak", 0) + 1
            else:
                self._stuck_streak = 0
            # The FIR-storm is needed at startup too: a viewer that connects after
            # the burst missed its IDR and needs a fresh one to begin decoding. It
            # just must be rate-limited to >=3s. The bug: the timestamp was only
            # stamped on request_fir() SUCCESS, which raises before the stream is
            # up, so the limit never engaged and it re-fired every tick (11 IDRs in
            # 12s). Stamp it BEFORE the request so >=3s holds regardless.
            last_fir_t = getattr(self, "_last_browser_fir_t", 0.0)
            if (
                self._session is not None
                and self._stuck_streak >= 2
                and (now - last_fir_t) >= 3.0
            ):
                self._last_browser_fir_t = now
                log.warning(
                    "browser reports %s for %ds with recent packet loss — "
                    "forcing IDR on all tiles", verdict, self._stuck_streak,
                )
                try:
                    self._session.request_fir()
                except Exception as e:
                    log.debug("request_fir from browser-stuck path failed: %s", e)
            return
        if kind == "set_resolution":
            # Browser-side runtime resolution control: changes the
            # virtual-display geometry advertised to the host so Apple
            # encodes at a smaller size before the wire. width=height=0
            # = clear advertise (host's panel default).
            try:
                target_w = max(0, int(msg.get("width", 0)))
                target_h = max(0, int(msg.get("height", 0)))
            except Exception:
                return
            from ...proxy.session import AdvertiseDims
            if target_w == 0 or target_h == 0:
                new_advertise = None
            else:
                new_advertise = AdvertiseDims(
                    width=target_w, height=target_h, hidpi_scale=1,
                )
            # No-op if it matches what we're already running with —
            # avoids the page-load applyRes() triggering a useless
            # reconnect cycle.
            if self._config.advertise == new_advertise:
                return
            self._config.advertise = new_advertise
            log.info("viewer requested resolution → advertise=%s", new_advertise)
            # Schedule the hot-reconnect on the asyncio loop. Use a
            # one-shot guard so concurrent set_resolution events
            # collapse into a single restart cycle.
            if getattr(self, "_reconnect_pending", False):
                return
            self._reconnect_pending = True
            def _do_reconnect():
                try:
                    self._teardown_session()
                    # Wait briefly for teardown to settle before fresh start.
                    time.sleep(0.5)
                    with self._viewers_lock:
                        should_start = (
                            self._session is None
                            and not self._session_starting
                            and bool(self.viewers)
                        )
                        if should_start:
                            self._session_starting = True
                    if should_start:
                        self._start_session_thread()
                finally:
                    self._reconnect_pending = False
            threading.Thread(
                target=_do_reconnect,
                name="wt-session-resreconnect", daemon=True,
            ).start()
            return
        if self._session is None:
            return
        try:
            if kind == "mouse":
                self._session.input.pointer_event(
                    int(msg.get("buttons", 0)),
                    int(msg["x"]), int(msg["y"]),
                )
            elif kind == "scroll":
                self._session.input.scroll_event(
                    int(msg["x"]), int(msg["y"]),
                    int(msg.get("dx", 0)), int(msg.get("dy", 0)),
                )
            elif kind == "key":
                self._session.input.key_event(
                    bool(msg.get("down")), int(msg["keysym"]),
                )
            elif kind == "resize":
                self._handle_resize(int(msg["w"]), int(msg["h"]))
        except Exception as e:
            log.debug("input dispatch error: %s", e)

    def _display_scale(self) -> float:
        """Numeric display scale from --hidpi: auto/on → 2×, off → 1×, or a
        custom float like 2.5. The browser advertises its window at
        logical = physical/scale so the host draws the UI at `scale` while the
        backing stays = the window's physical pixels (WYSIWYG bandwidth)."""
        h = str(getattr(self._config, "hidpi", "auto") or "auto").strip().lower()
        if h == "off":
            return 1.0
        if h in ("on", "auto"):
            return 2.0
        try:
            return min(4.0, max(1.0, float(h)))
        except ValueError:
            return 2.0

    def _handle_resize(self, w: int, h: int) -> None:
        """The browser reported a new viewport size — re-advertise the host
        display to match, so the stream is encoded at the window's resolution
        instead of the host's full canvas (dynamic-resolution parity with the
        desktop frontend; this is what makes the connect form's "Auto" actually
        track the browser window). Clamp to sane bounds and de-dup: the client
        debounces so it only sends the settled size once per resize gesture, so
        no server-side throttle is needed beyond ignoring an unchanged size."""
        if not self._config.dynamic_resolution:
            return  # fixed --advertise: respect the user's explicit resolution
        w = max(640, min(3840, w & ~1))
        h = max(480, min(2160, h & ~1))
        if (w, h) == getattr(self, "_last_resize_wh", None):
            return
        self._last_resize_wh = (w, h)
        if self._session is not None:
            try:
                # The host makes the backing canvas w*scale; the browser already
                # reported w = physical/scale (see index.html), so backing =
                # the window's physical pixels (crisp 1:1). `scale` is the user's
                # Display-scale (--hidpi); it sets the host UI size, and both ends
                # MUST use the same value or the UI size / bandwidth are wrong.
                self._session.send_dynamic_resolution(
                    w, h, hidpi_scale=self._display_scale())
                log.info("browser resize → re-advertising host display %dx%d", w, h)
            except Exception as e:
                log.debug("send_dynamic_resolution failed: %s", e)


# ── wire envelope ────────────────────────────────────────────────────

# Per-stream wire payload: 9-byte header + payload.
# Video/audio envelope wire format:
#   byte 0:    type (0=video delta, 1=video keyframe, 2=audio opus, 3=config)
#   byte 1:    tile_id (video only; 0 for audio/config)
#   bytes 2-9: presentation timestamp (microseconds, big-endian u64)
#   bytes 10+: payload (Annex-B H.264 AU / Opus packet / JSON config)
# Pass-through: the payload is Apple's H.264 access unit verbatim (no re-encode);
# SPS/PPS are prepended to keyframe AUs so the browser feeds plain Annex-B.
# Envelope: [tag(1)][tile_id(1)][seq(4,BE)][ts(8,BE)][payload]. The per-frame
# QUIC streams have NO cross-stream ordering guarantee, so the browser reorders
# video by `seq` (the order the AUs were forwarded = the native decode order)
# before feeding its single shared decoder — out-of-order feeding makes Apple's
# tiled stream conceal/drift ("fleas").
_HEADER_LEN = 14
_TYPE_VIDEO_DELTA = 0
_TYPE_VIDEO_KEY = 1
_TYPE_AUDIO_OPUS = 2
_TYPE_CONFIG = 3
_TYPE_CURSOR = 4


def _frame_envelope(au_bytes: bytes, is_key: bool, ts_us: int, tile_id: int,
                    seq: int) -> bytes:
    t = _TYPE_VIDEO_KEY if is_key else _TYPE_VIDEO_DELTA
    return (bytes([t, tile_id & 0xFF]) + (seq & 0xFFFFFFFF).to_bytes(4, "big")
            + ts_us.to_bytes(8, "big") + au_bytes)


def _audio_envelope(opus_bytes: bytes, ts_us: int) -> bytes:
    return (bytes([_TYPE_AUDIO_OPUS, 0]) + (0).to_bytes(4, "big")
            + ts_us.to_bytes(8, "big") + opus_bytes)


def _config_envelope(config_json: bytes) -> bytes:
    return (bytes([_TYPE_CONFIG, 0]) + (0).to_bytes(4, "big")
            + (0).to_bytes(8, "big") + config_json)


def _cursor_envelope(width: int, height: int, hx: int, hy: int, rgba: bytes) -> bytes:
    # 14-byte common header (type, then unused) + cursor metadata + RGBA8888.
    # The cursor is NOT baked into the framebuffer (RFB enc 1104); the browser
    # paints it as the canvas CSS cursor, positioned by the local pointer.
    return (bytes([_TYPE_CURSOR, 0]) + (0).to_bytes(4, "big") + (0).to_bytes(8, "big")
            + width.to_bytes(2, "big") + height.to_bytes(2, "big")
            + hx.to_bytes(2, "big") + hy.to_bytes(2, "big") + rgba)


# ── module entry ─────────────────────────────────────────────────────

def run(config: SessionConfig, *, port: int = 4433,
        trust_cert_for_safari: bool = False, **_unused: object) -> int:
    return WebTransportBridge(
        config, port=port, trust_cert_for_safari=trust_cert_for_safari,
    ).run()


__all__ = ["WebTransportBridge", "run"]
