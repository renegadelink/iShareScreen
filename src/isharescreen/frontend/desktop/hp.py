"""High Performance (Apple's UDP/SRTP video stream) as a session controller.

EXPERIMENTAL. Measured against an Intel iMac and an Apple Silicon Mac mini, the
bundled HP engine completed the login and negotiation but did not deliver video:
the iMac sent H.264 where HEVC was expected, then stalled in the hardware decoder
or never answered the canvas request, and the mini sent no RTP at all. A failed
attempt can also leave a Mac's Screen Sharing agent unresponsive for a couple of
minutes. So this path is off unless the user switches it on, every attempt has a
hard deadline, only one attempt per Mac may run at a time, and every failure is
reported plainly instead of leaving a black viewer.
"""
from __future__ import annotations
import importlib.util
import logging
import os
import threading
import time

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QImage

from .session import SessionController

log = logging.getLogger('iss.desktop.hp')

START_TIMEOUT_S = 45.0          # connect + first picture, then the attempt is abandoned
MIN_FRAME_INTERVAL_S = 1 / 30   # colour conversion is capped at about 30 pictures a second
WHEEL_TICKS = 3                 # HP ticks sent per wheel notch
RESOLUTIONS = (('auto', 'Automatic (1920 × 1080)'), ('1280x720', '1280 × 720'), ('1366x768', '1366 × 768'),
               ('1600x900', '1600 × 900'), ('1920x1080', '1920 × 1080'))
FAILURE_WARNING = ('A failed High Performance attempt can leave this Mac’s Screen Sharing unresponsive for a '
                   'couple of minutes.')
_ACTIVE: set = set()            # (host, port) with an attempt in progress
_ACTIVE_LOCK = threading.Lock()


def parse_resolution(value: str) -> tuple[int, int]:
    """'auto' or 'WxH' -> the logical size to ask the Mac for (within what it supports)."""
    if value and value != 'auto':
        try:
            width, height = (int(part) for part in value.lower().split('x'))
            if 640 <= width <= 1920 and 480 <= height <= 1080:
                return width, height
        except ValueError:
            pass
    return 1920, 1080


def hp_capability(result, enabled: bool) -> tuple[bool, str]:
    """Whether High Performance can be offered for a verified server, with a reason.

    A banner and security list only prove the login exists; whether video flows can
    only be learned by connecting, which is why this stays experimental."""
    if not enabled:
        return False, 'High Performance is coming in a later version.'
    if result.auth != 'account':
        return False, 'High Performance needs a Mac account sign-in.'
    if 33 not in result.security:
        return False, 'This computer does not offer the High Performance login.'
    if importlib.util.find_spec('av') is None:
        return False, 'High Performance needs the PyAV video library, which is not installed.'
    return True, 'Experimental: it may not deliver video from every Mac. ' + FAILURE_WARNING


# ── pictures ─────────────────────────────────────────────────────────

def compose_rgba(tiles: dict, canvas_w: int, canvas_h: int) -> np.ndarray:
    """Stack decoded YUV tiles into one RGBA picture (BT.709, full range, as Apple's
    stream uses). Tiles are horizontal strips: tile i starts at row i * (first tile height)."""
    first = tiles[min(tiles)]
    if first.is_nv12_passthrough:
        raise ValueError('NV12 passthrough tiles are not supported')
    subsampled = not first.is_yuv444
    y = np.full((canvas_h, canvas_w), 16, dtype=np.uint8)
    cw, ch = (canvas_w + 1) // 2, (canvas_h + 1) // 2
    u = np.full((ch, cw) if subsampled else (canvas_h, canvas_w), 128, dtype=np.uint8)
    v = u.copy()
    slot = first.height
    for index, tile in sorted(tiles.items()):
        top = index * slot
        if top >= canvas_h:
            continue
        rows, cols = min(tile.height, canvas_h - top), min(tile.width, canvas_w)
        plane = np.frombuffer(tile.y, dtype=np.uint8).reshape(-1, tile.y_stride)
        y[top:top + rows, :cols] = plane[:rows, :cols]
        c_top = top // 2 if subsampled else top
        c_rows = min(tile.chroma_height, u.shape[0] - c_top)
        c_cols = min(tile.chroma_width, u.shape[1])
        if c_rows > 0:
            for target, source in ((u, tile.u), (v, tile.v)):
                data = np.frombuffer(source, dtype=np.uint8).reshape(-1, tile.uv_stride)
                target[c_top:c_top + c_rows, :c_cols] = data[:c_rows, :c_cols]
    if subsampled:
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:canvas_h, :canvas_w]
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:canvas_h, :canvas_w]
    yi, ui, vi = y.astype(np.int32), u.astype(np.int32) - 128, v.astype(np.int32) - 128
    rgba = np.empty((canvas_h, canvas_w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(yi + ((vi * 403) >> 8), 0, 255)
    rgba[..., 1] = np.clip(yi - ((ui * 48 + vi * 120) >> 8), 0, 255)
    rgba[..., 2] = np.clip(yi + ((ui * 475) >> 8), 0, 255)
    rgba[..., 3] = 255
    return rgba


def classify_hp_error(error: BaseException) -> tuple[str, str]:
    """`(kind, message)` for a failed High Performance attempt, in plain language."""
    text = str(error)
    name = type(error).__name__
    if name == 'AuthError':
        return 'auth', 'The Mac rejected the user name or password. Check them and try again.'
    if 'never started' in text:
        return 'hp_no_video', ('High Performance did not start: the Mac accepted the login but sent no video. '
                               'Standard mode works with this computer. ' + FAILURE_WARNING)
    if 'another user' in text:
        return 'hp_console_user', text
    if 'decoder' in text.lower():
        return 'hp_decoder', f'This PC could not set up a video decoder for High Performance. ({text})'
    if isinstance(error, TimeoutError):
        return 'timeout', 'The Mac did not answer in time. ' + FAILURE_WARNING
    if isinstance(error, ConnectionError):
        return 'network', text or 'The connection was closed by the computer or the network.'
    return 'hp_error', f'High Performance stopped: {text or name}. ' + FAILURE_WARNING


class HPController(SessionController):
    """Same signals and methods as SessionController, over the HP engine.

    `session_factory(config)` exists so tests can stand in for the engine."""
    mode_label = 'High Performance'
    _timed_out = False

    def __init__(self, host, port, user, password, auth='account', resolution='auto', session_factory=None):
        super().__init__(host, port, user, password, auth)
        self.resolution = resolution
        self._factory = session_factory
        self.session = None
        self._timer = None
        self._owns_slot = False
        self._close_started = False

    def clone(self, secret):
        return HPController(self.host, self.port, self.user, secret, self.auth, self.resolution, self._factory)

    # ── lifecycle ────────────────────────────────────────────────────
    def stop(self):
        self.stopping = True
        timer, self._timer = self._timer, None
        if timer:
            timer.cancel()
        session = self.session
        if session is not None and not self._close_started:
            self._close_started = True
            # Session.close() joins its threads (seconds): never block the GUI on it.
            threading.Thread(target=self._close_session, args=(session,), daemon=True).start()

    @staticmethod
    def _close_session(session):
        try:
            session.close()
        except Exception as error:      # best effort: the connection is being abandoned anyway
            log.warning('closing the HP session: %s', error)

    def _expire(self):
        self._timed_out = True
        self.stop()

    def _claim(self) -> bool:
        with _ACTIVE_LOCK:
            key = (self.host, self.port)
            if key in _ACTIVE:
                return False
            _ACTIVE.add(key)
            self._owns_slot = True
            return True

    def _release(self):
        if self._owns_slot:
            with _ACTIVE_LOCK:
                _ACTIVE.discard((self.host, self.port))
            self._owns_slot = False

    def _build_config(self):
        from ...proxy.session import SessionConfig
        from ...proxy.protocol.negotiation import AdvertiseDims

        def refuse(console_user):
            # Never pop a "share your screen?" request up on someone else's Mac.
            raise RuntimeError(f'Another user ({console_user}) is signed in at this Mac. High Performance would '
                               'have to ask them to share their screen, so it was not attempted.')
        width, height = parse_resolution(self.resolution)
        return SessionConfig(host=self.host, username=self.user, password=self._password, port=self.port,
                             advertise=AdvertiseDims(width=width, height=height, hidpi_scale=1.0),
                             audio=False, hidpi='off', curtain=False, share_console=False, alt_session=False,
                             on_session_choice=refuse)

    def run(self):
        if not self._claim():
            self.failed.emit('hp_busy', 'A High Performance connection to this computer is already starting. '
                                        'Wait for it to finish before trying again.')
            return
        try:
            os.environ.setdefault('ISS_VIDEO_CODEC', 'avc')    # H.264 decodes everywhere; Intel Macs send it regardless
            os.environ['ISS_CLIP'] = '0'                       # clipboard is handled by the session window
            self.phase.emit('Starting High Performance…')
            config = self._build_config()
            self._password = ''
            if self._factory is not None:
                session = self._factory(config)
            else:
                from ...proxy.session import Session
                session = Session(config)
            self.session = session
            if self.stopping:
                return
            self._timer = threading.Timer(START_TIMEOUT_S, self._expire)
            self._timer.daemon = True
            self._timer.start()
            session.set_clipboard_text_callback(lambda text: None if self.stopping else self.clipboard.emit(text))
            session.connect()
            if self.stopping:
                return
            self.phase.emit('Waiting for the first screen update…')
            self._stream(session)
            if self._timed_out:
                raise TimeoutError('no picture before the deadline')
            if not self.stopping:
                raise ConnectionError('The High Performance connection was closed by the computer or the network.')
        except Exception as error:
            if self._timed_out:
                self.failed.emit('hp_timeout', f'High Performance did not start within {int(START_TIMEOUT_S)} seconds. '
                                               'Standard mode works with this computer. ' + FAILURE_WARNING)
            elif not self.stopping:
                self.failed.emit(*classify_hp_error(error))
        finally:
            self._password = ''
            timer, self._timer = self._timer, None
            if timer:
                timer.cancel()
            session, self.session = self.session, None
            if session is not None and not self._close_started:
                self._close_session(session)     # (after stop() the close already runs in the background)
            self._release()

    def _stream(self, session):
        tiles: dict = {}
        dirty, sent_first, last = False, False, 0.0
        while not self.stopping and session.is_connected:
            if session.wait_for_fresh_tile(timeout=0.2):
                for index in range(max(1, session.num_tiles)):
                    tile = session.get_frame(index)
                    if tile is not None:
                        tiles[index] = tile
                        dirty = True
            now = time.monotonic()
            if dirty and tiles and now - last >= MIN_FRAME_INTERVAL_S:
                width, height = session.canvas_dims
                rgba = compose_rgba(tiles, width, height)
                image = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
                self._raw_frame.emit(image)
                if not sent_first:
                    sent_first = True
                    timer, self._timer = self._timer, None
                    if timer:
                        timer.cancel()           # the picture arrived: the deadline no longer applies
                dirty, last = False, now

    # ── input ────────────────────────────────────────────────────────
    def send(self, method, *args):
        session = self.session
        if session is None or self.stopping:
            return
        try:
            if method == 'pointer':
                x, y, mask = args
                if mask & 8 or mask & 16:                                  # wheel: one notch per press
                    session.input.scroll_event(x, y, 0, -WHEEL_TICKS if mask & 8 else WHEEL_TICKS)
                session.input.pointer_event(mask & 7, x, y)
            elif method == 'key':
                keysym, down = args
                session.input.key_event(bool(down), keysym)
            elif method == 'send_clipboard':
                session.input.cut_text(args[0])
            elif method == 'request':
                session.request_fir(None)
        except (OSError, RuntimeError, ValueError) as error:
            log.debug('HP input %s dropped: %s', method, error)
