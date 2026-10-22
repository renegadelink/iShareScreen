"""Bounded transactional framebuffer receiver; no Qt or network dependencies."""
from __future__ import annotations
import numpy as np
from urllib.parse import urlsplit, urlunsplit
from . import wire as W


def endpoint(text):
    url = urlsplit(text.strip())
    if url.scheme not in ('ws', 'wss') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Enter ws://host:81/rvdp or a verified wss:// endpoint; credentials in URLs are not supported.')
    port = url.port or (443 if url.scheme == 'wss' else 80)
    if not 1 <= port <= 65535:
        raise ValueError('Invalid RVDP port')
    if url.path not in ('', '/rvdp'):
        raise ValueError('The RVDP path must be /rvdp')
    return urlunsplit((url.scheme, url.netloc, '/rvdp', '', '')), port


class Framebuffer:
    def __init__(self):
        self.sequence = W.Sequence()
        self.width = self.height = 0
        self.pixels = self.staged = self.coverage = None
        self.frame = self.last_frame = None
        self.count = 0
        self.needs_full = True
        self.granted = self.capable = False
        self.keyboard = False
        self.minor = 0

    def resync(self):
        self.needs_full = True
        self.frame = None
        self.count = 0

    def receive(self, data):
        packet = W.unpack(data)
        if self.sequence.last is None and packet.sequence != 0:
            raise ValueError('Session sequence must start at zero')
        self.sequence.accept(packet.sequence)
        p = packet.payload
        if packet.kind == W.Type.DISPLAY_INFO:
            if self.width:
                raise ValueError('Unexpected second display description')
            self.width, self.height, self.session, self.capable = W.read_display(p)
            self.minor = packet.minor
            self.keyboard = bool(W.DISPLAY.unpack(p)[-1] & 2)
            if self.keyboard and self.minor != 1:
                raise ValueError('Keyboard capability requires RVDP 1.1')
            self.pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            self.pixels[:, :, 3] = 255
            return 'display', None
        if not self.width:
            raise ValueError('DISPLAY_INFO must precede screen updates')
        if packet.minor != self.minor:
            raise ValueError('Session version changed')
        if packet.kind == W.Type.CONTROL_STATE:
            if len(p) != 1 or p[0] > 1 or (p[0] and not self.capable):
                raise ValueError('Invalid control grant')
            self.granted = bool(p[0])
            return 'control', self.granted
        if packet.kind == W.Type.RECT:
            frame, x, y, w, h, data = W.read_rectangle(p, self.width, self.height)
            if self.last_frame is not None and not 0 < ((frame - self.last_frame) & 0xffffffff) < 0x80000000:
                raise ValueError('Stale render batch')
            if self.frame is not None and frame != self.frame:
                self.resync()
                return 'resync', None
            if self.frame is None:
                self.frame = frame
                self.staged = self.pixels.copy()
                self.coverage = np.zeros((self.height, self.width), dtype=bool)
            self.count += 1
            if self.count > 512:
                raise ValueError('Too many rectangles')
            rgb = np.frombuffer(data, dtype='>u2').reshape(h, w).astype(np.uint32)
            dest = self.staged[y:y+h, x:x+w]
            dest[:, :, 0] = ((rgb >> 11) * 255 + 15) // 31
            dest[:, :, 1] = (((rgb >> 5) & 63) * 255 + 31) // 63
            dest[:, :, 2] = ((rgb & 31) * 255 + 15) // 31
            self.coverage[y:y+h, x:x+w] = True
            return None, None
        if packet.kind == W.Type.FRAME_END:
            frame, count, full, _ = W.read_end(p)
            if frame != self.frame or count != self.count or (full and not self.coverage.all()) or (self.needs_full and not full):
                self.resync()
                return 'resync', None
            self.pixels = self.staged
            self.last_frame = frame
            self.frame = None
            self.count = 0
            self.needs_full = False
            return 'frame', self.pixels
        if packet.kind in (W.Type.PING, W.Type.PONG):
            if len(p) != 4:
                raise ValueError('Invalid heartbeat')
            return packet.kind.name.lower(), p
        if packet.kind == W.Type.ERROR:
            if not 2 <= len(p) <= 130:
                raise ValueError('Invalid ERROR size')
            raise ConnectionError(p[2:].decode('utf-8', 'replace'))
        raise ValueError('Unexpected device message')
