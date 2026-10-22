"""RVDP 1.0 wire codec. Keep the iShareScreen wire.py copy byte-identical."""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import struct

MAX_PIXELS = 320 * 240
MAX_RECT = 16_384
MAX_PAYLOAD = MAX_RECT + 17
HEADER = struct.Struct('>4sBBBBII')
HELLO = struct.Struct('>BBBBIIB')
DISPLAY = struct.Struct('>HHBBII')
RECT = struct.Struct('>IHHHHBI')
END = struct.Struct('>IHBI')
POINTER = struct.Struct('>HHBB')
KEY = struct.Struct('>IB')
SPECIAL_KEYS = (0xff08, 0xff09, 0xff0d, 0xff1b, 0xff50, 0xff51, 0xff53, 0xff55, 0xff56, 0xff57, 0xffff)


class Type(IntEnum):
    HELLO = 1
    DISPLAY_INFO = 2
    RECT = 3
    FRAME_END = 4
    RESYNC = 5
    POINTER = 6
    KEY = 7
    PING = 8
    PONG = 9
    ERROR = 10
    CONTROL_STATE = 11


@dataclass(frozen=True)
class Packet:
    kind: Type
    sequence: int
    payload: bytes
    minor: int = 0


def pack(kind, sequence, payload=b'', *, minor=0):
    if len(payload) > MAX_PAYLOAD:
        raise ValueError('RVDP payload too large')
    if minor not in (0, 1):
        raise ValueError('Unsupported RVDP minor version')
    return HEADER.pack(b'RVDP', 1, minor, int(Type(kind)), 0, sequence, len(payload)) + payload


def unpack(data):
    if not isinstance(data, bytes) or not HEADER.size <= len(data) <= HEADER.size + MAX_PAYLOAD:
        raise ValueError('Invalid RVDP packet size')
    magic, major, minor, kind, flags, sequence, length = HEADER.unpack_from(data)
    if magic != b'RVDP' or major != 1 or minor not in (0, 1) or flags:
        raise ValueError('Unsupported RVDP header')
    if len(data) != HEADER.size + length:
        raise ValueError('RVDP length mismatch')
    try:
        kind = Type(kind)
    except ValueError:
        raise ValueError('Unsupported RVDP message type') from None
    if kind == Type.KEY and minor != 1:
        raise ValueError('Keyboard requires RVDP 1.1')
    return Packet(kind, sequence, data[HEADER.size:], minor)


def hello(control=False, keyboard=False):
    return HELLO.pack(1, 0, 1, int(keyboard), 1, MAX_RECT, bool(control))


def read_hello(payload):
    if len(payload) != HELLO.size:
        raise ValueError('Invalid HELLO size')
    lo, lm, hi, hm, enc, maximum, control = HELLO.unpack(payload)
    if (lo, lm, hi) != (1, 0, 1) or hm not in (0, 1) or enc != 1 or not 2 <= maximum <= MAX_RECT or control > 1:
        raise ValueError('Unsupported HELLO capabilities')
    return maximum, bool(control)


def display_info(width, height, session=1, control=False, keyboard=False):
    return DISPLAY.pack(width, height, 1, 0, session, int(bool(control)) | (2 if keyboard else 0))


def read_display(payload):
    if len(payload) != DISPLAY.size:
        raise ValueError('Invalid DISPLAY_INFO size')
    w, h, pixel, rotation, session, features = DISPLAY.unpack(payload)
    if not (0 < w <= 320 and 0 < h <= 240 and w * h <= MAX_PIXELS):
        raise ValueError('Invalid display dimensions')
    if pixel != 1 or rotation or features & ~3:
        raise ValueError('Unsupported display format/features')
    return w, h, session, bool(features & 1)


def rectangle(frame, x, y, width, height, pixels):
    if not width or not height or len(pixels) != width * height * 2 or len(pixels) > MAX_RECT:
        raise ValueError('Invalid rectangle pixels')
    return RECT.pack(frame, x, y, width, height, 1, len(pixels)) + pixels


def read_rectangle(payload, width, height):
    if len(payload) < RECT.size:
        raise ValueError('Truncated rectangle')
    frame, x, y, w, h, encoding, length = RECT.unpack_from(payload)
    if not w or not h or x + w > width or y + h > height or encoding != 1:
        raise ValueError('Invalid rectangle bounds/encoding')
    if length != w * h * 2 or length > MAX_RECT or len(payload) != RECT.size + length:
        raise ValueError('Invalid rectangle length')
    return frame, x, y, w, h, payload[RECT.size:]


def read_end(payload):
    if len(payload) != END.size:
        raise ValueError('Invalid FRAME_END size')
    frame, count, full, timestamp = END.unpack(payload)
    if not count or count > 512 or full > 1:
        raise ValueError('Invalid FRAME_END fields')
    return frame, count, bool(full), timestamp


def read_pointer(payload, width, height):
    if len(payload) != POINTER.size:
        raise ValueError('Invalid POINTER size')
    x, y, buttons, event = POINTER.unpack(payload)
    if x >= width or y >= height or buttons > 1 or event > 3 or (event in (2, 3) and buttons):
        raise ValueError('Invalid pointer event')
    return x, y, buttons, event


def valid_key(key):
    return isinstance(key, int) and (32 <= key <= 126 or key in SPECIAL_KEYS)


def read_key(payload):
    if len(payload) != KEY.size:
        raise ValueError('Invalid KEY size')
    key, down = KEY.unpack(payload)
    if not valid_key(key) or down > 1:
        raise ValueError('Invalid key event')
    return key, bool(down)


class Sequence:
    def __init__(self):
        self.last = None

    def accept(self, value):
        if self.last is not None and value != ((self.last + 1) & 0xffffffff):
            raise ValueError('Stale or skipped RVDP sequence')
        self.last = value
