"""Read-only diagnostics for Apple's length-prefixed media replies."""
import logging
import struct

log = logging.getLogger(__name__)


def parse_media_reply(payload: bytes) -> dict | None:
    if len(payload) < 2:
        return None
    size = struct.unpack_from(">H", payload)[0]
    if size < 8 or len(payload) < size + 2:
        return None
    body = payload[2:2 + size]
    kind = struct.unpack_from(">H", body)[0]
    if kind == 1 and len(body) >= 26:
        result = {"kind": "ports"}
        for name, offset in (("audio", 8), ("video", 14), ("video2", 20)):
            port, flags = struct.unpack_from(">HI", body, offset)
            result[name + "_port"] = port
            result[name + "_flags"] = flags
        return result
    if kind == 3 and len(body) >= 16:
        error_type, subcode = struct.unpack_from(">II", body, 8)
        return {"kind": "error", "type": error_type, "subcode": subcode}
    return None


def log_media_reply(payload: bytes) -> None:
    reply = parse_media_reply(payload)
    if reply is None:
        return
    if reply["kind"] == "error":
        log.warning("media server refusal: type=%d subcode=%d",
                    reply["type"], reply["subcode"])
        return
    log.info("media ports: audio=%d flags=0x%x video=%d flags=0x%x "
             "video2=%d flags=0x%x (flag bit 0 enables each leg)",
             reply["audio_port"], reply["audio_flags"],
             reply["video_port"], reply["video_flags"],
             reply["video2_port"], reply["video2_flags"])


def log_media_transport(message: bytes) -> None:
    """Inspect recognized FBU rectangles without consuming protocol state.

    Stop on unknown encodings: guessing their length could mistake pixel or
    cursor data for a transport announcement. No key/blob bytes are logged.
    """
    if len(message) < 4 or message[0] != 0:
        return
    count = struct.unpack_from(">H", message, 2)[0]
    pos = 4
    prefixed = {1010, 1011, 1107, 1109, 1110, 0x3ea, 0x453, 0x455, 0x456}
    for _ in range(count):
        if pos + 12 > len(message):
            return
        encoding = struct.unpack_from(">i", message, pos + 8)[0]
        pos += 12
        if encoding == 1103:
            pos += 36
        elif encoding in prefixed:
            if pos + 2 > len(message):
                return
            size = struct.unpack_from(">H", message, pos)[0]
            if pos + 2 + size > len(message):
                return
            if encoding in (1010, 1011):
                log_media_reply(message[pos:pos + 2 + size])
            pos += 2 + size
        else:
            return
