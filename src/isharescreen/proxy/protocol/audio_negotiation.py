"""Read-only diagnostics for AVConference audio capability answers."""

def _proto_fields(data: bytes):
    """Walk bounded protobuf fields; reject truncated or unsupported encodings."""
    pos = 0

    def varint():
        nonlocal pos
        value = 0
        for shift in range(0, 70, 7):
            if pos >= len(data):
                raise ValueError("truncated varint")
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("oversized varint")

    while pos < len(data):
        tag = varint()
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("invalid field number")
        if wire == 0:
            yield number, varint()
            continue
        size = varint() if wire == 2 else {1: 8, 5: 4}.get(wire)
        if size is None or size > len(data) - pos:
            raise ValueError("invalid field length")
        value = data[pos:pos + size]
        pos += size
        yield number, value


def audio_answer_fields(blob: bytes) -> dict[int, int] | None:
    """Read audio F3 scalars, excluding the video answer's audio placeholder.

    Field names beyond SSRC remain provisional; return numbers rather than
    presenting an unverified flag-to-codec mapping as a selected codec.
    """
    try:
        fields = list(_proto_fields(blob))
        if any(number == 5 for number, _ in fields):
            return None
        for number, value in fields:
            if number == 3 and isinstance(value, bytes):
                return {n: v for n, v in _proto_fields(value)
                        if 1 <= n <= 6 and isinstance(v, int)} or None
    except ValueError:
        pass
    return None
