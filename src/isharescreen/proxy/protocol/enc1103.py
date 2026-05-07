"""enc1103: Apple's bespoke RFB control-channel stream cipher.

After non-SRP / SRP authentication the server emits a `1103` encoding entry
containing 32 cipher-text bytes ECB-wrapped under the auth-derived key. We
unwrap to a 16-byte AES key + 16-byte CBC IV; from then on every RFB control
message in either direction is framed and AES-128-CBC encrypted with a
counter-keyed HMAC-SHA1-160 tag.

Wire format on the encrypted channel:

    u16 BE  ciphertext_length
    bytes   ciphertext      (multiple of 16)

Plaintext payload inside the cipher block:

    u16 BE  msg_length
    bytes   msg
    bytes   pad             (zero pad to 16-byte alignment after the 20-byte mac)
    bytes   mac             (HMAC-SHA1(counter || everything-before-mac))

Counter is per-direction, starts at 0, increments on each successful decrypt
or each encrypt. Decryption tolerates a small lookahead window because Apple
sometimes pipelines a re-key reply before the previous decrypt's counter has
been consumed.
"""
from __future__ import annotations

import hashlib
import logging
import struct
import threading
from typing import Optional

from Crypto.Cipher import AES


log = logging.getLogger(__name__)

_MAC_LEN = 20  # HMAC-SHA1-160
_BLOCK = 16    # AES-128 block size
_DECRYPT_COUNTER_WINDOW = 6  # forgive up to 6-message gaps from server


class StreamCipher:
    """Bidirectional enc1103 cipher state. Created from the 36-byte 1103 blob."""

    def __init__(self, key_blob_36: bytes, *, ecb_key: bytes) -> None:
        if len(key_blob_36) != 36:
            raise ValueError(f"enc1103 key blob must be 36 bytes, got {len(key_blob_36)}")
        ecb = AES.new(ecb_key, AES.MODE_ECB)
        cbc_key = ecb.decrypt(key_blob_36[4:20])
        cbc_iv = ecb.decrypt(key_blob_36[20:36])
        # Public alongside the wrapped CBC contexts because the alt-session
        # path's msg 0x10 input wrapper (HandleEncryptedEventMessage)
        # needs to AES-128-ECB-encrypt input blocks under THIS key. The
        # daemon's cryptor is replaced from `ecb_key` to `cbc_key`
        # inside HandleSetEncryptionMessage when iss sends
        # PostEncryptionToggle (msg 0x12), so all post-toggle msg-0x10
        # events must use this key, not the SRP-K-derived `ecb_key`.
        self.cbc_key = cbc_key

        # Apple uses the same key + IV for both directions but threads its own
        # CBC state per direction, so we need two independent contexts.
        self._enc = AES.new(cbc_key, AES.MODE_CBC, iv=cbc_iv)
        self._dec = AES.new(cbc_key, AES.MODE_CBC, iv=cbc_iv)
        self._enc_ctr = 0
        self._dec_ctr = 0
        self._lock = threading.Lock()

    def encrypt_message(self, plaintext: bytes) -> bytes:
        """Wrap a control-channel message (RFB body) for sending."""
        with self._lock:
            counter = self._enc_ctr
            pad = (-(2 + len(plaintext) + _MAC_LEN)) % _BLOCK
            framed = struct.pack(">H", len(plaintext)) + plaintext + b"\x00" * pad
            mac = hashlib.sha1(struct.pack(">I", counter) + framed).digest()
            block = framed + mac
            ciphertext = self._enc.encrypt(block)
            self._enc_ctr = counter + 1
        return struct.pack(">H", len(ciphertext)) + ciphertext

    def decrypt_message(self, ciphertext: bytes) -> Optional[bytes]:
        """Decrypt one ciphertext block. Returns the inner RFB body, or None
        if the MAC doesn't verify under any candidate counter.

        Counter recovery is empirically tolerant: we accept a hit anywhere in
        ``[ctr-1, ctr+5]`` (handles small server-side reorderings + a 5-msg
        future window). On MAC miss we still advance the receive counter by
        1, on the heuristic "we missed by more than 5 — nudge forward and
        let future messages resync." The AES-CBC decrypt MUST happen for
        every received block regardless of MAC outcome, since CBC consumes
        IV state across blocks.
        """
        if not ciphertext:
            return None
        with self._lock:
            plaintext = self._dec.decrypt(ciphertext)
            ctr_start = self._dec_ctr
            self._dec_ctr += 1
        if len(ciphertext) <= _MAC_LEN:
            return plaintext
        body, mac = plaintext[:-_MAC_LEN], plaintext[-_MAC_LEN:]
        for c in range(max(0, ctr_start - 1), ctr_start + _DECRYPT_COUNTER_WINDOW):
            if mac == hashlib.sha1(struct.pack(">I", c) + body).digest():
                # Resync decoder counter to the matched value (+1).
                with self._lock:
                    self._dec_ctr = c + 1
                inner_len = struct.unpack(">H", body[0:2])[0]
                return body[2:2 + inner_len]
        return None

    def decrypt_stream(self, data: bytes) -> tuple[list[bytes], int]:
        """Decrypt as many enc1103-framed messages as fit in `data`. Stops at
        the first incomplete frame.

        Returns ``(messages, consumed_bytes)`` so streaming consumers can
        keep the unconsumed tail in their buffer for the next call.
        """
        out: list[bytes] = []
        pos = 0
        while pos + 2 <= len(data):
            length = struct.unpack(">H", data[pos:pos + 2])[0]
            if length == 0 or length % _BLOCK != 0 or pos + 2 + length > len(data):
                break
            msg = self.decrypt_message(data[pos + 2:pos + 2 + length])
            pos += 2 + length
            if msg is not None:
                out.append(msg)
        return out, pos


__all__ = ["StreamCipher"]
