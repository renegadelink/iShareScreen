"""SRTP / SRTCP encrypt + decrypt using AES-256-CTR + HMAC-SHA1-80.

Uses ``cryptography``'s OpenSSL backend on the hot path; comfortably exceeds
Apple Screen Sharing's ~2.5 kpkt/s peak. PyCryptodome handles the slower
KDF setup so we avoid pulling in two AES backends when one already works.

Key derivation labels (RFC 3711 §4.3):
    0/1/2  →  RTP cipher / auth / salt
    3/4/5  →  RTCP cipher / auth / salt
"""
from __future__ import annotations

import hmac
import logging
import struct
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from Crypto.Cipher import AES
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


log = logging.getLogger(__name__)

_AUTH_TAG_LEN = 10  # HMAC-SHA1-80 truncated to 80 bits
_RTP_HEADER_MIN = 12

# AVCMediaStreamNegotiator hands us a 46-byte SRTP "key blob" per direction:
# 32 bytes of master key followed by 14 bytes of master salt.
SRTP_KEY_BLOB_LEN = 46
SRTP_MASTER_KEY_LEN = 32
SRTP_MASTER_SALT_LEN = 14


def _split_blob(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) != SRTP_KEY_BLOB_LEN:
        raise ValueError(
            f"SRTP key blob must be {SRTP_KEY_BLOB_LEN} bytes, got {len(blob)}"
        )
    return blob[:SRTP_MASTER_KEY_LEN], blob[SRTP_MASTER_KEY_LEN:SRTP_KEY_BLOB_LEN]


def _srtp_kdf(master_key: bytes, master_salt: bytes, label: int, out_len: int) -> bytes:
    """SRTP key derivation function (RFC 3711 §4.3.1) — AES-CM with the label
    XORed into the IV. Produces `out_len` bytes."""
    kid = bytearray(14)
    kid[7] = label
    iv0 = bytes(kid[i] ^ master_salt[i] for i in range(14))
    ecb = AES.new(master_key, AES.MODE_ECB)
    out = bytearray()
    counter = 0
    while len(out) < out_len:
        block = bytearray(iv0 + b"\x00\x00")
        c = counter
        for i in range(15, -1, -1):
            if c == 0:
                break
            c += block[i]
            block[i] = c & 0xFF
            c >>= 8
        out += ecb.encrypt(bytes(block))
        counter += 1
    return bytes(out[:out_len])


@dataclass
class _SsrcState:
    roc: int = 0
    max_seq: int = 0
    initialized: bool = False


class SRTPDecryptor:
    """SRTP receiver: AES-256-CTR + HMAC-SHA1-80 with per-SSRC ROC tracking.

    Apple's HP stream uses 4 SSRCs per quality tier whose seq spaces are
    independent, so ROC must be maintained per-SSRC; sharing a counter
    across SSRCs caused tile freezes after the first 65 k packets.
    """

    def __init__(self, master_key: bytes, master_salt: bytes) -> None:
        self._cipher_key = _srtp_kdf(master_key, master_salt, 0, 32)
        self._auth_key = _srtp_kdf(master_key, master_salt, 1, 20)
        self._salt = _srtp_kdf(master_key, master_salt, 2, 14)
        self._salt_int = int.from_bytes(self._salt + b"\x00\x00", "big")
        self._aes_key = algorithms.AES(self._cipher_key)
        self._states: dict[int, _SsrcState] = {}
        self._counts: defaultdict[int, int] = defaultdict(int)

    @classmethod
    def from_blob(cls, blob: bytes) -> "SRTPDecryptor":
        """Construct from a 46-byte AVCMediaStreamNegotiator key blob."""
        key, salt = _split_blob(blob)
        return cls(key, salt)

    @property
    def ssrc_counts(self) -> dict[int, int]:
        return dict(self._counts)

    def forget_ssrcs_except(self, keep: set[int]) -> None:
        """Drop count + cipher state for every SSRC not in `keep`.

        Used by `Session` after adopting a fresh SSRC group: any losing
        group still on the wire would otherwise keep accumulating counts
        and trigger another adoption the moment the cool-down expires.
        Resetting forces a clean restart for any later-arriving stream.
        """
        for s in list(self._counts.keys()):
            if s not in keep:
                self._counts.pop(s, None)
                self._states.pop(s, None)

    def get_primary_ssrc_group(self, tier: int = 0) -> set[int]:
        """Return one of the 4-consecutive-SSRC groups Apple emits per tier.
        Tier 0 = the highest-quality (most-packets) group."""
        if not self._counts:
            return set()
        sorted_ssrcs = sorted(self._counts.keys())
        groups: list[list[int]] = [[sorted_ssrcs[0]]]
        for s in sorted_ssrcs[1:]:
            if s - groups[-1][-1] <= 1 and len(groups[-1]) < 4:
                groups[-1].append(s)
            else:
                groups.append([s])
        groups.sort(key=lambda g: -sum(self._counts[s] for s in g))
        idx = min(tier, len(groups) - 1)
        return set(groups[idx])

    def decrypt(self, pkt: bytes) -> Optional[tuple[bytes, bytes]]:
        """Decrypt one SRTP packet. Returns (rtp_header, payload) or None
        if the auth tag fails for every candidate ROC."""
        if len(pkt) < _RTP_HEADER_MIN + _AUTH_TAG_LEN:
            return None

        body_len = len(pkt) - _AUTH_TAG_LEN
        seq = (pkt[2] << 8) | pkt[3]
        ssrc = int.from_bytes(pkt[8:12], "big")

        state = self._states.get(ssrc)
        if state is None or not state.initialized:
            roc_guess = 0
        else:
            diff = seq - state.max_seq
            if diff > 0x7FFF:
                roc_guess = max(0, state.roc - 1)
            elif diff < -0x7FFF:
                roc_guess = state.roc + 1
            else:
                roc_guess = state.roc

        # Try guess, current, guess+1, guess-1. Dedupe while preserving order.
        candidates = []
        seen: set[int] = set()
        for r in (roc_guess, state.roc if state else 0, roc_guess + 1, max(0, roc_guess - 1)):
            if r not in seen:
                seen.add(r)
                candidates.append(r)

        for roc in candidates:
            res = self._try_decrypt(pkt, body_len, seq, ssrc, roc)
            if res is not None:
                self._update_state(ssrc, roc, seq)
                self._counts[ssrc] += 1
                return res
        return None

    def _try_decrypt(
        self, pkt: bytes, body_len: int, seq: int, ssrc: int, roc: int,
    ) -> Optional[tuple[bytes, bytes]]:
        roc_be = roc.to_bytes(4, "big")
        h = hmac.new(self._auth_key, digestmod="sha1")
        h.update(memoryview(pkt)[:body_len])
        h.update(roc_be)
        if not hmac.compare_digest(h.digest()[:_AUTH_TAG_LEN], pkt[body_len:body_len + _AUTH_TAG_LEN]):
            return None

        first_byte = pkt[0]
        cc = first_byte & 0x0F
        hdr_len = _RTP_HEADER_MIN + cc * 4
        if (first_byte >> 4) & 1:  # extension bit
            if hdr_len + 4 > body_len:
                return None
            ext_len = (pkt[hdr_len + 2] << 8) | pkt[hdr_len + 3]
            hdr_len += 4 + ext_len * 4
        if hdr_len > body_len:
            return None

        header = bytes(pkt[:hdr_len])
        if hdr_len == body_len:
            return header, b""

        index = (roc << 16) | seq
        iv_int = self._salt_int ^ (ssrc << 64) ^ (index << 16)
        iv = iv_int.to_bytes(16, "big")
        dec = Cipher(self._aes_key, modes.CTR(iv)).decryptor()
        plaintext = dec.update(pkt[hdr_len:body_len]) + dec.finalize()
        return header, plaintext

    def _update_state(self, ssrc: int, roc: int, seq: int) -> None:
        state = self._states.setdefault(ssrc, _SsrcState())
        if not state.initialized:
            state.roc = roc
            state.max_seq = seq
            state.initialized = True
            return
        new_full = (roc << 16) | seq
        cur_full = (state.roc << 16) | state.max_seq
        if new_full > cur_full:
            state.roc = roc
            state.max_seq = seq

    def state_snapshot(self) -> dict[int, dict[str, int]]:
        """Per-SSRC stats for receiver reports."""
        return {
            ssrc: {"roc": s.roc, "max_seq": s.max_seq}
            for ssrc, s in self._states.items()
            if s.initialized
        }


class SRTPEncryptor:
    """SRTP sender. Used only for the PT=101 keepalive Apple expects from us."""

    def __init__(self, master_key: bytes, master_salt: bytes, ssrc: int) -> None:
        self._cipher_key = _srtp_kdf(master_key, master_salt, 0, 32)
        self._auth_key = _srtp_kdf(master_key, master_salt, 1, 20)
        self._salt = _srtp_kdf(master_key, master_salt, 2, 14)
        self._aes_key = algorithms.AES(self._cipher_key)
        self.ssrc = ssrc
        self._seq = 0
        self._roc = 0
        self._ts = 0
        self._lock = threading.Lock()

    @classmethod
    def from_blob(cls, blob: bytes, ssrc: int) -> "SRTPEncryptor":
        """Construct from a 46-byte AVCMediaStreamNegotiator key blob."""
        key, salt = _split_blob(blob)
        return cls(key, salt, ssrc)

    def encrypt(self, payload: bytes, *, pt: int = 101, marker: bool = False) -> bytes:
        with self._lock:
            seq = self._seq & 0xFFFF
            roc = self._roc
            ts = self._ts
            self._seq += 1
            if self._seq > 0xFFFF:
                self._seq = 0
                self._roc += 1
            self._ts += 480

        header = struct.pack(">BBHII", 0x80, (pt & 0x7F) | (0x80 if marker else 0), seq, ts, self.ssrc)
        index = (roc << 16) | seq
        iv_int = (
            int.from_bytes(self._salt + b"\x00\x00", "big")
            ^ (self.ssrc << 64)
            ^ (index << 16)
        )
        iv = iv_int.to_bytes(16, "big")
        ciphertext = Cipher(self._aes_key, modes.CTR(iv)).encryptor().update(payload)
        body = header + ciphertext
        tag = hmac.new(self._auth_key, body + roc.to_bytes(4, "big"), "sha1").digest()[:_AUTH_TAG_LEN]
        return body + tag


class SRTCPDecryptor:
    """SRTCP receiver. RTCP shares the same master key but uses labels 3/4/5."""

    def __init__(self, master_key: bytes, master_salt: bytes) -> None:
        self._cipher_key = _srtp_kdf(master_key, master_salt, 3, 32)
        self._auth_key = _srtp_kdf(master_key, master_salt, 4, 20)
        self._salt = _srtp_kdf(master_key, master_salt, 5, 14)
        self._aes_key = algorithms.AES(self._cipher_key)

    @classmethod
    def from_blob(cls, blob: bytes) -> "SRTCPDecryptor":
        key, salt = _split_blob(blob)
        return cls(key, salt)

    def unprotect(self, srtcp_pkt: bytes) -> Optional[bytes]:
        if len(srtcp_pkt) < 8 + 4 + _AUTH_TAG_LEN:
            return None
        body = srtcp_pkt[:-_AUTH_TAG_LEN]
        auth_tag = srtcp_pkt[-_AUTH_TAG_LEN:]
        e_index = struct.unpack(">I", srtcp_pkt[-_AUTH_TAG_LEN - 4:-_AUTH_TAG_LEN])[0]

        expected = hmac.new(self._auth_key, body, "sha1").digest()[:_AUTH_TAG_LEN]
        if not hmac.compare_digest(expected, auth_tag):
            return None

        encrypted = bool(e_index & 0x80000000)
        index = e_index & 0x7FFFFFFF
        hdr = srtcp_pkt[:8]
        ciphertext = srtcp_pkt[8:-_AUTH_TAG_LEN - 4]
        if not encrypted:
            return hdr + ciphertext

        ssrc = struct.unpack(">I", hdr[4:8])[0]
        iv = self._build_iv(ssrc, index)
        plaintext = Cipher(self._aes_key, modes.CTR(iv)).decryptor().update(ciphertext)
        return hdr + plaintext

    def _build_iv(self, ssrc: int, index: int) -> bytes:
        iv = bytearray(self._salt + b"\x00\x00")
        iv[4] ^= (ssrc >> 24) & 0xFF
        iv[5] ^= (ssrc >> 16) & 0xFF
        iv[6] ^= (ssrc >> 8) & 0xFF
        iv[7] ^= ssrc & 0xFF
        idx_be = index.to_bytes(4, "big")
        iv[10] ^= idx_be[0]
        iv[11] ^= idx_be[1]
        iv[12] ^= idx_be[2]
        iv[13] ^= idx_be[3]
        return bytes(iv)


class SRTCPEncryptor:
    """SRTCP sender. Wraps RR / FIR / PLI / NACK packets."""

    def __init__(self, master_key: bytes, master_salt: bytes) -> None:
        self._cipher_key = _srtp_kdf(master_key, master_salt, 3, 32)
        self._auth_key = _srtp_kdf(master_key, master_salt, 4, 20)
        self._salt = _srtp_kdf(master_key, master_salt, 5, 14)
        self._aes_key = algorithms.AES(self._cipher_key)
        self._index = 0
        self._lock = threading.Lock()

    @classmethod
    def from_blob(cls, blob: bytes) -> "SRTCPEncryptor":
        key, salt = _split_blob(blob)
        return cls(key, salt)

    def protect(self, rtcp_pkt: bytes) -> bytes:
        with self._lock:
            index = self._index
            self._index += 1

        hdr = rtcp_pkt[:8]
        plaintext = rtcp_pkt[8:]
        ssrc = struct.unpack(">I", hdr[4:8])[0]

        iv = bytearray(self._salt + b"\x00\x00")
        iv[4] ^= (ssrc >> 24) & 0xFF
        iv[5] ^= (ssrc >> 16) & 0xFF
        iv[6] ^= (ssrc >> 8) & 0xFF
        iv[7] ^= ssrc & 0xFF
        idx_be = index.to_bytes(4, "big")
        iv[10] ^= idx_be[0]
        iv[11] ^= idx_be[1]
        iv[12] ^= idx_be[2]
        iv[13] ^= idx_be[3]

        ciphertext = Cipher(self._aes_key, modes.CTR(bytes(iv))).encryptor().update(plaintext)
        e_index = struct.pack(">I", 0x80000000 | index)
        body = hdr + ciphertext + e_index
        tag = hmac.new(self._auth_key, body, "sha1").digest()[:_AUTH_TAG_LEN]
        return body + tag


__all__ = [
    "SRTCPDecryptor",
    "SRTCPEncryptor",
    "SRTPDecryptor",
    "SRTPEncryptor",
    "SRTP_KEY_BLOB_LEN",
    "SRTP_MASTER_KEY_LEN",
    "SRTP_MASTER_SALT_LEN",
]
