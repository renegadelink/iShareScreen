"""RFB authentication: Apple's SRP-RFC5054-4096-SHA512-PBKDF2 and the legacy
non-SRP RSA1+AES path.

Both flows share the post-banner socket, drive screensharingd through its
state machine, and return a 16-byte symmetric key suitable for the enc1103
stream cipher that follows.

The SRP variant is the modern path used by macOS 15+. Verifier lookup is
keyed on the username embedded in the RSA-encrypted modulus slot. The
non-SRP path (auth type 0x21, the asyncvnc-style RSA1 challenge) works
against every shipping macOS we've tested and is used as fallback.

Internals are decomposed into named phases per protocol step. Both
`do_srp_auth` and `do_nonsrp_auth` are thin orchestrators.
"""
from __future__ import annotations

import hashlib
import logging
import os
import socket
import struct
from dataclasses import dataclass

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.serialization import load_der_public_key

from Crypto.Cipher import AES

from .rfb import recv_exact


log = logging.getLogger(__name__)


# Apple's SRP profile: RFC 5054 4096-bit MODP group, SHA-512 hash, PBKDF2
# password expansion. Constants are spec-defined; documented here so the
# crypto isn't spread across function bodies.
_SRP_HASH_LEN = 64                  # SHA-512 output
_SRP_MODULUS_LEN = 512              # 4096 bits / 8
_SRP_PBKDF2_DK_LEN = 128            # bytes, per Apple's SASL srp.so
_NONSRP_AES_KEY_LEN = 16
_RSA_MODULUS_PAYLOAD_LEN = 256      # encrypted block size for 2048-bit RSA
_NONSRP_CRED_FIELD_LEN = 64         # ARD-style 64-byte slot per credential


class AuthError(Exception):
    """Authentication did not complete — server rejection or wire-format issue."""


# ── shared phase: RSA1 init ───────────────────────────────────────────

def _rsa1_init(sock: socket.socket) -> RSAPublicKey:
    """Send the 15-byte 'auth-type 0x21 + empty RSA1' init and read back the
    server's RSA public key. Used as the first phase by BOTH auth paths;
    server distinguishes SRP vs. non-SRP by the c2s1 that follows."""
    sock.sendall(b"\x21" + b"\x00\x00\x00\x0a\x01\x00RSA1\x00\x00\x00\x00")
    pkt_len = struct.unpack(">I", recv_exact(sock, 4))[0]
    pkt = recv_exact(sock, pkt_len)
    # 2-byte direction prefix, then u32 BE keyLength, then DER-encoded pubkey.
    key_len = struct.unpack(">I", pkt[2:6])[0]
    server_pub = load_der_public_key(pkt[6:6 + key_len])
    log.info("RSA1 init: server pubkey %d bits", server_pub.key_size)
    return server_pub


# ── non-SRP path ──────────────────────────────────────────────────────

def _pack_ard_credential(value: str | bytes) -> bytes:
    """ARD credential slot: NUL-terminated, random-padded to 64 bytes.

    The padding randomises the unused tail so the cipher block doesn't leak
    credential length. (Apple's ARD client fills with random bytes; macOS
    server doesn't care about the padding bytes themselves.)
    """
    data = value.encode() if isinstance(value, str) else value
    data += b"\x00"
    if len(data) < _NONSRP_CRED_FIELD_LEN:
        data += os.urandom(_NONSRP_CRED_FIELD_LEN - len(data))
    return data[:_NONSRP_CRED_FIELD_LEN]


def _send_nonsrp_credentials(
    sock: socket.socket, server_pub: RSAPublicKey, aes_key: bytes,
    username: str, password: str,
) -> None:
    """Build c2s1 (encrypted credentials + RSA-encrypted AES key) and send it."""
    creds = _pack_ard_credential(username) + _pack_ard_credential(password)
    enc_creds = AES.new(aes_key, AES.MODE_ECB).encrypt(creds)
    enc_aes_key = server_pub.encrypt(aes_key, rsa_padding.PKCS1v15())

    auth_blob = (
        b"\x01\x00RSA1"
        + b"\x00\x01" + enc_creds
        + b"\x00\x01" + enc_aes_key
    )
    sock.sendall(struct.pack(">I", len(auth_blob)) + auth_blob)


def _read_nonsrp_result(sock: socket.socket) -> None:
    """Read non-SRP auth result. Format: 4 padding bytes, then u32 BE result
    code (0 = success). Raises AuthError on non-zero result."""
    recv_exact(sock, 4)  # padding (M2 placeholder slot)
    result = struct.unpack(">I", recv_exact(sock, 4))[0]
    if result != 0:
        raise AuthError(f"non-SRP auth rejected: result={result}")


def do_nonsrp_auth(sock: socket.socket, username: str, password: str) -> bytes:
    """Asyncvnc-style RSA1+AES auth (RFB auth type 0x21). Returns the AES key,
    which doubles as the enc1103 master key."""
    server_pub = _rsa1_init(sock)
    aes_key = os.urandom(_NONSRP_AES_KEY_LEN)
    _send_nonsrp_credentials(sock, server_pub, aes_key, username, password)
    _read_nonsrp_result(sock)
    log.info("AUTH OK (non-SRP)")
    return aes_key


# ── SRP path ──────────────────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class _SrpChallenge:
    """Parsed s2c1 challenge body. All ints stored alongside their PAD-aligned
    byte form because both representations get used downstream."""
    N: int          # safe-prime modulus
    Nb: bytes       # N as 512-byte big-endian
    g: int
    salt: bytes
    B: int          # server ephemeral
    Bb: bytes
    iterations: int
    cap: bytes      # capability string (e.g. b"mda=SHA-512,replay_detection,...")


@dataclass(slots=True, frozen=True)
class _SrpProof:
    """Client-side outputs of the SRP exchange."""
    A: int
    Ab: bytes
    M1: bytes       # client proof of password knowledge
    K: bytes        # session key (SHA-512(S))


def _send_srp_modulus(
    sock: socket.socket, server_pub: RSAPublicKey, username: str,
) -> None:
    """Send c2s1 with a username-bearing payload RSA-encrypted into the
    'modulus' slot. Apple's SRP server requires this RSA decrypt to succeed
    before it routes to the SRP code path; a non-encrypted modulus → non-SRP
    fallback at the server side."""
    user_b = username.encode("utf-8")
    inner = struct.pack(">I", len(user_b)) + user_b + b"\x00\x00\x00"
    payload = struct.pack(">I", len(inner)) + inner
    encrypted = server_pub.encrypt(payload, rsa_padding.PKCS1v15())
    if len(encrypted) != _RSA_MODULUS_PAYLOAD_LEN:
        raise AuthError(
            f"expected {_RSA_MODULUS_PAYLOAD_LEN}B RSA block, got {len(encrypted)}"
        )

    c2s1 = (
        struct.pack("<H", 1) + b"RSA1\x00\x02"
        + struct.pack(">H", _RSA_MODULUS_PAYLOAD_LEN) + encrypted
        + b"\x00" * 384
    )
    sock.sendall(struct.pack(">I", 650) + c2s1)


def _read_srp_challenge(sock: socket.socket) -> _SrpChallenge:
    """Read s2c1, parse it. Raises AuthError if the response indicates the
    server fell back to non-SRP (i.e. our RSA decrypt didn't succeed on its
    end), or if the modulus isn't the expected 4096-bit shape."""
    s2c1_len = struct.unpack(">I", recv_exact(sock, 4))[0]
    s2c1 = recv_exact(sock, s2c1_len)
    if s2c1_len < 1000:
        raise AuthError(
            f"SRP challenge too short ({s2c1_len}B); server fell back to non-SRP path"
        )
    return _parse_apple_srp_challenge(s2c1)


def _parse_apple_srp_challenge(s2c1: bytes) -> _SrpChallenge:
    """Apple's SRP-RFC5054-4096-SHA512-PBKDF2 challenge layout. Reverse-
    engineered live from a Screen Sharing.app handshake; 1165 bytes total, fixed shape.

        offset  field
        0..11   TLV header (12 bytes — values are static; we skip past)
        12      0x00 (DER positive-int marker)
        13..524 N modulus (512 bytes)
        525..6  g_length u16 BE (= 1)
        527     g (= 5 for the 4096-bit group)
        528     salt_length (= 32)
        529..   salt
        ...     u16 BE B_length, B
                u64 BE iterations
                u16 BE cap_length, cap
    """
    p = 12
    if s2c1[p] != 0:
        raise AuthError(f"SRP parse: missing DER zero marker at offset 12, got {s2c1[p]:#x}")
    p += 1
    Nb = s2c1[p:p + _SRP_MODULUS_LEN]
    p += _SRP_MODULUS_LEN
    g_len = struct.unpack(">H", s2c1[p:p + 2])[0]
    p += 2
    g = int.from_bytes(s2c1[p:p + g_len], "big")
    p += g_len
    salt_len = s2c1[p]
    p += 1
    salt = s2c1[p:p + salt_len]
    p += salt_len
    B_len = struct.unpack(">H", s2c1[p:p + 2])[0]
    p += 2
    Bb = s2c1[p:p + B_len]
    p += B_len
    iterations = struct.unpack(">Q", s2c1[p:p + 8])[0]
    p += 8
    cap_len = struct.unpack(">H", s2c1[p:p + 2])[0]
    p += 2
    cap = s2c1[p:p + cap_len]

    if len(Nb) != _SRP_MODULUS_LEN:
        raise AuthError(f"expected 4096-bit SRP modulus, got {len(Nb) * 8}-bit")
    # Apple's daemon sends ~10000-150000 iterations in practice. Cap
    # at 1M to bound PBKDF2 work and prevent a malicious server from
    # hanging the client on the password-stretching path.
    if iterations > 1_000_000:
        raise AuthError(f"SRP iteration count {iterations} exceeds 1M cap")

    log.info(
        "SRP challenge: N=%db g=%d salt=%dB iters=%d cap=%r",
        len(Nb) * 8, g, len(salt), iterations,
        cap.decode("latin-1", errors="replace"),
    )
    return _SrpChallenge(
        N=int.from_bytes(Nb, "big"),
        Nb=Nb,
        g=g,
        salt=salt,
        B=int.from_bytes(Bb, "big"),
        Bb=Bb,
        iterations=iterations,
        cap=cap,
    )


def _derive_x(salt: bytes, iterations: int, password: str) -> int:
    """Apple's SALTED-SHA512-PBKDF2 KDF.

    Reverse-engineered from /usr/lib/sasl2/srp.so + corecrypto's
    `ccsrp_generate_x`. The inner SHA-512 takes an empty username — the
    actual identity travels in the c2s1 modulus payload, not in `x`.

        dk    = PBKDF2-HMAC-SHA512(password, salt, iterations, 128)
        inner = SHA512("" || ":" || dk)
        x     = SHA512(salt || inner) mod N
    """
    pw_b = password.encode("utf-8")
    dk = PBKDF2HMAC(
        hashes.SHA512(), _SRP_PBKDF2_DK_LEN, salt, iterations, default_backend(),
    ).derive(pw_b)
    inner = hashlib.sha512(b":" + dk).digest()
    return int.from_bytes(hashlib.sha512(salt + inner).digest(), "big")


def _solve_srp(challenge: _SrpChallenge, password: str) -> _SrpProof:
    """Run the SRP-6a computation. Produces (A, M1, K) — the client public,
    the proof of password knowledge, and the session key."""
    KL = _SRP_MODULUS_LEN
    g_padded = challenge.g.to_bytes(KL, "big")

    # k = H(N || PAD(g))
    k = int.from_bytes(hashlib.sha512(challenge.Nb + g_padded).digest(), "big")

    # Random ephemeral a, A = g^a mod N. 64 bytes of entropy is generous —
    # the SRP-6a security argument needs only ~256 bits.
    a = int.from_bytes(os.urandom(64), "big") % (challenge.N - 1) + 1
    A = pow(challenge.g, a, challenge.N)
    Ab = A.to_bytes(KL, "big")

    # u = H(A || B), x = derived from salt + iterations + password
    u = int.from_bytes(hashlib.sha512(Ab + challenge.Bb).digest(), "big")
    x = _derive_x(challenge.salt, challenge.iterations, password) % challenge.N

    # Premaster S = (B - k*g^x)^(a + u*x) mod N
    S = pow(
        (challenge.B - k * pow(challenge.g, x, challenge.N)) % challenge.N,
        a + u * x, challenge.N,
    )
    K = hashlib.sha512(S.to_bytes(KL, "big")).digest()

    # M1 = H(H(N) XOR H(g) || H("") || salt || A || B || K)
    h_n = hashlib.sha512(challenge.Nb).digest()
    h_g = hashlib.sha512(g_padded).digest()
    M1 = hashlib.sha512(
        bytes(p ^ q for p, q in zip(h_n, h_g))
        + hashlib.sha512(b"").digest()      # H(I) with empty I per Apple's SASL flow
        + challenge.salt + Ab + challenge.Bb + K
    ).digest()

    return _SrpProof(A=A, Ab=Ab, M1=M1, K=K)


def _send_srp_proof(
    sock: socket.socket, challenge: _SrpChallenge, proof: _SrpProof,
) -> None:
    """Build c2s2 (A + M1 + cap echo + civ) and send. Wire shape mirrors c2s1's
    RSA1-style envelope so the server's parser stays in the same code path."""
    civ = os.urandom(16)
    sd = (
        struct.pack(">H", _SRP_MODULUS_LEN) + proof.Ab
        + bytes([_SRP_HASH_LEN]) + proof.M1
        + struct.pack(">H", len(challenge.cap)) + challenge.cap
        + bytes([16]) + civ
    )
    pay = (
        struct.pack("<H", 1) + b"RSA1\x00\x02"
        + struct.pack(">H", len(sd) + 4)
        + struct.pack(">I", len(sd)) + sd
    )
    pay += b"\x00" * (1076 - len(pay))
    sock.sendall(struct.pack(">I", 1076) + pay)


def _read_srp_result(sock: socket.socket) -> None:
    """Read M2 (server's proof of session-key knowledge) and the auth result.
    We don't currently verify M2 — server's auth_result is canonical for
    pass/fail. M2 verification would catch a server impersonation at the
    SRP layer; worth adding later, not security-critical for pass/fail."""
    m2_len = struct.unpack(">I", recv_exact(sock, 4))[0]
    recv_exact(sock, m2_len)  # M2; not verified
    result = struct.unpack(">I", recv_exact(sock, 4))[0]
    if result != 0:
        raise AuthError(f"SRP auth rejected: result={result}")


def do_srp_auth(sock: socket.socket, username: str, password: str) -> bytes:
    """SRP-RFC5054-4096-SHA512-PBKDF2 against Apple screensharingd.

    Returns a 16-byte enc1103 master key derived from the SRP session key K.
    Even though Apple's cap string advertises ChaCha20-Poly1305 confidentiality
    + integrity, the post-SRP wire stays plaintext until enc1103 sets up
    AES-CBC, so deriving a 16-byte AES key is correct (verified empirically).
    """
    server_pub = _rsa1_init(sock)
    _send_srp_modulus(sock, server_pub, username)
    challenge = _read_srp_challenge(sock)
    proof = _solve_srp(challenge, password)
    _send_srp_proof(sock, challenge, proof)
    _read_srp_result(sock)
    log.info("AUTH OK (SRP)")
    return hashlib.sha256(proof.K).digest()[:16]


__all__ = [
    "AuthError",
    "do_nonsrp_auth",
    "do_srp_auth",
]
