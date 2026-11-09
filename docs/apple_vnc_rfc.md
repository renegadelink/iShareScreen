# Apple VNC High-Performance Extension

## Status of This Memo

This document is an **experimental, reverse-engineered project specification**. It is not an IETF standard, is not published or endorsed by Apple, and carries no standards-track status. It describes the observed and inferred behavior of Apple's macOS Screen Sharing high-performance VNC extension for the purpose of building an interoperable third-party client.

The specification is derived from packet captures, runtime instrumentation, static analysis of the server components, and an interoperable client implementation. Fields and behaviors that are not yet established are marked explicitly as **revision gaps** or as **implementation-defined**.

## Abstract

This document specifies an Apple-specific extension layered on the Remote Framebuffer (RFB) protocol. The extension adds Apple authentication branches, an encrypted control record layer, a virtual-display session model, vendor-specific framebuffer encodings, and an optional compressed-media transport. High-performance mode is a property of the session, negotiated after authentication; it is not implied by the selected authentication branch.

Major features:

- multiple Apple authentication branches selected by RFB security type: `30` (Diffie-Hellman), `33` (RSA1 / RSA-SRP), `35` (Kerberos GSS-API), and `36` (direct SRP);
- a post-authentication rekey message (`EncodeEncryptionInfo`, encoding `0x44f`) that activates an AES-128-CBC control record layer with a SHA-1 integrity trailer;
- a virtual-display configuration model carried by Apple control messages;
- vendor framebuffer-update encodings for cursor, layout, keyboard, device, and media-initialization metadata, plus Apple still-image codecs;
- an optional **Adaptive media** transport: HEVC video and AAC audio over SRTP/UDP, keyed in-band through the encrypted control channel.

## 1. Introduction

The Apple VNC high-performance extension augments a standard RFB session. After the RFB version exchange and Apple authentication, the session transitions through a cleartext Apple prelude into an encrypted record layer, configures a virtual display, and then operates in a steady state that may deliver content as RFB framebuffer rectangles, Apple still-image codec rectangles, or — when Adaptive media mode is negotiated — compressed HEVC/AAC over a separate SRTP/UDP transport.

High-performance mode is a **session property**, not an **authentication property**. A conforming implementation MUST NOT infer high-performance capability, media capability, or any session class from the selected authentication branch. The same high-performance session behavior is reachable through more than one authentication branch.

This document is organized for third-party implementers: each binary message is specified with its direction, phase, framing, byte order, field table, length rule, validation rule, treatment of unknown/reserved fields, and error behavior (§2.4). Conformance is expressed as testable requirement tables grouped into implementation profiles (§11).

## 2. Conventions and Terminology

### 2.1 Requirement Language

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in BCP 14 (RFC 2119, RFC 8174) when, and only when, they appear in all capitals.

Normative requirement language in this document constrains behavior **required for interoperability with this project's implementation profile**. It is not used for facts that are merely observed unless the document deliberately elevates an observation to a project requirement; such elevation is stated.

### 2.2 Terminology

- **client** / **viewer**: the party that initiates the TCP connection and renders the remote display.
- **server**: the party that authenticates the client and delivers display content; the macOS Screen Sharing service.
- **handshake stage**: the cleartext RFB exchange from ProtocolVersion through SecurityResult (§3).
- **session bootstrap stage**: ClientInit, ServerInit, and the cleartext Apple prelude (§5).
- **record layer**: the AES-128-CBC framed transport activated by the rekey message (§6).
- **cleartext prelude**: Apple control messages sent after ServerInit and before the record layer is active (§5.4).
- **encrypted preface**: the first client control messages carried inside the record layer (§6.5).
- **virtual display**: a server-side synthetic display created for the session.
- **high-performance mode**: the Apple session mode associated with virtual-display behavior, low-latency handling, and optional media-path negotiation.
- **Adaptive media mode**: the high-performance sub-mode in which visible content is delivered as compressed HEVC/AAC over SRTP/UDP (§10).
- **grammar atom**: a notation token used in the authentication grammar (§2.5).

### 2.4 Message Specification Template

Every binary message in this document is specified with: **direction**, **phase**, **transport/framing**, **byte order**, **field table**, **length rule**, **validation rule**, **receiver behavior on unknown/reserved fields**, and **error behavior**. Unless stated otherwise, all multi-byte integers are **big-endian** (network byte order); IEEE-754 floats are transmitted big-endian.

Reserved and unknown fields follow one of three sender/receiver rules, stated per field: **emit-zero** (sender MUST emit zero), **preserve** (sender MUST echo the most recently observed value), or **ignore-on-receive** (receiver MUST ignore the field's value).

### 2.5 Grammar Atoms

The authentication grammar uses the following atoms. The first three are length-prefixed variable-length atoms; the remainder are fixed-width integers.

- `%m`: `u16 length || byte[length] value`
- `%o`: `u8 length || byte[length] value`
- `%s`: `u16 length || byte[length] utf8_string`
- `%q`: `u64 value`
- `%u`: `u32 value`
- `%c`: `u8 value`

These atoms describe the Apple SRP branches (auth types `33` and `36`).

## 3. Protocol Overview and State Machine

### 3.1 Stage Model

A high-performance session proceeds through the following stages. The macro order is fixed; one intra-stage interleave is noted in §3.3.

1. RFB version negotiation (§4.1.3)
2. security-type negotiation (§4.1.4–§4.1.5)
3. selected authentication branch (§4.2)
4. SecurityResult (§4.2; per-branch)
5. ClientInit / ServerInit (§5.2–§5.3)
6. cleartext Apple prelude (§5.4)
7. rekey by `EncodeEncryptionInfo` (§6.1)
8. encrypted preface (§6.5)
9. server metadata burst (§8.2–§8.7)
10. steady-state framebuffer / control / media behavior (§8, §10)

### 3.2 State Machine

The following table is normative for client implementations. "Entering event" is the trigger that places the client in the state; "transition" gives the condition that advances to the next state; "failure" gives the required behavior on violation.

| State | Entering event | Valid incoming | Valid outgoing | Transition (→ next) | Failure behavior |
|---|---|---|---|---|---|
| S0 Connect | TCP established | server ProtocolVersion (12 B) | — | server banner received → S1 | close on timeout/short read |
| S1 Version | server banner read | — | client ProtocolVersion (12 B) | client banner sent → S2 | MUST close if banner ≠ `RFB 003.889` and client cannot speak it |
| S2 SecTypes | client banner sent | `u8 count` + `u8[count]` types | — | type list read → S3 | MUST close on `count = 0` (failure list) |
| S3 SecSelect | type list read | (branch-specific) | `u8 selected_type` (+ branch entry) | selector sent → S4 | MUST close if no advertised type is supported |
| S4 Auth | selector sent | branch challenge/result frames | branch response frames | branch completes → S5 | MUST close on branch error or non-zero SecurityResult |
| S5 SecResult | branch auth complete | `u32 SecurityResult` (per-branch placement) | — | `SecurityResult = 0` → S6 | MUST close on non-zero result |
| S6 Init | SecurityResult = 0 | ServerInit | ClientInit (`u8` shared-flag) | ServerInit read → S7 | close on malformed ServerInit |
| S7 Prelude | ServerInit read | — | ViewerInfo, SetEncryption(1), SetMode, SetEncryption(2) | rekey received → S8 | close on rekey-decrypt failure |
| S8 Record active | `0x44f` rekey processed | encrypted records | encrypted preface, then control | preface sent → S9 | MUST close on integrity-trailer failure (§6.4.5) |
| S9 Steady | encrypted preface sent | framebuffer/codec/metadata records; media (if negotiated) | control records; RTCP (if media) | session teardown | MUST ignore unknown encodings/messages (§12); MUST close on record-layer integrity failure |

The server MAY emit the rekey (`0x44f`) and begin its metadata burst at points the client treats as part of S7→S8→S9; the client MUST be prepared to receive the rekey as soon as it has sent `SetEncryption(command=1)` (§3.3).

### 3.3 Ordering Note (prelude/rekey interleave)

On the wire the server emits the `0x44f` rekey immediately after receiving the client's `SetEncryption(command=1)`, i.e. **before** the client's `SetEncryption(command=2)` (the fourth cleartext-prelude item, §5.4). The macro order (cleartext prelude region → rekey → encrypted records) holds, but a client MUST NOT assume the entire prelude is sent before the rekey arrives. A conforming client processes the rekey as soon as it is received and continues any unsent prelude items inside the record layer if required.

### 3.4 Roles and Session Classes

The client requests and renders; the server authenticates and delivers updates. Recognized session classes:

- framebuffer-backed virtual-display session;
- high-performance session with media-init signaling that remains on the framebuffer path;
- Adaptive media session (compressed HEVC/AAC over SRTP/UDP, §10).

Media-init signaling (advertising encoding `0x3f2`, §8.8) MUST NOT be treated as proof that visible content has switched to compressed media; the content switch is gated on the `0x1c` MediaStreamOptions exchange (§10.6).

## 4. Authentication

### 4.1 Handshake

#### 4.1.1 Overview

The handshake is the standard RFB cleartext exchange, extended only by Apple's security-type set and per-branch entry payloads.

#### 4.1.2 ProtocolVersion

- **Direction**: server→client, then client→server. **Phase**: handshake. **Framing**: raw 12-byte ASCII. **Byte order**: ASCII.
- Both banners are the 12-byte string `RFB 003.889\n` (`52 46 42 20 30 30 33 2e 38 38 39 0a`).
- **Validation**: a client that cannot speak `003.889` MUST close.

#### 4.1.3 Security-Type Advertisement

- **Direction**: server→client. **Framing**: `u8 count || u8[count] types`.
- Observed set: `count = 4`, types `30, 33, 36, 35` in that wire order (`04 1e 21 24 23`) — the packet capture (the 5900 stream) carries the server bytes `04 1e 21 24 23` byte-exact, preceded by both banners = `52 46 42 20 30 30 33 2e 38 38 39 0a`.
- **Validation**: `count = 0` is an RFB failure list; the client MUST close.
- **Unknown values**: a client MUST ignore advertised types it does not implement.

#### 4.1.4 Security-Type Selection

- **Direction**: client→server. **Framing**: `u8 selected_type`, optionally coalesced with the branch-entry payload in the same TCP segment.
- Selector-to-type map: `0x1e`→30, `0x21`→33, `0x23`→35, `0x24`→36.
- For type 33 the selector byte is immediately followed by the RSA1 envelope (§4.2.4).

### 4.2 Authentication Branches

#### 4.2.1 Flow Registry

| Type | Selector | Name | Section |
|---|---|---|---|
| `30` | `0x1e` | Diffie-Hellman | §4.2.3 |
| `33` | `0x21` | RSA1 / RSA-SRP | §4.2.4 |
| `35` | `0x23` | Kerberos GSS-API | §4.2.5 |
| `36` | `0x24` | Direct SRP | §4.2.6 |

Selecting a branch commits both endpoints to that branch's transcript, terminating in an RFB `SecurityResult` (§4.2.x). Types 33 and 36 share SRP-6a primitives; type 33 additionally wraps an identity envelope in RSA (RSA1).

#### 4.2.2 Common Branch Skeleton

All branches share: ProtocolVersion ×2 → security-type advertisement → security-type selection → branch transcript → `SecurityResult` (`u32 = 0` on success) → ClientInit → ServerInit. Each branch below specifies selector behavior, wire framing, transcript order, key derivation, SecurityResult behavior, error behavior, and the resulting initial wrap key (§6.2.2).

#### 4.2.3 Type 30: Diffie-Hellman

The challenge framing, group, credential cipher, and wrap key are established by handler `screensharingd::sub_100015ec0` (`HandleDHAskUserMessage`), challenge sender `sub_100017830`, params `sub_100066a80`, and key derivation `sub_100068030`. Not present in available captures.

- **Selector**: `0x1e`.
- **Server challenge** (server→client): `u16 generator || u16 keylen || byte[keylen] modulus_N || byte[keylen] server_public_B`, big-endian. The server uses a **fixed 1024-bit group**: `generator = 2`, `keylen = 0x80` (128 bytes), and a built-in 128-byte modulus (`sub_100066a80` returns `generator = 2`, `keylen = 0x80`, modulus `data_10007db30`).
- **Client response** (client→server): a 128-byte **AES-128-CBC (zero IV)** ciphertext over a fixed plaintext `byte[64] username || byte[64] password` (each NUL-terminated UTF-8, remaining bytes random padding), followed by the client public value `A` (`keylen` bytes). The AES-128 key is `k = MD5(shared)`, `shared = server_public_B ^ a mod modulus_N`. `HandleDHAskUserMessage` decrypts the 128-byte blob with `CCCryptorCreate(AES, options = 0 = CBC, key, IV = 0)`; ECB is used only by the post-auth record layer (§6.2.1), not the credential decrypt.
- **Key derivation / initial wrap key**: `MD5(shared)` (first 16 bytes), consumed by the record layer to decrypt the first rekey body (§6.2.2). `sub_100068030` computes `CC_MD5(shared)` and installs the 16-byte digest.
- **SecurityResult**: `u32 = 0` on success; then ClientInit.
- **Error behavior**: MUST close on non-zero SecurityResult.
- **Security note**: see §13.1 (MD5 + AES-ECB legacy weakness).

#### 4.2.4 Type 33: RSA1 / RSA-SRP

The type-33 transcript below is verified byte-exact against a captured session, except where noted.

##### 4.2.4.1 Transcript Order

1. client sends selector `0x21` and the RSA1 packet-1 (authtype 2) identity envelope;
2. server sends the SRP challenge (containing salt, `g`, `N`, `B`, iteration count, options);
3. client sends RSA1 packet-2 (`A`, `M1`, options echo, client random);
4. server **computes** `u`, `S`, `K`, expected `M1`, and `M2` after receiving `A`/`M1`, verifies `M1`, and sends the final proof (`M2`, server random);
5. server sends `SecurityResult` (`u32 = 0`);
6. client sends ClientInit.

The server generates its secret `b` and public `B` **before** sending the challenge; `B` is present in the challenge of step 2. The server's SRP computation that depends on `A` occurs in step 4.

> A minimal `authtype = 0` RSA public-key request followed by a DER `SubjectPublicKeyInfo` response MAY precede step 1; the reference client always performs it (the captured native session skipped it, encrypting under a cached key). The server's response framing is `u32_be (n + 7) || u32_le 0x100 || u16_be n || DER[n]` where `n` = DER length — handler `screensharingd::sub_10001872c` (`SendRSAResponseKeyRequest`); the `calloc(n + 0xb)` buffer also sends one trailing zero byte.

##### 4.2.4.2 RSA1 Envelope

- **Direction**: client→server. **Framing**: `u32 total_len || u16 version || "RSA1" || u16 authtype || u16 inner_len || body`. `total_len` counts the bytes after the leading `u32`. The field at the `aux` position is **not a constant**: it is a `u16` length of the meaningful body that follows. `screensharingd::sub_100018e60` (the authtype-2 SRP handler) loads this u16 (`ldrh [x20,#0x8]`, byteswapped) and passes it directly as the length argument to `dataWithBytes:length:` before RSA-decrypt; it is never compared to a constant.
- Packet-1: `version = 0x0100`, `authtype = 2`, `inner_len = 0x0100` (= 256, the RSA-ciphertext length — *not* a magic `aux` value), `total_len = 650`, `body = 640` (256-byte RSA ciphertext || 384 zero bytes). The 384-byte tail is all-zero.
- **RSA**: PKCS#1 v1.5 padding; ciphertext length 256 bytes (**RSA-2048**). The server keypair is **persisted, not per-boot** — `-[RSAKeyPair generateKeyPairWithKeySize:]` generates it once via `SecKeyGeneratePair` (`kSecAttrIsPermanent`, access group `com.apple.ARDAgent`), storing the private key in the keychain + `keysPath/PrivateKey` (chmod 0600) and the public key in `keysPath/PublicKey`; the size defaults to 2048 (`CFPreferences RSAKeySize`). The modulus/exponent reach the client via the §4.2.4.1 DER response, not the data channel.

##### 4.2.4.3 Packet-1 Identity Plaintext

The RSA plaintext encrypts the identity envelope. Field order:

```text
u32   payload_len            (= remaining length, e.g. 11)
u32   username_len
byte[username_len] username  (UTF-8)
u16   empty_string_len = 0
u8    empty_opaque_len = 0
```

> Note: the username field precedes the two empty fields.

##### 4.2.4.4 SRP Challenge

- **Direction**: server→client. **Outer framing**: `u32 total_len || u8 step || u32 x || u32 payload_len`, where `x = payload_len + 4`. **Inner grammar**: `%c control || %m N || %m g || %o salt || %m B || %q iterations || %s options`.
- Observed values: `control = 0x00`; `N` = 512 bytes = the well-known 4096-bit MODP safe prime (RFC 5054 / RFC 3526); `g = 0x05`; `salt` = 32 bytes; `B` = 512 bytes.
- `iterations`: a `u64` PBKDF2 cost stored with the account's verifier and read from the wire. It is **account-specific**, not a fixed constant.
- `options`: the ASCII string `mda=SHA-512,replay_detection,conf+int=ChaCha20-Poly1305,kdf=SALTED-SHA512-PBKDF2`. The client echoes this string verbatim in packet-2.

##### 4.2.4.5 Packet-2

- **Direction**: client→server. RSA1 envelope with `authtype = 2`. The `aux`-position `u16` is again a **length**, not a constant: it carries `inner_len + 4` (the sampled value `0x02aa` is just that length for the captured session and shifts with the options-string length). Body: `u16 preamble = 0 || u16 inner_len || inner || trailing`. **Inner grammar**: `%m A || %o M1 || %s options || %o client_random`. On the wire the bytes are `u16(inner_len+4) || u16(0) || u16(inner_len) || inner`. In the packet capture packet-2 begins `00000434 0100 52534131 0002 02aa 0000 02a6 0200 …` — `total_len=1076`, `authtype=2`, `aux=0x02aa=682`, then `preamble=0`, `inner_len=0x02a6=678`, and 682 = 678 + 4, with a 512-byte `A` following; the server's RSA1 parser `sub_100018e60` reads the field as a length (§4.2.4.2); and the reference client computes it as `inner_len+4`.
- Observed: `A` = 512 bytes (4096-bit), `M1` = 64 bytes (SHA-512), `options` = the verbatim challenge echo, `client_random` = 16 bytes.
- **Trailing bytes**: present and non-zero in the capture, but the bounded `inner` is authoritative; receivers MUST NOT require the trailing bytes.

##### 4.2.4.6 SRP-6a Parameters

Group RFC 5054 4096-bit MODP; `g = 5`; hash SHA-512; password pre-processing `P' = PBKDF2-HMAC-SHA512(password, salt, iterations, dkLen = 128)`. Standard SRP-6a: `x = H(salt || H(":" || P'))`, `k = H(PAD(N) || PAD(g))`, `u = H(PAD(A) || PAD(B))`, `S = (B − k·g^x)^(a + u·x)`, `K = H(PAD(S))`, `M1 = H(H(N) XOR H(g) || H(I) || salt || A || B || K)` with empty identity `I`, `M2 = H(PAD(A) || M1 || K)`. (Equations are derived from field sizes; not recomputable from capture without the secret exponent.)

##### 4.2.4.7 Final Proof and SecurityResult

- **Final proof** (server→client): outer framing as §4.2.4.4 (`x = payload_len + 4`); inner `%o server_proof_M2 || %o server_random || %s empty || %u zero`. Observed `M2` = 64 bytes, `server_random` = 16 bytes.
- **SecurityResult**: `u32 = 0` follows the final proof. On non-zero, the client MUST close (§12).

##### 4.2.4.8 Initial Wrap Key

The post-authentication session key is `SHA-256(K)[0:16]` (first 16 bytes). `screensharingd::sub_100018e60` calls `_CC_SHA256` over the corecrypto SRP session key and copies exactly the first 16 bytes of the 32-byte digest (`ldr q0`/`str q0`) into `SetupAESKeys` (`sub_100016fb8`) as the AES-128 key. This 16-byte value is the initial wrap key consumed by the record layer (§6.2). In the captured session the installed wrap key AES-128-ECB-decrypts the first rekey body to exactly the record-layer key/IV (§6.2.1), confirming the value's role even though `K` is not on the wire.

The record layer is AES-128 throughout. The `conf+int=ChaCha20-Poly1305` token in the options string does **not** key the record layer: ChaCha20-Poly1305 cryptors *are* instantiated, but only inside the **SASL/SRP security layer** (`LayerInit`, `sub_100012f9c`, cryptors at SASL-context `+0x178`/`+0x278`), which is separate from the viewer's AES-CBC record contexts (`+0x5e8`/`+0x5f0`). The post-auth control record layer (`EncryptOneMessage`/`DecryptNextMessage`) references only the AES-CBC contexts; `_CCHmac` and any AEAD are absent from it.

#### 4.2.5 Type 35: Kerberos GSS-API

Not present in available captures; the framing below is a revision gap pending a type-35 capture.

- **Selector / branch entry**: `0x23` followed by a 32-bit zero word; the server replies with a 32-bit word.
- **Transcript**: client length-prefixed GSS `AP-REQ` token (Kerberos V5 GSS mechanism OID `1.2.840.113554.1.2.2`, inner `APPLICATION 14`); server length-prefixed `AP-REP` (`APPLICATION 15`); server short per-message token with wrap prefix `05 04`.
- **Completion order**: a branch-local terminal 32-bit zero word, then the standard RFB `SecurityResult` (`u32 = 0`), then ClientInit. The terminal zero word does **not** replace SecurityResult.
- **Key derivation / initial wrap key**: on Kerberos success the server **generates a fresh random 16-byte AES key** (`AuthGetRandomBytes` ← `/dev/random`), installs it as the record-layer key, and **sends it to the viewer over the GSS-protected channel** (logged `CCCryptorCreate AESkeySend`); it does **not** derive the key from the GSS context. The GSS handling consumes an already-established/imported context (`KerberosServerAuthenticate`, `sub_10004b86c` → `gss_export_sec_context` + `bootstrap_look_up com.apple.RemoteDesktop.PrivilegeHelper`); there is no live `gss_accept_sec_context` in screensharingd. The bracketing zero-words / SecurityResult placement remain a revision gap.
- **Token framing note**: the bracketing 32-bit zero words are application-level framing, not GSS-API protocol elements; the server-side division of labor between the VNC server process and any system GSS service is not characterized here.
- **Error behavior**: MUST close on GSS failure or non-zero SecurityResult.

#### 4.2.6 Type 36: Direct SRP

The dispatch, SRP primitives, wrap key, and record-layer cipher come from selector `0x24` → `SendSRPChallenge` (`sub_1000172d0`); the cleartext-envelope specifics are a revision gap. Not present in available captures.

- **Selector / branch entry**: `0x24` followed by the cleartext identity envelope `u32 payload_len || u16 0 || u16 username_len || username || u16 0 || u8 0` (the same envelope used as the type-33 RSA plaintext, transmitted in cleartext rather than RSA-encrypted).
- **Challenge grammar**: `%c%m%m%o%m%q%s` (control, `N`, `g`, salt, `B`, iterations, options); group RFC 5054 4096-bit, `g = 5`, SHA-512, PBKDF2, same options string as type 33.
- **Key derivation / initial wrap key**: `SHA-256(K)[0:16]`, identical to type 33 (§4.2.4.8). `SendSRPChallenge` `sub_1000172d0` does `CC_SHA256(K)` and installs the first 16 bytes via `SetupAESKeys` (`sub_100016fb8`).
- **Record layer**: after the SRP exchange the session continues on the AES-128-CBC record layer of §6.4. Type 36 reuses the exact `ccsrp` primitives and the same `SetupAESKeys` AES-128-CBC contexts as type 33; `conf+int=ChaCha20-Poly1305` is advertised but does not key the record layer (§4.2.4.8).
- **SecurityResult / error**: `u32 = 0` then ClientInit; MUST close on non-zero.

#### 4.2.7 Branch Summary

All four branches converge on the common skeleton (§4.2.2). The initial wrap keys are summarized in §6.2.2. Only type 33 is byte-confirmed by capture in this revision; types 30/35/36 are flagged as revision gaps pending branch-specific captures.

## 5. Session Initialization

### 5.1 Post-Auth RFB Initialization

After `SecurityResult = 0`, the session performs standard RFB initialization (ClientInit, ServerInit), then the Apple cleartext prelude (§5.4), all before the record layer is active.

### 5.2 ClientInit

- **Direction**: client→server. **Phase**: session bootstrap. **Framing**: 1 byte. **Receiver**: server.
- The byte is the RFB shared-flag field. The interoperable value is `0xC1` (`0x80 | 0x40 | 0x01`). ClientInit is sent after `SecurityResult = 0` and **before** ServerInit.

### 5.3 ServerInit

- **Direction**: server→client. **Framing**: standard RFB ServerInit (`u16 width`, `u16 height`, 16-byte pixel format, `u32 name_len`, name).
- The name field MAY carry a vendor-specific value; clients MUST tolerate any name. **Receiver**: parse and use width/height/pixel-format for framebuffer setup.

### 5.4 Cleartext Apple Prelude

Native order: ViewerInfo (§5.5) → SetEncryption(command=1) (§5.6) → SetMode (§5.7) → SetEncryption(command=2) (§5.6). All four precede the first encrypted record; the server's rekey MAY arrive mid-prelude (§3.3). SetMode is **OPTIONAL** (§5.7): the reference client omits it and the server still proceeds, so a conforming client MAY emit just ViewerInfo → SetEncryption(1) → SetEncryption(2).

### 5.5 ViewerInfo

- **Direction**: client→server. **Framing**: `u8 type = 0x21 || u8 reserved || u16 body_len || body`. `body_len` counts only the bytes after the 4-byte prefix.
- Body: `u16 version = 1 || u16 viewer_app || version strings || byte[32] capability_bitmap` (bitmap MSB-first). Observed set bits include `{0, 2, 3, 20, 30, 31, 32, 35, 81}`.
- **Unknown bits**: `screensharingd` reads the bitmap MSB-first and the **only** bit it queries is **bit 20** (`0x14`) = "observe-only mode supported" (it gates a `0x14`-type control message); bits `30, 31, 32, 35, 81` are **never tested in `screensharingd`** — and **not in `ScreensharingAgent` either** (the Agent receives the viewer's capabilities already digested into a small flags scalar — 60 fps stream 1/2, do-not-send-cursor — not the raw 256-bit bitmap). Their meaning is therefore viewer-internal / unresolved from these binaries and remains a **revision gap**. Senders SHOULD emit the observed bitmap; receivers ignore unmodeled bits.

### 5.6 SetEncryptionMessage

- **Direction**: client→server. **Framing**: `u8 type = 0x12 || u8 reserved || u16 command || command-specific`.
- `command = 1`: `u16 method_count = 1 || u16 method = 1` (method 1 = AES-128). Full byte form `12 00 0001 0001 0001 00000001`.
- `command = 2`: short form `12 00 0002 || u16 value = 1 || u16 reserved = 0`.
- **Validation**: only method value `1` (AES-128) is accepted — `HandleSetEncryptionMessage` scans the `count`-entry method list for `1` and errors if it is absent; `command = 1` starts encryption, `command = 2` stops it. No other method value exists.

### 5.7 SetModeMessage

- **Direction**: client→server. **Framing**: `u8 type = 0x0a || u8 reserved || u16 mode`. `mode = 0` = observe, `mode = 1` = normal control. Observed `mode = 1` (`0a 00 0001`) emitted by native Screen Sharing.app.
- **Not required for interop.** The dispatcher `screensharingd::sub_1000352ac` has `case 0x0a` → `HandleSetModeMessage`, which only overwrites the session control word (`viewer+0x1a`) and fires `SSAgent_SetControl_rpc`. That control word is **independently established at ServerInit** (`SendServerInitialiation`, `sub_100033384`), so no streaming/start path depends on SetMode. The reference client omits SetMode and the server still streams; SetMode is therefore **OPTIONAL**, not a mandatory prelude step.

## 6. Rekey and Secure Transport

### 6.1 Rekey Message (`EncodeEncryptionInfo`, `0x44f`)

- **Direction**: server→client. **Phase**: prelude→record-layer transition. **Transport**: carried as a single-rectangle FramebufferUpdate with `x = y = w = h = 0` and encoding `0x44f`.
- The rekey activates (or re-keys) the AES-128-CBC record layer (§6.3).

### 6.2 Rekey Payload

- **Framing**: `u32 generation || byte[16] encrypted_key || byte[16] encrypted_iv` (36 bytes).
- `generation`: a 4-byte counter. The first rekey of a fresh session has `generation = 1`; broader semantics are a **revision gap**. Senders preserve the observed value; receivers MUST NOT rely on a specific initial value.

#### 6.2.1 Wrap-Key Decryption and Rotation

- The receiver decrypts `encrypted_key` and `encrypted_iv` **independently** using AES-128-ECB single-block decryption under the 16-byte **wrap key** (§6.2.2). ECB-decrypt of the rekey body under the installed wrap key yields exactly the record-layer key and IV.
- **Wrap-key rotation**: the recovered `next_key` becomes both the new AES-128-CBC content key (§6.4) and the wrap key used to decrypt the **next** rekey; `next_iv` becomes the new CBC IV in both directions. The server's rekey builder (`screensharingd::sub_100020ef8`, `SendFrameBuffer`) ECB-wraps the new key/iv under the persistent ECB context and then immediately rotates its own CBC send/recv contexts (`+0x5f0`/`+0x5e8`) to that new key/iv. Multi-rekey behavior across ≥2 rekeys in one session is not exercised by a capture, but the per-rekey install path is established from the server binary.
- **Receiver behavior on failure**: a decryption that does not yield a usable key/IV MUST cause the client to close.

#### 6.2.2 Initial Wrap Key Per Authentication Branch

| Branch | Initial wrap key | Source |
|---|---|---|
| `30` (DH) | `MD5(shared)[0:16]` | MD5 over the DH shared secret (§4.2.3); `sub_100068030` |
| `33` (RSA-SRP) | `SHA-256(K)[0:16]` | SHA-256 over the SRP shared secret `K` (§4.2.4.8); `sub_100018e60` |
| `35` (Kerberos) | a **fresh random 16-byte key** generated server-side (`/dev/random`) and sent to the viewer over the GSS channel (§4.2.5) | |
| `36` (Direct SRP) | `SHA-256(K)[0:16]`, identical to type 33 | |

### 6.3 Record-Layer Activation

After `0x44f`, both endpoints use AES-128-CBC with the rekey-distributed key and IV. The send and receive plaintext sequence counters (§6.4.4) MUST NOT be reset at activation; they are session-monotonic. No-reset across a subsequent rekey is a revision gap, there being no second rekey in the capture.

### 6.4 Record-Layer Properties

#### 6.4.1 Outer Wire Form

`u16 ciphertext_len || byte[ciphertext_len] ciphertext`. `ciphertext_len` is a non-zero multiple of 16. **Validation**: a record whose `ciphertext_len` is zero or not a multiple of 16 MUST cause the client to close. All captured records satisfy `len % 16 == 0`.

#### 6.4.2 Plaintext Layout

`u16 body_len || byte[body_len] body || byte[filler_len] filler || byte[20] integrity`, where `filler_len = (-(2 + body_len + 20)) mod 16` (minimal padding to a 16-byte multiple). This holds byte-exact across all captured records.

- **Filler value**: filler bytes are **random** (not derived from the body; specifically not a repeat of the preceding byte). Receivers MUST NOT validate filler contents. Senders MAY emit zero or random filler (implementation-defined).

#### 6.4.3 CBC State

Each direction is a single AES-128-CBC stream spanning the entire post-rekey session: the last 16 bytes of ciphertext from record N are the IV for record N+1. A receiver/sender MUST hold one persistent cipher context per direction and MUST NOT reset it between records. The initial IV is the rekey-distributed `next_iv`. This was independently reproduced by decrypting all captured records with a single persistent context per direction.

#### 6.4.4 Sequence Numbers

Independent `u32` send and receive counters, each starting at 0 and incrementing by one per record, never reset across the session (including across activation and rekey). No-reset across a second rekey is a revision gap.

#### 6.4.5 Integrity Trailer

`integrity = SHA-1( u32_be(seq) || plaintext[0 : ciphertext_len − 20] )` — plain SHA-1 (not HMAC) over the big-endian sequence number concatenated with the plaintext up to but excluding the trailer. **Validation**: on mismatch the receiver MUST close. `EncryptOneMessage` (`sub_100054888`) / `DecryptOneMessageWithComCryption` (`sub_100054a74`) call `_CC_SHA1_Init`/`Update`/`Final` with the byteswapped `u32` seq prepended (`_CC_SHA1_Update(&ctx,&seq_be,4)`) and a 20-byte tag; `_CCHmac` is absent from the import table. Send/recv sequence counters live at `viewer+0x914`/`+0x918` and only ever increment.

#### 6.4.6 Message Boundaries

For client→server control and small server messages, **one record carries exactly one higher-level message**; `body_len` delimits that single message. Large server payloads (e.g. zlib framebuffer rectangles) MAY span **consecutive** records; the higher-level payload is reassembled by concatenating successive record bodies in order. The record layer does **not** multiplex independent messages within one record and does not fragment a small control message across records. A client MUST treat `body_len` as the exact length of the record's body and MUST reassemble large server payloads by concatenation in record order.

### 6.5 Encrypted Preface

The first two client send-sequence records carry `SetDisplayConfiguration` (`0x1d`, §7.1) then `SetEncodings` (`0x02`) — decrypted record seq 0 and seq 1.

## 7. Display Configuration and Display Selection

### 7.1 SetDisplayConfiguration (`0x1d`)

- **Direction**: client→server. **Phase**: encrypted preface / steady state. **Transport**: record-layer body. **Byte order**: big-endian.

**Offset bases.** Three bases are used and are stated per field: **(M)** from the start of the full message; **(B)** from the start of the post-prefix body (i.e. the first byte after the `message_size` field); **(D)** from the start of a display descriptor.

**Front header** (offsets from M):

| Off (M) | Field | Type | Notes |
|---|---|---|---|
| +0x00 | `type` | u8 | `0x1d` |
| +0x01 | `reserved` | u8 | emit-zero |
| +0x02 | `message_size` | u16 | see length rule |
| +0x04 | `version` | u16 | `1` |
| +0x06 | `display_count` | u16 | observed `1` |
| +0x08 | `flags` | u32 | observed `0` |
| +0x0c | `display_descriptor[display_count]` | — | each begins a (D) frame |

**Length rule.** `message_size` is the length counted **from the byte after the `message_size` field** — i.e. the total message length minus the 4-byte prefix (`type`, `reserved`, `message_size`). For `display_count = 1`, `mode_count = 5`: total = `0x0c + 0x9c + 5×0x1c` = 308, and `message_size = 304`.

**Display descriptor** (offsets from D):

| Off (D) | Field | Type | Notes |
|---|---|---|---|
| +0x00 | `display_info_size` | u16 | descriptor length incl. mode table (`0x9c + mode_count×0x1c`). `screensharingd::sub_1000352ac` (msg `0x1d`) reads D+0x00 with a 16-bit load (`ldrh [x24]`, byteswapped) and uses it in the descriptor bound check; no 32-bit load of +0x00 exists. (`display_info_region` also begins at +0x02, leaving only 2 bytes.) |
| +0x02..+0x79 | `display_info_region` | byte[120] | opaque; NUL written at +0x79; MAY carry a display-name UTF-8 string. **Receiver: parse-and-ignore** (implementation-defined contents) |
| +0x7a | `display_flags` | u32 | bit `0x01` = dynamic-resolution; observed `0`. ScreensharingAgent `CreateVirtualDisplay` (`0x100026c28`) tests `display_flags & 1` and, when set, logs "set dynamic resolution" and calls `_SLSDisplaySetDynamicGeometryEnabled`. |
| +0x7e | `display_type` | u32 | observed value `0`; **value `4` = virtual display** (`screensharingd` keys its virtual-display path on a session-type field set to `4`; values `0` and `2` appear on other paths). Full enumeration is a **revision gap** |
| +0x82 / +0x86 | `physical_width_mm` / `physical_height_mm` | f32 (BE) | observed ≈ `369.45 × 207.82` for a 1920×1080 logical display |
| +0x8a / +0x8e | `max_width` / `max_height` | u32 | pixels |
| +0x92 / +0x94 | `current_mode_index` / `preferred_mode_index` | u16 | MUST be `< mode_count` |
| +0x96 | `rotations` (was `reserved`) | u32 | Display rotation setting; normal value `0`. ScreensharingAgent `-[SSAgentVirtualDisplay updateDisplayInfo:…]` (`0x100002a54`) logs this u32 as `displayInfo->rotations` and passes it straight into `SLVirtualDisplaySettings …rotations:`; `screensharingd` only byteswaps and forwards it. **Neither binary branches on +0x96/+0x99.** The reference client emits `7` here on an "alt-user login" path, but that path is **not** selected by this field (the login-window / blank-screen choice is a runtime session probe plus the `BlankScreen` CFPreference, not the descriptor). |
| +0x9a | `mode_count` | u16 | |
| +0x9c | `mode_table[mode_count]` | mode_entry[] | §7.2 |

**Validation.** `current_mode_index` and `preferred_mode_index` MUST be `< mode_count`, and `mode_count` MUST be non-zero. Server-side, `screensharingd::sub_1000352ac` (msg `0x1d`) rejects `mode_count == 0`, rejects `current_mode_index >= mode_count` and `preferred_mode_index >= mode_count` (unsigned compares), and bounds-checks the `mode_count × 0x1c` mode table against the message size. **Unknown/reserved fields**: `flags` = 0; the `+0x96` field is `rotations` (normally 0), not a reserved word; receivers ignore unmodeled bits.

### 7.2 Mode Table

Each entry is 28 bytes (`0x1c`), big-endian:

| Off | Field | Type |
|---|---|---|
| +0x00 | `width` | u32 |
| +0x04 | `height` | u32 |
| +0x08 | `scaled_width` | u32 |
| +0x0c | `scaled_height` | u32 |
| +0x10 | `refresh_rate_hz` | f64 (BE) |
| +0x18 | `flags` | u32 |

`width`/`height` are the source render resolution; `scaled_width`/`scaled_height` the logical resolution presented to the viewer. `refresh_rate_hz` is a big-endian IEEE-754 double (observed `60.0` = `40 4e 00 00 00 00 00 00`). `flags` bit `0` = HDR; higher bits are a **revision gap** (observed `0`). `HandleSetDisplayConfiguration` tests only bit 0 of the `+0x18` mode-entry flags (sets the per-display HDR flag); the rest is stored but unused. The separate "do not adjust refresh rate" bit is bit 1 of the **message-header** flags byte, not this mode-entry word.

### 7.3 Dynamic Resolution Behavior

A viewer that wants in-band mid-session resize MUST send a **full dynamic descriptor** on its `0x1d`: `display_flags` bit `0x01` set, `display_type = 4`, `reserved = 7`, `current_mode_index`/`preferred_mode_index` valid (`< mode_count`), `max_width`/`max_height` bounding the backing geometry, and a populated mode table. A descriptor that omits the dynamic flag is treated as an ordinary (non-resizable) display configuration — sending a bare 0x1d mid-session does not initiate a resize. Native Screen Sharing.app emits this full descriptor on **every** 0x1d (initial and steady-state); a viewer that only needs a fixed-resolution session, however, MAY send a **non-dynamic** descriptor — the reference client (iShareScreen) emits a bare descriptor (no dynamic flag, `display_type = 0`, zeroed mode indices) for static sessions and interoperates fully. The full dynamic descriptor is required only to *request* an in-band resize. The server later confirms each change via `AppleDisplayLayout` (§7.5, §8.4), which carries the authoritative `scaled`/`backing` geometry and MAY differ from (e.g. be smaller than) the request when it exceeds the host's backing cap. The full media-mode exchange the dynamic flag drives is specified in §10.9. This is confirmed against a native resize capture and reproduced by an interoperable client.

### 7.4 SetDisplayMessage (`0x0d`)

- **Direction**: client→server. **Framing**: `u8 type = 0x0d || u8 combine_all_displays || u16 reserved || u32 display_id`. Observed body `0d 01 0000 00000000` selects the combined-display aggregate; when `combine_all_displays ≠ 0`, `display_id` is ignored.

### 7.5 Display Layout Updates

The server MAY emit `AppleDisplayLayout` (§8.4) updates that change effective geometry mid-session. **Receiver behavior: action** — a client MUST treat layout updates as authoritative for framebuffer sizing and MUST resize local framebuffer state before applying rectangles that assume the new geometry. Observed: geometry reduced 3840×2160 → 1920×1080 mid-session.

## 8. Message and Encoding Registry

### 8.1 Standard RFB Messages

The session uses standard RFB messages (SetPixelFormat `0x00`, SetEncodings `0x02`, FramebufferUpdateRequest `0x03`, key/pointer events, etc.) unchanged, carried in record-layer bodies after activation.

### 8.2 Client-to-Server Message Registry

| Opcode | Name | Phase | Status |
|---|---|---|---|
| `0x00` | SetPixelFormat | steady | Standard RFB |
| `0x02` | SetEncodings | preface/steady | Standard RFB |
| `0x03` | FramebufferUpdateRequest | steady | Standard RFB |
| `0x08` | ScaleFactor | preface | §8.10 |
| `0x09` | AutoFrameBufferUpdate | steady | §8.11 |
| `0x0a` | SetMode | prelude | §5.7 |
| `0x0d` | SetDisplayMessage | preface/steady | §7.4 |
| `0x10` | EncryptedInputEvent | steady | 2-byte header + a 16-byte AES block (decrypted in place with the session ECB cryptor); a `0xff` marker byte selects the event: keyboard (`keysym u32`, down-flag) or pointer (button/scroll mask, x/y). `HandleEncryptedEventMessage`. |
| `0x12` | SetEncryptionMessage | prelude | §5.6 |
| `0x15` | AutoPasteboard | steady | §8.12 |
| `0x1c` | MediaStreamOptions | steady | **Partially specified / experimental** — §10.6; full schema a **revision gap** |
| `0x1d` | SetDisplayConfiguration | preface | §7.1 |
| `0x21` | ViewerInfo | prelude | §5.5 |
| `0x1a` | SetKeyboardInputSource (client→server) | steady | `u8 0x1a || u16 size || u16 message_version || u16 id_len || u8[id_len] source_id`; forwarded to the Agent (`SSAgent_SetKeyboardSourceID_rpc`). Carries the source-ID **string only — no flags**. The `keyboard_input_flags` echoed in `0x455` are **Agent-originated** (set via `SSDaemon_KeyboardInputSource`), opaque to `screensharingd`; their bit meanings remain a revision gap (§8.6). |

### 8.2.1 Server-to-Client Control Messages (non-encoding)

Beyond the framebuffer-update encodings (§8.13), the server sends a few **control** messages on the record-layer channel that the client→server registry (§8.2) does not cover.

| Opcode | Name | Dir | Notes |
|---|---|---|---|
| `0x14` | MiscStatus | S→C | 8 bytes: `u8 type=0x14 || u8 pad || u16 body_len (=4) || u16 flags (=1) || u16 cmd`. `cmd` dispatches: `12` = heartbeat (~2.1 s, gated by the viewer command-mask), `2` = remote clipboard changed (viewer replies with a `0x0b` fetch), `11` = UserSessionChanged. A live session also emitted **`cmd = 4`** (`14 00 0004 0001 0004`), meaning unknown — the `cmd` space is a **revision gap**. |
| `0x1f` | ClipboardSend (`HandleViewerClipboardSend`) | S→C | 16-byte header `u8 0x1f || u8 pad || u8 promise || u8 pad || u32 reserved || u32 uncompressed_size || u32 compressed_size`, then a zlib (`Z_SYNC_FLUSH`-framed) stream of an inner pasteboard archive. The compressed payload MAY span **multiple** record frames; reassemble to `compressed_size`. **Inner archive** (a 3-flavor pasteboard): `u32 item_count`, then per item `u32 uti_len || uti || u32 reserved(=0) || u32 alias_count || (u32 name_len || name || u32 val_len || val)×alias_count || u32 primary_data_len || primary_data`. Example flavors seen: `public.utf8-plain-text` (aliases `com.apple.ostype`="utf8", `com.apple.nspboard-type`="NSStringPboardType", `public.mime-type`="text/plain;charset=utf-8"), `com.apple.traditional-mac-plain-text` (ostype "TEXT"), `public.utf16-plain-text` (UTF-16-BE data). **Empty clipboard** is sent as `uncompressed_size=0` (the 7-byte deflate stream decompresses to nothing) — receivers MUST treat a zero-length inner archive as "no items" rather than failing to read `item_count`. |

Corresponding **client→server** control messages absent from §8.2: `0x06` ClientCutText (standard RFB, latin-1 text out) and `0x0b` ClipboardFetch (requests the full pasteboard after a `0x14 cmd=2`).

### 8.3 CursorImage (`0x450`)

- **Direction**: server→client. **Transport**: framebuffer-update rectangle. **Receiver behavior: render** (cursor pixmap). Encoder `EncodeCursorImageWithAlpha` (`0x100019aa8`). The rectangle header carries the cursor geometry (`x` = hotspot-x, `y` = hotspot-y, `w`/`h` = cursor size, per the RFB cursor convention); the payload is `u32 cache_id || u32 compressed_len || zlib_payload`. *(Confirmed: static analysis + a native Screen Sharing.app capture that carries the full STORE/SELECT cursor lifecycle, replayed end-to-end by an interoperable client.)*

- **Two forms, distinguished by `compressed_len` (a STORE/SELECT cursor cache):**
  - **STORE** (`compressed_len > 0`): the payload is a zlib (`deflate` level 9, `Z_SYNC_FLUSH`-framed) stream the client decompresses and **caches under `cache_id`**. The decompressed bytes are a `w·h·4` BGRA pixmap followed by a `w·h` 8-bit **alpha plane separated from the 24-bit RGB** (source is 32-bit BGRA; the alpha is *not* interleaved into RGBA). `cache_id`s are server-assigned, counting up from `0x3e8` (1000); they are **opaque session-local handles with no fixed shape meaning** — a given id is whatever shape the server happened to store into it this session, so a client MUST NOT special-case any id.
  - **SELECT** (`compressed_len = 0`): no pixels — the rectangle references a previously-STOREd `cache_id` (its geometry fields are zeroed) and the client re-applies that cached pixmap. This is how the host switches the cursor **shape** (e.g. arrow ↔ I-beam ↔ resize) without resending pixels; in steady state it is by far the common form.

- **Shape only, never position.** `0x450` conveys the cursor *shape*, not its screen location: the client draws the selected pixmap at its own locally-tracked pointer position (pointer motion travels client→server as ordinary RFB pointer events). A client that has not yet received the STORE for a SELECTed id MAY keep its last shape (or fall back to a local cursor) rather than blanking.

- **The SELECT stream is server-driven and must be kept armed.** After the initial `FramebufferUpdateRequest`, the server free-runs cursor STORE/SELECT updates only while its framebuffer sender is armed by **AutoFrameBufferUpdate (`0x09`, §8.11)**. That arming is dropped at a display/session transition (`AppleDisplayLayout`, §8.4) — a login, lock, or fast-user-switch agent handoff — after which SELECTs stop and the cursor shape **freezes on its last value**. A client MUST re-arm at every `0x451` (§8.4, §8.11) to keep the shape updating across these transitions. *(Confirmed: the post-login freeze reproduces without the re-arm and is resolved by re-sending `0x09` + a non-incremental `0x03` at each `0x451`, matching native.)*

### 8.4 AppleDisplayLayout (`0x451`)

- **Direction**: server→client. **Transport**: framebuffer-update rectangle with server-assigned `x,y,w,h`. **Receiver behavior: action** (framebuffer sizing). **Length rule**: the rectangle body begins with a `u16` payload-length prefix; a payload does not exceed 65535 bytes.
- The leading fields are interpretable: after the `u16` payload-length prefix and a `u16 version` (observed `5`), the body carries `u16 scaled_width`, `u16 scaled_height`, `u16 backing_width`, `u16 backing_height` (e.g. `1920, 1080, 3840, 2160`). `scaled_*` is the logical display geometry; `backing_*` is the pixel geometry the encoder renders at (HiDPI is `backing = 2 × scaled`).

**Provider-blob layout** from the sender `AppleVNCServer::EncodeDisplayInfo` → `AddNextDisplay_DisplayInf2Encoding` (`0x10004ef80`; `EncodeDisplayInfo.c`, a recent macOS build (minor version skew observed across builds; wire layout believed stable)). After the rectangle header + encoding tag the blob is:

```text
u16   payload_len      (= n_displays × 0x38 + 0x14)
u16   version          (= 5)
u32   current_display  (BE; 0xffffffff = none/-1 sentinel)
u32   flags            (bit 0x02000000 = CanBeModified)
u16   flag_word        (= 1)
display_record[n_displays]   -- 0x38 (56) bytes each:
  +0x00 f64 BE  scale/refresh    (CGDisplayMode resolution → double)
  +0x08 f64 BE  scale_factor     (viewer scale, else 1.0 = 3ff0000000000000)
  +0x10 u32 BE  display_id       (CGDirectDisplayID)
  +0x14 rect    global bounds    (x,y,w,h u16 BE; CGDisplayBounds)
  +0x1c rect    scaled bounds    (x,y,w,h u16 BE)
  +0x24 u32 BE  flags            (bit0 = main display, bit1 = in mirror set)
  +0x28 u8 bpp | u8 depth | u8 big_endian=0 | u8 true_colour=1
  +0x2c u16 BE redMax | u16 BE greenMax | u16 BE blueMax
  +0x32 u8 redShift | u8 greenShift | u8 blueShift (+pad to 0x38)
```

The leading `scaled_w/h`, `backing_w/h` are the first record's bounds rects; the two `3ff0000000000000` doubles in the live sample are the **scale factors (1.0)**, not refresh-Hz. The record body model comes from the sender. There is a minor offset discrepancy between the a recent macOS build disassembly and the live-host bytes near the geometry leader (likely version skew) — treat the live `0x451` geometry as authoritative and the record body as the field model.
- **Media-path obligation.** When this message arrives mid-session **on the media (HEVC) path** as the answer to a viewer-initiated resolution change (§10.9), framebuffer sizing is **not sufficient** — the viewer MUST follow it with a `MediaStreamOptions` (`0x1c`) re-offer to make the server resize the encoder canvas. A viewer that resizes its local framebuffer but never re-offers will see the media stream stall (the server emits the `0x451` and then stops sending media). On the pure framebuffer path this message is purely a sizing action with no re-offer.
- **Cursor re-arm obligation.** Independently of the media re-offer above — and on **every** `0x451`, including no-geometry-change layout events emitted at a login/lock/agent transition — a client MUST re-arm the server's framebuffer sender by re-sending `AutoFrameBufferUpdate` (`0x09`, §8.11) + a non-incremental `FramebufferUpdateRequest` (`0x03`). Without this the server stops emitting cursor (`0x450`) SELECTs after the transition and the cursor shape freezes on its last value (§8.3, §8.11). Unlike the media re-offer, the re-arm fires on every layout, not only on a geometry change.
- **Unknown fields**: receiver MUST tolerate and ignore trailing fields it does not interpret.

### 8.5 VendorKeysymEncoding (`0x453`)

- **Direction**: server→client. **Transport**: framebuffer-update rectangle (`x = y = w = h = 0`). **Receiver behavior: parse-and-ignore** (tolerate; do not disconnect on unknown values). **Length rule**: fixed **22 bytes** after the rectangle header. Encoder `EncodeVendorKeysyms` (`screensharingd` @ `0x100022bac`) writes `00 14 00 01 00 04` then 16 bytes copied from the fixed table `data_10007c250` = `1008fd00 1008fd01 1008fd02 1008fd03`; a live session emitted exactly `0014 0001 0004 1008fd00 1008fd01 1008fd02 1008fd03`.

```text
u16   header_count    (= 0x0014)
u16   header_version  (= 0x0001)
u16   value_count     (= 4)
u32   vendor_keysym_0 (= 0x1008FD00)
u32   vendor_keysym_1 (= 0x1008FD01)
u32   vendor_keysym_2 (= 0x1008FD02)
u32   vendor_keysym_3 (= 0x1008FD03)
```

The four values are a fixed server-side table, not session-derived. The symbolic meaning of each keysym is a **revision gap**. A client MUST NOT terminate the session because a vendor keysym's meaning is unknown.

### 8.6 KeyboardInputSource (`0x455`)

- **Direction**: server→client. **Transport**: framebuffer-update rectangle (`x = y = w = h = 0`). **Receiver behavior: parse-and-cache** (for local input mapping). Let `S` = byte length of the UTF-8 input-source identifier (no NUL).

```text
u16   prefix_length         (= S + 8)
u16   version_marker        (= 0x0001)
u32   keyboard_input_flags  (mirrors client-supplied flags)
u16   id_len                (= S)
u8[S] input_source_id       (UTF-8, no NUL)
```

Total payload after the rectangle header is `S + 10` bytes. `version_marker` is a fixed marker (wire value `0x0001`), not a bitmap. `keyboard_input_flags` is a single-bit **secure-event-input** indicator: the Agent's `SendKeyboardSourceInfoToDaemon` (`sub_10001bf08`) sets it from `_CGSIsSecureEventInputSet()` (`1` = server is in secure-terminal / secure-event-input mode, `0` = not) and sends it to the daemon via `SSDaemon_KeyboardInputSource` (MIG `0x1513`) alongside the input-source-ID string. No other bits are set. The client-to-server counterpart is `0x1a` SetKeyboardInputSource (§8.2), which carries the source-ID string only. Encoder `EncodeKeyboardInputSource` (`screensharingd` @ `0x100022cfc`) writes `prefix_length = S+8`, the literal `0x0001` marker (the `0x0100` immediate in disasm is a little-endian `strh` artifact), `keyboard_input_flags` from `viewer+0x1070`, then `id_len = S` and `S` raw bytes (no NUL). A live session emitted `001f 0001 00000000 0017` + `"com.apple.keylayout.ABC"` (S=23, prefix=0x1f=S+8, total 33=S+10).

### 8.7 DeviceInfo (`0x456`)

- **Direction**: server→client. **Transport**: framebuffer-update rectangle (`x = y = w = h = 0`). **Receiver behavior: parse-and-cache** (descriptive metadata).

```text
u16   message_size          (= info_block_size + 0x10)
u16   block_pair_count      (= 0x0002)
u32   structure_version     (= 0x00000001; low bit = server-supports-dynamic-drag)
u32   enclosure_rgb_color   (intended RGB int; current build never populates it → 0)
u16   device_identifier_len (incl. NUL)
u16   device_color_len      (incl. NUL)
u16   enclosure_color_len   (incl. NUL)
u8[]  device_identifier     (UTF-8, NUL-terminated; = sysctl "hw.model", else "unknown")
u8[]  device_color          (UTF-8, NUL-terminated; = MobileGestalt "DeviceColor")
u8[]  enclosure_color       (UTF-8, NUL-terminated; = MobileGestalt "DeviceEnclosureColor")
u32   housing_color         (signed BE int; = MobileGestalt kMGQDeviceHousingColor)
```

Encoder `screensharingd::sub_100020b4b` (`EncodeDeviceInfoMessage`, `RFBServerUtils.m`; writes the `0x456` tag as `0x00000456` and sits in the same `SendFrameBuffer` family as `0x453`/`0x455`), cross-checked against the decoder `-[SSSession handleDeviceInfoEncoding:]` in `ScreenSharing.framework` (`0x1c21808ec`, internal dispatch case `0x24`, mirroring `0x455`→case `0x22`).

Field notes:
- The `+0x08` word is **not** "reserved": it is an **enclosure RGB color** integer. The encoder fetches `DeviceEnclosureRGBColor` from MobileGestalt but only logs it and never stores it, so it is `0` in practice — the viewer files it under its `DeviceColor` key. Treat as ignore-on-receive but understand its intent.
- `message_size = info_block_size + 0x10` (info-block bytes plus the 16-byte fixed header).
- `structure_version` low bit is a capability flag: the viewer reads `version & 1` as `setServerSupportsDynamicDrag:`.
- Field sources: `device_identifier` = `sysctlbyname("hw.model")` with literal `"unknown"` fallback (Apple's log even has the typo "device identififer"); `device_color`/`enclosure_color` = MobileGestalt `DeviceColor`/`DeviceEnclosureColor`; `housing_color` = MobileGestalt `kMGQDeviceHousingColor` (a `CFNumber SInt32`).
- **Value space**: there is **no enumerated colour-name table** in either binary. `device_color`/`enclosure_color` are **free-form MobileGestalt UTF-8 strings** (whatever MG returns for the model); `enclosure_rgb_color` and `housing_color` are **`SInt32` integers** (a packed RGB and an opaque colour id respectively) with no integer→name mapping in the binaries. So a client should treat the strings as opaque labels and the integers as raw values.
- **Wire layout is encoder-canonical.** The viewer's dictionary *key names* do not line up with each field's semantic source (it effectively mislabels the strings), but the offsets/sizes are exactly as the encoder writes them, in source order identifier → device_color → enclosure_color, then the `housing_color` u32.

**Emission/bounds**: `block_pair_count` is hardcoded `2` and the `housing_color` u32 is always appended (the viewer consumes it only when `block_pair_count ≥ 2`, always true here). The encoder bounds `device_identifier_len ≤ 0x1387`, cumulative string length ≤ `0x1387`, and bails (logs an error) if the running offset would leave no room for `housing_color`.

### 8.8 RFBMediaStreamMessage1 (`0x3f2`)

- **Direction**: server→client (announcement). **Transport**: framebuffer-update rectangle with a `u16` payload-length prefix. **Receiver behavior: tolerate** (announcement only).
- Payload model after the `u16` prefix: `u32 version || u16 base_udp_port || u16 stream_count || u16 next_stream_port || reserved`. The baseline uses `version = 1`. Advertised but not emitted in the capture; field widths from the interoperable client. Later-field semantics are a **revision gap**.
- `0x3f2` is the **announcement** half of the media path; the **negotiation** half (SRTP keys, codec offers) is the separate `0x1c` MediaStreamOptions exchange (§10.6). Advertising `0x3f2` MUST NOT be treated as proof of a content switch (§3.4).

### 8.9 Apple Still-Image Codec Encodings

#### 8.9.1 Encoding Identifiers

| Encoding | Name | Class |
|---|---|---|
| `0x06` | Standard zlib | classical RFB pass-through |
| `0x3e8` | Low Quality | Apple codec, 4-bit color |
| `0x3e9` | Medium Quality | Apple codec, 8-bit YCoCg dithering |
| `0x3ea` | High Quality | Apple codec, 16-bit RGB 5-6-5 |
| `0x3f3` | Multi-Variant Scaled | per-tile DCT codec (TCP framebuffer) |

#### 8.9.2 Quality Tier Selection

| Tier | Encodings advertised |
|---|---|
| Full | `zlib, copyrect` |
| Low | `0x3e8, zlib, zrle` |
| Medium | `0x3e9, zlib, zrle` |
| High | `0x3f3, 0x3ea, zlib, zrle` |
| High + media-init | `0x3f2, 0x3f3, 0x3ea, zlib, zrle` |

A client MUST NOT advertise an encoding it cannot decode or safely handle (§11, Profile B).

#### 8.9.3 zlib Pipeline

`0x3e8`/`0x3e9`/`0x3ea`/`0x06` share a zlib pipeline differentiated by pixel pre-processing and deflate level: `0x3e8` 4-bit/16-color per 8-pixel block (deflate 9); `0x3e9` 8-bit YCoCg dither, 2 px/byte (deflate 6); `0x3ea` 16-bit RGB 5-6-5 (deflate 1); `0x06` 32-bit pass-through (deflate 1). (Deflate levels are read encoder-side.) A `0x06` rectangle's compressed payload MAY span consecutive records (§6.4.6).

#### 8.9.4 Multi-Variant Scaled (`0x3f3`)

`0x3f3` is a per-tile YCbCr/DCT codec delivered **in TCP framebuffer-update rectangles** — distinct from the UDP HEVC media path (§10). The High + media-init tier may advertise both `0x3f3` and `0x3f2` in one `SetEncodings`, but they are independent paths.

The body internals below come from the encoders `EncodePartialUpdateMVS` (`0x1000409dc`), `EncodeMVS` (`0x100042478`), `EncodeMVSQuantizationTables` (`0x1000421d4`), and bit-writer `BitWriteStoreBits` (`0x100041bf0`). The captured session negotiated UDP HEVC and emitted no `0x3f3` rectangle, so the per-field behavior is read from the encoder. The 3-bit command-code↔meaning mapping remains a **revision gap** (see below).

Rectangle body:

```text
u32   nbytes                (rectangle-body length)
u8    type_of_update        (0 = Partial, 1 = Full, 2 = Quantization-table upload)
u8    tile_width  / dct_p1  (tile width for Full; a DCT parameter for Partial)
u8    tile_height / dct_p2  (tile height for Full; a DCT parameter for Partial)
u8[3] data_off              (24-bit BE offset, from body start, to the data stream)
byte[] command_bitstream    (begins at body offset 6)
byte[] data_stream          (begins at body offset data_off)
```

Tile defaults: `16 × 16` for Full updates (when the advertised value is below 8), `8 × 8` for Partial. A `type_of_update = 2` message uploads the DCT quantization tables (64-byte luma then 64-byte chroma) and carries no tiles; it MUST precede any DCT (command 5) tile.

Command types: `0` Skip; `1` MatchPrevious; `2` MatchAbove; `3` TwoColor (8-byte mask + two YCbCr 8/6/6 colors); `4` Solid (one YCbCr color); `5` DCT (quantized YCbCr coefficients); `6` Cache6 (16-bit BE index); `7` Cache7 (implicit previous+1).

Commands are bit-packed **MSB-first** (`BitWriteStoreBits`, big-endian accumulator). The **repeat count** is gated: a 5-bit field `0x10 | (n − 1)` for `n ≤ 15`, otherwise the 5-bit escape `0x1f` followed by a base-128 varint of `n − 0x10` (7-bit groups, high bit = continuation); a run covers `repeat + 1` tiles. **Colors** are YCbCr 8/6/6, packed as a 20-bit value `Y(8) << 12 | Cb(6) << 6 | Cr(6)`. The codec emits **two parallel bitstreams** (luma + chroma); stream B is concatenated after stream A with a 24-bit split offset recorded in the header. Each stream is terminated by `0x6d`, and the block ends with the 3-byte marker `0x6d 0x76 0x73` (`"mvs"`). The tile grid is `ceil(rect.width / tile_width) × ceil(rect.height / tile_height)`, row-major. The **3-bit command type-code → meaning** mapping (Skip/MatchPrevious/MatchAbove/TwoColor/Solid/DCT/Cache6/Cache7) could not be isolated as a flat table and remains a **revision gap**.

### 8.10 ScaleFactor (`0x08`)

- **Direction**: client→server. **Framing**: `u8 type = 0x08 || u8 reserved || f64_be scale` (10 bytes). `HandleSetServerScalingMessage` reads the 8 bytes at offset 2 as a big-endian IEEE-754 **double** scale (validated `> 0.0`); the byte at offset 1 is reserved/uninterpreted (there is no `flags` enumeration). The server derives an internal downscaling flag from `scale < 1.0`.

### 8.11 AutoFrameBufferUpdate (`0x09`)

- **Direction**: client→server. **Framing** (16 bytes):

```text
u8    type = 0x09
u8    reserved        (= 0; emit-zero)
u16   version         (= 0x0001)
u32   selected_screen (0 = first; 0xffffffff = all/main, sent by native + interop client)
u16   x
u16   y
u16   w
u16   h
```

Switches the server to server-driven framebuffer streaming. `HandleAutoFrameBufferUpdateMessage`: `version` is the incremental/update flag; `selected_screen` is the target screen id, with `0xffffffff` as the sentinel for "all/main displays" (it clears the per-screen-selection bool). After sending this, a client SHOULD NOT continue to poll with `FramebufferUpdateRequest`. The region `x,y,w,h` may be the **logical** geometry (observed from native) or the **backing** geometry (an interoperable client sends backing); the daemon accepts either.

- **Cursor / pseudo-encoding arming.** Arming the framebuffer sender is also what keeps the server emitting the TCP-side cursor pseudo-encoding (`0x450` STORE/SELECT, §8.3) and the other server-driven control rects. Native Screen Sharing.app sends `0x09` at session start (paired with a non-incremental `FramebufferUpdateRequest`, §8.2) and **re-sends the same pair at every `AppleDisplayLayout` (`0x451`, §8.4)**. The arming is dropped across a display/session transition (login, lock, fast-user-switch agent handoff), so a client that does not re-arm at each `0x451` will see cursor SELECTs stop and the shape freeze. **A client MUST re-send `0x09` + a non-incremental `0x03` on every `0x451`** to keep cursor (and other server-driven pseudo-encoding) updates flowing. This re-arm pair is lightweight and **independent of** the §10.9 media re-offer (`0x1c`): the media re-offer is required only when the layout reflects an actual geometry change, whereas the cursor re-arm is required on **every** `0x451`, including no-change layout events. *(Confirmed against a native capture and an interoperable client: the post-login cursor freeze reproduces without the re-arm and is resolved by it.)*

### 8.12 AutoPasteboard (`0x15`)

- **Direction**: client→server. **Framing**: `u8 type = 0x15 || u8 reserved[2] || u8 selector || u8 reserved[4]`. The daemon's `HandleAutoPasteboardCommand` validates `selector ∈ {1, 2}` and forwards it to the Agent's `SSAgent_AutoPasteboardCommand` (`sub_100019134`), where **`selector = 1` = start monitoring the local pasteboard** (enable universal-clipboard sync; sets the autopasteboard-enabled global and begins change tracking) and **`selector = 2` = stop monitoring** (disable). Once enabled, pasteboard data flows over the separate `SSAgent_SetPasteboard`/`SetPasteboardText` RPCs (→ the `0x1f` ClipboardSend, §8.2.1).

### 8.13 Server-to-Client Encoding Registry

| Encoding | Name | Receiver behavior | Section |
|---|---|---|---|
| `0x06` | zlib | render | §8.9.3 |
| `0x3e8`/`0x3e9`/`0x3ea` | Apple still-image codecs | render | §8.9 |
| `0x3f2` | RFBMediaStreamMessage1 | tolerate (announcement) | §8.8 |
| `0x3f3` | Multi-Variant Scaled | render | §8.9.4 |
| `0x44f` | EncodeEncryptionInfo (rekey) | action (re-key) | §6.1 |
| `0x450` | CursorImage | render | §8.3 |
| `0x451` | AppleDisplayLayout | action (sizing) | §8.4 |
| `0x453` | VendorKeysymEncoding | parse-and-ignore | §8.5 |
| `0x455` | KeyboardInputSource | parse-and-cache | §8.6 |
| `0x456` | DeviceInfo | parse-and-cache | §8.7 |

"Receiver behavior" defines what "accept" means per message (§11 R-A7): **action** = parse and act on; **render** = decode and display; **parse-and-cache** = parse and retain; **parse-and-ignore** = parse and discard without disconnecting; **tolerate** = accept without requiring interpretation.

## 9. Startup Ordering

### 9.1 Canonical Order

The canonical startup order (prose and transcript MUST agree):

1. server ProtocolVersion
2. client ProtocolVersion
3. server security types
4. client security-type selection (+ branch entry)
5. authentication branch transcript
6. SecurityResult (`u32 = 0`)
7. ClientInit (`0xC1`)
8. ServerInit
9. ViewerInfo (`0x21`)
10. SetEncryption(command=1) (`0x12`)
11. SetMode(mode=1) (`0x0a`) — OPTIONAL; native sends it, the reference client omits it (§5.7)
12. SetEncryption(command=2) (`0x12`)
13. **`0x44f` rekey** (server→client) — MAY arrive between steps 10 and 12 (§3.3)
14. encrypted `SetDisplayConfiguration` (`0x1d`)
15. encrypted `SetEncodings` (`0x02`)
16. client arms the framebuffer sender: non-incremental `FramebufferUpdateRequest` (`0x03`) + `AutoFrameBufferUpdate` (`0x09`) — required for cursor (`0x450`) and other server-driven pseudo-encodings to free-run (§8.11)
17. server metadata burst (`0x451`, `0x453`, `0x455`, `0x456`, `0x450`) and steady state

Steps 1–12 are cleartext; steps 14+ are record-layer encrypted. Steps 7, 9–12, 14, 15 are byte-confirmed from capture; the rekey interleave is described in §3.3.

### 9.2 Transcript

```text
S->C  ProtocolVersion "RFB 003.889"
C->S  ProtocolVersion "RFB 003.889"
S->C  security types {30,33,36,35}
C->S  select 0x21 (+ RSA1 packet-1)
S->C  SRP challenge (B, salt, g, N, iterations, options)
C->S  RSA1 packet-2 (A, M1, options, client_random)
S->C  final proof (M2, server_random)
S->C  SecurityResult = 0
C->S  ClientInit 0xC1
S->C  ServerInit
C->S  ViewerInfo (0x21)
C->S  SetEncryption(cmd=1) (0x12)
S->C  EncodeEncryptionInfo (0x44f) rekey      ; interleave per §3.3
C->S  SetMode(mode=1) (0x0a)
C->S  SetEncryption(cmd=2) (0x12)
=== record layer active ===
C->S  [enc] SetDisplayConfiguration (0x1d)
C->S  [enc] SetEncodings (0x02)
C->S  [enc] FramebufferUpdateRequest (0x03, non-incremental) + AutoFrameBufferUpdate (0x09)
S->C  [enc] metadata burst (0x451,0x453,0x455,0x456,...)
...   re-arm: C->S 0x09 + 0x03 on every 0x451 (keeps 0x450 cursor SELECTs flowing)
```

## 10. High-Performance and Media Transport

### 10.1 Modes

High-performance mode has two content paths: a **framebuffer-backed** path (RFB rectangles + Apple still-image codecs, §8) and an **Adaptive media** path (compressed HEVC/AAC over SRTP/UDP, §10.5+). A session MAY remain on the framebuffer path even in high-performance mode. The Adaptive path is specified as a normative **Profile C** capability (§11) with the exceptions marked as revision gaps below; the `0x1c` schema beyond the confirmed fields is experimental.

### 10.2 Transition to Adaptive Media

The switch to compressed media is gated on completion of the `0x1c` MediaStreamOptions offer/answer (§10.6). Advertising `0x3f2` and entering high-performance mode do not by themselves switch content. Observed: `0x3f2`/`0x3f3`/`0x3ea` advertised before the `0x1c` exchange; UDP media began only after `0x1c`.

### 10.3 MediaStreamOptions (`0x1c`)

- **Direction**: client→server (offer); server→client (answer). **Transport**: record-layer body.
- **Not a protobuf.** The body is a **fixed-layout binary struct** — no varint / field-number decoding. All multi-byte fields are big-endian **except** the host-endian `flags` word (see the Flags bullet below). `screensharingd::sub_1000352ac` `case 0x1c` (`HandleServerMediaStreamConfiguration`) reads the 20-byte header (byteswapping `message_size`/`version`/the three offer-length `u16`s at +0x0a/+0x0c/+0x0e), lifts the `flags` `u32` at +0x06, and forwards the whole body plus the flags to ScreensharingAgent via the `SetMediaStreamConfiguration` MIG RPC. The Agent's `sub_10001c7e4` then re-reads the offer lengths (+0x0a/+0x0c/+0x0e), the UUID (+0x14), and the six 46-byte keys directly from the forwarded body — but takes the flags from the MIG argument, never from body +0x06.
- **Three streams, not two:** the message carries **audio + video1 + video2** offer blocks (a second video stream, tied to the second-60fps flag bit below). Each stream carries **two 46-byte SRTP key blobs** `key1` then `key2` (six `NSData …length:0x2e` reads in the Agent parser).
- `key2` is the receive (server→client) key; `key1` is the client→server key. `-[SSUDPSender sendToRemoteAddress:]` sets `setSendMediaKey:` = the ServerToViewer (2nd) blob and `setReceiveMediaKey:` = the ViewerToServer (1st) blob (`0x100008bdc`/`0x100008be8`). Also independently: decrypting captured server→client media with the video `key2` authenticates every packet; `key1` does not.
- **Flags** are a `u32` at body **+0x06**, read in **host (little-endian) byte order** — *not* big-endian like the `message_size`/`message_version`/offer-length fields. The daemon `screensharingd::sub_1000352ac` (`case 0x1c`) lifts it with an un-byteswapped word load (`ldur w3, [body+6]` @ `0x10003ce84`) and forwards it to ScreensharingAgent as a `SetMediaStreamConfiguration` MIG **argument** (`arg4`), not as part of the body the Agent re-parses; the Agent (`sub_10001c7e4`) then tests it `& 1 / & 2 / & 4 / & 8`. Bit `0x01` = stream-1 60 fps supported; `0x02` = stream-2 60 fps supported (second video stream); `0x04` = do-not-send-cursor; `0x08` = AVC client-name selector (`RemoteDesktopScreenSharing` vs `AppleRemoteDesktop`). Because the field is host-endian, the meaningful bits occupy the **byte at +0x06**: a conforming little-endian client writes `07 00 00 00`, **not** `00 00 00 07`. When `messageVersion ≤ 1` the daemon forces legacy 60 fps by OR-ing `|= 0x03` into that word in place (`orr w8, w8, #3; stur w8, [body+6]` @ `0x10003cdfc`). *(Offset corrected from an earlier revision that placed `flags` at +0x0c — that position is `video1_offer_len`. Confirmed against the a recent macOS build `screensharingd` (`sub_1000352ac`) and `ScreensharingAgent` (`sub_10001c7e4`) binaries via static disassembly, plus a packet capture whose length closes exactly at `0x80 + audio_offer_len + 0x5c + video1_offer_len`.)*
- **One UUID, dual-purpose:** there is a single 16-byte UUID at body **+0x14**; it serves as **both** the session identifier and the CallID (the per-stream audio/video CallIDs are derived from it by the Agent, not carried separately). `NSUUID initWithUUIDBytes:` → `setMediaStreamSessionID:`.
- **Top-level layout** (offsets from the body's first byte = message type):

```text
+0x00  u16  message_type   (= 0x1c)
+0x02  u16  message_size
+0x04  u16  message_version (≤1 → server force-sets flags |= 0x03)
+0x06  u32  flags          (host/little-endian; bits in byte +0x06 — see note)
+0x0a  u16  audio_offer_len
+0x0c  u16  video1_offer_len
+0x0e  u16  video2_offer_len (0 when no second video stream)
+0x10  u32  reserved       (observed 0)
+0x14  16B  session/CallID UUID
+0x24  46B  audio  key1 (viewer→server / server-receive)
+0x52  46B  audio  key2 (server→viewer / server-send)
+0x80  N    audio_offer    (audio_offer_len bytes; opaque, handed to AVConference)
       46B  video1 key1 ; 46B video1 key2 ; video1_offer (video1_offer_len bytes)
       46B  video2 key1 ; 46B video2 key2 ; video2_offer (video2_offer_len bytes; present only if non-zero)
```

- **Offer/answer state**: client sends `0x1c` offer → server returns `0x1c` answer → media transport begins. The per-stream `*_offer` blobs are opaque to the screen-sharing code (passed to AVConference's `AVCMediaStreamNegotiator`); the **video** offer internals and the full answer layout remain a **revision gap**. (The `u32` at +0x06 is the `flags` field above, not an unknown word; the `u32` at +0x10 is reserved/zero in every captured offer.)

- **`audio_offer` blob — capability fields (mapping provisional).** The audio offer is a zlib-compressed protobuf `MediaBlob` (negotiator mode 8), with audio settings in top-level F3. F3.f4 defaults to `24191` (`0x5e7f`), and F3.f5 defaults to zero. Earlier notes interpreted f4 as `preferredMediaBitRate` and a numerical cutoff as an AAC-ELD tier floor; that interpretation is withdrawn. Independent Screen Sharing RE names f4 `payloadFlags` and f5 `secondaryFlags`, matching properties exposed by Apple's `VCMediaNegotiationBlobAudioSettings`. Changing a number also changes its capability bits, so the prior silence experiment does not establish a bitrate floor or prove AAC-ELD is the only negotiable codec. The existing `--no-audio` value of 1000 remains an empirical behavior, not a verified mute flag. The audio section remains necessary in observed sessions. See [audio negotiation investigation](audio-negotiation.md) for sources and remaining gaps.

### 10.4 UDP Flows and RTP Framing

Two UDP flows relative to the control port `P` (default 5900):

- **`P+1` (5901): video** — RTP payload type `100` (HEVC), with the video stream's RTCP multiplexed on the same port;
- **`P` (5900): audio** — RTP payload type `101` (AAC-ELD-SBR), with the audio stream's RTCP multiplexed on the same port.

RTCP is rtcp-muxed onto each media port alongside that port's RTP; there is no separate RTCP port. RTCP (PT 200–207) is seen on both 5901 and 5900. Video is delivered as **four RTP streams** on four consecutive SSRCs, each a horizontal tile (§10.7). Each SSRC has an independent sequence space; receivers MUST track the SRTP rollover counter (ROC) **per SSRC**.

> A periodic client→server keepalive (RTP PT 101) was described by an alternative client but was **not present** in the captured native session; treat a media keepalive as OPTIONAL (implementation-defined).

### 10.5 Secure Media Transport (SRTP / SRTCP)

This was empirically authenticated and decoded from the capture and confirmed by live decode.

Media is protected with standard SRTP (RFC 3711):

- cipher **AES-256 in counter mode** (`AES_CM_256`); authentication **HMAC-SHA1**, 80-bit tag (10 bytes appended per packet);
- the 46-byte key blob (§10.3) splits as **32-byte master key || 14-byte master salt**;
- RFC 3711 §4.3.1 KDF (AES-CM with the label XORed into the IV): labels `0`/`1`/`2` derive the SRTP cipher key (32 B), auth key (20 B), and salt (14 B); labels `3`/`4`/`5` derive the SRTCP keys;
- per-packet keystream IV `= (salt << 16) XOR (SSRC << 64) XOR (index << 16)`, `index = (ROC << 16) | seq`;
- the 80-bit tag is computed over the SRTP packet (header + encrypted payload) concatenated with the 32-bit ROC; ROC is tracked per SSRC;
- SRTCP appends a 32-bit trailer word carrying a 31-bit SRTCP index plus a top-bit encrypt (E) flag.

**Replay**: receivers maintain per-SSRC ROC/sequence state; out-of-window or duplicate packets are discarded. This media cipher is distinct from, and independent of, the AES-128-CBC control record layer (§6.4); there is no SFrame layer or second encryption pass on the media payload.

### 10.6 HEVC RTP Payload Format (Apple RFC 7798 Variant)

After SRTP decryption, the payload carries H.265/HEVC NAL units in an Apple variant of RFC 7798 with a **DONL** present on every structure:

- **Single NAL**: 2-byte NAL header, 2-byte DONL, then payload; the DONL is stripped before decode.
- **Aggregation Packet (type 48)**: 2-byte header, one 2-byte DONL, then `[u16 size][NAL]` units with **no** per-unit DOND.
- **Fragmentation Unit (type 49)**: 2-byte header, 1-byte FU header, then a 2-byte DONL **in every fragment**; the original NAL header is reconstructed from the FU header on the start fragment.

NAL units are ordered across SSRCs by decoding-order number to feed a single decoder (§10.7). Honoring Apple's DONL placement is mandatory; a decoder assuming stock RFC 7798 mis-parses the NAL headers.

### 10.7 HEVC Codec Profile and Tiling

- Codec **HEVC Range Extensions (RExt), 4:4:4 chroma, 8-bit**; decoders deliver `kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` (`nv24`/`444f`) or `yuv444p`.
- The screen is partitioned into **four horizontal strips**, one per SSRC (e.g. 3840×544 strips for a 3840×2160 source; 4×540 = 2160 logical rows).
- All four tile streams MUST be fed to a **single** HEVC decoder instance: the encoder uses cross-tile picture references (a P-frame on one SSRC references picture-order-count values produced on another SSRC); per-tile decoders fail with missing-reference errors. The base SSRC carries IDRs; the other three carry only inter-coded frames; a single shared-DPB decoder fed all four in decoding order decodes every tile.
- IDR access units appear only on the **base SSRC** (tile 0); a client SHOULD treat any IDR as a DPB reset for all tiles. Presentation recomposites the four tiles vertically in SSRC order.
- The exact tile-to-screen geometry rules (strip ordering, CTU padding) beyond the observed four-strip model are a **revision gap**.

### 10.8 RTCP Feedback and Loss Recovery

The captured native client's feedback differs from an alternative client's; both interoperate with the server.

Media RTCP is rtcp-muxed onto each media port (§10.4). Observed native-client feedback:

- Receiver Reports (PT 201) and SDES (PT 202) from the client; Sender Reports (PT 200) from the server;
- **legacy Full-INTRA-request FIR (PT 192, RFC 2032)** to request a keyframe;
- an application-defined packet (PT 204) with a numeric subtype (`5`) and an incrementing counter payload — an LTR-style per-frame acknowledgment.

> A different interoperable client uses AVPF feedback instead: FIR (PT 206, FMT 4), PLI (PT 206, FMT 1), generic NACK (PT 205, FMT 1), an empty SR (PT 200), and an APP packet for long-term-reference acknowledgment. Servers accept both feedback styles. The exact server reference-selection policy on loss, and the APP/LTR acknowledgment semantics and cadence, are a **revision gap**.

A client without a live feedback loop (e.g. replaying a capture) cannot recover from a lost reference picture; a live client relies on FIR/NACK/LTR to keep the shared decoder synchronized.

### 10.9 Dynamic Resolution in Media Mode

The media canvas geometry is **not** fixed for the lifetime of a media session. A viewer changes the resolution mid-session **in-band on the existing connection** — no TCP reconnect, no re-authentication, no record-layer rekey, and the UDP media flow is never torn down. A native resize capture is a single TCP stream over its lifetime with zero additional SYN/RST and continuous UDP video; a runtime trace of the native client shows one RSA operation and one record-layer key setup for the whole session; and an independent client reproduces the exchange live.

The exchange is **viewer-driven**:

1. The viewer sends a steady-state `SetDisplayConfiguration` (`0x1d`, §7.1) carrying the new logical size, with `display_flags` bit `0x01` (dynamic-resolution) set. The descriptor MUST be a full dynamic descriptor (`display_type = 4`, `reserved = 7`, a valid `current/preferred_mode_index`, `max_width/max_height` bounding the backing size, and a populated mode table); a bare descriptor without the dynamic flag is not treated as a resize request.
2. The server answers with `AppleDisplayLayout` (`0x451`, §8.4) carrying the new `scaled_width/height` (logical) and `backing_width/height` (pixel) geometry.
3. The viewer **MUST re-offer the media session**: send a fresh `MediaStreamOptions` (`0x1c`, §10.3). This is the step that makes the server resize the encoder canvas — **without the `0x1c` re-offer the server stops emitting media** after the `0x451` and the stream stalls. The re-offer MAY reuse the existing SRTP key blobs (an interoperable client does so and the server accepts it) or carry fresh key blobs (native Screen Sharing.app generates new `session_id`, `CallID`, and per-stream keys on every round).
4. The server restarts the HEVC encoder at the new canvas: a **new four-SSRC group**, a **new in-band VPS/SPS/PPS** sized to the new backing geometry, and a fresh IDR. The change is carried by *both* a `0x1c` re-offer **and** new in-band parameter sets.

**Receiver obligations.** A viewer MUST size its framebuffer / decode surfaces to the **backing** geometry from `0x451` (the decoded picture dimensions; e.g. `3144×1792`), while using the **scaled** geometry for window sizing and input mapping (e.g. `1572×896`, a 2× HiDPI ratio). On the SSRC switch the viewer MUST adopt the new group, harvest the new parameter sets from the new stream's burst (the prior set is stale and would reject every frame), and restart its decoder with them.

**Backing cap.** The server caps the backing canvas at a host-dependent ceiling (observed `3840×2160` on the test panel); a request whose backing would exceed it is answered with a server-chosen fallback geometry rather than the requested size, so a conforming viewer treats the `0x451` / `0x1c`-answer geometry as authoritative rather than assuming its request was honoured. (Observed on one host panel.)

Residual gaps: the full `0x1c` answer schema (§10.3) and the exact server key-lifecycle expectation across rounds (reuse vs. rotate) remain **revision gaps**.

## 11. Conformance

Conformance is expressed as three cumulative profiles. Each requirement has a stable ID, a normative level, the profile(s) it applies to, and a test method. "Test method" names how an implementation demonstrates the requirement: **capture-replay** (verify against a recorded session), **decrypt-verify** (re-derive keys and verify the record layer/integrity), **decode** (decode payload to valid output), **interop** (sustain a live session against a reference server), **negative** (induce the error and confirm the required failure behavior).

### 11.1 Profile A — Minimal TCP Framebuffer Client

A Profile A client authenticates with type 33, activates the record layer, configures a virtual display, and sustains a framebuffer-backed session.

| ID | Requirement | Level | Profile | Test method |
|---|---|---|---|---|
| R-A1 | Speak `RFB 003.889`; close if unable | MUST | A,B,C | capture-replay / negative |
| R-A2 | Parse the security-type list; close on a failure list (`count = 0`) | MUST | A,B,C | negative |
| R-A3 | Implement type-33 (RSA1/RSA-SRP) authentication per §4.2.4 | MUST | A,B,C | interop |
| R-A4 | Read `iterations` from the SRP challenge as account-specific; never assume a fixed value | MUST | A,B,C | capture-replay |
| R-A5 | Treat the record layer as AES-128-CBC and ignore the advertised `conf+int=ChaCha20-Poly1305` token | MUST | A,B,C | decrypt-verify |
| R-A6 | Send ClientInit (`0xC1`) after `SecurityResult = 0` and before ServerInit | MUST | A,B,C | capture-replay |
| R-A7 | Emit the cleartext prelude (ViewerInfo, SetEncryption×2; SetMode OPTIONAL — §5.7) and tolerate the rekey arriving mid-prelude | MUST | A,B,C | capture-replay |
| R-A8 | AES-128-ECB-unwrap the `0x44f` key/IV under the current wrap key; install as the CBC content key/IV | MUST | A,B,C | decrypt-verify |
| R-A9 | Maintain one persistent CBC context per direction; never reset between records | MUST | A,B,C | decrypt-verify |
| R-A10 | Verify the SHA-1 integrity trailer on every record; close on mismatch | MUST | A,B,C | decrypt-verify / negative |
| R-A11 | Maintain non-resetting per-direction sequence counters | MUST | A,B,C | decrypt-verify |
| R-A12 | Treat `body_len` as the exact body length; reassemble large server payloads by record concatenation; assume no intra-record multiplexing | MUST | A,B,C | capture-replay |
| R-A13 | Emit encrypted `SetDisplayConfiguration` then `SetEncodings` as the encrypted preface | MUST | A,B,C | capture-replay |
| R-A14 | Compute and validate `SetDisplayConfiguration` `message_size` as total minus the 4-byte prefix | MUST | A,B,C | capture-replay |
| R-A15 | Act on `AppleDisplayLayout` (`0x451`) for framebuffer sizing, including mid-session reductions | MUST | A,B,C | capture-replay |
| R-A16 | Parse the metadata burst (`0x450/0x451/0x453/0x455/0x456`) and not disconnect on unknown values; apply the per-message receiver behavior of §8.13 | MUST | A,B,C | capture-replay |
| R-A16a | Maintain the `0x450` cursor cache (STORE on `compressed_len > 0`, SELECT on `= 0`) and render the selected shape at the locally-tracked pointer position (§8.3) | MUST | A,B,C | capture-replay |
| R-A16b | Arm the server framebuffer sender with `AutoFrameBufferUpdate` (`0x09`) at session start, and re-send `0x09` + a non-incremental `0x03` on **every** `AppleDisplayLayout` (`0x451`) to keep cursor SELECTs flowing across login/lock/agent transitions (§8.11, §8.4) | MUST | A,B,C | capture-replay |
| R-A17 | Set `SetDisplayConfiguration` `+0x96` (`rotations`) to the desired display rotation — normally `0` | SHOULD | A,B,C | decrypt-verify |
| R-A18 | Advertise only encodings it can decode or safely handle | MUST | A,B,C | interop |

### 11.2 Profile B — Apple Codec Client

Profile A plus Apple framebuffer still-image codecs.

| ID | Requirement | Level | Profile | Test method |
|---|---|---|---|---|
| R-B1 | Decode `0x3ea` (High Quality) and/or `0x3f3` (Multi-Variant Scaled) if advertised | MUST | B,C | decode |
| R-B2 | Decode `0x3f3` per §8.9.4 (process `type_of_update = 2` quant-table uploads before DCT tiles) | MUST (if `0x3f3` advertised) | B,C | decode |
| R-B3 | Decode `0x06` zlib rectangles, reassembling payloads spanning consecutive records | MUST | B,C | decode |
| R-B4 | MUST NOT advertise a codec encoding it cannot decode | MUST | B,C | interop |

### 11.3 Profile C — Adaptive Media Client

Profile A plus the Adaptive media transport.

| ID | Requirement | Level | Profile | Test method |
|---|---|---|---|---|
| R-C1 | Advertise `0x3f2` to request media-init; treat it as signaling, not a content switch | MUST | C | capture-replay |
| R-C2 | Send a `0x1c` MediaStreamOptions offer and process the answer per §10.3 | MUST | C | interop |
| R-C3 | Derive SRTP keys from the 46-byte blobs (32-byte key + 14-byte salt) and the RFC 3711 KDF; use `key2` for receive | MUST | C | decrypt-verify |
| R-C4 | Decrypt SRTP with AES-256-CTR + HMAC-SHA1-80, tracking ROC per SSRC | MUST | C | decrypt-verify |
| R-C5 | Depayload HEVC per the Apple RFC 7798 variant (DONL on single/AP/FU) | MUST | C | decode |
| R-C6 | Feed all four tile SSRCs to a single HEVC decoder; treat any IDR as a DPB reset; composite tiles vertically | MUST | C | decode |
| R-C7 | Decode HEVC RExt 4:4:4 8-bit | MUST | C | decode |
| R-C8 | Emit RTCP receiver feedback and a keyframe request (legacy FIR PT 192 or AVPF FIR) on loss | SHOULD | C | interop |
| R-C9 | Emit a media keepalive | MAY | C | interop |
| R-C10 | On media negotiation failure or sustained decode failure, fall back to the framebuffer path (§12) | MUST | C | negative |

## 12. Error Handling and Fallback

A general principle: failures SHOULD close the connection (or, for media, fall back) **without** emitting detailed on-wire diagnostics, to avoid error- and padding-oracle exposure (§13.6).

| Condition | Required behavior |
|---|---|
| Server security-type list is a failure list (`count = 0`) | MUST close |
| No advertised security type is supported | MUST close |
| Malformed authentication message / branch error | MUST close; MUST NOT emit a diagnostic distinguishing failure causes |
| `SecurityResult ≠ 0` | MUST close |
| Rekey (`0x44f`) body fails to AES-128-ECB-unwrap to a usable key/IV | MUST close |
| Record `ciphertext_len` is zero or not a multiple of 16 | MUST close |
| Record-layer integrity-trailer mismatch | MUST close; MUST NOT continue to process the record |
| Unknown client→server message type (server side) / unknown server message type (client side) | MUST ignore the message and continue; MUST NOT close solely for unknown type |
| Unknown or reserved field within a known message | MUST ignore the field's value per its rule (emit-zero / preserve / ignore-on-receive) |
| Unknown advertised encoding | MUST NOT advertise it; if received unexpectedly, MUST tolerate (skip the rectangle) without closing |
| Codec advertised but undecodable | MUST NOT advertise it (R-A18/R-B4) |
| Media negotiation (`0x1c`) failure or timeout | Profile C: SHOULD abandon the media path and continue on the framebuffer path |
| Sustained media decode failure (no recoverable reference) | Profile C: SHOULD request a keyframe (R-C8); on persistent failure SHOULD fall back to the framebuffer path |
| SRTP authentication failure / replayed packet | MUST discard the packet; MUST NOT close the media transport for a single failure |

## 13. Security Considerations

### 13.1 Type 30 (Diffie-Hellman) Weakness

Type 30 derives its key with **MD5** over a Diffie-Hellman shared secret and encrypts credentials with **AES-128-CBC under a zero IV** over a fixed-layout plaintext (§4.2.3). MD5, a zero IV, and a fixed unauthenticated **1024-bit** group are legacy constructions: a zero IV over a constant-prefixed plaintext leaks equality of the leading block across sessions, and the group is small and server-fixed. Type 30 SHOULD be treated as a legacy/weak path and avoided when a stronger branch is offered.

### 13.2 Type 33 Server Public-Key Trust

Type 33 encrypts the identity envelope to an RSA public key (RSA1, §4.2.4). In the observed flow the server public key was not delivered on the wire (the client used a cached/pre-provisioned key); even when delivered via the DER response (a revision gap), the protocol provides no certificate chain. The RSA step therefore provides **passive confidentiality of the identity envelope to whoever holds the matching private key**, not server authentication. Without key pinning or trust-on-first-use (TOFU), a client cannot distinguish the genuine server from a party that substitutes its own RSA key. Implementations SHOULD pin or TOFU the server's RSA key and SHOULD treat an unexpected key change as a potential man-in-the-middle.

### 13.3 Authentication Downgrade

The server advertises multiple branches of differing strength (§4.1.3). The advertised list is unauthenticated until a branch completes, so an on-path attacker could strip stronger branches to force type 30. Clients SHOULD prefer the strongest offered branch (SRP branches 33/36 over 30) and SHOULD refuse to downgrade to type 30 when a stronger branch was previously seen for the same host.

### 13.4 Legacy Cryptographic Constructions

The record layer authenticates with **plain SHA-1** (not HMAC) over `u32_be(seq) || plaintext` (§6.4.5). SHA-1 is collision-weak; while the construction is keyed implicitly by the secret content key, implementations SHOULD treat it as integrity-but-not-collision-resistant and MUST close on any mismatch. The advertised `conf+int=ChaCha20-Poly1305` token is **not** the record-layer cipher (which is AES-128-CBC + SHA-1); implementations MUST NOT assume AEAD protection on the control channel. The server *does* build ChaCha20-Poly1305 cryptors, but only in the SASL/SRP security layer (§4.2.4.8), which does not protect the post-auth VNC record stream — so the AEAD token's presence must not be mistaken for AEAD on the control channel.

### 13.5 Fresh Randomness

All authentication branches require fresh per-session randomness: DH `a`, SRP `a` and `client_random`, RSA padding, and the SRTP master keys. Implementations MUST use a cryptographic RNG; reuse of `a`, salts, or SRTP keys across sessions breaks confidentiality and, for SRP, the proof of password possession.

### 13.6 Error- and Padding-Oracle Exposure

The record layer is CBC with a separate SHA-1 trailer (encrypt-and-MAC-like layout). To avoid padding/format oracles, a receiver MUST verify the integrity trailer and MUST close on any malformed record (`ciphertext_len`, body-length, or trailer failure) **without** signaling the specific cause on the wire (§12). The filler bytes are not authenticated structure and MUST NOT be validated (§6.4.2).

### 13.7 SRTP Key Distribution and Channel Separation

The Adaptive media keys are 46-byte SRTP blobs distributed **in-band inside the AES-128-CBC control channel** (§10.3). Media-channel confidentiality therefore depends entirely on control-channel secrecy: an observer who recovers the control-channel keys also recovers the SRTP blobs and can decrypt the media. SRTP here protects the media against on-path attackers who do not hold the control-channel keys; it is **not** end-to-end protection independent of the control channel. Control-channel security (AES-128-CBC + SHA-1, §6) and media-channel security (AES-256-CTR + HMAC-SHA1-80, §10.5) are distinct mechanisms with distinct keys and distinct threat coverage; an implementation MUST NOT treat one as implying the other.

## 14. IANA Considerations

This document has no IANA actions.

## 15. References

### 15.1 Normative References

- [RFB] T. Richardson, J. Levine, "The Remote Framebuffer Protocol", RFC 6143.
- [BCP14-1] S. Bradner, "Key words for use in RFCs to Indicate Requirement Levels", RFC 2119.
- [BCP14-2] B. Leiba, "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words", RFC 8174.
- [SRP] D. Taylor et al., "Using the Secure Remote Password (SRP) Protocol for TLS Authentication", RFC 5054.
- [SRTP] M. Baugher et al., "The Secure Real-time Transport Protocol (SRTP)", RFC 3711.
- [HEVC-RTP] Y.-K. Wang et al., "RTP Payload Format for High Efficiency Video Coding (HEVC)", RFC 7798.
- [H265] ITU-T Recommendation H.265, "High efficiency video coding".
- [KRB-CFX] L. Zhu et al., "The Kerberos Version 5 GSS-API Mechanism: Version 2", RFC 4121.
- [KRB-1964] J. Linn, "The Kerberos Version 5 GSS-API Mechanism", RFC 1964.
- [DEFLATE] P. Deutsch, "DEFLATE Compressed Data Format", RFC 1951; "ZLIB Compressed Data Format", RFC 1950.

### 15.2 Informative References

- [AVPF] J. Ott et al., "Extended RTP Profile for RTCP-Based Feedback (RTP/AVPF)", RFC 4585.
- [CCM] S. Wenger et al., "Codec Control Messages in the RTP/AVPF", RFC 5104.
- [H261-RTP] T. Turletti, C. Huitema, "RTP Payload Format for H.261 Video Streams", RFC 2032 (legacy FIR, PT 192).
- [CHACHA] Y. Nir, A. Langley, "ChaCha20 and Poly1305 for IETF Protocols", RFC 8439 (advertised in the SRP option string but not used by the control record layer).

## Appendix A. Encoding Registry

| Value | Name | Class |
|---|---|---|
| `0x06` | Standard zlib | inherited from RFB |
| `0x3e8` | Low Quality codec | Apple codec |
| `0x3e9` | Medium Quality codec | Apple codec |
| `0x3ea` | High Quality codec | Apple codec |
| `0x3f2` | RFBMediaStreamMessage1 | media-init metadata |
| `0x3f3` | Multi-Variant Scaled | Apple codec, per-tile adaptive |
| `0x44f` | EncodeEncryptionInfo | rekey |
| `0x450` | CursorImage | cursor metadata |
| `0x451` | AppleDisplayLayout | display metadata |
| `0x453` | VendorKeysymEncoding | keyboard capability metadata |
| `0x455` | KeyboardInputSource | keyboard metadata |
| `0x456` | DeviceInfo | device metadata |

## Appendix B. Client URL Conventions (Informative)

This appendix is informative. URL parameters are **not** on-wire protocol; they are client-side configuration. Several deterministically affect on-wire behavior (e.g. a `quality` parameter selects the §8.9.2 tier advertised in `SetEncodings`); others affect only local policy. The `quality`→tier mapping is observed on the wire; other parameter effects are read from client configuration, and value enumeration is a **revision gap**.

### B.1 Parameter Registry

| Parameter | On-wire effect | Notes |
|---|---|---|
| `quality` | Selects the `SetEncodings` tier (§8.9.2) | |
| `control` | `SetModeMessage` mode (§5.7) | |
| `numVirtualDisplays` | `display_count` (§7.1) | |
| `displayID` | `SetDisplayMessage` combine/target (§7.4) | |
| others (`encrypt`, `auth`, `hdr`, `panning`, …) | local policy; value enumeration | **revision gap** |

### B.2 Quality Mapping

`quality` maps to the §8.9.2 tier set; the High / media-init tier advertises `0x3f2, 0x3f3, 0x3ea, zlib, zrle`.

### B.3 Conformance Note

URL conventions are not required for interoperability and are out of scope for §11.

## Appendix C. Test Vectors

This appendix defines test-vector slots for implementers. Vectors that depend on a specific account's secret material are left as TODO; supply them from a controlled test account rather than a production credential. Do not fabricate values.

### C.1 Record-Layer Rekey Vector

Demonstrates AES-128-ECB unwrap of the `0x44f` body and CBC activation.

```text
INPUT:
  wrap_key        : 16 bytes              TODO: from a controlled session (SHA-256(K)[0:16])
  rekey_body      : 36 bytes              TODO: u32 generation || 16B enc_key || 16B enc_iv
EXPECTED:
  content_key     = AES128-ECB-dec(wrap_key, enc_key)   TODO
  content_iv      = AES128-ECB-dec(wrap_key, enc_iv)    TODO
PROCEDURE: ECB-decrypt each 16-byte half independently; install as the CBC key/IV (both directions).
```

### C.2 First Encrypted Record Vector

```text
INPUT:
  content_key, content_iv : from C.1     TODO
  record0_ciphertext      : u16 len || ciphertext   TODO
EXPECTED:
  plaintext   = AES128-CBC-dec(content_key, content_iv, ciphertext)
  body_len    = u16(plaintext[0:2])
  body        = plaintext[2:2+body_len]              EXPECT: SetDisplayConfiguration (0x1d)
  integrity   = plaintext[len-20:len]
  CHECK       : SHA1( u32_be(0) || plaintext[0:len-20] ) == integrity
```

### C.3 Type-33 SRP Vector

```text
INPUT (non-secret, from challenge):
  N (4096-bit MODP), g=5, salt (32B), iterations (u64)   TODO
SECRET (controlled account):
  password                                               TODO
EXPECTED:
  P'  = PBKDF2-HMAC-SHA512(password, salt, iterations, 128)   TODO
  x   = H(salt || H(":" || P'))                               TODO
  ... A, M1, K, M2 per §4.2.4.6                               TODO
```

### C.4 SetDisplayConfiguration Parser Vector

```text
INPUT:  a full 0x1d message (308 bytes for display_count=1, mode_count=5)   TODO
EXPECT: message_size = 304; descriptor display_info_size = 296;
        mode_table[0] = { width=3840, height=2160, scaled=1920x1080, refresh=60.0, flags=0 };
        reserved(+0x96) = 7.
```

### C.5 Vendor Metadata Parser Vectors

```text
0x453 VendorKeysymEncoding: 22-byte payload
  EXPECT: header_count=0x0014, header_version=0x0001, value_count=4,
          keysyms = {0x1008FD00, 0x1008FD01, 0x1008FD02, 0x1008FD03}
0x455 KeyboardInputSource:
  EXPECT: version_marker=0x0001; prefix_length=S+8; total payload = S+10
0x456 DeviceInfo:
  EXPECT: block_pair_count=2; structure_version=1; a u32 enclosure-RGB-color word (0 in practice);
          message_size = info_block_size + 0x10; three NUL-terminated strings; u32 housing_color
0x451 AppleDisplayLayout:
  EXPECT: u16 payload_len; u16 version=5; u32 current_display (-1 sentinel); u32 flags;
          then 0x38-byte per-display records (scale f64 x2, display_id, global+scaled bounds, flags, pixel format)
```

### C.6 SRTP Media Vector

```text
INPUT:
  key_blob (46B) = 32B master_key || 14B master_salt   TODO: from 0x1c video key2
  srtp_packet                                           TODO
EXPECT:
  HMAC-SHA1-80 over (packet || u32_be(ROC)) == trailing 10 bytes;
  AES-256-CTR keystream IV = (salt<<16) XOR (SSRC<<64) XOR (index<<16);
  decrypted payload begins a valid HEVC NAL structure (single / AP-48 / FU-49 with DONL).
```

## Change Log

Derived from packet captures, runtime traces, and static analysis of `screensharingd`, `ScreensharingAgent`, and `AppleVNCServer` (baseline a recent macOS build). Fields not yet established are marked as a revision gap or as unspecified.

### Remaining Revision Gaps

- **Type 30/35/36 live captures.** The transcripts and key-derivations are established by static analysis (§4.2.3/§4.2.5/§4.2.6); only on-wire captures of those branches are still absent (the test host offers only type 33).
- **Wrap-key rotation across ≥2 rekeys** and sequence-counter non-reset across a second rekey — the per-rekey install path is established (§6.2.1); multi-rekey behavior is unexercised (needs a ≥2-rekey session).
- **`0x1c` per-stream offer-blob internals** (handed to AVConference) and the server *answer* layout — top-level offer framing (including the `+0x06` `flags` `u32` and the `+0x0a`/`+0x0c`/`+0x0e` offer-length words) is recovered (§10.3). The audio capability-bit-to-codec mapping remains unresolved (§10.3); the earlier field-4 bitrate interpretation is withdrawn. The full offer/answer semantics remain incomplete.
- **`0x3f3` 3-bit command-code → meaning mapping** — all other MVS framing is recovered (§8.9.4).
- `0x3ea` rectangle body beyond pre-processing (§8.9.3).
- **ViewerInfo bits 30/31/32/35/81** (§5.5) — viewer-internal: tested neither in `screensharingd` nor `ScreensharingAgent` (the Agent receives digested flags, not the raw bitmap).
- **`0x14` MiscStatus `cmd` space** (a live `cmd=4` is unexplained, §8.2.1); **RTCP APP/LTR** ack semantics + server reference-selection on loss (§10.8); **§10.9** dynamic-resolution renegotiation specifics (needs a resize harness); `generation` counter increment semantics (§6.2); and URL-parameter value enumeration (Appendix B).
