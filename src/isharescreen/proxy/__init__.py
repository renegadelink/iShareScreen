"""The "proxy" — engine that brokers between the user-facing frontend
and Apple's screensharingd on the host Mac.

It owns the network connection, the SRP / SRTP / RFB wire protocol,
the HEVC + AAC-ELD decoders, the RTCP / FIR / NACK feedback path,
and the input forwarder. Frontends consume it as a service:
construct a `Session(SessionConfig(...))`, call `connect()`, poll
`get_frame(tile_idx)` for decoded video, and call
`session.input.pointer_event(...)` to send input back.

Layout:

    session         Session, SessionConfig — the public surface
    input           InputController — encrypted-RFB input forwarding
    media.*         HEVC, AAC-ELD, NAL-unit, tile-frame primitives
    protocol.*      wire protocol (auth, SRTP, RFB, RTCP, enc1103,
                    initial-burst parser, AVC offer build/parse,
                    Apple constants)
"""
