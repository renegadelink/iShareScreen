"""`iss` command-line entry point.

Modes:

  ``iss [flags]``
      Default. Opens the native desktop viewer (wgpu + glfw) and streams
      the host Mac's screen at lowest latency, hardware HEVC RExt 4:4:4
      decode where the platform supports it.

  ``iss --headless [flags]``
      Protocol smoke test: connect, decode for ``--auto-quit-secs`` seconds,
      print stats, exit. No window. Useful for verifying the decoder
      against a host without UI plumbing.

Password input, in order of precedence:
  ``--password-stdin`` (read the first line of stdin — the
  recommended unattended path: ``echo "$PW" | iss ... --password-stdin``).
  Otherwise the connect prompt asks for it interactively via getpass.

There is no ``-p`` / ``--password`` flag on purpose — anything in argv
shows up in ``ps`` and shell history.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from typing import Optional

from . import __version__
from .proxy.protocol.negotiation import AdvertiseDims
from .proxy.session import Session, SessionConfig


log = logging.getLogger("iss.cli")


# ── argparse construction ────────────────────────────────────────────

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="iss",
        description=(
            "Cross-platform client for Apple's macOS Screen Sharing "
            "High Performance mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  iss                                       # interactive prompt\n"
            "  iss --host mac.lan -u user                # asks for password\n"
            "  echo \"$PW\" | iss --host mac.lan -u user --password-stdin\n"
            "  iss --host mac.lan -u user --password-stdin < pw.txt\n"
            "  iss --host mac.lan -u user --headless --auto-quit-secs 30\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"iss {__version__}")

    g = p.add_argument_group("connection")
    g.add_argument("--host", required=False, help="hostname or IP of the Mac")
    g.add_argument("-u", "--user", required=False, help="username")
    # No `--password PASSWORD` flag on purpose: anything passed on the
    # command line shows up in `ps` and gets recorded in shell history.
    # For unattended use, pipe via --password-stdin
    # (`echo "$PW" | iss ... --password-stdin`); for interactive use,
    # let the connect prompt handle it via getpass.
    g.add_argument(
        "--password-stdin", action="store_true",
        help="read password from the first line of stdin",
    )
    g.add_argument("--port", type=int, default=5900, help="TCP port (default 5900)")
    g.add_argument(
        "--codec", choices=("auto", "hevc", "avc"), default="auto",
        help=(
            "video codec to negotiate with the host. 'auto' (default) probes "
            "whether this GPU can hardware-decode HEVC 4:4:4 and uses it if so, "
            "otherwise falls back to H.264 4:2:0. 'hevc' = force Apple HEVC RExt "
            "4:4:4 (best quality; slow CPU fallback on GPUs without 4:4:4 HW "
            "decode). 'avc' = force H.264 High 4:2:0 (lower quality, but "
            "hardware-decodes on virtually any GPU: D3D11VA / VAAPI / "
            "VideoToolbox)."
        ),
    )
    g.add_argument(
        "--decoder", metavar="NAME", default="auto",
        help=(
            "video decoder to use. 'auto' (default) picks the best available "
            "decoder for the negotiated codec; or force one by name — "
            "vt-hevc444 / libav-hevc444 / qsv-hevc444 / libav-avc420 (legacy "
            "aliases vt / qsv / libav also accepted). Run --list-decoders to "
            "see the matrix and which are available on this machine."
        ),
    )
    g.add_argument(
        "--list-decoders", action="store_true",
        help=("print the decoder capability matrix (with live availability on "
              "this machine) and exit"),
    )
    g.add_argument(
        "--auth", choices=("srp", "nonsrp"), default="srp",
        help="authentication mode (default srp; falls back to nonsrp on rejection)",
    )
    g.add_argument(
        "--frontend", choices=("browser", "desktop"), default="browser",
        help="browser = WebTransport+WebCodecs in a browser tab, H.264 "
        "pass-through (default); desktop = native wgpu/Qt window",
    )
    g.add_argument(
        "--bridge-port", type=int, default=4433,
        help="browser frontend: WebTransport/HTTP3 listen port (default 4433)",
    )

    g = p.add_argument_group("display")
    g.add_argument(
        "--advertise", metavar="WxH[@HIDPI]", default=None,
        help=(
            "virtual display geometry advertised to the host "
            "(e.g. '2560x1440' or '1920x1200@2'). When omitted, the "
            "interactive prompt asks for a resolution preset "
            "(default 1920x1200)."
        ),
    )
    g.add_argument(
        "--hdr", action="store_true",
        help="advertise HDR-capable viewer to the host",
    )
    g.add_argument(
        "--curtain", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "blank the host's physical screen via a SkyLight virtual "
            "display while we view (default on; --no-curtain to mirror "
            "the physical display instead)"
        ),
    )
    g.add_argument(
        "--share-console", action="store_true",
        help=(
            "when authenticating as a non-console user, ask the currently "
            "logged-in user to share their session (Apple's 'Ask to share' "
            "choice). The console user gets a permission popup; on accept, "
            "this viewer joins their existing session in observe-only mode "
            "instead of starting a separate alt-user session."
        ),
    )
    g.add_argument(
        "--alt-session", action="store_true",
        help=(
            "when authenticating as a non-console user, log them in to a "
            "fresh virtual display via Apple's cmd=2 SessionSelect path. "
            "No popup. Daemon spawns the alt-user's vdisplay and we get "
            "their desktop. Mutually exclusive with --share-console."
        ),
    )

    g = p.add_argument_group("audio")
    g.add_argument(
        "--audio", action=argparse.BooleanOptionalAction, default=True,
        help=(
            "play the host's audio through the local sound device "
            "(default on; --no-audio for video-only)"
        ),
    )

    g = p.add_argument_group("mode")
    g.add_argument(
        "--headless", action="store_true",
        help="run a protocol smoke test instead of opening the viewer window",
    )
    g = p.add_argument_group("logging")
    g.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")
    g.add_argument("-q", "--quiet", action="store_true", help="warnings + errors only")
    g.add_argument("--log-file", metavar="PATH", help="tee logs to a file")
    g.add_argument(
        "--control-socket", metavar="PATH", default=None,
        help=(
            "open a local control socket so a TUI / monitor can subscribe "
            "to live session stats. POSIX: UDS path; Windows: the real TCP "
            "port is written to '<PATH>.port'."
        ),
    )
    g.add_argument(
        "--record", metavar="PATH", default=None,
        help=(
            "capture the whole session to a libpcap file at PATH. Every byte "
            "on the TCP control socket and the two UDP media sockets is "
            "written exactly as it goes on the wire (still encrypted), with "
            "synthetic Ethernet/IP/TCP|UDP framing — byte-identical to a "
            "tcpdump capture. Open it in Wireshark or feed it to the Python "
            "dissector (the connect log prints the exact decode command)."
        ),
    )
    g.add_argument(
        "--auto-quit-secs", type=int, default=0,
        help=(
            "exit after N seconds. In --headless mode, this is the smoke-test "
            "duration (default 10s). For the viewer, 0 means run forever."
        ),
    )

    return p


# ── value parsing ────────────────────────────────────────────────────

def _parse_advertise(spec: Optional[str]) -> Optional[AdvertiseDims]:
    if not spec:
        return None
    geom_part, _, hidpi_part = spec.partition("@")
    try:
        w_str, h_str = geom_part.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
        hidpi = int(hidpi_part) if hidpi_part else 1
    except ValueError as e:
        raise SystemExit(
            f"invalid --advertise value {spec!r}: expected 'WxH' or 'WxH@HIDPI' ({e})"
        ) from e
    return AdvertiseDims(width=width, height=height, hidpi_scale=hidpi)


def _password_from_args(args: argparse.Namespace) -> Optional[str]:
    """Read --password-stdin, or return None so the connect prompt
    handles it interactively. We deliberately don't accept a password
    on the command line — argv shows up in `ps` and shell history."""
    if args.password_stdin:
        line = sys.stdin.readline()
        if not line:
            raise SystemExit("--password-stdin: no input on stdin")
        return line.rstrip("\r\n")
    return None


# ── logging setup ────────────────────────────────────────────────────

def _setup_logging(args: argparse.Namespace) -> None:
    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        h.setLevel(level)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers[:] = handlers


# ── config build ─────────────────────────────────────────────────────

def _build_session_config(args: argparse.Namespace) -> SessionConfig:
    """Build a SessionConfig from CLI flags. Requires every input to come
    from a flag (no stdin/terminal prompts) -- this entry point serves
    `iss --headless` (CI / scripted smoke) and the TUI's spawned child
    viewer, both of which always pass the full set of flags. Interactive
    use goes through `isharescreen.tui` instead."""
    cli_password = _password_from_args(args)
    cli_advertise = _parse_advertise(args.advertise)
    missing = [
        label for label, value in
        (("--host", args.host), ("-u/--user", args.user),
         ("--password-stdin", cli_password))
        if not value
    ]
    if missing:
        raise SystemExit(
            "missing required arg(s): " + ", ".join(missing) +
            " (interactive prompt is the TUI -- run plain `iss` for that)"
        )
    return SessionConfig(
        host=args.host, port=args.port,
        username=args.user, password=cli_password or "",
        auth_mode=args.auth,
        advertise=cli_advertise,
        hdr=args.hdr,
        video_codec=args.codec,
        curtain=args.curtain,
        audio=args.audio,
        share_console=args.share_console, alt_session=args.alt_session,
        control_socket=args.control_socket,
        record_pcap=args.record,
    )


# ── mode runners ─────────────────────────────────────────────────────

_DEFAULT_SMOKE_DURATION_S = 10


def _run_smoke(config: SessionConfig, args: argparse.Namespace) -> int:
    """Connect, decode for N seconds, report frame counts per tile."""
    duration = args.auto_quit_secs or _DEFAULT_SMOKE_DURATION_S
    log.info("headless smoke test against %s for %ds", config.host, duration)
    deadline = time.monotonic() + duration

    with Session(config) as session:
        log.info(
            "connected: server=%dx%d  canvas=%dx%d  hw=%s",
            *session.server_dims, *session.canvas_dims, session.hw_accel,
        )
        n_tiles = session.num_tiles
        frames_per_tile = [0] * n_tiles
        last_report = time.monotonic()

        while time.monotonic() < deadline:
            if not session.is_connected:
                log.error("session disconnected mid-smoke")
                return 1
            if session.wait_for_fresh_tile(timeout=0.5):
                for ti in range(n_tiles):
                    if session.get_frame(ti) is not None:
                        frames_per_tile[ti] += 1
            now = time.monotonic()
            if now - last_report > 2.0:
                log.debug("frames so far: %s", frames_per_tile)
                last_report = now

        total = sum(frames_per_tile)
        log.info("smoke complete: %d frames in %ds %s", total, duration, frames_per_tile)
        return 0 if total > 0 else 2


def _run_frontend(config: SessionConfig, args: argparse.Namespace) -> int:
    """Open the selected frontend. Default is the browser (WebTransport +
    WebCodecs, H.264 pass-through) since every machine has a browser; the
    native wgpu/desktop viewer stays available via --frontend desktop."""
    if args.frontend == "desktop":
        from isharescreen.frontend.desktop.app import run as run_desktop
        return run_desktop(config, auto_quit_secs=args.auto_quit_secs)
    # browser (default): H.264 pass-through needs the AVC codec path. The
    # Session reads ISS_VIDEO_CODEC at construction, so force it unless the
    # user explicitly chose a codec.
    if args.codec == "auto":
        os.environ["ISS_VIDEO_CODEC"] = "avc"
    # Use the cursor pseudo-encoding (RFB enc 1104), same as the wgpu viewer:
    # the daemon does NOT bake the cursor into the framebuffer, it sends cursor
    # pixmaps which the browser paints as the canvas CSS cursor. This also means
    # moving the mouse doesn't dirty the framebuffer / wake the encoder.
    # ISS_LEGACY_CURSOR=1 reverts to the cursor-in-framebuffer path.
    # Request a SINGLE picture per frame (tilesPerFrame=1) instead of Apple's
    # default 4 tiles. Browser WebCodecs can't follow the cross-tile reference
    # structure of the 4-tile stream (drift/"fleas"); one stream decodes
    # cleanly with one decoder, no compositing.
    os.environ.setdefault("ISS_TILES_PER_FRAME", "1")
    from isharescreen.frontend.wt.server import run as run_browser
    return run_browser(config, port=args.bridge_port)


# ── entry point ──────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    args = _make_parser().parse_args(argv)
    _setup_logging(args)
    if args.list_decoders:
        from .proxy.media import registry
        print(registry.describe())
        return 0
    # `--decoder` feeds the registry override via the env var session.py reads.
    if getattr(args, "decoder", "auto") and args.decoder != "auto":
        os.environ["ISS_DECODER"] = args.decoder
    signal.signal(signal.SIGINT, signal.default_int_handler)

    # Surface tracebacks for any thread that crashes — without this,
    # an unhandled exception in a worker (decoder, pump, RX loop,
    # asyncio task) prints nothing and the process can appear to die
    # silently. Mostly diagnostic.
    import threading
    def _thread_excepthook(args):
        log.exception(
            "unhandled exception in thread %s — process state may be compromised",
            args.thread.name if args.thread else "<unknown>",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _thread_excepthook

    try:
        config = _build_session_config(args)
    except SystemExit:
        raise
    except Exception as e:
        log.error("config error: %s", e)
        if args.verbose:
            log.exception("traceback:")
        return 1

    try:
        return _run_smoke(config, args) if args.headless else _run_frontend(config, args)
    except KeyboardInterrupt:
        log.info("interrupted")
        return 130
    except Exception as e:
        # Bad credentials are a user error, not a bug — clean message,
        # no traceback even in --verbose, distinct exit code.
        from isharescreen.proxy.protocol.auth import AuthError
        if isinstance(e, AuthError):
            log.error("authentication failed — check username and password")
            return 2
        log.error("fatal: %s", e)
        if args.verbose:
            log.exception("traceback:")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
