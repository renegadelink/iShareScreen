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
import signal
import sys
import time
from typing import Optional

from . import __version__
from .frontend.connect_prompt import ConnectChoice, UserCancelled, prompt as connect_prompt
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
        "--auth", choices=("srp", "nonsrp"), default="srp",
        help="authentication mode (default srp; falls back to nonsrp on rejection)",
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
        datefmt="%H:%M:%S",
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
    """Build a SessionConfig from CLI flags, prompting for anything missing.

    Smoke-test (`--headless`) refuses to prompt -- it's meant to run
    unattended in CI / scripted captures, so we bail loudly instead of
    blocking on user input. For interactive runs the connect prompt
    fills in whatever the CLI didn't supply.
    """
    cli_password = _password_from_args(args)
    cli_advertise = _parse_advertise(args.advertise)

    if args.headless:
        # Unattended path: no terminal prompt; everything must come
        # from flags or be reported as a config error.
        missing = [
            label for label, value in
            (("--host", args.host), ("-u/--user", args.user),
             ("--password-stdin", cli_password))
            if not value
        ]
        if missing:
            raise SystemExit(
                f"--headless requires {', '.join(missing)}"
            )
        return SessionConfig(
            host=args.host, port=args.port,
            username=args.user, password=cli_password or "",
            auth_mode=args.auth,
            advertise=cli_advertise,
            hdr=args.hdr,
            curtain=args.curtain,
            audio=args.audio,
            share_console=args.share_console, alt_session=args.alt_session,
        )

    prefill = ConnectChoice(
        host=args.host or "",
        username=args.user or "",
        password=cli_password or "",
        advertise=cli_advertise,
    )
    try:
        choice = connect_prompt(prefill)
    except UserCancelled:
        # User cancelled the prompt -- treat as a clean exit, not an
        # error. Returning a non-zero code would surprise scripts that
        # simply re-launch with new args.
        raise SystemExit(0)

    # Explicit --advertise from the CLI always wins over the prompt's pick;
    # anything else, take what the prompt returned.
    advertise = cli_advertise if cli_advertise is not None else choice.advertise

    return SessionConfig(
        host=choice.host,
        port=args.port,
        username=choice.username,
        password=choice.password,
        auth_mode=args.auth,
        advertise=advertise,
        hdr=args.hdr,
        curtain=args.curtain,
        audio=args.audio,
        share_console=args.share_console,
        alt_session=args.alt_session,
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
                log.info("frames so far: %s", frames_per_tile)
                last_report = now

        total = sum(frames_per_tile)
        log.info("smoke complete: %d frames in %ds %s", total, duration, frames_per_tile)
        return 0 if total > 0 else 2


def _run_frontend(config: SessionConfig, args: argparse.Namespace) -> int:
    """Open the desktop viewer."""
    from isharescreen.frontend.desktop.app import run as run_desktop
    return run_desktop(config, auto_quit_secs=args.auto_quit_secs)


# ── entry point ──────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    args = _make_parser().parse_args(argv)
    _setup_logging(args)
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
        log.error("fatal: %s", e)
        if args.verbose:
            log.exception("traceback:")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
