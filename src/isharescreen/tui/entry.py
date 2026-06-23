"""`iss` console-script entry. Routes between the TUI (default) and the
script-friendly CLI (`--headless` / `--help` / `--version`).

CLI flags supplied alongside the bare `iss` invocation pre-fill the
connect form — `iss --host mac.local -u me --advertise 1920x1080` opens
the TUI with those values already filled in. Last-session storage fills
anything the CLI didn't touch; defaults fill the rest.
"""
from __future__ import annotations

import sys
from typing import Any, Optional


# Flags that route to `cli.py` instead of the TUI. Anything else goes
# to the TUI (with the typed values pre-filled into the connect form).
_CLI_ROUTING_FLAGS = {"--headless", "--help", "-h", "--version", "--list-decoders"}


def main() -> int:
    argv = sys.argv[1:]
    # `--frontend browser` runs the WebTransport bridge, which serves its own
    # browser UI and needs no terminal TUI — route it straight to the cli.
    # (The TUI wraps the native wgpu viewer only.)
    _browser = (
        "browser" in (argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--frontend")
    )
    if _browser or any(a in _CLI_ROUTING_FLAGS for a in argv):
        from isharescreen.cli import main as cli_main
        return cli_main()
    # TUI default. Parse argv with cli's parser so unrecognised flags
    # still raise the same helpful argparse error, and build an override
    # dict the App merges into the connect-form prefill.
    overrides = _build_cli_overrides(argv)
    from isharescreen.tui.app import main as tui_main
    return tui_main(overrides)


def _build_cli_overrides(argv: list[str]) -> Optional[dict[str, Any]]:
    """Parse `argv` with cli.py's argparse and return the just-typed
    fields as an override dict for the TUI's connect form. Returns None
    when argv is empty (TUI falls back to last-session / defaults only).

    Only fields the user actually mentioned end up in the dict —
    argparse defaults would otherwise overwrite saved preferences. For
    BooleanOptionalAction flags (`--audio`/`--no-audio`,
    `--curtain`/`--no-curtain`), either form counts as "user set"."""
    if not argv:
        return None
    from isharescreen.cli import _make_parser, _parse_advertise, _password_from_args

    args = _make_parser().parse_args(argv)
    typed = set(argv)
    overrides: dict[str, Any] = {}

    if args.host:
        overrides["host"] = args.host
    if args.user:
        overrides["user"] = args.user
    # --password-stdin lets a launcher pipe in a password while still
    # showing the TUI. Without it, the password field stays whatever the
    # last-session / default flow produced (i.e., empty).
    pwd = _password_from_args(args)
    if pwd:
        overrides["password"] = pwd
    if args.advertise:
        d = _parse_advertise(args.advertise)
        if d is not None:
            overrides["advertise"] = (
                f"{d.width}x{d.height}"
                + (f"@{d.hidpi_scale}" if d.hidpi_scale != 1 else "")
            )
    if "--audio" in typed or "--no-audio" in typed:
        overrides["audio"] = args.audio
    if "--curtain" in typed or "--no-curtain" in typed:
        overrides["curtain"] = args.curtain
    if "--hdr" in typed:
        overrides["hdr"] = args.hdr
    if "--share-console" in typed:
        overrides["share_console"] = args.share_console
    if "--alt-session" in typed:
        overrides["alt_session"] = args.alt_session
    # --verbose / --log-file are not connect-form fields; they live on
    # ViewerArgs so the supervisor can forward them to the worker.
    if args.verbose:
        overrides["verbose"] = args.verbose
    if args.log_file:
        overrides["log_file"] = args.log_file
    if "--codec" in typed:
        overrides["codec"] = args.codec
    return overrides or None


if __name__ == "__main__":
    raise SystemExit(main())
