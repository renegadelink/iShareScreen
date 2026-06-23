"""Spawn and manage the child iss viewer process.

The TUI process owns the terminal; the actual wgpu desktop viewer + proxy
is a separate child process so the two libraries don't fight for the
main thread (GLFW on macOS requires Cocoa main thread; Textual likewise
expects the main thread of its own process). The child is invoked as
`python -m isharescreen.cli` with whatever CLI args the connect form
produced, plus an auto-generated `--control-socket` path the TUI then
subscribes to for live stats.

The supervisor captures the child's stderr line-by-line and pipes it to
a callback (the TUI's log widget). On stop/quit/reconnect it sends
SIGTERM and reaps.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Optional


log = logging.getLogger("iss.tui.sup")


def _viewer_control_socket_path() -> str:
    """Generate a unique control-socket path for this viewer instance.
    Lives in a per-user temp dir so multiple TUI sessions don't collide
    and so a crash leaves at most one stale file per session."""
    base = Path(tempfile.gettempdir()) / f"iss-tui-{os.getuid() if hasattr(os, 'getuid') else os.getpid()}"
    base.mkdir(parents=True, exist_ok=True)
    return str(base / f"viewer-{os.getpid()}-{int(time.time())}.sock")


@dataclass
class ViewerArgs:
    """The argv-equivalent values produced by the connect form. The
    supervisor turns this into a real argv for the child python."""
    host: str
    user: str
    password: str
    advertise: str         # e.g. "1920x1080" or "1920x1200@2"
    frontend: str = "desktop"   # "browser" | "desktop" — which viewer to launch
    audio: bool = True
    curtain: bool = True
    hdr: bool = False
    share_console: bool = False
    alt_session: bool = False
    dynamic_resolution: bool = False  # "Auto" resolution: re-render host on resize
    hidpi: str = "auto"               # "auto" | "on" | "off"
    port: int = 5900
    auth: str = "srp"
    verbose: int = 0           # -v count to forward to worker
    log_file: Optional[str] = None  # --log-file to forward to worker
    codec: Optional[str] = None     # --codec to forward to worker
    decoder: Optional[str] = None   # --decoder to forward to worker ("auto" → omit)
    extra_env: dict[str, str] = field(default_factory=dict)


class Supervisor:
    """Owns one child viewer process at a time. `start()` spawns; the
    child's stderr is streamed to `on_stderr_line` from a background
    asyncio task. `stop()` sends SIGTERM and waits; `is_running` reflects
    the child's liveness."""

    def __init__(
        self,
        on_stderr_line: Callable[[str], Awaitable[None]],
        on_exit: Callable[[int], Awaitable[None]],
    ) -> None:
        self._on_stderr_line = on_stderr_line
        self._on_exit = on_exit
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._wait_task: Optional[asyncio.Task] = None
        self._control_socket_path: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def control_socket(self) -> Optional[str]:
        return self._control_socket_path

    async def start(self, args: ViewerArgs) -> None:
        """Spawn the child viewer with the connect-form args. Returns once
        the process is launched; live stats start flowing later when the
        control socket binds (the TUI subscribes via ControlClient)."""
        if self.is_running:
            raise RuntimeError("a viewer is already running")
        sock = _viewer_control_socket_path()
        self._control_socket_path = sock
        argv = self._build_argv(args, sock)
        env = os.environ.copy()
        env.update(args.extra_env)
        # The viewer needs a TTY-less stdin/stdout so it doesn't fight the
        # parent's terminal. We feed the password via stdin (-- once --
        # then close), and discard the child's stdout (the TUI consumes
        # only stderr, where the logger writes).
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        # Push the password and close stdin so the child's
        # `--password-stdin` read returns.
        assert proc.stdin is not None
        try:
            proc.stdin.write((args.password + "\n").encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        try:
            proc.stdin.close()
        except OSError:
            pass
        self._proc = proc
        self._stderr_task = asyncio.create_task(
            self._pump_stderr(), name="iss-tui-sup-stderr",
        )
        self._wait_task = asyncio.create_task(
            self._wait_exit(), name="iss-tui-sup-wait",
        )
        log.info("viewer spawned pid=%d argv=%s", proc.pid, " ".join(argv))

    async def stop(self, *, timeout: float = 4.0) -> None:
        """Stop the child gracefully (SIGTERM → wait → SIGKILL fallback).
        Idempotent. Cleans up the control-socket file too."""
        proc = self._proc
        if proc is None:
            return
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("viewer didn't exit on SIGTERM; killing")
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
        for t in (self._stderr_task, self._wait_task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._stderr_task = None
        self._wait_task = None
        # Best-effort cleanup of the control socket file & its parent dir
        # if empty.
        if self._control_socket_path is not None:
            try:
                os.unlink(self._control_socket_path)
            except OSError:
                pass
            try:
                os.unlink(self._control_socket_path + ".port")
            except OSError:
                pass
        self._proc = None
        self._control_socket_path = None

    # ── internals ──────────────────────────────────────────────────

    def _build_argv(self, args: ViewerArgs, sock_path: str) -> list[str]:
        argv = [
            sys.executable, "-m", "isharescreen.cli",
            "--host", args.host,
            "-u", args.user,
            "--password-stdin",
            "--advertise", args.advertise,
            "--port", str(args.port),
            "--auth", args.auth,
            "--control-socket", sock_path,
            # Which viewer the connect button chose. Forced explicitly so the
            # cli's default doesn't override the user's pick. "desktop" is the
            # native wgpu child the TUI supervises; "browser" launches the
            # WebTransport bridge (the TUI shows its log incl. the viewer URL).
            "--frontend", args.frontend,
        ]
        argv += ["--audio"] if args.audio else ["--no-audio"]
        argv += ["--curtain"] if args.curtain else ["--no-curtain"]
        # "Auto" resolution → --dynamic (host re-renders to the viewer window);
        # a fixed WxH → --no-dynamic. HiDPI scale forwarded explicitly.
        argv += ["--dynamic"] if args.dynamic_resolution else ["--no-dynamic"]
        argv += ["--hidpi", args.hidpi]
        if args.hdr:
            argv.append("--hdr")
        if args.share_console:
            argv.append("--share-console")
        if args.alt_session:
            argv.append("--alt-session")
        # Forward --verbose / --log-file so the worker's session log lands
        # on disk; without this the worker only writes to stderr, which the
        # TUI captures into its in-memory log panel and not the user's
        # --log-file.
        argv += ["-v"] * max(0, args.verbose)
        if args.log_file:
            argv += ["--log-file", args.log_file]
        if args.codec:
            argv += ["--codec", args.codec]
        if args.decoder and args.decoder != "auto":
            argv += ["--decoder", args.decoder]
        return argv

    async def _pump_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                try:
                    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                except Exception:
                    continue
                try:
                    await self._on_stderr_line(text)
                except Exception:
                    log.exception("on_stderr_line handler raised")
        except asyncio.CancelledError:
            raise

    async def _wait_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            rc = await proc.wait()
        except asyncio.CancelledError:
            raise
        try:
            await self._on_exit(rc)
        except Exception:
            log.exception("on_exit handler raised")


# Re-export for convenience.
__all__ = ["Supervisor", "ViewerArgs"]
