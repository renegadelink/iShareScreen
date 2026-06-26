"""The Textual App. Owns the supervisor + control client + screens.

Lifecycle:
    user lands on ConnectScreen
    -> on Connect message: build ViewerArgs, spawn child via supervisor,
       push SessionScreen, start ControlClient, route messages -> screen
    -> on Disconnect / Reconnect: stop supervisor + client, pop session
       screen (or respawn if Reconnect), reset
    -> on app exit: stop everything cleanly
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from typing import Any, Optional

from textual.app import App
from textual.binding import Binding

from .connect_screen import ConnectFormValues, ConnectScreen
from .control_client import ControlClient
from .session_screen import SessionScreen
from .storage import bug_snapshot_path, load_last, save_last
from .supervisor import Supervisor, ViewerArgs
from ..proxy.media.registry import all_specs as _all_decoder_specs

# decoder name → codec, built once from the registry so it stays in sync.
_DECODER_TO_CODEC: dict[str, str] = {s.name: s.codec for s in _all_decoder_specs()}


log = logging.getLogger("iss.tui.app")


class IssTuiApp(App):
    """Top-level Textual app."""

    TITLE = "iShareScreen"
    SUB_TITLE = "terminal control"

    CSS = ""  # screen-level styles live with each Screen.

    BINDINGS = [
        Binding("ctrl+c", "quit_app", "Quit", show=False, priority=True),
    ]

    def __init__(self, cli_overrides: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self._supervisor: Optional[Supervisor] = None
        self._ctrl: Optional[ControlClient] = None
        self._session_screen: Optional[SessionScreen] = None
        self._last_form: Optional[ConnectFormValues] = None
        # Fields the user supplied on the CLI (`iss --host x -u y ...`).
        # Split into: (a) form-prefill fields, applied on top of last-session
        # storage so launcher scripts work end-to-end; (b) viewer-only flags
        # like --verbose / --log-file that don't belong on the connect form
        # but should pass through to the worker subprocess.
        raw = dict(cli_overrides or {})
        self._cli_viewer_flags: dict[str, Any] = {}
        for k in ("verbose", "log_file", "codec"):
            if k in raw:
                self._cli_viewer_flags[k] = raw.pop(k)
        self._cli_overrides = raw

    # ── lifecycle ─────────────────────────────────────────────────

    async def on_mount(self) -> None:
        # Prefill priority (highest wins): CLI args > last-session > blank.
        # Last-session never persists the password (see storage.save_last),
        # so passwords only land here if --password-stdin was piped.
        last = load_last()
        if last is None and not self._cli_overrides:
            prefill = None
        else:
            base = last or ConnectFormValues(
                host="", user="", password="",
                advertise="auto",
                audio=True, curtain=True, hdr=False, hidpi="auto",
                share_console=False, alt_session=False,
            )
            prefill = dataclasses.replace(base, **self._cli_overrides)
        await self.push_screen(ConnectScreen(prefill=prefill))

    async def on_unmount(self) -> None:
        await self._teardown_session()

    # ── messages from ConnectScreen ───────────────────────────────

    async def on_connect_screen_connect(self, msg: ConnectScreen.Connect) -> None:
        await self._start_session(msg.values)

    # ── messages from SessionScreen ───────────────────────────────

    async def on_session_screen_disconnect(self, _msg: SessionScreen.Disconnect) -> None:
        await self._teardown_session()
        # Drop SessionScreen and replace the original ConnectScreen with
        # one pre-filled from the last submitted form (incl. password) so
        # the user can tweak + reconnect. `switch_screen` replaces the
        # now-top screen atomically; pop+push would briefly leave the
        # stack empty, which Textual rejects.
        await self.pop_screen()
        if self._last_form is not None:
            await self.switch_screen(ConnectScreen(self._last_form))

    async def on_session_screen_reconnect(self, _msg: SessionScreen.Reconnect) -> None:
        if self._last_form is None:
            return
        if self._session_screen is not None:
            self._session_screen.set_state("RECONNECTING")
            self._session_screen.append_log("info: reconnecting…")
        await self._teardown_session(pop_screen=False)
        await self._start_session(self._last_form, push_screen=False)

    async def on_session_screen_force_idr(self, _msg: SessionScreen.ForceIdr) -> None:
        if self._ctrl is not None:
            await self._ctrl.send_command("fir")
            if self._session_screen is not None:
                self._session_screen.append_log("info: sent force-IDR command")

    async def on_session_screen_bug_snapshot(self, msg: SessionScreen.BugSnapshot) -> None:
        """Dump scrubbed state + log tail to a timestamped file the user
        can safely attach to a public bug report. Safe to call repeatedly.

        Privacy: the user's host, username, and any IPv4 literals are
        redacted from BOTH the form and the log-tail. We keep enough
        signal to diagnose ("was it a hostname or an IPv4 literal?",
        "was the resolved host the same the whole session?") without
        leaking who the user is or what their network looks like."""
        path = bug_snapshot_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            host = self._last_form.host if self._last_form else ""
            user = self._last_form.user if self._last_form else ""
            # If we somehow snapshot before the user submitted the
            # connect form (shouldn't happen interactively, but defensive),
            # emit a marker so consumers know nothing useful is here
            # rather than failing on a missing key.
            if self._last_form is None:
                form_block: dict[str, Any] = {"_note": "no connect form yet"}
            else:
                form_block = {
                    # Literal host / user redacted; we keep just whether
                    # the host *looked* like a hostname vs. an IPv4 (the
                    # IPv6-resolution bug class hinges on this).
                    "host_is_ipv4_literal": _is_ipv4_literal(host),
                    "host_is_dot_local":    host.endswith(".local"),
                    "advertise":     self._last_form.advertise,
                    "audio":         self._last_form.audio,
                    "curtain":       self._last_form.curtain,
                    "hdr":           self._last_form.hdr,
                    "share_console": self._last_form.share_console,
                    "alt_session":   self._last_form.alt_session,
                }
            payload = {
                "form": form_block,
                "latest_snapshot": msg.latest_snapshot,
                "log_tail": [_redact(line, host=host, user=user) for line in msg.log_tail],
                "redactions": {
                    "host": "<HOST>",
                    "user": "<USER>",
                    "ipv4": "<IPV4>",
                    "note": "Host, username, and any IPv4 literals in form / log lines were replaced before write; safe to attach to a public issue.",
                },
            }
            path.write_text(json.dumps(payload, indent=2))
            if self._session_screen is not None:
                self._session_screen.append_log(f"info: wrote scrubbed bug snapshot -> {path}")
            self.notify(f"saved bug snapshot to {path}", timeout=6)
        except OSError as e:
            self.notify(f"bug snapshot failed: {e}", severity="error", timeout=6)

    # ── core ──────────────────────────────────────────────────────

    async def _start_session(
        self,
        form: ConnectFormValues,
        *,
        push_screen: bool = True,
    ) -> None:
        self._last_form = form
        # Persist the connect form (sans password) so next launch
        # prefills these values. Best-effort -- a save failure just
        # means no prefill next time.
        try:
            save_last(form)
        except Exception:
            log.exception("save_last failed")
        # Spawn the child.
        self._supervisor = Supervisor(
            on_stderr_line=self._handle_stderr,
            on_exit=self._handle_child_exit,
        )
        args = ViewerArgs(
            host=form.host,
            user=form.user,
            password=form.password,
            advertise=form.advertise,
            frontend=form.frontend,
            audio=form.audio,
            curtain=form.curtain,
            hdr=form.hdr,
            share_console=form.share_console,
            alt_session=form.alt_session,
            dynamic_resolution=form.dynamic_resolution,
            hidpi=form.hidpi,
            verbose=int(self._cli_viewer_flags.get("verbose", 0) or 0),
            log_file=self._cli_viewer_flags.get("log_file"),
            codec=(self._cli_viewer_flags.get("codec")
                   or _DECODER_TO_CODEC.get(form.decoder or "")),
            decoder=form.decoder if form.decoder and form.decoder != "auto" else None,
        )
        try:
            await self._supervisor.start(args)
        except Exception as e:
            self.notify(f"failed to spawn viewer: {e}", severity="error", timeout=6)
            self._supervisor = None
            return
        # Push the session screen (or reuse the existing one on reconnect).
        if push_screen:
            self._session_screen = SessionScreen()
            await self.push_screen(self._session_screen)
        elif self._session_screen is not None:
            self._session_screen.set_state("CONNECTING")
        if form.frontend == "browser":
            # The browser bridge serves a page for the user's OWN browser; it
            # publishes no control socket, so there are no live stats to
            # subscribe to. Its stderr (incl. the viewer URL + rx/decode info)
            # streams to the log panel.
            url = "http://localhost:4433/"  # cli's default --bridge-port
            # The session screen was just pushed and isn't mounted yet, so
            # touching its widgets now (set_state→#hdr, append_log→#log) raises
            # NoMatches. Defer to after the next refresh, when it's mounted.
            def _announce(screen: "SessionScreen" = self._session_screen) -> None:
                if screen is not None:
                    screen.set_state("BROWSER")
                    screen.append_log(f"info: browser frontend — opening {url}")
            self.call_after_refresh(_announce)
            self.notify(f"Browser frontend — opening {url}", timeout=10)
            if push_screen:
                # Pop open the user's default browser at the bridge URL so they
                # don't have to. Only on the initial connect (not reconnect,
                # where a tab is likely already open).
                asyncio.create_task(
                    self._open_browser(url), name="iss-tui-open-browser")
            return
        # Desktop: subscribe to the child's control socket as soon as it's up.
        sock_path = self._supervisor.control_socket
        assert sock_path is not None
        self._ctrl = ControlClient(sock_path, self._handle_ctrl_message)
        # Run the subscriber in the background — its start() blocks until
        # the socket binds.
        asyncio.create_task(self._safe_ctrl_start(), name="iss-tui-ctrl-start")

    async def _safe_ctrl_start(self) -> None:
        if self._ctrl is None:
            return
        try:
            await self._ctrl.start()
        except TimeoutError as e:
            if self._session_screen is not None:
                self._session_screen.append_log(f"warn: {e}")
        except Exception as e:
            log.exception("control client start failed")
            if self._session_screen is not None:
                self._session_screen.append_log(f"warn: control client failed: {e}")

    async def _open_browser(self, url: str) -> None:
        """Best-effort: open the user's default browser at the bridge URL after
        a short delay so the bridge has bound its port (avoids a too-early
        connection-refused). No-ops cleanly if there's no GUI browser (e.g. a
        headless/SSH session) — the URL is still in the log + notification."""
        import webbrowser
        await asyncio.sleep(2.5)
        try:
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, webbrowser.open, url)
            if not ok and self._session_screen is not None:
                self._session_screen.append_log(
                    f"info: couldn't auto-open a browser — open {url} yourself")
        except Exception:
            log.debug("auto-open browser failed", exc_info=True)

    async def _handle_stderr(self, line: str) -> None:
        if self._session_screen is not None:
            self._session_screen.append_log(line)

    async def _handle_child_exit(self, rc: int) -> None:
        if self._session_screen is not None:
            self._session_screen.append_log(f"info: viewer exited (rc={rc})")
            self._session_screen.set_state("DISCONNECTED")

    async def _handle_ctrl_message(self, msg: dict[str, Any]) -> None:
        if self._session_screen is None:
            return
        t = msg.get("type")
        if t == "hello":
            self._session_screen.apply_hello(msg.get("session", {}))
        elif t == "snapshot":
            self._session_screen.apply_snapshot(msg.get("data", {}))
        elif t == "event":
            kind = msg.get("kind", "?")
            extra = {k: v for k, v in msg.items() if k not in ("type", "ts", "kind")}
            self._session_screen.apply_event(kind, extra)

    async def _teardown_session(self, *, pop_screen: bool = False) -> None:
        if self._ctrl is not None:
            try:
                await self._ctrl.stop()
            except Exception:
                pass
            self._ctrl = None
        if self._supervisor is not None:
            try:
                await self._supervisor.stop()
            except Exception:
                pass
            self._supervisor = None

    async def action_quit_app(self) -> None:
        await self._teardown_session()
        self.exit()


async def amain(cli_overrides: Optional[dict[str, Any]] = None) -> int:
    app = IssTuiApp(cli_overrides=cli_overrides)
    await app.run_async()
    return 0


def main(cli_overrides: Optional[dict[str, Any]] = None) -> int:
    """Sync entry point — runs the Textual app. `cli_overrides` is the
    just-typed-flags dict produced by `tui.entry._build_cli_overrides`
    and gets merged into the connect-form prefill on mount."""
    return asyncio.run(amain(cli_overrides))


# ── PII redaction for bug-report snapshots ──────────────────────────

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _is_ipv4_literal(s: str) -> bool:
    """True iff `s` is exactly four dotted decimal octets (0-255). Used
    to record the *shape* of the host the user typed without persisting
    the literal value."""
    if not s:
        return False
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or not (0 <= int(p) <= 255):
            return False
    return True


def _redact(line: str, *, host: str, user: str) -> str:
    """Scrub host, username, and any IPv4 literal from one line.

    Order matters: the host substring replace runs first, so any IPv4
    appearing inside the host (rare) is already inside `<HOST>` and
    won't be double-redacted. Word boundaries on `user` so short names
    like `user` don't redact words like `username`. The user pass runs
    AFTER the host pass; the placeholder `<HOST>` has no `user`
    substring at a word boundary, so the two passes never collide."""
    if host:
        line = line.replace(host, "<HOST>")
    if user:
        line = re.sub(rf"\b{re.escape(user)}\b", "<USER>", line)
    line = _IPV4_RE.sub("<IPV4>", line)
    return line


__all__ = ["IssTuiApp", "main", "amain"]
