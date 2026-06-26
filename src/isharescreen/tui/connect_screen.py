"""Pre-flight form. Collects host / user / password / resolution /
options and posts a `Connect` message when the user submits. Replaces the
old stdin/stdout `connect_prompt.py`.

Resolution presets mirror the ones the old prompt offered, so users with
muscle memory for "1920×1080 (FHD)" find the same picks here. The
default is the same too (`1600 × 900`); the form is laid out so Enter
on a freshly opened screen connects with sensible defaults the moment
the host field is filled in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import asyncio
import socket
from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Button, Footer, Header, Input, Label, Select, Static, Switch,
)


# How long after the last host-field keystroke before we run a reachability
# probe. Short enough to feel live, long enough to coalesce typing.
_PROBE_DEBOUNCE_S: float = 0.5
# TCP connect timeout for the probe. The Mac is on a LAN; >1 s means the
# host is genuinely unreachable.
_PROBE_TIMEOUT_S: float = 1.5


# (label, "WxH", optional hidpi). Keep in sync with the old
# `connect_prompt._RESOLUTION_PRESETS`.
_RESOLUTION_PRESETS: list[tuple[str, str]] = [
    ("Auto — track window size",  "auto"),
    ("3840 × 2160 (4K UHD)",   "3840x2160"),
    ("3440 × 1440 (UWQHD)",    "3440x1440"),
    ("2560 × 1600 (WQXGA)",    "2560x1600"),
    ("2560 × 1440 (QHD)",      "2560x1440"),
    ("1920 × 1200 (WUXGA)",    "1920x1200"),
    ("1920 × 1080 (FHD)",      "1920x1080"),
    ("1680 × 1050 (WSXGA+)",   "1680x1050"),
    ("1600 ×  900 (HD+)",      "1600x900"),
    ("1366 ×  768 (WXGA)",     "1366x768"),
    ("1280 ×  720 (HD)",       "1280x720"),
    ("1024 ×  768 (XGA)",      "1024x768"),
    (" 800 ×  600 (SVGA)",     "800x600"),
]
_DEFAULT_RESOLUTION = "1920x1080"  # native-FHD = the safest pick

_DECODER_PRESETS: list[tuple[str, str]] = [
    ("Auto",                 "auto"),
    ("HEVC — VideoToolbox",  "vt-hevc444"),
    ("HEVC — Generic HW",    "libav-hevc444"),
    ("HEVC — Intel QSV",     "qsv-hevc444"),
    ("HEVC — Software",      "libav-hevc444-sw"),
    ("H.264",                "libav-avc420"),
]
_DEFAULT_DECODER = "auto"


@dataclass(slots=True)
class ConnectFormValues:
    """Snapshot of the form, posted with the `Connect` message."""
    host: str
    user: str
    password: str
    advertise: str
    audio: bool
    curtain: bool
    hdr: bool
    share_console: bool
    alt_session: bool
    # Which viewer to launch: "browser" (WebTransport + WebCodecs in a browser
    # tab) or "desktop" (native wgpu window). Chosen by which Connect button the
    # user presses; persisted so reconnect reuses the same one.
    frontend: str = "browser"
    # Dynamic resolution: re-render the host to match the viewer window on
    # resize (the "Auto" resolution pick). When True, `advertise` is only the
    # initial size. HiDPI scale: "auto" (match local display) / "on" / "off".
    dynamic_resolution: bool = False
    hidpi: str = "auto"
    decoder: str = "auto"  # decoder name or "auto"


class ConnectScreen(Screen):
    """The pre-flight form. Posts a `Connect` message on submit."""

    CSS = """
    ConnectScreen {
        align: center middle;
    }
    #form-card {
        width: 70;
        height: auto;
        max-height: 100%;
        overflow-y: auto;
        padding: 1 2;
        background: $panel;
        border: round $primary;
    }
    #title { content-align: center middle; padding-bottom: 1; }
    Label { padding-top: 1; }
    Input, Select { width: 100%; }
    #host-row { height: auto; }
    #host { width: 1fr; }
    #host-probe { width: 14; padding-left: 1; content-align: left middle; }
    .switch-row { padding: 1 0 0 0; height: auto; }
    .switch-row Switch { margin-right: 2; }
    .switch-row Label { padding: 0 1 0 0; }
    #frontend-hint { padding-top: 1; text-align: right; }
    #buttons { padding-top: 0; height: auto; align-horizontal: right; }
    Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
    ]

    class Connect(Message):
        """Posted when the user submits the form."""

        def __init__(self, values: ConnectFormValues) -> None:
            self.values = values
            super().__init__()

    def __init__(self, prefill: Optional[ConnectFormValues] = None) -> None:
        super().__init__()
        self._prefill = prefill or ConnectFormValues(
            host="", user="", password="",
            advertise=_DEFAULT_RESOLUTION,
            audio=True, curtain=True, hdr=False,
            share_console=False, alt_session=False,
            decoder=_DEFAULT_DECODER,
        )
        self._probe_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Footer()
        with Container(id="form-card"):
            yield Static("[b]iShareScreen[/b] — connect", id="title")
            yield Label("Host")
            with Horizontal(id="host-row"):
                yield Input(value=self._prefill.host, placeholder="Mac hostname or IP", id="host")
                yield Static("", id="host-probe")
            yield Label("Username")
            yield Input(value=self._prefill.user, placeholder="macOS account", id="user")
            yield Label("Password")
            yield Input(value=self._prefill.password, password=True, id="password")
            yield Label("Resolution")
            yield Select(
                options=[(label, key) for label, key in _RESOLUTION_PRESETS],
                value=("auto" if self._prefill.dynamic_resolution
                       else self._prefill.advertise),
                id="resolution",
                allow_blank=False,
            )
            yield Label("HiDPI")
            yield Select(
                options=[("Auto (match display)", "auto"),
                         ("On (2×)", "on"), ("Off (1×)", "off")],
                value=self._prefill.hidpi,
                id="hidpi",
                allow_blank=False,
            )
            yield Label("Decoder")
            yield Select(
                options=_DECODER_PRESETS,
                value=self._prefill.decoder,
                id="decoder",
                allow_blank=False,
            )
            with Horizontal(classes="switch-row"):
                yield Label("Audio"); yield Switch(value=self._prefill.audio, id="audio")
                yield Label("Curtain"); yield Switch(value=self._prefill.curtain, id="curtain")
                yield Label("HDR"); yield Switch(value=self._prefill.hdr, id="hdr")
            with Horizontal(classes="switch-row"):
                yield Label("Share console"); yield Switch(value=self._prefill.share_console, id="share-console")
                yield Label("Alt session"); yield Switch(value=self._prefill.alt_session, id="alt-session")
            yield Static(
                "[dim]Browser = open in a browser tab · Desktop = native window[/]",
                id="frontend-hint",
            )
            with Horizontal(id="buttons"):
                yield Button("Quit", id="quit", variant="error")
                yield Button("Desktop", id="connect-desktop")
                yield Button("Browser", id="connect-browser", variant="success")

    def on_mount(self) -> None:
        # Kick a probe for whatever host was prefilled (if any).
        if self._prefill.host:
            self._schedule_probe(self._prefill.host)
        # Land focus on whichever field is empty, in order.
        for fid in ("host", "user", "password"):
            inp = self.query_one(f"#{fid}", Input)
            if not inp.value:
                inp.focus()
                return
        # All fields filled: land on the button for the last-used frontend.
        btn = "connect-desktop" if self._prefill.frontend == "desktop" else "connect-browser"
        self.query_one(f"#{btn}", Button).focus()

    @on(Input.Changed, "#host")
    def _host_changed(self, event: Input.Changed) -> None:
        self._schedule_probe(event.value.strip())

    def _schedule_probe(self, host: str) -> None:
        """(Re)schedule a reachability probe after the debounce. Cancels
        any pending probe so we don't chase intermediate keystrokes."""
        if self._probe_task is not None and not self._probe_task.done():
            self._probe_task.cancel()
        badge = self.query_one("#host-probe", Static)
        if not host:
            badge.update("")
            return
        badge.update("[dim]…checking[/]")
        self._probe_task = asyncio.create_task(self._probe(host))

    async def _probe(self, host: str) -> None:
        try:
            await asyncio.sleep(_PROBE_DEBOUNCE_S)
        except asyncio.CancelledError:
            return
        badge = self.query_one("#host-probe", Static)
        # Resolve and TCP-connect to host:5900 with a short timeout.
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, _tcp_probe, host, 5900),
                timeout=_PROBE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            badge.update("[red]✗ timeout[/]")
            return
        except (OSError, socket.gaierror) as e:
            short = type(e).__name__
            badge.update(f"[red]✗ {short}[/]")
            return
        badge.update("[green]✓ reachable[/]")

    @on(Input.Submitted)
    def _input_submitted(self, event: Input.Submitted) -> None:
        # Tab through inputs on Enter; submit when on the password field
        # if all required fields are populated.
        order = ["host", "user", "password"]
        cur = event.input.id
        if cur not in order:
            return
        idx = order.index(cur)
        if idx < len(order) - 1:
            self.query_one(f"#{order[idx + 1]}", Input).focus()
        else:
            # Enter on the password field connects with the last-used frontend.
            self._submit(self._prefill.frontend)

    @on(Button.Pressed, "#connect-browser")
    def _connect_browser(self) -> None:
        self._submit("browser")

    @on(Button.Pressed, "#connect-desktop")
    def _connect_desktop(self) -> None:
        self._submit("desktop")

    @on(Button.Pressed, "#quit")
    def _quit_clicked(self) -> None:
        self.app.exit()

    def action_quit(self) -> None:
        self.app.exit()

    def _submit(self, frontend: str) -> None:
        host = self.query_one("#host", Input).value.strip()
        user = self.query_one("#user", Input).value.strip()
        password = self.query_one("#password", Input).value
        if not host or not user or not password:
            self.notify(
                "Host, username and password are all required.",
                severity="warning", timeout=4,
            )
            return
        res = self.query_one("#resolution", Select).value or _DEFAULT_RESOLUTION
        # "Auto" = dynamic resolution: forward advertise="auto" so the frontend
        # monitor-auto-sizes the initial window (the CLI maps 'auto' -> None),
        # then re-renders to match the window on resize. Sending a concrete WxH
        # here would skip that auto-size path and open at a fixed size.
        dynamic = res == "auto"
        advertise = "auto" if dynamic else str(res)
        values = ConnectFormValues(
            host=host,
            user=user,
            password=password,
            advertise=advertise,
            audio=self.query_one("#audio", Switch).value,
            curtain=self.query_one("#curtain", Switch).value,
            hdr=self.query_one("#hdr", Switch).value,
            share_console=self.query_one("#share-console", Switch).value,
            alt_session=self.query_one("#alt-session", Switch).value,
            frontend=frontend,
            dynamic_resolution=dynamic,
            hidpi=str(self.query_one("#hidpi", Select).value or "auto"),
            decoder=str(self.query_one("#decoder", Select).value or _DEFAULT_DECODER),
        )
        self.post_message(self.Connect(values))


def _tcp_probe(host: str, port: int) -> None:
    """Blocking helper run in a worker thread. Raises on failure; returns
    None on success."""
    with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_S):
        pass


__all__ = ["ConnectScreen", "ConnectFormValues"]
