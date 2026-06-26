"""Live-session screen. Renders reactive panels bound to the snapshot
stream coming off the child's ControlServer, plus a tail of the child's
stderr log. Footer keybindings drive force-IDR / reconnect / quit; the
parent App owns the supervisor + control client (those are not screen
state).

Layout (full-screen):

    ┌─ header ─────────────────────────────────────────────────────────┐
    │ state · host · canvas · decoder · uptime                         │
    ├─ tiles ──────────────────┬─ rates ──────────────────────────────┤
    │ tile fps loss state      │ video / ctrl bandwidth bars          │
    ├──────────────────────────┴─ health / queue / cursor ────────────┤
    │ udp queue · drops · last publish · cursor age                    │
    ├─ log ──────────────────────────────────────────────────────────────┤
    │ stderr tail (level-coloured)                                     │
    └──────────────────────── q quit · r reconnect · f force-IDR ──────┘
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time

from collections import deque

# Textual handles text selection natively (App.ALLOW_SELECT), so selecting is
# just a plain mouse drag — no Shift/Option/Fn modifier (those interfere). Copy
# the selection with C (action_copy_log_selection → App.copy_to_clipboard).
_SELECT_HINT = "M to select/copy text"


def _copy_to_system_clipboard(text: str) -> bool:
    """Put `text` on the LOCAL system clipboard via the OS clipboard tool.
    Works regardless of terminal (Terminal.app has no OSC52, so the app-level
    copy_to_clipboard can't reach the clipboard — but pbcopy can, since the TUI
    runs locally). Returns True on success."""
    if sys.platform == "darwin":
        tool = ["pbcopy"]
    elif sys.platform == "win32":
        tool = ["clip"]
    else:
        tool = next(([t] for t in (["wl-copy"], ["xclip", "-selection", "clipboard"])
                     if shutil.which(t[0])), None) or None
        if tool is None:
            return False
    if not shutil.which(tool[0]):
        return False
    try:
        subprocess.run(tool, input=text.encode("utf-8"), check=True, timeout=5)
        return True
    except Exception:
        return False
from dataclasses import dataclass
from typing import Any, Deque, Optional, Tuple

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    DataTable, Footer, Header, Label, RichLog, Static,
)


def _fmt_uptime(seconds: float) -> str:
    if seconds < 0:
        return "—"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _fmt_bytes_rate(mbps: float, kbps: float | None = None) -> str:
    """Pick the friendlier unit between Mbps and kbps for display."""
    if mbps >= 1:
        return f"{mbps:.1f} Mbps"
    if kbps is not None and kbps >= 1:
        return f"{kbps:.1f} kbps"
    return f"{mbps * 1000:.0f} kbps"


def _classify_loglevel(line: str) -> tuple[str, str]:
    """Return (level, css-style) for one stderr line. The proxy logger
    formats lines as `YYYY-MM-DD HH:MM:SS.mmm LEVEL  name | message`,
    so we look for the LEVEL token."""
    m = re.search(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL|FATAL)\b", line[:60])
    lvl = m.group(1) if m else "INFO"
    style = {
        "DEBUG":    "dim",
        "INFO":     "white",
        "WARNING":  "yellow",
        "ERROR":    "bold red",
        "CRITICAL": "bold red",
        "FATAL":    "bold red",
    }.get(lvl, "white")
    return lvl, style


# ── widgets ──────────────────────────────────────────────────────────


class HeaderBar(Static):
    """Top status line: connection state · host · canvas · decoder · uptime
    · warnings counter (when > 0)."""

    state: reactive[str] = reactive("CONNECTING")
    host: reactive[str] = reactive("")
    canvas: reactive[str] = reactive("")
    decoder: reactive[str] = reactive("")
    uptime: reactive[float] = reactive(-1.0)
    warnings: reactive[int] = reactive(0)
    errors: reactive[int] = reactive(0)

    def render(self) -> str:
        badge = {
            "CONNECTING":   "[on yellow] CONNECTING [/]",
            "LIVE":         "[on green] LIVE [/]",
            "RECONNECTING": "[on yellow] RECONNECTING [/]",
            "DISCONNECTED": "[on red] DISCONNECTED [/]",
        }.get(self.state, f"[on grey] {self.state} [/]")
        bits = [badge]
        if self.host:
            bits.append(f"host: [b]{self.host}[/]")
        if self.canvas:
            bits.append(f"canvas: [b]{self.canvas}[/]")
        if self.decoder:
            bits.append(f"decoder: [b]{self.decoder}[/]")
        if self.uptime >= 0:
            bits.append(f"up [b]{_fmt_uptime(self.uptime)}[/]")
        if self.errors > 0:
            bits.append(f"[on red] {self.errors} err [/]")
        elif self.warnings > 0:
            bits.append(f"[on yellow] {self.warnings} warn [/]")
        return "  ·  ".join(bits)


class TilesPanel(Container):
    """Per-tile fps / loss / state table."""

    def compose(self) -> ComposeResult:
        yield Label("[b]Tiles[/]")
        yield DataTable(id="tiles-table", zebra_stripes=False, cursor_type="none")

    def on_mount(self) -> None:
        t = self.query_one("#tiles-table", DataTable)
        t.add_columns("tile", "fps", "loss/s", "frames")

    def apply(self, tiles: list[dict[str, Any]]) -> None:
        t = self.query_one("#tiles-table", DataTable)
        t.clear()
        for i, tile in enumerate(tiles):
            fps = tile.get("fps", 0.0)
            loss = tile.get("loss_s", 0)
            frames = tile.get("frames", 0)
            fps_style = (
                "green"  if fps >= 45 else
                "yellow" if fps >= 20 else
                "red"
            )
            loss_style = "red" if loss > 0 else "dim"
            t.add_row(
                f"t{i}",
                f"[{fps_style}]{fps:.1f}[/]",
                f"[{loss_style}]{loss}[/]",
                str(frames),
            )


class RatesPanel(Container):
    """Bandwidth + packets-per-second summary."""

    def compose(self) -> ComposeResult:
        yield Label("[b]Throughput[/]")
        yield Static("", id="rates-text")

    def apply(self, rx: dict[str, Any], tx: dict[str, Any]) -> None:
        video_mbps = rx.get("video_mbps", 0.0)
        ctrl_kbps = rx.get("ctrl_kbps", 0.0)
        video_pps = rx.get("video_pps", 0.0)
        tx_pps = tx.get("pps", 0.0)
        lines = [
            f"video  rx  [b]{video_pps:6.0f}[/] pps  [b]{video_mbps:5.1f}[/] Mbps",
            f"ctrl   rx  [b]{rx.get('ctrl_pps', 0.0):6.0f}[/] pps  [b]{ctrl_kbps:5.1f}[/] kbps",
            f"        tx  [b]{tx_pps:6.2f}[/] pps",
        ]
        self.query_one("#rates-text", Static).update("\n".join(lines))


_HISTORY_WINDOW_S: float = 30.0


def _stats(window: "Deque[Tuple[float, float]]", now: float) -> Optional[Tuple[float, float, float]]:
    """Return (min, avg, max) of values in the last 30 s, or None if empty."""
    vals = [v for t, v in window if now - t <= _HISTORY_WINDOW_S]
    if not vals:
        return None
    return min(vals), sum(vals) / len(vals), max(vals)


class HealthPanel(Container):
    """UDP queue depth, drops, cursor age, last-publish age."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Rolling 30-second history: (monotonic_time, value)
        self._lat_hist: Deque[Tuple[float, float]] = deque()
        self._dq_hist: Deque[Tuple[float, float]] = deque()

    def compose(self) -> ComposeResult:
        yield Label("[b]Health[/]")
        yield Static("", id="health-text")

    def apply(
        self,
        udp_q: dict[str, Any],
        cursor: dict[str, Any],
        last_publish_age_s: float,
        loss_total: int,
        loss_unmapped: int,
        decode_latency_ms: Optional[float] = None,
        decode_queue: Optional[dict] = None,
    ) -> None:
        now = time.monotonic()
        v = udp_q.get("video", {}); c = udp_q.get("ctrl", {})
        v_drop = v.get("drop", 0); c_drop = c.get("drop", 0)
        cur_age = cursor.get("age_ms", -1)
        cur_age_s = "—" if cur_age < 0 else (
            f"{cur_age/1000:.1f}s" if cur_age < 60000 else f"{cur_age//60000}m"
        )
        drop_style = "red" if (v_drop + c_drop) > 0 else "green"
        loss_style = "red" if loss_total > 0 else "green"

        # ── dec pipe latency ────────────────────────────────────────
        if decode_latency_ms is not None and decode_latency_ms > 0:
            self._lat_hist.append((now, decode_latency_ms))
        # prune old samples
        while self._lat_hist and now - self._lat_hist[0][0] > _HISTORY_WINDOW_S:
            self._lat_hist.popleft()
        lat_stats = _stats(self._lat_hist, now)
        if decode_latency_ms is not None and decode_latency_ms > 0:
            lat_style = "yellow" if decode_latency_ms > 30 else "green"
            cur_str = f"[{lat_style}]{decode_latency_ms:.1f}[/] ms"
            if lat_stats:
                lo, avg, hi = lat_stats
                cur_str += f"  [dim]↓{lo:.1f} ⌀{avg:.1f} ↑{hi:.1f}[/]"
            lat_str = cur_str
        else:
            lat_str = "[dim]—[/]"

        # ── dec queue depth ─────────────────────────────────────────
        if decode_queue is not None:
            dq_depth = decode_queue.get("depth", 0)
            dq_cap = decode_queue.get("cap", 512)
            dq_drop = decode_queue.get("drop", 0)
            self._dq_hist.append((now, float(dq_depth)))
            while self._dq_hist and now - self._dq_hist[0][0] > _HISTORY_WINDOW_S:
                self._dq_hist.popleft()
            pct = dq_depth / dq_cap if dq_cap > 0 else 0
            filled = round(pct * 10)
            bar = "█" * filled + "░" * (10 - filled)
            bar_style = "red" if pct > 0.5 else ("yellow" if pct > 0.1 else "green")
            drop_str = f"  drop [red]{dq_drop}[/]" if dq_drop > 0 else ""
            dq_str = f"[{bar_style}]{bar}[/] {dq_depth}/{dq_cap}{drop_str}"
            dq_stats = _stats(self._dq_hist, now)
            if dq_stats:
                lo, avg, hi = dq_stats
                dq_str += f"  [dim]↓{int(lo)} ⌀{avg:.1f} ↑{int(hi)}[/]"
        else:
            dq_str = "[dim]—[/]"

        lines = [
            f"udp_q     video [b]{v.get('depth', 0):>4}/{v.get('cap', 0)}[/]  ctrl [b]{c.get('depth', 0):>4}/{c.get('cap', 0)}[/]",
            f"drops     video [{drop_style}]{v_drop}[/]  ctrl [{drop_style}]{c_drop}[/]",
            f"loss      total [{loss_style}]{loss_total}[/]  unmapped [{loss_style}]{loss_unmapped}[/]",
            f"cursor    [b]{cur_age_s}[/] ago  (count {cursor.get('count', 0)})",
            f"publish   {last_publish_age_s:.1f}s ago",
            f"dec pipe  {lat_str}",
            f"dec q     {dq_str}",
        ]
        self.query_one("#health-text", Static).update("\n".join(lines))


_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_LEVEL_LABEL = {"DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARN", "ERROR": "ERROR"}


class _AutoPauseRichLog(RichLog):
    """`RichLog` with an explicit pause toggle (not an auto-detector).

    Earlier versions tried to detect user scroll-up via mouse / key
    handlers on the widget and auto-pause when scroll_y dropped below
    max. That claimed mouse events and broke text selection — so this
    widget intentionally adds NO event handlers, leaving Textual's native
    text selection (a plain mouse drag, App.ALLOW_SELECT) intact. Copy the
    selection with C (SessionScreen.action_copy_log_selection).

    The parent `SessionScreen` provides a screen-level key (default
    `space`) that toggles `auto_scroll` directly. `_pending_while_paused`
    is bumped by `LogPanel.append` on every write that lands while
    paused so the title can surface "↓ N new" as a nudge."""

    _pending_while_paused: int = 0

    def get_selection(self, selection):
        """Extract text under a mouse selection. RichLog renders line Strips
        (not a single Text/Content), so Textual's base Widget.get_selection
        returns None — meaning drag-select yields nothing. Override it to pull
        plain text from the rendered strips, the way the Log widget does for
        its lines. This is what makes click-drag selection actually work in
        the log window."""
        try:
            text = "\n".join(strip.text for strip in self.lines)
        except Exception:
            return None
        return selection.extract(text), "\n"


class LogPanel(Container):
    """Tail of the child's stderr, level-coloured + level-filtered.

    The panel only displays lines at-or-above `min_level`, but the
    parent screen keeps every line in its ring buffer (for bug-snapshot
    dumps) regardless of this filter. Default = INFO. The underlying
    RichLog auto-pauses scroll-to-bottom whenever you scroll up so you
    can read / copy text without the stream yanking the viewport."""

    min_level: reactive[str] = reactive("INFO")

    def compose(self) -> ComposeResult:
        yield Label("[b]Log[/] [dim](level: INFO — L to cycle)[/]", id="log-title")
        yield _AutoPauseRichLog(
            id="log-view", wrap=False, highlight=False, markup=True, max_lines=2000,
        )

    def append(self, line: str) -> bool:
        """Render the line if it passes the level filter; return True if
        it was rendered."""
        lvl, style = _classify_loglevel(line)
        if _LEVEL_ORDER.get(lvl, 20) < _LEVEL_ORDER.get(self.min_level, 20):
            return False
        from rich.markup import escape
        view = self.query_one("#log-view", _AutoPauseRichLog)
        view.write(f"[{style}]{escape(line)}[/]")
        if not view.auto_scroll:
            view._pending_while_paused += 1
            self._refresh_title()
        return True

    def watch_min_level(self, _old: str, new: str) -> None:
        self._refresh_title()

    def _refresh_title(self) -> None:
        try:
            view = self.query_one("#log-view", _AutoPauseRichLog)
        except Exception:
            return
        lvl = _LEVEL_LABEL.get(self.min_level, self.min_level)
        pending = view._pending_while_paused
        if not view.auto_scroll and pending > 0:
            suffix = f"  [reverse yellow] {pending} new — End to jump [/]"
        else:
            suffix = ""
        # Textual selects text natively on a plain mouse drag (no modifier);
        # C copies the selection (see action_copy_log_selection). P pauses the
        # live stream so you can scroll back and select without it scrolling.
        paused = not view.auto_scroll
        state = "[reverse yellow] PAUSED [/]  " if paused else ""
        self.query_one("#log-title", Label).update(
            f"[b]Log[/] {state}[dim](level: {lvl} — L to cycle, "
            f"P to pause, End to jump, {_SELECT_HINT})[/]{suffix}"
        )

    def jump_to_bottom(self) -> None:
        """Scroll the underlying RichLog to the bottom and re-arm auto-
        scroll. Bound to End in the parent SessionScreen."""
        view = self.query_one("#log-view", _AutoPauseRichLog)
        view.scroll_end(animate=False)
        view.auto_scroll = True
        view._pending_while_paused = 0
        self._refresh_title()


# ── screen ───────────────────────────────────────────────────────────


class SessionScreen(Screen):
    """The live-session view. Receives snapshots, events, and log lines
    via methods the parent App calls."""

    CSS = """
    SessionScreen { layout: vertical; }
    HeaderBar { dock: top; height: 1; padding: 0 1; background: $panel; color: $text; }
    #middle { height: auto; }
    TilesPanel, RatesPanel { width: 1fr; padding: 0 1; height: auto; }
    HealthPanel { padding: 0 1; height: auto; border-top: solid $primary 30%; }
    LogPanel { height: 1fr; padding: 0 1; border-top: solid $primary 30%; }
    Label { padding: 0 0 1 0; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reconnect", "Reconnect"),
        Binding("f", "force_idr", "Force IDR"),
        Binding("d", "disconnect", "Disconnect"),
        Binding("l", "cycle_log_level", "Log level"),
        Binding("p", "toggle_log_pause", "Pause log"),
        Binding("end", "log_bottom", "Log → bottom"),
        Binding("c", "copy_log_selection", "Copy selection"),
        Binding("m", "toggle_mouse", "Select mode"),
        Binding("ctrl+b", "bug_snapshot", "Bug snapshot"),
    ]

    # Cycle order for the `l` keybinding. Lands on INFO by default; one
    # press → DEBUG (everything), another → WARNING (only problems),
    # another → back to INFO.
    _LOG_LEVEL_CYCLE = ("INFO", "DEBUG", "WARNING")

    class Disconnect(Message):
        """User asked to drop the current session."""

    class Reconnect(Message):
        """User asked to drop + restart with the same form values."""

    class ForceIdr(Message):
        """User asked for an immediate IDR refresh."""

    class BugSnapshot(Message):
        """User asked to dump the current state for a bug report. The
        message carries the latest snapshot and a tail of recent log
        lines; the App writes them to disk and surfaces the path."""

        def __init__(
            self,
            latest_snapshot: Optional[dict],
            log_tail: list[str],
        ) -> None:
            self.latest_snapshot = latest_snapshot
            self.log_tail = log_tail
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._connect_started_at = time.monotonic()
        self._latest_snapshot: Optional[dict[str, Any]] = None
        # Ring buffer of recent log lines for the bug-snapshot dump.
        # 1000 lines covers the most useful post-mortem window without
        # bloating the snapshot file.
        self._log_ring: list[str] = []
        self._LOG_RING_MAX: int = 1000
        # Whether we've released the terminal mouse (M) for native selection.
        self._mouse_released: bool = False

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="hdr")
        with Horizontal(id="middle"):
            yield TilesPanel(id="tiles")
            yield RatesPanel(id="rates")
        yield HealthPanel(id="health")
        yield LogPanel(id="log")
        yield Footer()

    # ── App-facing API ────────────────────────────────────────────

    def apply_hello(self, session: dict[str, Any]) -> None:
        hdr = self.query_one("#hdr", HeaderBar)
        hdr.host = session.get("host") or session.get("dest_host", "")
        canvas = session.get("canvas") or {}
        if canvas:
            hdr.canvas = f"{canvas.get('w', '?')}x{canvas.get('h', '?')} ({canvas.get('tiles', '?')} tiles)"
        hdr.decoder = session.get("decoder", "")
        hdr.state = "LIVE"

    def apply_snapshot(self, data: dict[str, Any]) -> None:
        self._latest_snapshot = data
        hdr = self.query_one("#hdr", HeaderBar)
        hdr.uptime = float(data.get("uptime_s", -1))
        hdr.decoder = data.get("decoder", hdr.decoder)
        self.query_one("#tiles", TilesPanel).apply(data.get("tiles", []))
        self.query_one("#rates", RatesPanel).apply(data.get("rx", {}), data.get("tx", {}))
        self.query_one("#health", HealthPanel).apply(
            data.get("udp_q", {}),
            data.get("cursor", {}),
            float(data.get("last_publish_age_s", -1)),
            int(data.get("loss_total", 0)),
            int(data.get("loss_unmapped", 0)),
            decode_latency_ms=data.get("decode_latency_ms"),
            decode_queue=data.get("decode_q"),
        )

    def apply_event(self, kind: str, fields: dict[str, Any]) -> None:
        hdr = self.query_one("#hdr", HeaderBar)
        if kind == "connected":
            hdr.state = "LIVE"
        elif kind == "disconnected":
            hdr.state = "DISCONNECTED"
        self.append_log(f"event: {kind} {fields if fields else ''}")

    def append_log(self, line: str) -> None:
        # Always keep in the ring (for bug-snapshot completeness), even
        # if the panel filter hides it visually.
        self._log_ring.append(line)
        if len(self._log_ring) > self._LOG_RING_MAX:
            self._log_ring = self._log_ring[-self._LOG_RING_MAX:]
        try:
            self.query_one("#log", LogPanel).append(line)
        except NoMatches:
            # Called before the screen finished mounting (e.g. the browser
            # frontend logs the viewer URL right after push_screen). The line
            # is preserved in _log_ring above; skip the visual append + the
            # header bump rather than crashing.
            return
        # Bump the header chips so a normal user sees that something
        # warrants attention even if their eyes are on the wgpu window.
        lvl, _ = _classify_loglevel(line)
        if lvl not in ("ERROR", "CRITICAL", "FATAL", "WARNING"):
            return
        try:
            hdr = self.query_one("#hdr", HeaderBar)
        except NoMatches:
            return
        if lvl in ("ERROR", "CRITICAL", "FATAL"):
            hdr.errors = hdr.errors + 1
        else:
            hdr.warnings = hdr.warnings + 1

    def set_state(self, state: str) -> None:
        try:
            self.query_one("#hdr", HeaderBar).state = state
        except NoMatches:
            pass  # called before mount; the header reflects state once mounted

    # ── bindings ────────────────────────────────────────────────

    def action_quit(self) -> None:
        self.app.exit()

    def action_disconnect(self) -> None:
        self.post_message(self.Disconnect())

    def action_reconnect(self) -> None:
        self.post_message(self.Reconnect())

    def action_force_idr(self) -> None:
        self.post_message(self.ForceIdr())

    def action_bug_snapshot(self) -> None:
        self.post_message(self.BugSnapshot(
            latest_snapshot=self._latest_snapshot,
            log_tail=list(self._log_ring),
        ))

    def action_toggle_mouse(self) -> None:
        """`M` → release / re-grab the terminal mouse. The TUI normally captures
        the mouse (for scroll/click), which suppresses the terminal's own text
        selection — and terminals like macOS Terminal.app don't feed Textual the
        drag events it would need to select itself. Releasing the mouse hands it
        back to the terminal so you can select log text the normal way (drag,
        then the terminal's copy — Cmd+C in Terminal.app), exactly like you can
        anywhere else. Press M again to restore TUI mouse control."""
        drv = getattr(self.app, "_driver", None)
        enable = getattr(drv, "_enable_mouse_support", None)
        disable = getattr(drv, "_disable_mouse_support", None)
        if enable is None or disable is None:
            self.notify("mouse toggle not supported on this build", timeout=3)
            return
        try:
            if self._mouse_released:
                enable()
                self._mouse_released = False
                self.notify("mouse restored — TUI controls active", timeout=3)
            else:
                disable()
                self._mouse_released = True
                self.notify(
                    "mouse released — select log text with drag + Cmd/Ctrl+C; "
                    "press M to restore TUI control", timeout=6)
        except Exception as e:
            self.notify(f"mouse toggle failed: {e}", timeout=3)

    def action_copy_log_selection(self) -> None:
        """`C` → copy the mouse-selected text to the clipboard. Drag over the
        log to select first (RichLog selection works now via the subclass's
        get_selection). Copies through the local OS clipboard tool (pbcopy on
        macOS) because Terminal.app has no OSC52, so Textual's own ⌘C copy
        can't reach the clipboard; we also fire OSC52 for the remote/SSH case."""
        try:
            text = self.get_selected_text()
        except Exception:
            text = None
        if not text:
            self.notify("nothing selected — drag over the log first", timeout=2)
            return
        clipped = _copy_to_system_clipboard(text)
        try:
            self.app.copy_to_clipboard(text)
        except Exception:
            pass
        self.notify(
            f"copied {len(text)} chars to clipboard" if clipped
            else f"selected {len(text)} chars (clipboard tool unavailable)",
            timeout=2,
        )

    def action_log_bottom(self) -> None:
        """End → jump the log panel back to the latest line and re-arm
        auto-scroll. Used after you've paused + read / selected text and
        want to resume following the live stream."""
        self.query_one("#log", LogPanel).jump_to_bottom()

    def action_toggle_log_pause(self) -> None:
        """`P` → pause / resume the log panel's auto-scroll. Pausing
        lets you scroll, read, and bypass-select text without the live
        stream yanking you back to the bottom. Resume snaps to the end
        and clears the pending counter."""
        panel = self.query_one("#log", LogPanel)
        view = panel.query_one("#log-view", _AutoPauseRichLog)
        if view.auto_scroll:
            view.auto_scroll = False
            panel._refresh_title()
            self.notify("log paused — P to resume, End to jump to bottom", timeout=2)
        else:
            panel.jump_to_bottom()
            self.notify("log resumed", timeout=2)

    def action_cycle_log_level(self) -> None:
        """Cycle the log-panel display filter (INFO → DEBUG → WARNING →
        back to INFO). Doesn't replay history; only future lines are
        affected. The ring buffer + bug-snapshot stays comprehensive."""
        panel = self.query_one("#log", LogPanel)
        cur = panel.min_level
        cycle = self._LOG_LEVEL_CYCLE
        try:
            nxt = cycle[(cycle.index(cur) + 1) % len(cycle)]
        except ValueError:
            nxt = cycle[0]
        panel.min_level = nxt
        self.notify(f"log level → {_LEVEL_LABEL.get(nxt, nxt)}", timeout=2)


__all__ = ["SessionScreen"]
