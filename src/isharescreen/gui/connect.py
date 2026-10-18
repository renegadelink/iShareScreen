"""Browser connect form + live diagnostics dashboard — stdlib only.

A small local web server that is the iss control surface:

1. serves the connect form (native text selection / copy-paste in the browser);
2. on submit, launches the iss session with a ``--control-socket`` and keeps
   running;
3. turns the tab into a live dashboard — structured stat panels (stream /
   network / loss / queues / per-tile), fed by iss's ``ControlServer`` snapshot
   stream, plus a selectable log — all streamed to the browser over SSE.

The video opens separately (a native window for ``--frontend desktop``, or
iss's own browser tab for ``--frontend browser``). Zero extra dependencies.
"""
from __future__ import annotations

import http.server
import json
import mimetypes
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs

try:
    from isharescreen.tui.storage import load_last
except Exception:  # pragma: no cover
    load_last = None  # type: ignore[assignment]

# ── shared state, streamed to any connected dashboard tab ──────────────
_LOG: "deque[str]" = deque(maxlen=4000)
_LATEST: dict = {}          # most recent snapshot (for tabs opened mid-session)
_HELLO: dict = {}           # static session header
_STATE = {"host": "", "frontend": "", "status": "idle", "pid": 0}
_LOCK = threading.Lock()
_SUBSCRIBERS: "list[queue.Queue]" = []
_PROC = None                # current iss subprocess (Reconnect / new replaces it)
_CTRL_SOCK = None           # connected control socket, for sending commands (fir)
_LAST: dict = {}            # last form values, for Reconnect
_LAUNCH_N = 0               # per-launch counter → unique control-socket path
_BRIDGE_PORT = 4433         # iss --bridge-port default; the GUI never overrides it
_VIDEO_URL = None           # browser-frontend video page URL → the Open-video button
_GUI_PORT = 0               # this server's port, for the iss → GUI choice callback
_CHOICE_EVENT = threading.Event()   # set when the user answers the session modal
_CHOICE_RESULT = "share"            # "alt" | "share" — the user's last answer

_STATIC_DIR = Path(__file__).with_name("static")


def _emit(kind: str, payload) -> None:
    if kind == "log":
        with _LOCK:
            _LOG.append(payload)
    elif kind == "snapshot":
        with _LOCK:
            _LATEST.clear(); _LATEST.update(payload)
    elif kind == "hello":
        with _LOCK:
            _HELLO.clear(); _HELLO.update(payload)
    for q in list(_SUBSCRIBERS):
        try:
            q.put_nowait((kind, payload))
        except Exception:
            pass


def _control_reader(sock_path: str) -> None:
    """Connect to iss's control socket, relay its newline-delimited JSON
    (hello / snapshot / event) to the dashboard, and keep the socket open so
    the dashboard can send commands (force-IDR). Retries until iss binds."""
    global _CTRL_SOCK
    s = None
    for _ in range(120):  # ~60s of 0.5s retries while iss handshakes
        try:
            if sys.platform == "win32":
                # On Windows iss's ControlServer binds AF_INET on 127.0.0.1 and
                # writes the chosen port to a "<path>.port" sidecar (control.py).
                with open(sock_path + ".port") as pf:
                    port = int(pf.read().strip())
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", port))
            else:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(sock_path)
            break
        except (OSError, ValueError):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass
            s = None
            time.sleep(0.5)
    if s is None:
        _emit("log", "[gui] could not reach iss control socket for stats")
        return
    _CTRL_SOCK = s
    buf = b""
    try:
        while True:
            chunk = s.recv(16384)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                t = msg.get("type")
                if t == "snapshot":
                    _emit("snapshot", msg.get("data", {}))
                elif t == "hello":
                    _emit("hello", msg.get("session", {}))
                elif t == "event":
                    _emit("event", {k: v for k, v in msg.items() if k != "type"})
    except OSError:
        pass
    finally:
        if _CTRL_SOCK is s:
            _CTRL_SOCK = None
        try:
            s.close()
        except Exception:
            pass


def _send_command(cmd: dict) -> bool:
    """Send a JSON control command (e.g. force-IDR) to the running session."""
    s = _CTRL_SOCK
    if s is None:
        _emit("log", "[gui] no control connection — is a session running?")
        return False
    try:
        s.sendall((json.dumps(cmd) + "\n").encode())
        _emit("log", f"[gui] sent: {cmd.get('action')}")
        return True
    except OSError as e:
        _emit("log", f"[gui] command failed: {e}")
        return False


def _set_session_choice(choice: str) -> None:
    """Record the user's share/alt answer and wake the blocked /session-choice
    request — which is what iss is parked on over HTTP."""
    global _CHOICE_RESULT
    _CHOICE_RESULT = "alt" if choice == "alt" else "share"
    _CHOICE_EVENT.set()


def _kill_current() -> None:
    """Terminate the current session, if any (Reconnect / new connection).

    Waits for the child to actually exit so its bridge port (and control
    socket) is released before a relaunch — otherwise a fresh launch races
    the old process and fails to bind with 'address already in use'. Escalates
    to SIGKILL if it doesn't stop promptly (a WebTransport bridge whose viewer
    never connected — e.g. a failed Safari attempt — otherwise lingers)."""
    global _PROC, _VIDEO_URL
    _VIDEO_URL = None
    p, _PROC = _PROC, None
    if p is not None and p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=2)
            except Exception:
                pass


def _free_bridge_port(port: int) -> None:
    """Kill any orphaned iss bridge still LISTENing on `port` (a prior GUI
    session that crashed / had a failed WebTransport handshake). Scoped by port
    AND to iss processes, so it never touches a bridge on another port or any
    non-iss process — and skips our own pid."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=3).stdout
    except Exception:
        return
    killed = False
    for pid_s in out.split():
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=2).stdout
        except Exception:
            cmd = ""
        if "isharescreen" in cmd:
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
            except Exception:
                pass
    if killed:
        time.sleep(0.4)  # let the socket release before we bind


def _reconnect() -> None:
    """Drop the current session and relaunch with the last form values."""
    if not _LAST:
        _emit("log", "[gui] nothing to reconnect to yet")
        return
    _emit("log", "[gui] reconnecting…")
    _launch(dict(_LAST))


def _browser_is_safari(ua: str) -> bool:
    """True for desktop/iOS Safari from its User-Agent. Safari's UA
    contains 'Safari' + 'Version/', but so does Chrome/Edge/etc. — they
    additionally carry their own token, so exclude those."""
    u = ua.lower()
    if "safari" not in u:
        return False
    return not any(t in u for t in
                   ("chrome", "chromium", "crios", "edg", "fxios", "android", "opr"))


def _launch(values: dict) -> None:
    global _PROC, _LAST, _LAUNCH_N
    host = values.get("host", "").strip()
    user = values.get("user", "").strip()
    password = values.get("password", "")
    if not (host and user and password):
        _emit("log", "[gui] host, user, and password are all required")
        _STATE["status"] = "idle"
        _emit("status", dict(_STATE))
        return
    _kill_current()            # Reconnect / new connection replaces the old one
    # Also clear any ORPHANED iss bridge (a previous GUI session that crashed,
    # was force-quit, or had a failed WebTransport handshake) still holding the
    # bridge port, else it blocks this launch with 'address already in use'.
    # Scoped to the SPECIFIC bridge port and to iss processes only, so it can't
    # touch a bridge on another port or any non-iss process.
    _free_bridge_port(_BRIDGE_PORT)
    _LAST = dict(values)
    frontend = values.get("frontend", "browser")
    _STATE.update(host=host, frontend=frontend, status="connecting")
    _emit("status", dict(_STATE))

    _LAUNCH_N += 1
    sock_path = os.path.join(
        os.path.expanduser("~/.iss"), f"gui-{os.getpid()}-{_LAUNCH_N}.sock")
    log_path = sock_path[:-5] + ".log"   # gui-<pid>-<n>.log, next to the socket
    cmd = ([sys.executable] if getattr(sys, "frozen", False) else
           [sys.executable, "-m", "isharescreen.cli"]) + [
        "--host", host, "-u", user, "--password-stdin",
        "--frontend", frontend, "--control-socket", sock_path,
        "--log-file", log_path,
    ]
    # Resolution = the encoded BACKING (bandwidth); Display scale = UI size.
    # For a fixed resolution we advertise logical points = backing ÷ scale, so
    # the host encodes exactly the chosen backing while drawing the UI at
    # `scale`. For "Auto" (track-window) the scale rides --hidpi.
    advertise = values.get("advertise", "").strip()
    scale = values.get("scale", "2.0").strip() or "2.0"
    if advertise and advertise.lower() != "auto":
        try:
            bw, bh = (int(x) for x in advertise.lower().split("x", 1))
            s = float(scale)
            lw = max(2, int(round(bw / s / 2)) * 2)   # even logical dims
            lh = max(2, int(round(bh / s / 2)) * 2)
            cmd += ["--advertise", f"{lw}x{lh}@{scale}"]
        except (ValueError, ZeroDivisionError):
            cmd += ["--advertise", advertise]
    else:
        cmd += ["--hidpi", scale]
    decoder = values.get("decoder", "auto").strip()
    if decoder and decoder != "auto":
        cmd += ["--decoder", decoder]        # explicit pin — session honours it
    elif frontend == "desktop" and "ISS_VIDEO_CODEC" not in os.environ:
        # "Auto" on the desktop frontend: the GUI already probed hardware at
        # startup, so pin the DECODER here → the session skips its decoder-build
        # probe (and "hevc444 probe: unavailable" log noise) at stream start.
        # Only pass --codec for AVC: that matches what the cli already forces on
        # a non-HEVC box (where the probe noise actually is), and it lets the
        # session skip resolve_codec too. For HEVC we deliberately do NOT pass
        # --codec — forcing it would change the media offer from the default
        # "both"-bank (byte-identical to Apple's) to a single-HEVC bank
        # (offers.py DANGER note); the session resolves the codec itself
        # (cheap on a HEVC-capable box). Skipped if the user set ISS_VIDEO_CODEC.
        try:
            from isharescreen.proxy.media.registry import resolve_codec, select
            codec = resolve_codec("auto")
            spec = select(codec)
            if spec is not None:
                cmd += ["--decoder", spec.name]
                if codec == "avc":
                    cmd += ["--codec", "avc"]
        except Exception:
            pass  # fall back to the session's own auto resolution
    if values.get("curtain") != "on":
        cmd.append("--no-curtain")
    if values.get("audio") != "on":
        cmd.append("--no-audio")
    if values.get("safari") == "on" and frontend == "browser":
        cmd.append("--trust-cert-for-safari")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    if _GUI_PORT:
        # iss POSTs here (blocking) to ask share-vs-alt when a different user is
        # at the console; the /session-choice handler answers from the modal.
        env["ISS_SESSION_CHOICE_URL"] = f"http://127.0.0.1:{_GUI_PORT}/session-choice"
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env,
        start_new_session=True, bufsize=1, text=True,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    _PROC = proc
    _STATE["pid"] = proc.pid
    assert proc.stdin is not None
    proc.stdin.write(password + "\n")
    proc.stdin.flush()
    proc.stdin.close()
    _STATE["status"] = "running"
    _emit("status", dict(_STATE))

    threading.Thread(target=_control_reader, args=(sock_path,), daemon=True).start()

    def _pump() -> None:
        global _VIDEO_URL
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            _emit("log", line)
            if frontend == "browser" and "open this URL" in line:
                i = line.find("http")
                if i != -1:
                    _VIDEO_URL = line[i:].strip()
                    _emit("video_url", {"url": _VIDEO_URL})
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        _STATE["status"] = "ended"
        _emit("status", dict(_STATE))
        _emit("log", "[gui] iss process exited")

    threading.Thread(target=_pump, daemon=True).start()


# ── HTML ───────────────────────────────────────────────────────────────
_HEAD = (_STATIC_DIR / "head.html").read_text(encoding="utf-8")
_VIDEO_WAIT = (_STATIC_DIR / "waiting.html").read_text(encoding="utf-8").replace("__HEAD__", _HEAD)
_DASH = (_STATIC_DIR / "dashboard.html").read_text(encoding="utf-8")


# Friendly labels for the registry's decoder specs. The dropdown only lists
# decoders SUPPORTED ON THIS CLIENT PLATFORM (via the registry), so e.g.
# VideoToolbox never shows on Windows/Linux and Intel QSV never shows on macOS.
_DECODER_LABELS = {
    "vt-hevc444":       "HEVC — VideoToolbox (Mac GPU)",
    "libav-hevc444":    "HEVC — Generic GPU (d3d11va / vaapi / cuda)",
    "qsv-hevc444":      "HEVC — Intel Quick Sync",
    "libav-hevc444-sw": "HEVC — Software (CPU)",
    "libav-avc420":     "H.264",
}


# Hardware-probe state. `_warm_decoder_probe` runs the real availability probe
# ONCE at iss startup (background), so (a) the dropdown lists only decoders that
# actually work here, not just platform-appropriate ones, and (b) the session
# doesn't re-probe at stream start (which logged "hevc444 probe: unavailable"
# noise). The form render waits for the probe (normally already done by the
# time the browser connects), so even the first page load is filtered to the
# hardware that's really present.
_PROBE_DONE = threading.Event()


def _warm_decoder_probe() -> None:
    """Probe decoder hardware once at startup and cache it (hwcaps caches
    per-process). Best-effort — failures just leave the platform filter."""
    try:
        from isharescreen.proxy.media.registry import all_specs, resolve_codec
        resolve_codec("auto")                     # warms the codec / HEVC-444 probe
        for s in all_specs():
            if s.supported_here():
                try:
                    s.available()                 # warms this decoder's probe
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        _PROBE_DONE.set()


def _spec_offerable(s, probe_ready: bool) -> bool:
    """Show a decoder if it's supported on this platform, and — once the startup
    probe has completed — actually available (real hardware present)."""
    if not s.supported_here():
        return False
    if not probe_ready:
        return True   # probe still running: fall back to the platform filter
    try:
        return bool(s.available())
    except Exception:
        return True   # probe error → don't hide it


def _decoder_options_html() -> str:
    """<option>s for the decoder dropdown, filtered to decoders available on
    THIS machine. Waits for the startup hardware probe so the FIRST page load
    already shows the correctly filtered list (the probe runs while the
    browser is still launching, so this wait is normally instant). Only if the
    probe is somehow still running after the cap do we fall back to the
    platform filter rather than hang the form."""
    import html as _h
    opts = ['<option value="auto" selected>Auto (best available)</option>']
    probe_ready = _PROBE_DONE.wait(timeout=5.0)
    try:
        from isharescreen.proxy.media.registry import all_specs
        for s in all_specs():
            if not _spec_offerable(s, probe_ready):
                continue
            label = _DECODER_LABELS.get(s.name, f"{s.codec.upper()} — {s.name}")
            opts.append(f'<option value="{_h.escape(s.name, quote=True)}">'
                        f'{_h.escape(label)}</option>')
    except Exception:
        pass  # registry unavailable → just offer Auto
    return "\n   ".join(opts)


def _form_page() -> bytes:
    host = user = ""
    frontend = "browser"
    if load_last is not None:
        try:
            last = load_last()
        except Exception:
            last = None
        if last is not None:
            host = getattr(last, "host", "") or ""
            user = getattr(last, "user", "") or ""
            frontend = getattr(last, "frontend", "browser") or "browser"
    import html as _html
    return ((_STATIC_DIR / "connect.html").read_text(encoding="utf-8")
            .replace("__HEAD__", _HEAD)
            .replace("__HOST__", _html.escape(host, quote=True))
            .replace("__USER__", _html.escape(user, quote=True))
            .replace("__BROWSER_SEL__", "selected" if frontend == "browser" else "")
            .replace("__DESKTOP_SEL__", "selected" if frontend == "desktop" else "")
            .replace("__DECODER_OPTIONS__", _decoder_options_html())).encode()


def _dash_page() -> bytes:
    return _DASH.replace("__HEAD__", _HEAD).encode()


# ── server ─────────────────────────────────────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):
    def _html(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _local_request(self) -> bool:
        """Reject requests not addressed to our own localhost origin — guards the
        control endpoints against DNS-rebinding / cross-site driving."""
        hosts = {f"127.0.0.1:{_GUI_PORT}", f"localhost:{_GUI_PORT}"}
        if self.headers.get("Host", "") not in hosts:
            return False
        origin = self.headers.get("Origin")
        return not origin or origin in {f"http://{h}" for h in hosts}

    def do_GET(self) -> None:  # noqa: N802
        if not self._local_request():
            self._html(b"forbidden", 403); return
        if self.path.startswith("/static/"):
            target = (_STATIC_DIR / self.path[8:].split("?", 1)[0]).resolve()
            if not target.is_relative_to(_STATIC_DIR.resolve()) or not target.is_file():
                self._html(b"not found", 404); return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            self._html(_form_page() if _STATE["status"] == "idle" else _dash_page())
        elif self.path == "/new":
            self._html(_form_page())
        elif self.path == "/dashboard":
            self._html(_dash_page())
        elif self.path == "/events":
            self._sse()
        elif self.path == "/video-wait":
            self._html(_VIDEO_WAIT.encode("utf-8"))
        elif self.path == "/video-url":
            self._html(json.dumps({"url": _VIDEO_URL}).encode())
        else:
            self._html(b"not found", 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._local_request():
            self._html(b"forbidden", 403); return
        if self.path == "/quit":
            self._html(b"closed")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        values = {k: v[0] for k, v in parse_qs(raw).items()}
        if self.path == "/action":
            action = values.get("action")
            if action == "fir":
                _send_command({"action": "fir"})
            elif action == "reconnect":
                threading.Thread(target=_reconnect, daemon=True).start()
            elif action == "disconnect":
                _kill_current()
                _STATE["status"] = "ended"
                _emit("status", dict(_STATE))
            elif action == "session_choice":
                _set_session_choice(values.get("choice", "share"))
            self._html(b"ok")
            return
        if self.path == "/session-choice":
            # iss (the child) blocks here until the user answers the modal.
            console_user = values.get("console_user", "")
            _CHOICE_EVENT.clear()
            _emit("session_choice", {"console_user": console_user})
            got = _CHOICE_EVENT.wait(timeout=50)
            self._html((_CHOICE_RESULT if got else "share").encode())
            return
        global _VIDEO_URL
        _VIDEO_URL = None   # clear the prior session's URL so the waiting tab waits for THIS one
        # The video page auto-opens in the SAME browser as this connect form,
        # so the connect request's User-Agent tells us whether the viewer is
        # Safari — which needs the cert keychain-trusted (WebKit has no
        # serverCertificateHashes). Auto-enable it; the install is idempotent
        # (no prompt once trusted), so no checkbox is needed.
        if _browser_is_safari(self.headers.get("User-Agent", "")):
            values.setdefault("safari", "on")
        threading.Thread(target=_launch, args=(values,), daemon=True).start()
        self._html(_dash_page())

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q: "queue.Queue" = queue.Queue()
        with _LOCK:
            backlog = list(_LOG)
        _SUBSCRIBERS.append(q)
        try:
            with _LOCK:
                state, hello, latest = dict(_STATE), dict(_HELLO), dict(_LATEST)
            self._event("status", state)
            if hello:
                self._event("hello", hello)
            if latest:
                self._event("snapshot", latest)
            if _VIDEO_URL:
                self._event("video_url", {"url": _VIDEO_URL})
            for line in backlog:
                self._event("log", line)
            while True:
                kind, payload = q.get()
                self._event(kind, payload)
        except Exception:
            pass
        finally:
            try:
                _SUBSCRIBERS.remove(q)
            except ValueError:
                pass

    def _event(self, kind: str, payload) -> None:
        self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def log_message(self, *_a) -> None:  # quiet
        pass


def main() -> int:
    global _GUI_PORT
    # Probe decoder hardware once, now, in the background — so the connect form's
    # dropdown reflects real availability and the session doesn't re-probe at
    # stream start (the "hevc444 probe: unavailable" log noise).
    threading.Thread(target=_warm_decoder_probe, daemon=True).start()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _GUI_PORT = server.server_address[1]
    url = f"http://127.0.0.1:{_GUI_PORT}/"
    print(f"iShareScreen → {url}", flush=True)
    if os.environ.get("ISS_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _kill_current()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
