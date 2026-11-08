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
    cmd = [
        sys.executable, "-m", "isharescreen.cli",
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
    auto_chroma = None
    decoder = values.get("decoder", "auto").strip()
    if decoder and decoder != "auto":
        cmd += ["--decoder", decoder]        # explicit pin — session honours it
    elif frontend == "desktop" and "ISS_VIDEO_CODEC" not in os.environ:
        # Reuse the GUI's cached hardware probes to pin the child decoder.
        # Preserve the native both-bank 4:4:4 offer; 4:2:0 needs an explicit
        # HEVC-only offer and matching chroma in the child's environment.
        try:
            from isharescreen.proxy.media.registry import resolve_video_format, select
            codec, chroma = resolve_video_format("auto")
            spec = select(codec, chroma=chroma)
            if spec is not None:
                cmd += ["--decoder", spec.name]
                if codec == "avc":
                    cmd += ["--codec", "avc"]
                elif chroma == "420":
                    cmd += ["--codec", "hevc"]
                    auto_chroma = "420"
        except Exception:
            pass  # fall back to the session's own auto resolution
    if values.get("curtain") != "on":
        cmd.append("--no-curtain")
    if values.get("audio") != "on":
        cmd.append("--no-audio")
    if values.get("safari") == "on" and frontend == "browser":
        cmd.append("--trust-cert-for-safari")

    env = dict(os.environ)
    if auto_chroma is not None:
        env["ISS_HEVC_CHROMA"] = auto_chroma
    if _GUI_PORT:
        # iss POSTs here (blocking) to ask share-vs-alt when a different user is
        # at the console; the /session-choice handler answers from the modal.
        env["ISS_SESSION_CHOICE_URL"] = f"http://127.0.0.1:{_GUI_PORT}/session-choice"
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, env=env,
        start_new_session=True, bufsize=1, text=True,
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
_HEAD = """<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style> :root { color-scheme: dark; } * { box-sizing: border-box; }
 body { margin:0; background:#16171a; color:#e8e8ea; font:15px -apple-system,"Segoe UI",system-ui,sans-serif; } </style>"""

_FORM = """<!doctype html><html><head><title>iShareScreen — Connect</title>__HEAD__
<style>
 body { min-height:100vh; display:grid; place-items:center; }
 .card { width:360px; background:#1e1f22; border:1px solid #2c2e33; border-radius:16px; padding:28px 28px 24px; box-shadow:0 12px 40px #0008; }
 h1 { margin:0 0 2px; font-size:22px; } .sub { margin:0 0 20px; color:#9aa0a6; font-size:13px; }
 label { display:block; margin:0 0 6px; color:#c7c9cf; font-size:13px; }
 input[type=text],input[type=password],select { width:100%; margin:0 0 14px; padding:10px 12px; font-size:15px; color:#e8e8ea; background:#2b2d31; border:1px solid #3a3d42; border-radius:9px; outline:none; }
 input:focus,select:focus { border-color:#4f8cff; }
 .row { display:flex; gap:10px; align-items:center; margin:2px 0 12px; } .row input { width:auto; margin:0; }
 button { width:100%; margin-top:8px; padding:12px; font-size:15px; font-weight:600; color:#fff; background:#4f8cff; border:0; border-radius:9px; cursor:pointer; } button:hover { background:#5f97ff; }
</style></head><body>
 <form class="card" method="post" action="/connect" onsubmit="return prepVideo()">
  <h1>iShareScreen</h1><p class="sub">Connect to a Mac</p>
  <label>Host</label><input type="text" name="host" placeholder="Mac hostname or IP" value="__HOST__" autofocus>
  <label>User</label><input type="text" name="user" placeholder="macOS account" value="__USER__">
  <label>Password</label><input type="password" name="password" placeholder="password">
  <label>Frontend</label>
  <select name="frontend"><option value="browser" __BROWSER_SEL__>browser (H.264)</option><option value="desktop" __DESKTOP_SEL__>desktop (native window, H.264 / HEVC)</option></select>
  <label>Resolution</label>
  <select name="advertise" title="The encoded pixel resolution = how much bandwidth you use. Higher = sharper + more bandwidth. UI size is set separately by Display scale.">
   <option value="auto" selected>Auto — track window size</option>
   <option value="3840x2160">3840 × 2160 (4K UHD)</option>
   <option value="3440x1440">3440 × 1440 (UWQHD)</option>
   <option value="2560x1600">2560 × 1600 (WQXGA)</option>
   <option value="2560x1440">2560 × 1440 (QHD)</option>
   <option value="1920x1200">1920 × 1200 (WUXGA)</option>
   <option value="1920x1080">1920 × 1080 (FHD)</option>
   <option value="1680x1050">1680 × 1050 (WSXGA+)</option>
   <option value="1600x900">1600 × 900 (HD+)</option>
   <option value="1366x768">1366 × 768 (WXGA)</option>
   <option value="1280x720">1280 × 720 (HD)</option>
   <option value="1024x768">1024 × 768 (XGA)</option>
   <option value="800x600">800 × 600 (SVGA)</option>
  </select>
  <label>Display scale</label>
  <select name="scale" title="How large the host UI is drawn — like Windows display scaling (100%–400%). Higher = bigger text/UI. This does NOT change bandwidth; Resolution does.">
   <option value="1.0">100% — smallest UI, most desktop space</option>
   <option value="1.5">150%</option>
   <option value="2.0" selected>200% — Retina (normal UI, crisp)</option>
   <option value="2.5">250%</option>
   <option value="3.0">300% — large UI</option>
   <option value="3.5">350%</option>
   <option value="4.0">400% — largest UI</option>
  </select>
  <label>Decoder</label>
  <select name="decoder" title="Only decoders available on this computer are listed. 'Auto' picks the best one; the rest are for pinning a specific path when debugging.">
   __DECODER_OPTIONS__
  </select>
  <div class="row"><input type="checkbox" name="audio" id="audio" checked><label for="audio" style="margin:0">Audio</label></div>
  <div class="row"><input type="checkbox" name="curtain" id="curtain" checked><label for="curtain" style="margin:0">Curtain (private virtual display)</label></div>
  <button type="submit">Connect</button>
 </form>
 <script>
 // Open the video tab DURING the Connect click (a real user gesture, so no
 // popup-block) — but the URL doesn't exist yet, so point it at a local
 // "Connecting…" page that polls for the real URL and redirects itself. Only
 // for the browser frontend; the desktop frontend has no video page.
 function prepVideo(){ try{ var f=document.querySelector('[name=frontend]');
   if(f&&f.value==='browser') window.open('/video-wait','_blank'); }catch(e){} return true; }
 </script></body></html>"""

_VIDEO_WAIT = """<!doctype html><html><head><meta charset="utf-8"><title>Connecting…</title>
<style>body{background:#16171a;color:#c7c9cf;font:15px -apple-system,system-ui,sans-serif;display:flex;height:100vh;margin:0;align-items:center;justify-content:center;flex-direction:column;gap:16px}.s{width:34px;height:34px;border:3px solid #2c2e33;border-top-color:#4f8cff;border-radius:50%;animation:r .8s linear infinite}@keyframes r{to{transform:rotate(360deg)}}</style></head>
<body><div class="s"></div><div>Connecting to the stream…</div>
<script>(function p(){fetch('/video-url',{cache:'no-store'}).then(r=>r.json()).then(d=>{if(d&&d.url){location.replace(d.url);}else{setTimeout(p,400);}}).catch(function(){setTimeout(p,700);});})();</script>
</body></html>"""

_DASH = """<!doctype html><html><head><title>iShareScreen — Diagnostics</title>__HEAD__
<style>
 body { display:flex; flex-direction:column; height:100vh; }
 .top { display:flex; gap:12px; align-items:center; padding:10px 16px; background:#1e1f22; border-bottom:1px solid #2c2e33; }
 .top b { font-size:15px; } .dot { width:9px; height:9px; border-radius:50%; background:#666; }
 .dot.running{background:#4ec97a;} .dot.connecting{background:#e0b341;} .dot.ended,.dot.idle{background:#e06c75;}
 .meta { color:#9aa0a6; font-size:13px; } .grow { flex:1; }
 a.new { color:#9aa0a6; text-decoration:none; font-size:13px; } a.new:hover { color:#e8e8ea; }
 .btn { background:#2b2d31; color:#e8e8ea; border:1px solid #3a3d42; border-radius:8px; padding:5px 11px; font-size:13px; cursor:pointer; }
 .btn:hover { background:#34373c; border-color:#4f8cff; }
 .kv span, .card h3, .loghdr span, [data-tip] { cursor:help; }
 /* Styled hover tooltip (native title is delayed + unstyled + width-capped, and
    long troubleshooting text renders badly there). Shows instantly on the whole
    row; .card/.cards have no overflow so it isn't clipped. */
 [data-tip] { position:relative; }
 [data-tip]:hover::after {
   content:attr(data-tip); position:absolute; left:0; top:calc(100% + 5px); z-index:30;
   width:max-content; max-width:270px; background:#0f1113; color:#dfe1e4;
   border:1px solid #3a3d42; border-radius:8px; padding:8px 10px;
   font-size:12px; line-height:1.45; font-weight:400; white-space:normal; text-align:left;
   box-shadow:0 8px 22px rgba(0,0,0,.6); pointer-events:none; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; padding:14px 16px 4px; }
 .card { background:#1e1f22; border:1px solid #2c2e33; border-radius:12px; padding:12px 14px; }
 .card h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#9aa0a6; }
 .kv { display:flex; justify-content:space-between; padding:3px 0; font-size:13.5px; }
 .kv span { color:#9aa0a6; } .kv b { font-variant-numeric:tabular-nums; }
 .tiles { padding:0 16px; } .tilegrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
 .tile { background:#1e1f22; border:1px solid #2c2e33; border-radius:10px; padding:9px 11px; font-size:13px; }
 .tile .t { color:#9aa0a6; font-size:11px; } .bar { height:5px; border-radius:3px; background:#2b2d31; margin-top:6px; overflow:hidden; }
 .bar > i { display:block; height:100%; background:#4ec97a; } .bar.loss > i { background:#e06c75; }
 .logwrap { flex:1; display:flex; flex-direction:column; min-height:120px; padding:8px 16px 14px; }
 .loghdr { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:#9aa0a6; margin:6px 0; }
 #log { flex:1; margin:0; padding:10px 12px; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#0f1012; border:1px solid #2c2e33; border-radius:10px; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }
 .lvl-ERROR{color:#ff7b72;} .lvl-WARN{color:#e0b341;} .lvl-DEBUG{color:#7f848e;}
 #modal { position:fixed; inset:0; background:#000a; display:none; align-items:center; justify-content:center; z-index:20; }
 #modal.show { display:flex; }
 .modalbox { width:430px; background:#1e1f22; border:1px solid #2c2e33; border-radius:14px; padding:22px 24px; box-shadow:0 16px 50px #000a; }
 .modalbox h2 { margin:0 0 8px; font-size:18px; } .modalbox p { color:#c7c9cf; font-size:14px; line-height:1.45; margin:0 0 18px; }
 .modalbtns { display:flex; gap:10px; justify-content:flex-end; }
 .btn.primary { background:#4f8cff; color:#fff; border-color:#4f8cff; } .btn.primary:hover { background:#5f97ff; }
</style></head><body>
 <div class="top"><span class="dot idle" id="dot"></span><b>iShareScreen</b>
  <span class="meta" id="host" title="The Mac you're connected to, the frontend in use, and the iss process ID."></span><span class="grow"></span>
  <span class="meta" id="uptime" title="How long this session has been connected."></span><span class="meta" id="decoder" title="Active decoder + hardware path (also shown in the Stream card)."></span>
  <button class="btn" onclick="act('fir')" title="Request a fresh keyframe (IDR) on every tile. Use when a tile is frozen, gray, or showing stale / garbled content.">Force IDR</button>
  <button class="btn" onclick="act('reconnect')" title="Drop and re-establish the session with the same settings. Use after a stall the stream can't recover from on its own.">Reconnect</button>
  <a id="videobtn" class="btn primary" style="display:none;text-decoration:none" target="_blank" rel="noopener" title="Open the live video stream in a new tab — in this same browser.">▶ Open video</a>
  <a class="new" href="/new">+ new connection</a></div>
 <div class="cards">
  <div class="card"><h3 title="What's being streamed and how it's decoded.">Stream</h3>
   <div class="kv" data-tip="Session lifecycle: connecting → running → ended. Stuck on 'connecting' → check the host address / username / password, or a firewall blocking TCP 5900. 'ended' unexpectedly → see the log below.">
     <span>Status</span><b id="s_status">—</b></div>
   <div class="kv" data-tip="Host's encoded canvas size and tile count (WIDTH×HEIGHT · N tiles). The screen is split into independently-decoded tiles, so packet loss in one only corrupts that region. On the desktop frontend, resize the window to renegotiate this (dynamic resolution).">
     <span>Canvas</span><b id="s_canvas">—</b></div>
   <div class="kv" data-tip="Number of independently-encoded tiles (usually 4). Loss in one tile only corrupts its region, not the whole frame — a single stuck/gray tile is recoverable with Force IDR.">
     <span>Tiles</span><b id="s_tiles">—</b></div>
   <div class="kv" data-tip="Active decoder + hardware path. videotoolbox (Mac) / d3d11va / qsv / nvdec = GPU, low latency. 'software' = CPU decode: higher latency + CPU, lower fps — usually means the codec (e.g. HEVC 4:4:4) has no HW decoder on this machine; try --codec avc. 'n/a' on browser sessions (the browser decodes via WebCodecs).">
     <span>Decoder</span><b id="s_dec">—</b></div></div>
  <div class="card"><h3 title="Inbound and outbound packet activity.">Network in</h3>
   <div class="kv" data-tip="Inbound video packet rate / bitrate. Drops to near zero on a static screen — that is NORMAL, not a stall. Sustained 0 while the screen is actively changing = a real stall: try Force IDR, then Reconnect.">
     <span>Video</span><b id="n_vid">—</b></div>
   <div class="kv" data-tip="Inbound system-audio stream (AAC-ELD-SBR) from the host. With audio enabled, macOS screen-share sends this continuously — ~100 pps / ~21 kbps even in silence (no silence suppression / DTX), so a constant idle rate here is NORMAL, not a leak. --no-audio turns it off at the SOURCE (the offer gates the host's audio transmitter below its tier floor): the rate drops to ~0.4 kbps (2 pps residual), not just muting local playback.">
     <span>Audio</span><b id="n_audio">—</b></div>
   <div class="kv" data-tip="Inbound RTCP feedback (sender reports) from the host — a few pps, muxed on the audio socket. Near-zero is fine.">
     <span>RTCP</span><b id="n_rtcp">—</b></div>
   <div class="kv" data-tip="RFB control packets over TCP: cursor shape, clipboard sync, and input acknowledgements. Low and bursty.">
     <span>TCP</span><b id="n_tcp">—</b></div>
   <div class="kv" data-tip="Packets iss sends back (NACKs / RTCP / input). Staying at 0 usually means a one-way network path or a firewall blocking the return — input and loss-recovery won't work. Check that outbound UDP to the host isn't blocked.">
     <span>Uplink</span><b id="n_tx">—</b></div></div>
  <div class="card"><h3 title="Packet loss and stream-health signals.">Loss &amp; health</h3>
   <div class="kv" data-tip="RTP packets detected lost (sequence-number gaps). A few is fine; climbing steadily on an active screen = a lossy link (Wi-Fi interference, congestion). If it's paired with UDP-queue drops, raise the OS receive buffer (net.core.rmem_max on Linux).">
     <span>Loss total</span><b id="h_loss">—</b></div>
   <div class="kv" data-tip="Lost packets on SSRCs not mapped to a tile. Non-zero during a gray-out means the host is publishing extra SSRC groups iss isn't tracking yet — usually self-heals within a second or two.">
     <span>Unmapped</span><b id="h_unmap">—</b></div>
   <div class="kv" data-tip="Time since the host last delivered a frame. Large while the screen is idle is NORMAL. Large while the screen is actively changing = a stall — Force IDR, or Reconnect if it persists. (An idle remote whose display has slept also looks like this — move the mouse.)">
     <span>Last publish</span><b id="h_pub">—</b></div>
   <div class="kv" data-tip="Number of RTP SSRC groups (quality tiers) the host is streaming — typically 1–3. Rapid churn here during gray-outs means the host is re-grouping streams mid-session.">
     <span>SSRC groups</span><b id="h_ssrc">—</b></div></div>
  <div class="card"><h3 title="Internal buffers — growth here means iss can't keep up.">Queues</h3>
   <div class="kv" data-tip="Video receive-queue depth / capacity + drop count. Depth should sit near 0; sustained drops mean iss can't drain UDP fast enough — raise the OS receive buffer (net.core.rmem_max on Linux, to ~33 MB).">
     <span>UDP video</span><b id="q_vid">—</b></div>
   <div class="kv" data-tip="Audio/RTCP receive-queue depth / capacity + drops. Same remedy as UDP video if it drops.">
     <span>UDP ctrl</span><b id="q_ctrl">—</b></div>
   <div class="kv" data-tip="Frames waiting to be decoded. Growing = the decoder can't keep up (common with software decode — switch to a HW decoder or lower the resolution). 'n/a' on browser sessions.">
     <span>Decode q</span><b id="q_dec">—</b></div>
   <div class="kv" data-tip="Time from a complete frame arriving to it being decoded. Sub-millisecond on a GPU decoder; high or growing = a decode bottleneck (software decode, or an overloaded GPU).">
     <span>Decode lat</span><b id="q_lat">—</b></div></div>
 </div>
 <div class="tiles"><div class="tilegrid" id="tiles"></div></div>
 <div class="logwrap"><div class="loghdr">Log <span class="meta">— drag to select &amp; copy</span></div><pre id="log"></pre></div>
 <div id="modal"><div class="modalbox">
   <h2>Another user is at the console</h2>
   <p><span id="modaluser">Someone is signed in at this Mac.</span> Share their existing session, or sign in as your own user on a separate display (alt-session)?</p>
   <div class="modalbtns"><button class="btn" onclick="choose('share')">Share their session</button><button class="btn primary" onclick="choose('alt')">Sign in separately</button></div>
 </div></div>
<script>
 const $ = id => document.getElementById(id);
 function act(a){ fetch('/action',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'action='+a}); }
 function choose(c){ fetch('/action',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'action=session_choice&choice='+c}); document.getElementById('modal').classList.remove('show'); }
 const set = (id,v) => { const e=$(id); if(e) e.textContent = (v==null||v==='')?'—':v; };
 const upt = s => { s=Math.floor(s||0); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60; return (h?h+'h ':'')+(h||m?m+'m ':'')+x+'s'; };
 const qf = q => q ? `${q.depth}/${q.cap}` + (q.drop?` ⓧ${q.drop}`:'') : '—';
 function tiles(ts){
   const host=$('tiles'); if(!ts||!ts.length){ host.innerHTML=''; return; }
   const maxf = Math.max(1, ...ts.map(t=>t.fps||0));
   host.innerHTML = ts.map((t,i)=>{
     const fps=t.fps||0, loss=t.loss_s||0;
     return `<div class="tile"><div class="t">tile ${i}</div><div>${fps.toFixed(1)} fps`+
            (loss?` · <span style="color:#e06c75">${loss} lost</span>`:'')+`</div>`+
            `<div class="bar"><i style="width:${Math.min(100,fps/maxf*100)}%"></i></div></div>`;
   }).join('');
 }
 const es = new EventSource('/events');
 es.addEventListener('status', e => { const s=JSON.parse(e.data);
   $('dot').className='dot '+(s.status||'idle'); set('s_status', s.status);
   set('host', (s.host||'') + (s.frontend?' · '+s.frontend:'') + (s.pid?'  (pid '+s.pid+')':'')); });
 es.addEventListener('hello', e => { const h=JSON.parse(e.data);
   if(h.canvas) set('s_canvas', h.canvas.w+'×'+h.canvas.h + (h.canvas.tiles?'  ·  '+h.canvas.tiles+' tiles':''));
   if(h.decoder){ set('decoder', h.decoder); set('s_dec', h.decoder); } });
 es.addEventListener('snapshot', e => { const d=JSON.parse(e.data), rx=d.rx||{}, uq=d.udp_q||{};
   const pub = d.last_publish_age_s;
   set('uptime', upt(d.uptime_s));
   set('decoder', d.decoder); set('s_dec', d.decoder);
   set('s_tiles', (d.tiles||[]).length);
   set('n_vid', `${rx.video_pps||0} pps · ${rx.video_mbps||0} Mbps`);
   set('n_audio', `${rx.audio_pps||0} pps · ${rx.audio_kbps||0} kbps`);
   set('n_rtcp', `${rx.rtcp_pps||0} pps · ${rx.rtcp_kbps||0} kbps`);
   set('n_tcp', `${rx.tcp_pps||0} pps`); set('n_tx', `${(d.tx||{}).pps||0} pps`);
   set('h_loss', d.loss_total); set('h_unmap', d.loss_unmapped);
   set('h_pub', (pub!=null && pub>=0) ? pub+'s ago' : '—'); set('h_ssrc', d.ssrc_groups);
   set('q_vid', qf(uq.video)); set('q_ctrl', qf(uq.ctrl));
   set('q_dec', qf(d.decode_q));
   set('q_lat', d.decode_latency_ms!=null?d.decode_latency_ms+' ms':'—');
   tiles(d.tiles); });
 const logEl=$('log');
 es.addEventListener('log', e => { const line=JSON.parse(e.data);
   const atBottom = logEl.scrollHeight-logEl.scrollTop-logEl.clientHeight < 40;
   const lvl = line.includes(' ERROR ')?'lvl-ERROR':line.includes(' WARNING ')?'lvl-WARN':line.includes(' DEBUG ')?'lvl-DEBUG':'';
   const s=document.createElement('span'); s.className=lvl; s.textContent=line+'\\n'; logEl.appendChild(s);
   if(atBottom) logEl.scrollTop=logEl.scrollHeight; });
 es.addEventListener('video_url', e => { try{ const u=(JSON.parse(e.data)||{}).url; const b=document.getElementById('videobtn'); if(u&&b){ b.href=u; b.style.display=''; } }catch(_){} });
 es.addEventListener('session_choice', e => { let u=''; try{ u=(JSON.parse(e.data)||{}).console_user||''; }catch(_){}
   document.getElementById('modaluser').textContent = u ? (u+' is signed in at this Mac.') : 'Someone is signed in at this Mac.';
   document.getElementById('modal').classList.add('show'); });
</script></body></html>"""


# Friendly labels for the registry's decoder specs. The dropdown only lists
# decoders SUPPORTED ON THIS CLIENT PLATFORM (via the registry), so e.g.
# VideoToolbox never shows on Windows/Linux and Intel QSV never shows on macOS.
_DECODER_LABELS = {
    "libav-hevc420":    "HEVC 4:2:0 — GPU with CPU fallback",
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
    if os.environ.get("ISS_VIDEO_CODEC", "").lower() == "hevc":
        frontend = "desktop"
    import html as _html
    return (_FORM
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
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        values = {k: v[0] for k, v in parse_qs(raw).items()}
        if self.path == "/action":
            action = values.get("action")
            if action == "fir":
                _send_command({"action": "fir"})
            elif action == "reconnect":
                threading.Thread(target=_reconnect, daemon=True).start()
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
    print(f"iShareScreen → {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
