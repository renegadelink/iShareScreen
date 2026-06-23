"""Desktop frontend: glfw window + wgpu render loop + input forwarding.

The loop pumps glfw events, drains fresh tiles from `Session`, uploads
them to the GPU, and presents. The decoder runs on its own thread
inside `Session`; audio decode + playback also live off-thread (see
`audio_sink.AudioSink`), so neither competes with the input forwarder
on the render thread.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import threading
import time
from typing import Optional

import glfw
import wgpu
from rendercanvas.glfw import RenderCanvas

from ...proxy.protocol.negotiation import AdvertiseDims
from ...proxy.session import Session, SessionConfig
from .audio_sink import make_audio_sink
from .gpu import Renderer
from .keymap import GLFW_KEY_TO_X11, glfw_button_to_rfb_bit


# X11 keysyms for the Cmd-equivalent (Super_L / Super_R). When Ctrl→Cmd
# remap is on we substitute these for KEY_LEFT_CONTROL / KEY_RIGHT_CONTROL
# so the user's Linux Ctrl+C / Ctrl+A / Ctrl+V etc. land as Cmd+C/A/V
# on the macOS host. The Linux WM (GNOME) typically grabs the actual
# Super key for the activities overview, so without this remap there's
# no usable way to trigger Cmd-shortcuts on the Mac.
_KEYSYM_SUPER_L = 0xffeb
_KEYSYM_SUPER_R = 0xffec


log = logging.getLogger(__name__)


# How long the loop blocks waiting for the next tile before pumping
# events again. Short enough to stay responsive, long enough that we
# don't busy-spin and starve the decoder thread.
_FRESH_TILE_WAIT_S = 0.005


# ── dynamic resolution ───────────────────────────────────────────────
# When the viewer window is resized we re-advertise the host's virtual
# display to match (Session.send_dynamic_resolution), so the remote
# re-renders sharp at the new size instead of stretching a fixed canvas.
# This is an IN-BAND request on the live connection (no reconnect), but
# the server still has to restart the HEVC encoder, so we debounce +
# threshold the requests so a click-drag resize doesn't fire one per pixel.
_RESIZE_DEBOUNCE_S = 0.5        # window must hold a new size this long
_RESIZE_MIN_DELTA_PX = 32       # ignore sub-threshold jitter on either axis
_RESIZE_MIN_INTERVAL_S = 2.5    # floor between consecutive resize requests
_MIN_ADVERTISE_W = 640
_MIN_ADVERTISE_H = 480
_MAX_ADVERTISE_W = 1920  # server backing 3840 / mode-table ratio 2
_MAX_ADVERTISE_H = 1080


def _even(n: int) -> int:
    """Round down to an even number. HEVC tile geometry prefers even
    dimensions; odd sizes can confuse the encoder's CTU padding."""
    return n - (n & 1)


def _auto_advertise_dims() -> tuple[int, int]:
    """Initial virtual-display size when no fixed --advertise was given:
    ~85% of the primary monitor's work area (its usable region, minus
    menu bar / dock / taskbar) so the viewer opens clearly windowed and
    resizable. Returns logical screen-coordinate units — the same space
    glfw.get_window_size() reports, which the input mapper and the
    resize tracker both use. Falls back to 1280x800 when no monitor can
    be enumerated (headless / unusual WM)."""
    ww = wh = 0
    try:
        glfw.init()  # idempotent; rendercanvas re-inits when it builds the window
        mon = glfw.get_primary_monitor()
        if mon:
            area = glfw.get_monitor_workarea(mon)
            if area and area[2] > 0 and area[3] > 0:
                ww, wh = area[2], area[3]
            else:
                vm = glfw.get_video_mode(mon)
                ww, wh = vm.size.width, vm.size.height
    except Exception as e:
        log.warning("monitor auto-detect failed (%s); falling back to 1280x800", e)
    if ww <= 0 or wh <= 0:
        ww, wh = 1280, 800
    # Clamp to the same cap the resize path uses (_MAX_ADVERTISE). On a
    # large monitor an un-clamped 0.85×work-area would build a 0x1d whose
    # scaled mode-table rows exceed the advertised max_width/height, which
    # the server answers with a degenerate fallback canvas.
    w = max(_MIN_ADVERTISE_W, min(_even(int(ww * 0.85)), _MAX_ADVERTISE_W))
    h = max(_MIN_ADVERTISE_H, min(_even(int(wh * 0.85)), _MAX_ADVERTISE_H))
    return w, h


# The host caps the virtual display's backing (pixel) canvas at this size; a
# request whose backing would exceed it is rejected/degraded by the host (so
# we must never ask for more — same as the native viewer, which clamps the
# window to what the remote supports).
_HOST_MAX_BACKING_W = 3840
_HOST_MAX_BACKING_H = 2160


def _display_scale(glfw_window=None) -> int:
    """The LOCAL display's backing-scale factor: 2 = Retina/HiDPI, 1 = standard.

    HiDPI 'auto' matches the host render to this so the stream maps 1:1 to the
    client's pixels — crisp on Retina (2×) and on non-Retina (1×, e.g. most
    Linux/Windows displays) alike, with the right bandwidth for each. Reads
    GLFW's content scale (per-window once the window exists, else the primary
    monitor for the pre-window initial advertise). Falls back to 2 (the
    macOS-Retina common case) if detection fails.
    """
    try:
        if glfw_window is not None:
            sx, _sy = glfw.get_window_content_scale(glfw_window)
        else:
            glfw.init()  # idempotent
            mon = glfw.get_primary_monitor()
            sx, _sy = glfw.get_monitor_content_scale(mon) if mon else (2.0, 2.0)
        return 2 if int(round(sx)) >= 2 else 1
    except Exception as e:
        log.debug("display-scale detect failed (%s); assuming 2× Retina", e)
        return 2


def _resolve_hidpi_request(
    mode: str, win_w: int, win_h: int, client_scale: int = 2,
) -> tuple[int, int, int]:
    """Map (HiDPI mode, window logical size, client display scale) →
    (req_w, req_h, hidpi_scale).

    The request preserves the window's aspect and is clamped so the resulting
    backing (= request × scale) never exceeds the host's 3840×2160 ceiling —
    so we never over-stretch past what the remote can render.

      mode 'on'   → always 2× (Retina); logical ceiling 1920×1080.
      mode 'off'  → always 1× (flat);    logical ceiling 3840×2160.
      mode 'auto' → match the LOCAL display: 2× on a Retina client (host
                    backing maps 1:1 to the client's Retina pixels), 1× on a
                    non-Retina client (1:1 to the client's pixels — crisp and
                    low-bandwidth). 2× is downgraded to 1× when it wouldn't fit
                    the host backing cap (window logical > 1920×1080), so a
                    fullscreen Retina client still gets the true large desktop
                    rather than an upscaled small one.
    """
    win_w = max(1, win_w)
    win_h = max(1, win_h)
    if mode == "off":
        scale = 1
    elif mode == "on":
        scale = 2
    else:  # auto: match the client display scale, but only 2× when it fits
        scale = 2 if (client_scale >= 2
                      and win_w * 2 <= _HOST_MAX_BACKING_W
                      and win_h * 2 <= _HOST_MAX_BACKING_H) else 1
    max_w = _HOST_MAX_BACKING_W // scale
    max_h = _HOST_MAX_BACKING_H // scale
    fit = min(max_w / win_w, max_h / win_h, 1.0)  # shrink-only, keep aspect
    req_w = max(_MIN_ADVERTISE_W, _even(int(win_w * fit)))
    req_h = max(_MIN_ADVERTISE_H, _even(int(win_h * fit)))
    return req_w, req_h, scale


def run(
    config: SessionConfig,
    *,
    title: str = "iShareScreen",
    auto_quit_secs: int = 0,
    **_unused: object,
) -> int:
    """Open the window streaming `config`. Blocks until close (or
    `auto_quit_secs` elapses; 0 = forever)."""
    log.info("opening desktop frontend → %s", config.host)
    # Dynamic resolution: with no fixed --advertise, size the host's
    # virtual display to the local monitor before connecting; if dynamic
    # tracking is on, the render loop re-advertises on window resize.
    dynamic = config.dynamic_resolution
    if config.advertise is None:
        aw, ah = _auto_advertise_dims()
        rw, rh, scale = _resolve_hidpi_request(
            config.hidpi, aw, ah, _display_scale())
        config = dataclasses.replace(
            config, advertise=AdvertiseDims(
                width=rw, height=rh, hidpi_scale=scale),
        )
        log.info("auto-detected initial viewer %dx%d → request %dx%d @%dx "
                 "(hidpi=%s dynamic=%s)",
                 aw, ah, rw, rh, scale, config.hidpi, dynamic)
    session = Session(config)
    session.connect()

    # Audio playback. `make_audio_sink()` returns None when the OS
    # has no usable output device or sounddevice is missing — we
    # silently continue with video-only in that case. When the sink
    # is up, the proxy's audio rx thread feeds it directly; the
    # render thread never touches it.
    audio_sink = make_audio_sink() if config.audio else None
    if audio_sink is not None:
        session.set_audio_callback(audio_sink.feed)

    # Clipboard sync (both directions) lives in Session — every frontend
    # gets it for free.

    # Cursor pseudo-encoding: by default the daemon stops baking the
    # system cursor into the encoded frames (saves ~0.5-1 Mbps of
    # cursor-motion bandwidth) and instead sends cursor pixmaps over
    # the RFB control channel. We hand each one to GLFW so the local
    # OS cursor adopts the host's shape — I-beam over text, resize
    # cursors over window edges, busy spinner, etc. Set
    # ISS_LEGACY_CURSOR=1 to revert to the cursor-in-framebuffer path.
    # Captured-by-reference so we can flip cursors from the proxy rx
    # thread; GLFW calls below all run on the render thread.
    # Sentinel that distinguishes "no pending change" from "pending =
    # None (revert to OS default)" — both pass `is None` so we use a
    # unique object instead.
    _NO_PENDING = object()
    _pending_cursor: dict[str, "Optional[object]"] = {"img": _NO_PENDING}
    _cursor_lock = threading.Lock()

    # GLFW set_cursor thread-safety:
    #   macOS: Cocoa requires NSCursor calls on main thread
    #   Linux X11 / Windows: GLFW handles cross-thread set_cursor
    # On Linux/Windows we apply directly from the proxy thread so we
    # don't lose intermediate cursor frames between video-frame-paced
    # render iterations. On macOS we defer to the render loop.
    # `_set_cursor_now` is defined later (after the GLFW window setup);
    # Python closure resolution happens at call time, by which point
    # the callback will only fire post-connect when everything's set up.
    import sys as _sys
    _cursor_direct_apply = _sys.platform in ("linux", "win32")

    # Cursor: by default we hide the local OS cursor and render the host's
    # separately-sent cursor (enc 1104) ourselves as a wgpu overlay — the way
    # the native viewer does. The pointer is never the local system cursor
    # floating over the remote screen; it's a crisp client-rendered sprite at
    # the pointer position. On the HP/curtain path the host *also* bakes a
    # cursor into the video (Apple's AVCScreenCapture composites it on the
    # SkyLight virtual display regardless of the do_not_send_cursor flag,
    # confirmed in screensharingd/ScreensharingAgent); our rendered cursor
    # sits on top of it, so you see one crisp cursor — a faint baked ghost
    # may trail it only during fast motion. ISS_LOCAL_CURSOR=1 reverts to the
    # legacy behaviour (reshape the local OS cursor; no overlay).
    _canvas_cursor = os.environ.get("ISS_LOCAL_CURSOR") != "1"
    _last_cursor_img: dict[str, object] = {"img": None}

    def _on_cursor(img):
        if _canvas_cursor:
            # Defer to the render thread — the overlay texture upload has to
            # run on the thread that owns the wgpu device.
            with _cursor_lock:
                _pending_cursor["img"] = img
            return
        if _cursor_direct_apply:
            _set_cursor_now(img)
        else:
            with _cursor_lock:
                _pending_cursor["img"] = img

    session.set_cursor_callback(_on_cursor)

    canvas_w, canvas_h = session.canvas_dims  # backing/pixel size for GPU
    scaled_w, scaled_h = session.scaled_dims  # logical size for window
    server_w, server_h = session.server_dims
    num_tiles = session.num_tiles
    # Apple's HEVC encodes each tile padded up to a CTU boundary —
    # typically 304 rows for a 1920×1200/4-tile canvas, not the 300
    # you'd get from canvas_h//num_tiles. Refined to tile.height on the
    # first tile that arrives.
    slot_h = canvas_h // num_tiles
    log.info("session ready: canvas=%dx%d scaled=%dx%d server=%dx%d tiles=%d hw=%s",
             canvas_w, canvas_h, scaled_w, scaled_h, server_w, server_h,
             num_tiles, session.hw_accel)

    # ── window + wgpu surface ──────────────────────────────────────────
    win_w = scaled_w or canvas_w
    win_h = scaled_h or canvas_h
    window = RenderCanvas(title=title, size=(win_w, win_h), max_fps=120)
    glfw_window = window._window  # for raw glfw input callbacks only
    # In canvas-cursor mode we render the host's cursor as a wgpu overlay and
    # hide the local system pointer — but NOT yet. Hiding it now, before the
    # overlay has both a shape (a cursor pixmap arrived) and a position (the
    # pointer has moved over the window at least once), would leave the user
    # with no visible pointer at all on a static / pre-first-pixmap screen.
    # The render loop hides it the moment the overlay can actually draw (see
    # the `_os_cursor_hidden` block below), so the OS cursor stays as a
    # fallback until the host cursor is ready to take over.
    _os_cursor_hidden = {"v": False}
    # Window resizes freely. When the host fell back to a frame smaller
    # than the advertised canvas, the renderer (see gpu.Renderer.draw)
    # centers that real frame in the window at its native on-screen size
    # so the leftover shows as symmetric black bars on all four sides,
    # rather than pinning the image to the top-left corner. Matches
    # Apple's native viewer.

    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    surface_ctx = window.get_context("wgpu")
    # Pick the linear variant so we don't double-encode sRGB.
    preferred = surface_ctx.get_preferred_format(adapter)
    surface_format = (
        preferred[: -len("-srgb")] if preferred.endswith("-srgb") else preferred
    )
    surface_ctx.configure(
        device=device, format=surface_format, alpha_mode="opaque",
    )

    renderer = Renderer(device, surface_format, canvas_w, canvas_h)

    def _cursor_render_scale() -> float:
        """On-screen size factor for the cursor sprite.

        Draw the cursor pixmap 1:1 by default. The host delivers it at its
        *backing* (device-pixel) resolution — a 17x23 arrow is already the
        Retina 2x arrow — and the wgpu surface is the logical (point) size,
        so one sprite pixel maps to one logical point, i.e. the host's true
        device-pixel arrow. Scaling it up by the Retina factor (as an earlier
        revision did, via glfw content scale or the backing/scaled ratio)
        double-counts that 2x and makes the cursor twice too large.

        ISS_CURSOR_SCALE multiplies this for manual tuning."""
        mult = 1.0
        override = os.environ.get("ISS_CURSOR_SCALE")
        if override:
            try:
                mult = max(0.1, float(override))
            except ValueError:
                pass
        return mult

    if _canvas_cursor:
        renderer.set_cursor_scale(_cursor_render_scale())

    # ── input forwarding ───────────────────────────────────────────────
    button_mask = 0
    cursor: Optional[tuple[int, int]] = None
    # Whether the local pointer is currently over the window content. When it
    # leaves, GLFW stops firing cursor-pos events, so without this the overlay
    # would freeze its last in-window sprite at the edge (a ghost cursor). The
    # draw callback suppresses the overlay while the pointer is outside.
    _pointer_in_window = {"v": True}

    def to_canvas(wx: float, wy: float) -> Optional[tuple[int, int]]:
        """Map glfw cursor (in glfw window coords) → canvas coords for
        `InputController.pointer_event`.

        This MUST invert the exact transform in `gpu.Renderer.draw`: the
        decoded content (`content_dims()`, pinned to the top-left of the
        canvas textures) is scaled by ONE uniform factor — the largest that
        fits the window — then centered, with black bars filling the rest.
        So here: undo the centering offset, divide into the centered content
        rect, then map onto the content's texels. A cursor over a black bar
        clamps to the nearest content edge. The window→content fractions are
        identical whether measured in logical points (this function) or
        physical surface pixels (draw's target), so `glfw.get_window_size`
        points are the right space.

        Maps to canvas dims, NOT server-init dims: the daemon's
        composite ServerInit (e.g. 2940×1912 when a SkyLight virtual
        display is active alongside the panel) doesn't correspond to
        what's rendered in the iss window — only the canvas does.
        """
        win_w, win_h = glfw.get_window_size(glfw_window)
        if win_w == 0 or win_h == 0:
            return None
        cw, ch = renderer.content_dims()  # real frame size, canvas texels
        if cw <= 0 or ch <= 0 or canvas_w <= 0 or canvas_h <= 0:
            return None
        # Mirror draw()'s uniform-scale fit: one factor on both axes (so the
        # mapping never skews), sized to the largest content rect that fits
        # the window, then centered.
        scale = min(win_w / cw, win_h / ch)
        rect_w = cw * scale
        rect_h = ch * scale
        ox, oy = (win_w - rect_w) / 2.0, (win_h - rect_h) / 2.0
        u = (wx - ox) / rect_w if rect_w else 0.0
        v = (wy - oy) / rect_h if rect_h else 0.0
        u = min(1.0, max(0.0, u))
        v = min(1.0, max(0.0, v))
        sx = min(cw - 1, int(u * cw))
        sy = min(ch - 1, int(v * ch))
        return (max(0, min(canvas_w - 1, sx)),
                max(0, min(canvas_h - 1, sy)))

    def on_cursor_pos(_w, x, y):
        nonlocal cursor
        cursor = to_canvas(x, y)
        if cursor is not None:
            session.input.pointer_event(button_mask, cursor[0], cursor[1])

    def on_mouse_button(_w, button, action, _mods):
        nonlocal button_mask
        bit = glfw_button_to_rfb_bit(button)
        if bit == 0:
            return
        if action == glfw.PRESS:
            button_mask |= bit
        elif action == glfw.RELEASE:
            button_mask &= ~bit
        if cursor is not None:
            session.input.pointer_event(button_mask, cursor[0], cursor[1])

    # Wheel velocity acceleration. macOS's native scroll amplifies
    # consecutive fast events exponentially; RFB doesn't carry that
    # across the wire, so we recreate the curve here. Without it,
    # RFB scroll feels glacial vs the OS-native scroll the user is
    # used to.
    _wheel_accum = [0.0]
    _wheel_last_t = [0.0]

    def on_scroll(_w, _x, dy):
        if cursor is None or dy == 0:
            return
        now = time.monotonic() * 1000.0  # ms
        dt = now - _wheel_last_t[0]
        _wheel_last_t[0] = now
        if dt < 25:    mult = 10.0
        elif dt < 50:  mult = 6.0
        elif dt < 100: mult = 3.5
        elif dt < 200: mult = 2.0
        elif dt < 350: mult = 1.4
        else:          mult = 1.0
        # glfw's dy is in wheel ticks already (1.0 per notch on a
        # discrete wheel; fractional on trackpads). `scroll_event`
        # interprets dy<0 as scroll-up.
        _wheel_accum[0] += -dy * mult
        ticks = int(_wheel_accum[0])
        if ticks == 0:
            ticks = -1 if dy > 0 else 1
            _wheel_accum[0] = 0.0
        else:
            _wheel_accum[0] -= ticks
        ticks = max(-50, min(50, ticks))
        session.input.scroll_event(cursor[0], cursor[1], 0, ticks)

    # Ctrl→Cmd remap: rewrite local Ctrl into the Mac's Cmd modifier
    # so Ctrl+C / Ctrl+V "feel right". Off by default now — the OS
    # keyboard grab below routes the actual Win/Super key to iss
    # without WM interception, so users get the Mac's native Cmd by
    # pressing Win/Super directly. The remap conflicted with bare-Ctrl
    # shortcuts (Ctrl+W expecting word-delete became Cmd+W close-tab,
    # etc.). Opt back in with ISS_CTRL_AS_CMD=1 if you prefer the
    # rewrite (e.g. on a Mac without keyboard-grab support).
    ctrl_as_cmd = os.environ.get("ISS_CTRL_AS_CMD", "0") == "1"
    if ctrl_as_cmd:
        log.info("input: Ctrl→Cmd remap on (set ISS_CTRL_AS_CMD=0 to disable)")

    def on_key(_w, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            return
        # Force-IDR (formerly local F12) now lives in the TUI's session
        # screen as the `f` keybinding -- it sends an `{"action":"fir"}`
        # command over the control socket which the session's command
        # handler translates into the same `request_fir(None)` call.
        # The desktop window stays a pure viewer; all keys forward.
        # Ctrl→Cmd: substitute Super for the Control modifier-key
        # events so when iss tells the Mac "Control_L is down" it
        # actually says "Super_L is down" → macOS treats it as Cmd.
        if ctrl_as_cmd:
            if key == glfw.KEY_LEFT_CONTROL:
                session.input.key_event(action != glfw.RELEASE, _KEYSYM_SUPER_L)
                return
            if key == glfw.KEY_RIGHT_CONTROL:
                session.input.key_event(action != glfw.RELEASE, _KEYSYM_SUPER_R)
                return
        keysym = GLFW_KEY_TO_X11.get(key, 0)
        if keysym == 0:
            # Printable key. Normally `on_char` handles these so the
            # user's keyboard layout (Dvorak / AZERTY / Shift) is
            # respected. But `on_char` does NOT fire when a modifier
            # is held (e.g. Ctrl+C, Cmd+V), so we'd drop the letter
            # and only forward the modifier press — host sees Ctrl
            # held + Ctrl release, no C in between. Catch the
            # modifier-held case here and synthesize a keysym for the
            # raw ASCII letter / digit. X11 keysyms for A–Z and 0–9
            # are just their ASCII codepoints.
            held = mods & (glfw.MOD_CONTROL | glfw.MOD_SUPER | glfw.MOD_ALT)
            if held and glfw.KEY_A <= key <= glfw.KEY_Z:
                keysym = ord("a") + (key - glfw.KEY_A)
            elif held and glfw.KEY_0 <= key <= glfw.KEY_9:
                keysym = ord("0") + (key - glfw.KEY_0)
            elif held and key == glfw.KEY_SPACE:
                keysym = ord(" ")
            else:
                return
        session.input.key_event(action != glfw.RELEASE, keysym)

    def on_char(_w, codepoint):
        session.input.key_event(True, codepoint)
        session.input.key_event(False, codepoint)

    def on_cursor_enter(_w, entered):
        # entered != 0 → pointer entered the content area; 0 → left it.
        _pointer_in_window["v"] = bool(entered)
        # Force one repaint so the overlay clears (on leave) or reappears (on
        # enter) even over a static screen with no fresh video tile.
        _cursor_dirty["v"] = True

    glfw.set_cursor_pos_callback(glfw_window, on_cursor_pos)
    glfw.set_mouse_button_callback(glfw_window, on_mouse_button)
    glfw.set_scroll_callback(glfw_window, on_scroll)
    glfw.set_cursor_enter_callback(glfw_window, on_cursor_enter)
    glfw.set_key_callback(glfw_window, on_key)
    glfw.set_char_callback(glfw_window, on_char)

    # Grab the keyboard at the OS level so Win/Super combos and
    # Alt+Tab go to the remote Mac instead of the local desktop. On by
    # default; opt out with ISS_GRAB_KEYBOARD=0 (e.g. when debugging
    # local shortcuts while iss is foreground).
    if os.environ.get("ISS_GRAB_KEYBOARD", "1") != "0":
        from .keyboard_grab import make_grab
        # The Win32 backend needs to deliver intercepted keys directly
        # to iss's input pipeline (it bypasses GLFW for hooked keys —
        # see keyboard_grab.py for why). Hand it a sender bound to the
        # session's InputController so Win+C/V/etc. reach the Mac.
        kbd_grab = make_grab(
            glfw_window,
            key_sender=lambda is_down, ks: session.input.key_event(is_down, ks),
        )
        # Initial enable if window already focused at startup.
        if glfw.get_window_attrib(glfw_window, glfw.FOCUSED):
            kbd_grab.enable()

        def on_window_focus(_w, focused):
            if focused:
                kbd_grab.enable()
            else:
                kbd_grab.disable()

        glfw.set_window_focus_callback(glfw_window, on_window_focus)
    else:
        kbd_grab = None

    # ── render loop ────────────────────────────────────────────────────
    first_seen: list[bool] = [False] * num_tiles
    slot_h_resolved = False
    # Last cursor position we painted, so a pointer move over a static
    # screen still repaints the overlay (the cursor isn't baked into video).
    _last_drawn_cursor: "Optional[tuple[int, int]]" = None
    deadline = time.monotonic() + auto_quit_secs if auto_quit_secs > 0 else float("inf")

    # rendercanvas presents only from inside the draw callback (it owns
    # the swap chain). We do the GPU work here and trigger via
    # `force_draw()` from the main loop so render is paced to tile
    # arrivals, not the rendercanvas internal animation timer.
    # Set by the draw callback when wgpu reports the GPU device is
    # lost (Windows DXGI TDR, driver reset, display mode change). Main
    # loop checks this and shuts down cleanly instead of letting the
    # underlying Rust panic kill the process with an unhelpful trace.
    _device_lost: dict[str, bool] = {"v": False}

    def draw_callback():
        try:
            target = surface_ctx.get_current_texture()
            if _canvas_cursor:
                # Suppress the overlay while the pointer is outside the window
                # (None hides it) so it doesn't freeze as a ghost at the edge.
                renderer.set_cursor_pos(
                    cursor if _pointer_in_window["v"] else None)
            renderer.draw(target.create_view(), target.width, target.height)
        except Exception as e:
            msg = str(e)
            if "device is lost" in msg.lower() or "Validation Error" in msg:
                if not _device_lost["v"]:
                    log.error(
                        "GPU device lost (likely Windows DXGI TDR / driver "
                        "reset / display change) — closing viewer cleanly. "
                        "Restart iss to recover. wgpu: %s", msg[:200],
                    )
                    _device_lost["v"] = True
                return
            raise

    window.request_draw(draw_callback)

    # Currently-applied cursor + GLFW handle. We hold the GLFW cursor
    # alive in a ref because glfw.set_cursor doesn't take ownership;
    # destroying the cursor while it's set crashes the X server.
    _cursor_state: dict[str, "Optional[object]"] = {
        "img": None, "handle": None,
    }

    import ctypes

    def _scale_cursor_rgba(rgba: bytes, w: int, h: int,
                           hx: int, hy: int, scale: int,
                           ) -> tuple[bytes, int, int, int, int]:
        """Integer-scale RGBA cursor pixmap (nearest-neighbor). Integer
        scale keeps edges crisp; non-integer would blur small bitmap
        cursors. Inner loop uses row-replication + bytes slicing
        instead of a per-pixel Python loop."""
        if scale <= 1:
            return rgba, w, h, hx, hy
        nw, nh = w * scale, h * scale
        out = bytearray(nw * nh * 4)
        row_bytes = nw * 4
        for sy in range(h):
            scaled_row = bytearray(row_bytes)
            for sx in range(w):
                src_off = (sy * w + sx) * 4
                pix = rgba[src_off:src_off + 4]
                for k in range(scale):
                    o = (sx * scale + k) * 4
                    scaled_row[o:o + 4] = pix
            for k in range(scale):
                base = (sy * scale + k) * row_bytes
                out[base:base + row_bytes] = scaled_row
        return bytes(out), nw, nh, hx * scale, hy * scale

    # Minimum on-screen cursor size in device pixels (Linux/Windows
    # only). Some host cursors are very small in their native pixmap
    # (the I-beam ships at 9x18 from Apple's daemon) and on a
    # non-HiDPI Linux desktop that would render as 9x18 device pixels
    # — practically invisible. Bump small cursors up to this floor
    # via integer scaling so every shape is recognisable.
    _MIN_CURSOR_DEVICE_PX = 24

    def _build_glfw_cursor(img):
        """Translate our `_CursorImage` into a GLFW cursor handle sized
        to match the local OS cursor.

        GLFW's cursor image dimensions are interpreted differently per
        platform:
          - macOS: dimensions are LOGICAL POINTS (NSCursor expects
            point-sized images and the OS upscales to device pixels).
            The daemon already ships at host-pixel size which matches
            point-level expectations on Retina-to-Retina, so we pass
            through unmodified.
          - Linux X11 / Windows: dimensions are DEVICE PIXELS. We
            integer-scale by max(content_scale, ceil(MIN/longer_side))
            so HiDPI displays match the local cursor's footprint AND
            tiny cursors (like the 9×18 I-beam) get bumped up enough
            to be visible. Without the floor, the I-beam was effectively
            invisible on non-HiDPI Linux."""
        if _sys.platform == "darwin":
            scale = 1
        else:
            try:
                sx, _sy = glfw.get_window_content_scale(glfw_window)
                dpi_scale = max(1, int(round(sx)))
            except Exception:
                dpi_scale = 1
            long_side = max(img.width, img.height)
            min_scale = -(-_MIN_CURSOR_DEVICE_PX // long_side)  # ceil div
            scale = max(dpi_scale, min_scale)
        rgba, w, h, hx, hy = _scale_cursor_rgba(
            img.rgba, img.width, img.height,
            img.hotspot_x, img.hotspot_y, scale,
        )
        n_bytes = len(rgba)
        arr_type = ctypes.c_ubyte * n_bytes
        buf = arr_type.from_buffer_copy(rgba)
        gimg = glfw._GLFWimage()
        gimg.width = w
        gimg.height = h
        gimg.pixels = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
        gimg.pixels_array = buf
        handle = glfw._glfw.glfwCreateCursor(
            ctypes.pointer(gimg), hx, hy,
        )
        return handle, gimg

    # GLFW cursor handle cache, keyed by id(_CursorImage). The host's
    # daemon sends rapid alternating cache hits at UI element
    # boundaries (20-50 ms swap intervals). On X11 each set_cursor
    # with a fresh handle does 4 X-server round trips (create + load
    # + define + free old), producing visible tearing under rapid
    # alternation. Caching the handle collapses set_cursor to a
    # single XDefineCursor — fast enough that rapid alternations
    # don't visibly tear.
    #
    # Each entry stores (img, handle, gimg). The img reference
    # both keeps the _CursorImage alive (preventing id() reuse) AND
    # acts as an identity guard: if the proxy-side cursor cache evicts
    # an entry and Python recycles its id() for a new _CursorImage, we
    # detect that mismatch on lookup and rebuild the handle.
    _glfw_cursor_handle_cache: dict[int, "tuple[object, object, object]"] = {}

    def _set_cursor_now(img):
        """Apply a cursor immediately — direct path used on platforms
        where GLFW is thread-safe enough for set_cursor off the main
        thread (Windows + Linux/X11). Avoids the per-frame delay of
        the render-loop deferral, which was losing intermediate
        cursor frames when shape changed faster than video FPS.

        `img is None` reverts to the local OS default cursor."""
        if img is _cursor_state["img"]:
            return
        try:
            if img is None:
                glfw.set_cursor(glfw_window, None)
                handle = None
            else:
                cached = _glfw_cursor_handle_cache.get(id(img))
                if cached is None or cached[0] is not img:
                    handle, gimg = _build_glfw_cursor(img)
                    _glfw_cursor_handle_cache[id(img)] = (img, handle, gimg)
                else:
                    handle = cached[1]
                glfw.set_cursor(glfw_window, handle)
            _cursor_state["img"] = img
            _cursor_state["handle"] = handle
            # Don't destroy old handles — the cache owns them and the
            # daemon may reference any of them again at any moment via
            # cache hit. Total memory is bounded by the proxy-side
            # `_cursor_cache` cap (64 entries) since handles are keyed
            # by id() of those entries; once they get GC'd, the next
            # _set_cursor_now never finds them by id() and we'd build
            # fresh anyway. Worst case = ~64 small X cursors.
        except Exception as e:
            log.debug("cursor apply failed: %s", e)

    # Set when the overlay cursor shape changes so the render loop repaints
    # even with no fresh video tile. On a static screen (e.g. just after
    # login) the host sends a new cursor pixmap (I-beam → arrow) but no
    # pixel update — and since `do_not_send_cursor` keeps the cursor out of
    # the video, the shape change would otherwise never reach the screen.
    _cursor_dirty: dict[str, bool] = {"v": False}

    def _apply_pending_cursor():
        """Pull a pending cursor shape from the slot on the render thread and
        apply it. In canvas-cursor mode this uploads the overlay texture
        (which must happen on the device-owning thread); in legacy mode it
        sets the local OS cursor. On Linux/Windows the legacy path also
        applies directly from the proxy thread, so the slot is usually empty.

        `_pending_cursor["img"]` may legitimately be None (revert to OS
        default) — distinguished from "no pending update" by a sentinel."""
        with _cursor_lock:
            img = _pending_cursor["img"]
            _pending_cursor["img"] = _NO_PENDING
        if img is _NO_PENDING:
            return
        if _canvas_cursor:
            # img is None means the proxy hit a cursor cache-miss and asked
            # us to "revert to the OS default". In canvas mode the local OS
            # cursor is hidden, so clearing the overlay would make the
            # pointer vanish entirely (no OS cursor underneath). Keep the
            # last shape instead — a stale-but-visible cursor beats an
            # invisible one, and it self-corrects on the next real pixmap.
            # The native viewer likewise never shows an empty cursor.
            if img is None:
                return
            _last_cursor_img["img"] = img  # re-applied after a canvas rebuild
            renderer.set_cursor_image(img)
            _cursor_dirty["v"] = True
        else:
            _set_cursor_now(img)

    # ── dynamic-resolution resize tracking ─────────────────────────────
    # Mid-session resolution change works on the existing TCP connection:
    # send_dynamic_resolution() → 0x1d → server sends 0x451 → client
    # re-offers 0x1c with same keys → server restarts HEVC encoder at
    # new resolution → param sets harvested → decoder restarted → video
    # resumes sharp. No reconnect, no re-auth. The canvas change is
    # visible to the render loop via `canvas_dims` which checks
    # `_runtime_canvas_w/h` updated by the 0x451 handler.
    cur_adv_w, cur_adv_h = glfw.get_window_size(glfw_window)
    pending_size: Optional[tuple[int, int]] = None
    pending_since = 0.0
    last_resize_t = 0.0

    def _apply_new_canvas() -> None:
        nonlocal canvas_w, canvas_h, num_tiles, slot_h, slot_h_resolved
        nonlocal first_seen, renderer, cur_adv_w, cur_adv_h
        canvas_w, canvas_h = session.canvas_dims
        scaled_w, scaled_h = session.scaled_dims
        num_tiles = session.num_tiles
        slot_h = canvas_h // num_tiles if num_tiles else canvas_h
        slot_h_resolved = False
        first_seen = [False] * num_tiles
        renderer = Renderer(device, surface_format, canvas_w, canvas_h)
        # The new renderer has no cursor texture/scale; re-apply both so the
        # overlay doesn't vanish or mis-size across a resolution change.
        if _canvas_cursor:
            renderer.set_cursor_scale(_cursor_render_scale())
            if _last_cursor_img["img"] is not None:
                renderer.set_cursor_image(_last_cursor_img["img"])
        # Do NOT snap the window to the server's reply. Native keeps the
        # window where the user put it (including fullscreen) and centers
        # the remote frame inside it; when the remote can't fulfil the
        # request it returns a smaller / different-aspect canvas and the
        # renderer letter/pillarboxes it (see gpu.Renderer.draw). Snapping
        # here would instead shrink a fullscreen window down to the remote's
        # reply — the opposite of the native behaviour we're matching.
        #
        # Re-sync the resize guard to the ACTUAL window size (not the
        # server's reply). The window hasn't moved, so a reply smaller than
        # the window must NOT read as a fresh resize — otherwise _poll_resize
        # would re-fire every interval forever.
        ww, wh = glfw.get_window_size(glfw_window)
        cur_adv_w, cur_adv_h = _even(ww), _even(wh)
        log.info("dynamic resize: canvas=%dx%d scaled=%dx%d window=%dx%d "
                 "tiles=%d hw=%s",
                 canvas_w, canvas_h, scaled_w, scaled_h, cur_adv_w, cur_adv_h,
                 num_tiles, session.hw_accel)

    def _poll_resize() -> None:
        nonlocal pending_size, pending_since, last_resize_t, cur_adv_w, cur_adv_h
        now = time.monotonic()
        win_w, win_h = glfw.get_window_size(glfw_window)
        if win_w <= 0 or win_h <= 0:
            return
        tw, th = _even(win_w), _even(win_h)
        if (abs(tw - cur_adv_w) >= _RESIZE_MIN_DELTA_PX
                or abs(th - cur_adv_h) >= _RESIZE_MIN_DELTA_PX):
            if pending_size != (tw, th):
                pending_size = (tw, th)
                pending_since = now
        else:
            pending_size = None
        if (pending_size is not None
                and now - pending_since >= _RESIZE_DEBOUNCE_S
                and now - last_resize_t >= _RESIZE_MIN_INTERVAL_S):
            # Resolve the window's logical size + the HiDPI mode into the
            # request resolution AND the backing scale, preserving the window
            # aspect and clamping so backing never exceeds the host's
            # 3840×2160 cap (see _resolve_hidpi_request). The request IS the
            # viewport: when the host satisfies it the remote fills the window
            # 1:1; the renderer letterboxes only a genuine shortfall.
            nw, nh, scale = _resolve_hidpi_request(
                config.hidpi, pending_size[0], pending_size[1],
                _display_scale(glfw_window))
            # Advance the guard to the WINDOW size we acted on (tw,th),
            # NOT the clamped request (nw,nh). If the window is larger
            # than the cap, the request is clamped but the window stays
            # big; comparing future polls against the clamped value would
            # leave abs(window - cur_adv) above threshold forever and
            # re-fire a resize every interval. Tracking the window size
            # means a stable (if oversized) window never re-fires.
            cur_adv_w, cur_adv_h = tw, th
            pending_size = None
            last_resize_t = now
            log.info("resize → requesting %dx%d @%dx (hidpi=%s)",
                     nw, nh, scale, config.hidpi)
            try:
                session.send_dynamic_resolution(nw, nh, hidpi_scale=scale)
            except Exception as e:
                log.error("send_dynamic_resolution failed: %s", e)
                return

    while time.monotonic() < deadline:
        glfw.poll_events()
        _apply_pending_cursor()
        # Hide the local OS cursor only once the overlay can actually draw —
        # a shape has been uploaded (_last_cursor_img) AND a pointer position
        # is known (cursor, set on the first in-window move). Until then keep
        # the OS cursor as a visible fallback so a static / pre-first-pixmap
        # screen never shows an invisible pointer. One-shot.
        if (_canvas_cursor and not _os_cursor_hidden["v"]
                and _last_cursor_img["img"] is not None
                and cursor is not None):
            try:
                glfw.set_input_mode(
                    glfw_window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
                _os_cursor_hidden["v"] = True
            except Exception as e:
                log.debug("could not hide local cursor: %s", e)
        if window.get_closed() or glfw.window_should_close(glfw_window):
            break
        if _device_lost["v"]:
            break

        if dynamic:
            _poll_resize()
        if not session.is_connected:
            log.error("connection lost — closing viewer")
            break

        # Detect canvas changes from 0x451 handler.
        new_cw, new_ch = session.canvas_dims
        if (new_cw, new_ch) != (canvas_w, canvas_h) and new_cw and new_ch:
            _apply_new_canvas()

        session.wait_for_fresh_tile(timeout=_FRESH_TILE_WAIT_S)

        # Drain fresh decoded frames + upload.
        any_fresh = False
        for ti in range(num_tiles):
            tf = session.get_frame(ti)
            if tf is None:
                continue
            first_seen[ti] = True
            # Refine slot_h on the first tile — encoder's CTU-padded
            # picture height (typically 4 rows taller than canvas_h//
            # num_tiles). See comment at session-ready above.
            if not slot_h_resolved:
                slot_h = tf.height
                slot_h_resolved = True
            renderer.upload_tile(ti, tf, slot_h)
            any_fresh = True

        # Present after a fresh upload, OR when the overlay cursor moved or
        # changed shape — otherwise a pointer that moves/reshapes over a
        # static screen (no fresh tile) would never repaint, freezing the
        # last-drawn cursor (e.g. an I-beam stuck after login).
        cursor_moved = _canvas_cursor and cursor != _last_drawn_cursor
        if (any_fresh or cursor_moved or _cursor_dirty["v"]) and any(first_seen):
            window.force_draw()
            _last_drawn_cursor = cursor
            _cursor_dirty["v"] = False

    log.info("desktop frontend closing")
    if kbd_grab is not None:
        kbd_grab.disable()
    if audio_sink is not None:
        audio_sink.stop()
    session.close()
    # Explicit window close — triggers orderly wgpu cleanup. Without
    # this we'd rely on __del__ when locals fall out of scope, and the
    # interpreter's GC ordering can deadlock the wgpu native poller
    # (observed: main thread stuck in wgpu _release proxy_func at exit).
    try:
        window.close()
    except Exception as e:
        log.debug("window.close() failed: %s", e)
    # Fail-safe: bypass Python's interpreter-shutdown phase. We've
    # already released the network sockets, audio device, and X11
    # grab; anything left to clean up belongs to the kernel and gets
    # reclaimed by the process exit. Avoids the rare wgpu-__del__
    # deadlock at scope unwind that leaves the process hung.
    os._exit(0)


__all__ = ["run"]
