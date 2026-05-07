"""Desktop frontend: glfw window + wgpu render loop + input forwarding.

The loop pumps glfw events, drains fresh tiles from `Session`, uploads
them to the GPU, and presents. The decoder runs on its own thread
inside `Session`; audio decode + playback also live off-thread (see
`audio_sink.AudioSink`), so neither competes with the input forwarder
on the render thread.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import glfw
import wgpu
from rendercanvas.glfw import RenderCanvas

from ...proxy.session import Session, SessionConfig
from .audio_sink import make_audio_sink
from .gpu import Renderer, letterbox
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

    canvas_w, canvas_h = session.canvas_dims
    server_w, server_h = session.server_dims
    num_tiles = session.num_tiles
    # Apple's HEVC encodes each tile padded up to a CTU boundary —
    # typically 304 rows for a 1920×1200/4-tile canvas, not the 300
    # you'd get from canvas_h//num_tiles. Refined to tile.height on the
    # first tile that arrives.
    slot_h = canvas_h // num_tiles
    log.info("session ready: canvas=%dx%d server=%dx%d tiles=%d hw=%s",
             canvas_w, canvas_h, server_w, server_h, num_tiles,
             session.hw_accel)

    # ── window + wgpu surface ──────────────────────────────────────────
    window = RenderCanvas(title=title, size=(canvas_w, canvas_h), max_fps=120)
    glfw_window = window._window  # for raw glfw input callbacks only

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

    # ── input forwarding ───────────────────────────────────────────────
    button_mask = 0
    cursor: Optional[tuple[int, int]] = None

    def to_canvas(wx: float, wy: float) -> Optional[tuple[int, int]]:
        """Map glfw cursor (in glfw window coords) → canvas coords for
        `InputController.pointer_event`.

        glfw's cursor callback delivers coords in the same coordinate
        space as `glfw.get_window_size()` — no HiDPI conversion needed
        here regardless of the OS scaling factor; both ends of the
        ratio are in the same space. Earlier versions tried to
        re-scale via `rendercanvas.get_logical_size()` /
        `get_physical_size()`, which double-scaled on HiDPI laptops
        and threw the mapping off.

        Maps to canvas dims, NOT server-init dims: the daemon's
        composite ServerInit (e.g. 2940×1912 when a SkyLight virtual
        display is active alongside the panel) doesn't correspond to
        what's rendered in the iss window — only the canvas does.

        Returns None if the cursor is outside the letterboxed video.
        """
        win_w, win_h = glfw.get_window_size(glfw_window)
        if win_w == 0 or win_h == 0:
            return None
        vx, vy, vw, vh = letterbox(canvas_w, canvas_h, win_w, win_h)
        if wx < vx or wy < vy or wx >= vx + vw or wy >= vy + vh:
            return None
        sx = int((wx - vx) * canvas_w / vw)
        sy = int((wy - vy) * canvas_h / vh)
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

    # Default-on Ctrl→Cmd remap: Linux/Windows users controlling a Mac
    # almost always want Ctrl+C / Ctrl+A / Ctrl+V to map to the Mac's
    # Cmd+C / Cmd+A / Cmd+V (the WM grabs the actual Super/Win key on
    # most desktops). Disable with `ISS_CTRL_AS_CMD=0` for users who
    # genuinely want raw Ctrl on the Mac (e.g. emacs Ctrl-A = beginning
    # of line).
    ctrl_as_cmd = os.environ.get("ISS_CTRL_AS_CMD", "1") != "0"
    if ctrl_as_cmd:
        log.info("input: Ctrl→Cmd remap on (set ISS_CTRL_AS_CMD=0 to disable)")

    def on_key(_w, key, _scancode, action, mods):
        if action not in (glfw.PRESS, glfw.RELEASE, glfw.REPEAT):
            return
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
            else:
                return
        session.input.key_event(action != glfw.RELEASE, keysym)

    def on_char(_w, codepoint):
        session.input.key_event(True, codepoint)
        session.input.key_event(False, codepoint)

    glfw.set_cursor_pos_callback(glfw_window, on_cursor_pos)
    glfw.set_mouse_button_callback(glfw_window, on_mouse_button)
    glfw.set_scroll_callback(glfw_window, on_scroll)
    glfw.set_key_callback(glfw_window, on_key)
    glfw.set_char_callback(glfw_window, on_char)

    # ── render loop ────────────────────────────────────────────────────
    first_seen: list[bool] = [False] * num_tiles
    slot_h_resolved = False
    deadline = time.monotonic() + auto_quit_secs if auto_quit_secs > 0 else float("inf")

    # rendercanvas presents only from inside the draw callback (it owns
    # the swap chain). We do the GPU work here and trigger via
    # `force_draw()` from the main loop so render is paced to tile
    # arrivals, not the rendercanvas internal animation timer.
    def draw_callback():
        target = surface_ctx.get_current_texture()
        vp = letterbox(canvas_w, canvas_h, target.width, target.height)
        renderer.draw(target.create_view(), vp)

    window.request_draw(draw_callback)

    while time.monotonic() < deadline:
        glfw.poll_events()
        if window.get_closed() or glfw.window_should_close(glfw_window):
            break
        if not session.is_connected:
            log.error("connection lost — closing viewer")
            break
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

        # Present after a fresh upload. No fresh content = skip the
        # draw, swap chain holds the previous good frame.
        if any_fresh and any(first_seen):
            window.force_draw()

    log.info("desktop frontend closing")
    if audio_sink is not None:
        audio_sink.stop()
    session.close()
    return 0


__all__ = ["run"]
