# Native Windows desktop build

Run `dist/iShareScreen-Desktop.exe` (also copied to `dist/iShareScreen.exe`).
Python is bundled. This is a native Qt application laid out like macOS Screen
Sharing: a library window of saved computers, small "Connect To:" and sign-in
dialogs, and one window per remote desktop. It opens no browser and no HTTP
server. See the top-level README for the flow.

Enable Screen Sharing on the Mac under System Settings → General → Sharing. The
default (Standard) mode uses Apple Remote Desktop authentication (RFB security
type 30) and raw/zlib framebuffer rectangles over TCP 5900, with no UDP, virtual
display or curtain mode. High Performance is not released: its engine code ships
inside the package (PyAV is included for it, which is why the executable is about
95 MB) but it is hidden unless `ISS_EXPERIMENTAL_HP=1`; see
`docs/DESKTOP_REDESIGN_HANDOFF.md` for what testing showed and how it is guarded.

Standard mode has received visible desktop images from an Intel and an Apple
Silicon Mac. Implemented: keyboard, pointer, dragging, scroll, full screen, zoom
and actual size with edge scrolling, view-only, Mission Control / Show Desktop /
Launchpad buttons, two-way plain-text clipboard (per-session toggle), typing the
clipboard as keystrokes, refresh, disconnect and reconnect. Audio is not
implemented.

Use **Ctrl+Alt+Enter** to enter/exit full screen. Keyboard input goes to the Mac
while that session's remote screen is focused; held keys and buttons are
released on focus loss, disconnect and close. Standard mode uses the host's
existing resolution and does not change its display configuration.

**Passwords:** kept in memory only, cleared from the form when a session opens,
and never put in arguments, logs, the library file or error messages. If the user
ticks Remember password, the password is stored in Windows Credential Manager
(DPAPI, bound to the signed-in Windows account; any process running as that user
could read it, as with any credential vault) under a per-computer key, and only a
`password_saved` flag is written to `library.json`. It is deleted when the entry
is removed or the user chooses Forget, and forgotten automatically when the
address, port or username changes.

**State files** (`%LOCALAPPDATA%\iss\`): `library.json` (name, address, port,
username, mode, view, resolution, last connected, the `password_saved` flag) and
`settings.json` (the experimental High Performance switch). Both are written
atomically.

Key mapping (forwarded as X11 keysyms; measured on a live Intel Mac running
Apple's Screen Sharing server, not assumed): Ctrl → Control, **Alt → Option**
(sent as Meta_L), the **Windows key → Command** (Super_L). Apple's server treats
Alt_L, Super_L and Hyper_L as Command and only Meta_L as Option, which is why Alt
is sent as Meta_L. The same mapping holds on the Apple Silicon Mac. The ⋯ menu's
**Ctrl = Command** switch (off by default) sends Ctrl as Command, so Ctrl+C,
Ctrl+V and Ctrl+Z work as they do on Windows. **Swap Alt/Win** (off by default)
reverses the other two: Alt is sent as Command and the Windows key as Option. The
two switches combine, and flipping either releases any key currently held.
Typing with a modifier held sends the unshifted letter; the server adds Shift
for capitals and symbols itself. Not forwarded: Ctrl+Alt+Enter (local
full-screen toggle) and Windows-level shortcuts such as Alt+Tab, Win+L and
Ctrl+Alt+Del. Shortcuts such as Ctrl+N and Ctrl+Q exist only in the library
window, never in a session window.

The toolbar's shortcut buttons send raw keysyms, so the Ctrl/Alt options never
change them: Mission Control is Control+Up, Show Desktop is F11 (a toggle), both
measured on the Intel and Apple Silicon Macs. Launchpad answers no plain key on
either Mac, so it presses Command+Space, and only if the screen then visibly
changes (and is not too busy to tell) types "Launchpad" and Return.

**Type clipboard** types the local clipboard text on the Mac as Unicode
keystrokes, up to 20,000 characters, paced so the window stays responsive. It
works for accented letters, CJK and symbols (verified), and where pasting is not
possible. Emoji and other characters above U+FFFF cannot be typed: the server
drops them or types broken halves, so they are skipped and counted in the status
bar. Typed text is real keystrokes, so Mac text substitutions apply (for example
macOS's "add period with double-space").

**Look:** macOS Screen Sharing's layout and symbol toolbar, drawn in code (no
image assets), in light or dark following Windows (`ISS_THEME=light|dark` to
force). Remote pixels never get a styled background.

Build on Windows x64 with Python 3.14:

```powershell
./packaging/build.ps1
.venv/Scripts/python.exe packaging/check_exe.py
```

The build installs pinned dependencies, runs the test suite, and creates a
windowed single-file executable with Qt libraries and the app icon. The file
is unsigned. No administrator rights or system-wide firewall change is needed
for the outbound TCP connection (High Performance would also receive UDP on
5900/5901). The smoke check starts the frozen app, initialises Qt and verifies
that the High Performance engine and PyAV import from inside the package.

`packaging/iShareScreen.spec`, the browser UI, and the dissector are legacy
development code. They are not the default entry point or this desktop build.
