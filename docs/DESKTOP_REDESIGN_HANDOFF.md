# iShareScreen: native desktop redesign handoff

Updated: 19 October 2026

## Status update (19 October 2026, later): Phase 1 implemented

The library, staged wizard and independent session windows described below now
exist; the "Current native application" section further down describes the
earlier single-window shell and is kept for history. Where it differs, the code
is authoritative.

Implemented, under `src/isharescreen/`:

- `proxy/address.py`: `parse_address` and `probe_server`. A probe resolves,
  connects, reads the RFB banner, answers with the client version and reads the
  offered security types, then **closes** the socket. It never chooses a
  security type or authenticates, so no server session exists afterwards;
  credentials use a fresh connection. The result says what to ask for
  (`account`, `password` or `none`) and unsupported servers never reach the
  credentials step.
- `proxy/standard.py`: the ARD path is unchanged and still preferred whenever
  type 30 is offered. Added VNC password (type 2, DES per RFC 6143) and None
  (type 1) for generic servers, RFB 3.7/3.8, the server's failure reason (read
  with a 1.5 s cap), typed errors (`AuthRejected`, `UnsupportedSecurity`,
  `UnsupportedEncoding`, `DesktopUnavailable`, `ServerRefused`), an `on_socket`
  hook so a cancel can interrupt a blocked connect, and `on_phase` for truthful
  progress text.
- `frontend/desktop/library.py`: versioned JSON store, atomic writes, rollback on
  failed save, corrupt file moved aside, newer schema read-only. No password
  field exists. Identity is host (case-insensitive) + port + account + mode.
- `frontend/desktop/session.py`: `SessionController` (one thread and socket per
  session), `Screen` (fit and actual size), `SessionWindow` (banner, reconnect
  with a fresh password prompt).
- `frontend/desktop/wizard.py`: address check, then credentials and settings,
  then connecting. Standard is enabled; High Performance and display
  count/resolution are shown disabled with an explanation.
- `frontend/desktop/library_window.py`, `theme.py`, `standard_app.py`
  (`AppController`: owns library, wizards and sessions; quit shuts all down).

Success gate: the first decoded framebuffer update is received by the session
controller on the GUI thread (`established`). Only then is the session window
opened and the library entry written; a failed write shows a notice and leaves
the live session alone. An all-black desktop is a valid success.

Later in the same day: local-to-remote clipboard (`StandardClient.send_clipboard`,
Latin-1, 1 MB cap, per-session "Share clipboard" toggle, echo guard, and the
clipboard contents at session start are never sent) and actual-size edge
scrolling (`SessionWindow.pan_step`, which re-sends the pointer because the image
moves under a still cursor). The known-gaps list further down predates these.

Live validation (19 Oct 2026, against the two supplied Macs, credentials passed
through environment variables and never written): both answer on TCP 5900 with
RFB 3.889 and offer type 30. Intel (3840×1920) and Apple Silicon (1920×1080)
each passed: address check, a wrong password returning to sign-in with nothing
saved, correct sign-in with the first screen update in about 1.4 s (Intel) and
1.9 s (Apple Silicon), real desktop content, library save without a password,
refresh and harmless input (pointer move, Shift) with the connection staying up,
a 20 s soak with both sessions open, actual-size scrolling to the far corner,
and closing one session leaving the other connected. **Not live-tested:** typing,
clicks, the Windows-key-as-Command mapping and clipboard (the last would
overwrite the Macs' clipboards); those are covered by the fake server only.

Keyboard validation (Intel iMac, same day, via Spotlight as a harmless visible
target, nothing submitted; raw frames and crops were deleted afterwards):
- Measured keysym behaviour of Apple's Screen Sharing server: Alt_L/Alt_R,
  Super_L/Super_R and Hyper_L act as **Command**; Meta_L/Meta_R act as
  **Option**; Control_L is Control; Mode_switch and ISO_Level3_Shift do nothing.
- **Bug found and fixed:** the app sent Alt as Alt_L, which is Command, so
  Option was unreachable. Alt now goes out as Meta_L. Windows key (Super_L) is
  Command. Verified through the real `Screen` handlers: Alt+Backspace deleted a
  word, Win+Backspace a line, Ctrl+A moved to line start.
- New: toolbar "Ctrl = Command" (off by default) verified: Ctrl+Backspace
  deleted the line and Ctrl+A then X replaced everything.
- New: "Type clipboard" types Unicode keystrokes. Verified for upper/lower case,
  all ASCII symbols, Äö é, 日本語, ✓ and a 150-character string. The server
  supplies Shift itself. Characters above U+FFFF (emoji) are dropped as one
  keysym and typed as broken halves as a surrogate pair, so they are skipped and
  counted in the status bar.
- **Apple Silicon Mac, same checks, later the same day:** identical keysym
  behaviour (Meta = Option; Alt, Super, Hyper = Command; Control = Control),
  identical typing results (case, symbols, Äö é 日本語 ✓, 150 characters, emoji
  skipped). Method note: on that Mac's newer Spotlight, modifier+Backspace does
  nothing for any modifier, so Backspace is not a valid probe there; the
  select-all and word-jump (Left arrow) probes work on both. The "✓. end" seen
  once was macOS's own "add period with double-space" acting on two typed spaces.
- New, on request: toolbar **Swap Alt/Win** (Alt -> Command, Windows key ->
  Option; off by default). Verified live on Apple Silicon through the real
  handlers in both states: default Alt+Left jumped a word (Option) and Win+Left
  went to line start (Command); swapped, those reversed; Alt+A then X replaced
  everything; switching back restored the default. The setting is per session
  window and not yet remembered between sessions.
- Not tested: real Windows-key delivery (Windows may keep the Start menu for
  itself, so Ctrl = Command is the dependable route); clipboard sync against a
  real Mac.

Verified: 193 automated tests pass (105 new, as of the keyboard work), using a fake server with a real
ARD key exchange plus VNC-password, no-auth, RFB 3.7, refusal and unsupported
variants. They cover address checks, wrong password, cancel, two independent
sessions, reconnect, save failure, quit and the wizard adapting to the server's
authentication. `dist/` was rebuilt with `./packaging/build.ps1` (the script
was fixed for Windows PowerShell 5.1, which treated PyInstaller's stderr as
fatal) and `packaging/check_exe.py` passes against it. Offscreen renders of
every screen were reviewed. The previous build was saved outside the repo
before overwriting.

**Not verified:** nothing in this phase has been run against a real Mac or a
real third-party VNC server. Live Intel/Apple Silicon checks of the new flow,
real keyboard/modifier behaviour, the Windows-key-as-Command assumption, and
VNC-password/no-auth interop (the fake server restates the RFC independently,
which is not the same as a real implementation) remain open.

Open follow-ups: more generic security types (TLS/VeNCrypt), Phase 2 polish
(actual-size wheel scrolling, Unicode paste), Phase 3 High Performance.

## Status update (19 October 2026, evening): macOS layout, saved passwords, experimental HP

**Decisions that supersede the text further down**
- *Layout and look.* The user supplied six screenshots of macOS Screen Sharing
  (library with sidebar and ⓘ button, the ⓘ sheet, "Connect To:", the sign-in
  dialog, and the session toolbar with Launchpad / Mission Control / Show Desktop)
  and asked to copy them, with **symbols instead of text**. The sage/Liquid Glass
  palette is therefore gone; the app uses a macOS-style light/dark theme that
  follows Windows. Symbols are drawn in code (`icons.py`), so no image assets.
- *Saved passwords.* The "never save passwords" rule below was reversed at the
  user's explicit request. Passwords are **opt-in per computer**, stored only in
  Windows Credential Manager (`credentials.py`), never in `library.json` (only a
  `password_saved` flag). Changing address/port/username forgets it; a rejected
  saved password falls back to a prompt. Double-click then connects by itself.
- *Edit.* Right-click → Edit…, the ⓘ button and Ctrl+E open a sheet like macOS's.

**New modules** (`frontend/desktop/`): `icons.py`, `theme.py` (rewritten),
`library_window.py` / `edit_dialog.py` / `wizard.py` (rewritten), `credentials.py`,
`settings.py`, `hp.py`. Session window: segmented symbol toolbar, view-only
Control, zoom (25%–400%, fit, actual) with edge scrolling, Mission Control
(Control+Up) and Show Desktop (F11) measured on both Macs, Launchpad via Spotlight
with a "did Spotlight really open" guard (refuses on a busy screen), Type
clipboard, Ctrl = Command, Swap Alt/Win. Action feedback now survives screen
updates (connection info has its own status-bar label).

**High Performance: built, experimental, off by default — and not verified live.**
Live attempts (one at a time, no curtain, no console-sharing prompt):
- Intel iMac: (1) default HEVC offer, audio off → zero RTP; (2) audio on → H.264
  RTP arrived but the engine expected HEVC ("missing-param-sets"); (3) H.264
  offered → connected in 47 s (16 slow 0x1c re-queries), but the d3d11va decoder
  stalled ("FIR storm", zero frames); (4) the Mac's Screen Sharing then **wedged**
  (TCP accepted, no RFB banner) for about two minutes and recovered by itself;
  (5) H.264 + software decode → the Mac never answered the canvas request.
- Apple Silicon Mac mini: login and canvas (1920×1080, 4 tiles) negotiated, but
  zero RTP in three fresh handshakes, with audio on or off.
- Ruled out: this PC's firewall (inbound UDP allowed for Python), Tailscale routing
  (both direct, 4 ms and 34 ms). Root cause unknown; the engine's own comments call
  it a host-side stall. **A failed attempt can wedge the Mac's Screen Sharing for
  minutes**, so do not loop live attempts.
**Release decision: High Performance comes later.** The ⋯-menu switch is hidden
unless `ISS_EXPERIMENTAL_HP=1` (or it was already switched on), and the Screen
Sharing Type entry reads "coming in a later version". The engine code and PyAV
stay in the package.
Hence `hp.py`: opt-in switch (⋯ menu, confirms), capability = switch on + account
auth + login type 33 offered + PyAV present, 45 s deadline, one attempt per Mac,
H.264 offer, curtain off, console-user refusal instead of a share request, plain
failure messages, Reconnect keeps the kind. Tested only against a simulated engine
(33 tests). Next steps if someone continues: capture the Mac side (Console logs
for screensharingd/ScreensharingAgent) during an attempt, and try an Apple-silicon
host on the same LAN as this PC.

**Bug found by the tests and fixed:** the address check ran in a `QThread`; quitting
mid-check made Qt abort the process at exit (exit code 0xC0000409). It now runs on a
daemon thread, and HP's `stop()` never blocks on the engine's slow `close()`.

**Verification, this evening:** 286 automated tests pass (the build runs them and
checks the exit code). Live, through the real windows, on both Macs: Standard
connect/sign-in/save, refresh and input, 20 s soak with both open, zoom and edge
scrolling to the far corner (27/27), and the real Mission Control, Show Desktop and
Launchpad buttons plus zoom and view-only (10/10 on each Mac). `dist/` was rebuilt
(90 MB, PyAV included) and the frozen smoke check passes, including importing the
HP engine from inside the package. Remember password and double-click were also
run end to end against the iMac with the **real** Credential Manager (9/9: saved to
the vault and not the library file, double-click connected in 1.6 s with no typing,
a stale saved password was reported and replaceable, Forget cleaned the vault;
nothing left behind). **Not verified:** High Performance end to end; the macOS-look
pixel match beyond offscreen renders.

## Goal and user requirements

Build a native desktop remote-screen client with an experience similar to
macOS Screen Sharing. The main window is a **library of saved Macs or other VNC
systems**. Creating a session opens a **new window**, walks through address
verification, credentials, and session settings, then connects. A successful
connection is saved to the library and its remote desktop opens in a session
window.

This must be a real desktop application. Do not use a browser tab, localhost
website, or webview wrapper as the primary application. The user needs it for
actual work and prioritizes a working remote screen over cosmetic work.

The user's requested sequence is:

1. Open the computer/VNC library.
2. Choose **New session**; open a separate connection window.
3. Enter an **IP address** (allowing a hostname is also useful).
4. Verify that the address points to an available VNC service.
5. Only after verification, prompt for **username and password**.
6. Specify **Standard or High Performance** mode and, where supported, the
   **number of displays and resolution**.
7. Connect and show real connection progress/errors.
8. On a successful connection, save the connection to the library and open
   the remote desktop in its own session window.

The library remains available while session windows are open. Each session
must have its own state and lifecycle; opening another Mac must not replace or
disconnect an existing session.

## Important scope decision: Standard first

The user explicitly instructed: **“we want this compatible with both intel and
apple silicon macs. no high performance for now.”**

Implement the library, connection flow, and native windows with **Standard VNC
working first**. Keep a mode-selection location in the design, but make High
Performance unavailable until its implementation and host capabilities are
verified. Explain that availability beside the option. Do not silently launch
the old HP transport when the user selects Standard.

The final product is intended to offer Standard and High Performance. The
requested display-count/resolution configuration belongs in the planned flow;
it must not imply that unsupported settings already work. Add the operational
HP option in a later phase, with capability detection and live validation.

## Current working state

### Black-screen investigation and actual fix

The repository originally supported Apple's High Performance protocol: extended
RFB 3.889 authentication/session setup, encrypted control, and UDP/SRTP video.
The first packaged UI used a browser bridge. The user's Macs authenticated but
sent no usable HP video, leaving a black browser canvas.

A first native attempt reused the extended SRP authentication path and also
showed black updates. **Merely receiving updates or seeing a pixel maximum of
255 was not sufficient proof of a usable desktop.**

The working path now uses:

- RFB **3.8** (`RFB 003.008\n`) negotiation.
- Apple Remote Desktop **security type 30** authentication.
- Shared ordinary framebuffer session (`ClientInit = 0x01`).
- Standard raw/zlib framebuffer rectangles over **TCP 5900**.
- A native Qt framebuffer widget, not a browser/WebCodecs canvas.
- No HP negotiation, UDP video, virtual-display creation, or curtain mode.

The ordinary ARD authentication sequence follows the implementation in
[LibVNCClient](https://github.com/LibVNC/libvncserver/blob/master/src/libvncclient/rfbclient.c),
`HandleARDAuth`: DH exchange, MD5 of the padded shared secret, AES-128-ECB
credential encryption, then the standard SecurityResult. Do not change its
credential cipher based solely on contradictory legacy notes in this repo;
the current path has been tested against the user's real Macs.

Visible desktop images were verified from both supplied Apple Silicon and
Intel systems. The user confirmed **“ok, i see my desktop now.”** No passwords
are recorded in this document, source, or test fixtures.

### Current native application

- `src/isharescreen/frontend/desktop/standard_app.py`
  - Native PySide6 connection form and viewer.
  - `Worker`: network/receive thread, frame/status/error signals, shutdown.
  - `Screen`: QImage rendering, fit-to-window coordinate mapping, keyboard,
    pointer buttons, dragging, scrolling, and releasing held inputs on blur.
  - `Window`: current single-session shell, refresh, disconnect, full screen.
  - `Ctrl+Alt+Enter` toggles full screen.
- `src/isharescreen/proxy/standard.py`
  - Standard ARD authentication and TCP framebuffer transport.
  - Continuous zlib stream decoding, raw rectangles, desktop-size updates.
  - Thread-serialized writes; no HP/media setup.
- `src/isharescreen/tui/entry.py`
  - `iss` / `isharescreen` now route to the native standard-VNC application.
- `tests/test_standard_vnc.py`
  - Local fake server verifies RFB 3.8 + type 30 selection, credentials,
    ordinary encoding negotiation, and actual coloured pixels.
  - Continuous zlib and desktop-size regression coverage.
  - Native painter regression test checks visible red/green pixels.

The current window contains the form and one remote screen. **The library,
staged address-first wizard, and independent multiple session windows have not
yet been implemented.** That is the next product work.

Known feature gaps: no standard-VNC audio, no local-to-remote clipboard, no
generic VNC password/type negotiation beyond the tested Mac ARD path. Basic
remote-to-local clipboard text is implemented. The window fits the host's
current desktop; it does not reconfigure its resolution.

### Build and verification

The current native executables are:

- `dist/iShareScreen-Desktop.exe`
- `dist/iShareScreen.exe` — identical native executable, replacing the former
  browser launcher.

Build on Windows x64 with Python 3.14:

```powershell
./packaging/build.ps1
.venv/Scripts/python.exe packaging/check_exe.py
```

Relevant files:

- `packaging/desktop_launcher.py`: windowed frozen entry point.
- `packaging/iShareScreen-Desktop.spec`: native Qt single-file build.
- `packaging/requirements-build.txt`: pinned build dependencies.
- `packaging/make_icon.py`: executable icon generator.
- `packaging/README.md`: current native behavior and build notes.
- `packaging/probe_standard.py`: interactive diagnostic using `getpass`; can
  verify framebuffer statistics without saving credentials. `--capture`
  explicitly saves a diagnostic frame to the ignored build directory; treat
  those images as private and delete when finished.

At handoff, **88 tests pass**, and the frozen native GUI initializes Qt and
exits cleanly under the executable smoke test. Live framebuffer verification
was performed separately. Do not confuse the smoke test with a live connection
or complete remote-input validation.

## Target desktop UX

### 1. Computer library — primary window

Use macOS Screen Sharing's library concept, with the user's chosen Liquid Glass
visual language adapted to native desktop controls.

- Window title: **iShareScreen**.
- Clear **New session** action; support `Ctrl+N` on Windows.
- Saved systems as an accessible list/grid with name, address, mode, and last
  connected timestamp. Provide a search field once there is saved content.
- Empty state explains the app and offers **New session**.
- Double-click/Enter on an entry opens its connection flow in a new window,
  with saved address/username/settings prefilled and password empty.
- Context actions: Connect, Rename/Edit, and Remove from library. Removing an
  entry must not shut down its active remote session.
- Show active-session state only when backed by actual session state. Do not
  invent online badges from a previous connection or run constant login probes.
- Make it possible to keep multiple sessions open to different computers.

### 2. New session window — address stage

- Label: **IP address**; optionally support hostnames and an advanced port
  field defaulting to 5900.
- Primary action: **Continue** / **Check connection**.
- Validate formatting locally, then resolve and connect asynchronously.
- Verify TCP reachability **and an RFB server banner**. Ping alone is not a
  reliable test of whether a VNC system exists or can accept a session.
- Keep the GUI responsive; enforce timeouts and cancellation.
- Show Checking, Ready, Unreachable, DNS failure, Wrong service, and Retry
  states with actionable messages. Do not prompt for credentials on an
  unverified address.
- Preserve the address when returning from a later stage.
- A temporary probe should not leak a socket or leave a half-created server
  session. Decide explicitly whether to close it or hand it to authentication.

### 3. Credentials and session settings stage

- Show the verified address prominently, with Back/Edit.
- Ask for **Username** and **Password**. Password is masked by default, with
  a labeled show/hide action and Enter-to-connect behavior.
- Keep passwords out of command lines, logs, Markdown, settings, crash
  reports, diagnostics, and screenshots.
- Do not add password persistence as part of “save connection.”
- Offer a **Connection mode** control:
  - **Standard**: enabled now, compatibility path for Intel/Apple Silicon Macs.
  - **High Performance**: future capability-gated option; disabled for now with
    a brief explanation. It must not break or replace Standard.
- Include the planned **display count** and **resolution** configuration in
  the mode/settings design. Only enable controls that the selected transport
  and remote system actually support.
- In Standard mode, explain **Use the Mac's existing display(s)** and
  **Fit to window / actual size** as local viewing choices. Do not advertise
  a virtual display or silently change the Mac's physical resolution.
- For future HP mode, expose supported virtual-display count and per-display
  resolution after capability detection. Preserve the requested settings for
  reconnect. Validate allocation and surface unsupported settings before
  connection. Do not invent universal host limits.
- **Display count is not codec tile count.** The old HP stream's 4 tiles are
  independently encoded regions, not four monitors.
- Generic VNC systems may require only a VNC password, rather than a macOS
  account. The authentication UI should adapt to advertised supported security
  types when that support is added; do not claim all VNC systems already work.

### 4. Connecting stage

- Show truthful phases such as Connecting, Authenticating, Requesting desktop,
  and Waiting for screen. Avoid fabricated percentages.
- Preserve address/user/settings across failures; never expose the password
  in an error message.
- Rejected credentials return focus to the credentials stage.
- Distinguish authorization rejection, unavailable desktop/session, transport
  disconnect, unsupported encoding, timeout, and local rendering failure.
- Cancel must stop network work and clean up the partial session.
- Prevent duplicate submissions and concurrent retries for the same wizard.
- Do not save a failed/half-created connection to the library as successful.

### 5. Success, persistence, and session window

- Success requires authenticated session setup and a valid framebuffer update
  delivered through the renderer, not merely an open socket, child process,
  status string, or allocated blank canvas.
- Do not use a “nonblack pixel” heuristic as the protocol's success gate: an
  intentionally black remote desktop is valid. Verify decoding/rendering and
  make waiting/blank-screen diagnostics available separately.
- Save the connection only after the first successful usable session setup.
  Save address, port, optional friendly name, username, mode, supported
  display settings, and last successful connection time. **Never save password.**
- Avoid duplicate entries on reconnect; use a stable connection identifier and
  an explicit policy for address/port/username/mode identity.
- Make persistence atomic; a write failure must not tear down a working remote
  session. Show a recoverable “could not save to library” notice.
- Open the remote screen in a separate native session window. The connection
  wizard can become that window after success, or hand off cleanly to a new
  window; avoid leaving an extra blank wizard behind.
- Session title identifies the host and connection state.
- Provide compact controls for full screen, fit/actual size, refresh,
  reconnect, disconnect, and access to display settings where supported.
- Keep the screen itself visually clean, with no decorative background or
  blur competing with pixels.
- Route keyboard/mouse only while that session's screen has focus. Release
  held buttons/modifiers on focus loss and disconnect. Document how Ctrl,
  Command/Windows, Alt/Option, and reserved local shortcuts map.
- Closing a session closes its transport and workers, and leaves the library
  and other sessions running. Library/application Quit should shut down every
  session cleanly.

## Design reference and native translation

The user explicitly asked to make heavy use of:

**https://github.com/pointlessduffin-21/liquid-glass-design**

The reference checkout is in ignored `.design-reference/` if still present.
The copied web kit is under `src/isharescreen/gui/static/glass/`, with attribution
in `THIRD_PARTY_NOTICES.md`. It was taken from revision
`55b438d7e432f4958ce3a5759e578a61a16962ef`.

Read that reference's `SKILL.md` and
`references/liquid-glass-design-guidelines.md` before visual work. Use it as
design guidance for **native widgets**, not as a reason to reintroduce a website.

Carry through:

- Light sage background, green accent, calm dark-green text.
- Frosted/reflective outer navigation surfaces and restrained highlights.
- Rounded, precise cards and controls with consistent spacing.
- Solid, legible form fields and readable saved-computer rows.
- A clear primary action and progressive disclosure of advanced settings.
- Short entrance/state transitions and hover/press feedback, respecting
  reduced motion. Never move an active control's hit target.
- Keyboard navigation, visible focus, labels, text-plus-colour status, and
  usable scaling on HiDPI Windows displays.

Native opaque fallbacks are preferable to expensive fake refraction. Do not
add a browser engine, nested blur layers, a 3D field behind remote pixels, or a
large marketing hero to the everyday library/session workflow.

The current source has an initial native sage theme; it is not the finished
library redesign. The earlier browser HTML/CSS is useful visual reference only.

## Suggested implementation plan

### Phase 1 — native library and exact Standard connection flow

1. Refactor the present `Window` into library, connection wizard, and session
   window components without regressing `StandardClient`.
2. Add a per-session controller owning its worker/socket and all cleanup.
3. Implement asynchronous address verification and the separate credentials
   stage, with cancellation/back/retry and explicit connection states.
4. Add saved-connection storage. Prefer a versioned local JSON store with
   atomic writes unless SQLite is justified by actual requirements. Integrate
   platform state paths from `tui/storage.py`; do not reuse the old single-form
   schema as if it were already a multi-connection library.
5. Persist only after successful session setup; reopen entries without storing
   passwords. Support multiple independent session windows.
6. Apply the native Liquid Glass-inspired library and wizard design.
7. Verify with the user's real Standard sessions, while leaving an already
   working session running whenever possible.

### Phase 2 — everyday remote-work polish and broader VNC compatibility

- Test/fix keyboard layouts and modifier mapping, paste, Unicode text,
  clipboard directionality, multi-monitor cropping/selection, actual-size
  scrolling, reconnect, and display-size changes.
- Negotiate additional supported VNC authentication/encodings for non-Mac
  systems, with bounded parsers and real interoperability tests.
- Add accessible connection editing, sorting/search, error recovery, and
  screenshots of native library/connection states without private desktops.
- Preserve a reliable Standard path throughout.

### Phase 3 — optional High Performance

- Revisit the retained HP code only after Standard is stable.
- Gate on actual host/client capability; do not assume Intel and Apple Silicon
  hosts have identical HP support.
- Implement actual display-count/resolution negotiation and its lifecycle.
- Validate a real visible stream, input, resizing, reconnect, resource cleanup,
  and fallback. Keep Standard directly selectable and functional.
- Show failure in the connection window instead of opening an indefinitely
  black viewer or launching overlapping automatic handshake attempts.

## Acceptance checks for the next implementation

- Fresh launch opens a native library and no browser.
- New session opens a separate window.
- Unreachable/wrong-service addresses never advance to credentials.
- Verified VNC address advances to credentials and supported settings.
- Standard is enabled; HP is explicitly unavailable until implemented.
- Wrong password is recoverable without losing address/settings or saving a
  successful library record.
- A successful session appears in the library only after valid screen setup.
- Relaunch restores saved systems/settings, with no persisted password.
- At least two independent session windows can remain open; disconnecting one
  does not affect the other.
- Test coloured image content through the native renderer, continuous zlib
  updates, coordinate scaling, key/button release, full screen, and resize.
- Verify visible, interactive desktops against both Intel and Apple Silicon
  Macs in Standard mode. A handshake-only test does not satisfy this check.
- Fake-server tests cover address verification, auth errors, malformed sizes,
  worker cancellation, storage success/failure, and independent lifecycles.
- The rebuilt `.exe` starts on the Windows machine without Python installed
  separately and loads its Qt plugins. Run the frozen GUI smoke check.
- Never include real account passwords or private desktop captures in commits,
  fixtures, the handoff, or the saved library.

## Instructions for the continuing coding agent

Start by reading this document, `README.md`, the native app/transport, and the
new Standard tests. Preserve the working standard ARD path. Do not “simplify”
it back to the old SRP/HP handshake that produced black screens.

Implement the requested native library and staged flow before optional HP work.
The user has authorized this direction and prefers action over repeated design
confirmation. Make concrete, reviewable changes and run appropriate native,
protocol, persistence, and packaged-app checks. Report actual live validation
separately from synthetic tests. Do not disrupt a working remote session merely
to take screenshots or restart the UI during development.
