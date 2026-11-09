# iShareScreen

Cross-platform Python client for Apple's macOS Screen Sharing **High
Performance** mode (HEVC RExt 4:4:4 over UDP/SRTP). Renders the host
Mac's screen in a native wgpu window with hardware decode where
available.

## Setup

### On the host Mac (the target you want to view)

1. Any modern macOS that supports the Screen Sharing app's High
   Performance mode (Apple Silicon, recent macOS).
2. *System Settings → General → Sharing → **Screen Sharing*** → toggle on.

### On the viewing machine

#### macOS

1. Install Python 3.10 or later (from [python.org](https://www.python.org/downloads/),
   `brew install python`, or `uv python install 3.13`).
2. Install iShareScreen:
   ```sh
   pip install git+https://github.com/renegadelink/iShareScreen.git
   ```

#### Windows

1. Install Python 3.10 or later from [python.org](https://www.python.org/downloads/)
   (the installer includes pip; check "Add python.exe to PATH" during install).
2. Open PowerShell or cmd and run:
   ```sh
   pip install git+https://github.com/renegadelink/iShareScreen.git
   ```

3. Audio uses the included PyAV plain AAC-ELD decoder automatically, without
   an environment variable or another DLL. Audible playback was confirmed
   with a macOS 27.2 host. If PyAV cannot initialize, the client tries
   **libfdk-aac**, available through [MSYS2](https://www.msys2.org):
   ```sh
   pacman -Sy --noconfirm mingw-w64-x86_64-fdk-aac
   ```
   This drops `libfdk-aac-2.dll` at `C:\msys64\mingw64\bin\`, which iss
   searches automatically. If you already have [scoop](https://scoop.sh),
   `scoop install msys2` followed by the same `pacman` command also
   works — iss looks under `%USERPROFILE%\scoop\apps\msys2\current\mingw64\bin`
   too. Without a usable audio decoder, video continues and a warning is logged.
   To select FDK explicitly for a different host configuration, set
   `$env:AAC_BACKEND = 'fdk'` before launching from the same PowerShell window.
   See [audio configuration and negotiation notes](docs/audio-negotiation.md).

#### Linux (Debian / Ubuntu)

1. Install Python and the system libraries that the GPU + window stack
   need (Vulkan loader, OpenGL, X11 / Wayland surfaces, PortAudio,
   AAC-ELD-SBR audio decoder):
   ```sh
   sudo apt install python3 python3-pip python3-venv \
       libvulkan1 libgl1 libegl1 \
       libxrandr2 libxinerama1 libxcursor1 libxi6 \
       libportaudio2 libfdk-aac2 \
       xclip
   ```
   `libfdk-aac2` is optional: audio tries the included PyAV decoder first,
   then FDK if PyAV cannot initialize. The PyAV path was verified on Windows
   against macOS 27.2. Video works without an audio decoder.

   `xclip` enables bidirectional clipboard sync with the macOS host. On
   Wayland desktops use `wl-clipboard` instead. If neither is installed
   iss logs a one-line warning at startup and runs without clipboard
   sync — everything else still works.
2. (Optional, for hardware HEVC decode on Intel GPUs) install vaapi:
   ```sh
   sudo apt install vainfo intel-media-va-driver-non-free
   ```
   For AMD, swap the driver: `sudo apt install mesa-va-drivers`.
3. Install iShareScreen (in a venv recommended):
   ```sh
   python3 -m venv ~/.venvs/iss
   ~/.venvs/iss/bin/pip install git+https://github.com/renegadelink/iShareScreen.git
   ~/.venvs/iss/bin/iss     # or symlink to ~/.local/bin
   ```

For Fedora / Arch / openSUSE, translate the apt package names with your
distro's package manager (most are named the same or very close).

### Firewall

iss connects to the host Mac over **TCP 5900** (control) plus two UDP
flows: **5900** (audio + RTCP) and **5901** (video). It sends from both
UDP ports during connect. If a firewall still blocks the stream, allow
UDP 5900–5901 inbound.

## Usage

```sh
iss
```

Opens the terminal UI: a connect form for host / username / password
(masked) and resolution, then a live session view with per-tile fps and
loss, throughput, UDP queue health, and a log tail. The actual screen
streams in a separate window. Last-session values pre-fill the form on
the next launch; passwords are never persisted.

Any flag accepted by `iss --headless` also pre-fills the form, so
launchers can do:

```sh
iss --host mac.local -u me --advertise 1920x1080 --no-curtain
```

Useful keys in the live session: **f** force IDR refresh (fixes the
rare gray patch the auto-recovery doesn't catch), **r** reconnect,
**d** disconnect, **Ctrl-B** save a bug-report snapshot, **q** quit.

For CI / scripted use:

```sh
echo "$PASSWORD" | iss --headless --host mac.local -u me --password-stdin --auto-quit-secs 30
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

This is an independent reverse-engineering of a publicly-documented
network protocol. No Apple source code, headers, or symbols are
included.
