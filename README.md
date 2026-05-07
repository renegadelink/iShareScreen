# iShareScreen

Cross-platform Python client for Apple's macOS Screen Sharing **High
Performance** mode (HEVC RExt 4:4:4 over UDP/SRTP). Renders the host
Mac's screen in a native wgpu window with hardware decode where
available.

## Setup

### On the host Mac (the target you want to view)

1. Update to macOS Ventura, Sonoma, or Sequoia — High Performance mode
   needs the Apple Silicon HEVC encoder shipped with these.
2. *System Settings → General → Sharing → **Screen Sharing*** → toggle on.
3. Click the *(i)* next to Screen Sharing → *Allow access for*: pick
   "All users" or "Only these users" and add the account you'll log in as.
4. Note the Mac's hostname (`scutil --get LocalHostName`.local) or IP.

### On the viewing machine

#### macOS

1. Install Python 3.10 or later (from [python.org](https://www.python.org/downloads/),
   `brew install python`, or `uv python install 3.13`).
2. Install iShareScreen:
   ```sh
   pip install git+https://github.com/renegadelink/iShareScreen.git
   ```

That's all — every native dependency (FFmpeg, wgpu's Metal backend, libglfw,
PortAudio) is either bundled in a wheel or part of the OS. Audio uses the
OS-supplied AudioToolbox decoder so there's nothing extra to install.

#### Windows

1. Install Python 3.10 or later from [python.org](https://www.python.org/downloads/)
   (the installer includes pip; check "Add python.exe to PATH" during install).
2. Open PowerShell or cmd and run:
   ```sh
   pip install git+https://github.com/renegadelink/iShareScreen.git
   ```

3. (Optional, only if you want audio) install **libfdk-aac** — Apple's
   PT=101 audio uses AAC-ELD-SBR, which Windows Media Foundation can't
   decode. The cleanest source is [MSYS2](https://www.msys2.org):
   ```sh
   pacman -Sy --noconfirm mingw-w64-x86_64-fdk-aac
   ```
   This drops `libfdk-aac-2.dll` at `C:\msys64\mingw64\bin\`, which iss
   searches automatically. Without libfdk-aac, video works as normal
   and audio is silently skipped.

Every other native dependency (FFmpeg, wgpu's D3D12 backend, libglfw,
PortAudio) is bundled in a wheel or part of the OS.

#### Linux (Debian / Ubuntu)

1. Install Python and the system libraries that the GPU + window stack
   need (Vulkan loader, OpenGL, X11 / Wayland surfaces, PortAudio,
   AAC-ELD-SBR audio decoder):
   ```sh
   sudo apt install python3 python3-pip python3-venv \
       libvulkan1 libgl1 libegl1 \
       libxrandr2 libxinerama1 libxcursor1 libxi6 \
       libportaudio2 libfdk-aac2
   ```
   `libfdk-aac2` is only used for audio; if you skip it, video still
   works and audio is silently disabled.
2. (Optional, for hardware HEVC decode on Intel GPUs) install vaapi:
   ```sh
   sudo apt install vainfo intel-media-va-driver-non-free
   ```
   For AMD, swap the driver: `sudo apt install mesa-va-drivers`.
3. Bump the kernel UDP receive-buffer ceiling so Apple's HEVC RTP
   firehose isn't dropped:
   ```sh
   echo 'net.core.rmem_max=33554432' | sudo tee /etc/sysctl.d/99-isharescreen.conf
   sudo sysctl --system
   ```
4. Install iShareScreen (in a venv recommended):
   ```sh
   python3 -m venv ~/.venvs/iss
   ~/.venvs/iss/bin/pip install git+https://github.com/renegadelink/iShareScreen.git
   ~/.venvs/iss/bin/iss     # or symlink to ~/.local/bin
   ```

For Fedora / Arch / openSUSE, translate the apt package names with your
distro's package manager (most are named the same or very close).

### Running from a clone (developers)

```sh
git clone https://github.com/renegadelink/iShareScreen.git
cd iShareScreen
pip install -e .
```

## Usage

```sh
iss
```

Prompts for host / username / password (not echoed) and a resolution
(defaults to 1920 × 1200).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

This is an independent reverse-engineering of a publicly-documented
network protocol. No Apple source code, headers, or symbols are
included.
