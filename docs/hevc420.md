# Hardware HEVC 4:2:0 negotiation

Status: a live macOS 27.2 / Radeon 8065S session confirmed HEVC 4:2:0
hardware decoding through D3D11VA. Subsequent live trials also confirmed
decoder resets on encoder generation changes, without the earlier pool
exhaustion errors in the supplied logs. Extended stability is not established.

The video settings in the AVConference media offer include a pixel-format
bitmap in protobuf field 8. iShareScreen's original value, 63, includes
4:4:4 formats. The HEVC 4:2:0 path sends 3 instead: only
8-bit full-range and video-range 4:2:0. It retains the HEVC codec bank.
It also defaults to one picture per frame and selects `libav-hevc420`,
which attempts platform hardware decoding with software fallback.
The one-picture HEVC 4:2:0 offer uses AVConference's Nearby profile (mode 9),
receive direction 2, its 40 Mbit/s bandwidth entry with marker 4096, and
the HEVC `SW:0` feature. It disables RTCP feedback in video settings field 2
as in the reference offer. Explicit multi-picture offers retain mode 7.

The field mapping is described in the independent
[open-screen-sharing negotiation implementation](https://github.com/nekohasekai/open-screen-sharing/blob/main/internal/rfb/media/offer.go),
including analysis of AVConference on macOS 27.2. The
[sharelinux client](https://github.com/lucid-fabrics/sharelinux)
reports using a 4:2:0-only HEVC offer against macOS 27.
Live results from this checkout are summarized below.

Early live trials against the user's macOS 27.2 host returned payload 100 and a
pixel-format bitmap of 3, but no video packets were received, with both the
initial mode-7 offer and the revised mode-9 offer. Later trials below established transmission and hardware decoding after
receiver-report and packet-framing corrections.
Startup diagnostics now log media-port announcements (including enabled flags)
and explicit server refusal codes even when the handshake otherwise discards
the messages. These diagnostics do not change media sockets.

Mac logs also exposed an independent audio liveness bug: the viewer sent
unsolicited payload-101 audio RTP, but no audio RTCP, and sent video receiver
reports to the audio port. The client now sends authenticated audio receiver
reports with the audio SSRC/key on the audio leg, and video reports on the
video leg. Reports begin during the initial burst wait, before the regular
transmit loop starts. Later trials received and decoded HEVC successfully after these corrections;
the earlier Mac encoder statistics had been for H.264.

A later Desktop trial received over 1,100 packets per attempt, but exposed a
single-picture packet-framing bug. The reference uses DONL decoding-order
fields only for multi-band streams; the client had stripped two bytes even
from single-picture HEVC packets. Reassembly now handles the omitted fields
in both the initial burst and streaming path, and startup also harvests
complete parameter packets without requiring an end-of-frame marker.
Codec diagnostics prefer the observed RTP payload types (100 = HEVC,
123/126 = H.264) over ambiguous NAL header bytes.

The 2026-11-08 13:34 Desktop trial confirmed received `chroma_format_idc=1`
and D3D11VA hardware decoding on Radeon 8065S: all 11 startup pictures
decoded. This establishes the HEVC 4:2:0 path on that Mac and AMD system.
Later in the same trial, an SSRC generation change produced missing-reference
errors and exhausted the default hardware frame pool. The
single-picture path now resets the DPB at each adopted encoder generation
and requests 32 extra hardware frames before opening the codec. These
corrections were exercised in later supplied logs: the new encoder generation
reset the DPB and resumed D3D11VA output without the earlier errors. The
13:58 trial also decoded all 116 pictures collected in its startup burst.

## Automatic selection

Choose the **Desktop** frontend and **Auto** decoder. No codec environment
variables are required. Auto prefers hardware HEVC 4:4:4, then hardware
HEVC 4:2:0, then H.264. The browser frontend continues to use H.264.

The 4:2:0 check decodes a packaged 64x64 IDR through the platform hardware
backends with software fallback disabled. A produced frame confirms basic
hardware decoding, even if PyAV downloads it into NV12. The result is cached
for the process lifetime. The sample is documented in
[the probe assets](../src/isharescreen/proxy/media/samples/README.md).

The selected chroma is passed consistently to the offer builder and decoder.
The GUI passes it to the child process without changing its own configuration.
Explicit codec and chroma overrides remain available.

## Force HEVC 4:2:0 for diagnostics

From the directory containing this checkout's `src` folder, run in PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:ISS_VIDEO_CODEC = "hevc"
$env:ISS_HEVC_CHROMA = "420"
iss
```

In the connection form, choose the **Desktop** frontend and **Auto** decoder.
The form now defaults to Desktop when `ISS_VIDEO_CODEC=hevc` is set.
The browser frontend forces H.264 and cannot exercise this path. Explicit
overrides bypass normal automatic selection and may use software decoding
when the requested hardware path is unavailable.

The offer log should say `chroma=420 pixel-format bitmap=0x3 tiles=1 mode=9`.
The video-answer log also shows selected payloads and pixel formats. A
canvas size alone does not prove that the server enabled a video stream.
The received HEVC SPS log must say `chroma_format_idc=1` for 4:2:0;
`3` means the server still sent 4:4:4. The decoder status must also
remain on a hardware backend such as `d3d11va` for hardware acceleration
to be confirmed. The first successfully decoded hardware frame logs
`HEVC hardware decode confirmed: d3d11va` (or another backend).
Merely seeing `libav-hevc420` in the registry does
not prove hardware support; it includes a software fallback.

This requires a live Mac session to establish interoperability. The offer
and registry regression tests do not prove the server honors this request.
For a zero-packet failure, collect the `media ports:` / `media server refusal:`
lines and the Mac's encoder logs immediately after the attempt:

```sh
log show --last 5m --style compact --predicate '(process == "ScreensharingAgent" OR process == "avconferenced" OR process == "screensharingd")'
```

To restore the usual codec behavior in the same shell:

```powershell
Remove-Item Env:ISS_VIDEO_CODEC
Remove-Item Env:ISS_HEVC_CHROMA
```
