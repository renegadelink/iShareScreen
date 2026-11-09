# Audio codec negotiation investigation

The receiver supports FDK AAC and a PyAV plain-ELD backend.
Audible PyAV playback was confirmed on Windows with a macOS 27.2 host. This does **not** establish
that Apple's High Performance Screen Sharing protocol can negotiate only ELD.
No alternative audio codec has yet been verified in a live session here.

## Evidence

- Apple's captured audio offer uses negotiator mode 8 and a protobuf F3 audio
  record: f1 SSRC, f2=0, f3=0, f4=`0x5e7f`, f5=0, f6=0.
- Independent [open-screen-sharing server negotiation code](https://github.com/nekohasekai/open-screen-sharing/blob/main/internal/rfb/media/server_negotiation.go)
  writes an audio answer with f4=`0x1600` (`answerAudioPayloads`),
  f5=`0x600` (`answerAudioSecondary`), and f6=1. The user's macOS 27.2
  session at 2026-11-08 13:53:37 returned exactly these values, with
  f2=f3=0. This confirms the values, not the codec-to-bit mapping.
- The historical [AVConference audio settings runtime header](https://github.com/nst/iOS-Runtime-Headers/blob/master/PrivateFrameworks/AVConference.framework/VCMediaNegotiationBlobAudioSettings.h)
  exposes `payloadFlags`, `secondaryFlags`, `supportFlags`, `useSBR`, and
  payload-to-negotiation-payload conversion methods. The header alone does
  not reveal their protobuf field numbers or codec bit assignments, and is
  from iOS rather than the current macOS build.

Together these support interpreting f4/f5 as capability flags. They do not
prove that the Screen Sharing operating mode enables other AVConference codecs.
The earlier documentation calling f4 a bitrate and the silence threshold a
tier floor was too strong: varying f4 also varies its bits. For example, 4000
and 6000 differ in bit `0x1000` as well as other bits.

## Diagnostics and next experiment

Normal connections now log `audio offer: F3.f4=... F3.f5=...` and
`audio answer: F3.f2=... F3.f3=... F3.f4=... F3.f5=... F3.f6=...` when present.
The video answer's placeholder audio record is excluded. Keys and media blob
contents are not dumped. Start with the normal offer and capture the answer
from the target Mac before changing flags.

`ISS_AUDIO_F4` accepts a decimal or `0x`-prefixed unsigned 32-bit value for
controlled experiments. The default remains unchanged. An override does not
select a matching decoder: the receiver still assumes AAC-ELD. A silent or
failed connection after a mask change cannot by itself identify a codec.

To establish an alternative, map the flag conversion functions or inspect
the current framework's negotiation logs, then vary one identified capability
and verify the answer, incoming RTP payload, codec configuration and decoded
audio. Do not infer codec identity from a dynamic RTP payload number alone.

## Verified plain ELD decoder

The local PyAV 19.0.2 / libavcodec 63.1.102 probe opens the native `aac`
decoder with `f8e65000` successfully, but raises `PatchWelcomeError` for
our existing ELD-SBR config `f8e65132e000`. The assertion that FFmpeg
cannot decode *any* AOT 39 was incorrect. The independent implementation's
[audio format](https://github.com/nekohasekai/open-screen-sharing/blob/main/internal/rfb/media/audio.go)
uses `f8e65000`, 48 kHz stereo and 480 samples per packet.

PyAV now uses that configuration automatically on Windows and Linux,
without an environment variable or extra DLL. macOS retains AudioToolbox
as its default. The 2026-11-08 13:58 session decoded 480-sample
48 kHz stereo blocks; the user confirmed audible playback after unmuting
Windows. Other host versions and long-session behavior remain unverified.
The audio offer is unchanged. If the default decoder cannot initialize,
the client falls back to FDK; decode errors during playback do not trigger
an automatic backend switch.

Explicit overrides remain available in PowerShell:

```powershell
$env:AAC_BACKEND = 'fdk' # Or 'pyav-eld' to select PyAV explicitly.
# Launch the client using your usual command from this same shell.
```

Play audio on the Mac and listen for correct sound and speed. Look for
`PyAV plain ELD: decoded first PCM block`, or
`PyAV plain ELD decode failed`. Decoder initialization alone is
not success. The test requires no DLL installation and leaves the normal
audio offer unchanged. To restore the default for subsequent launches:

```powershell
Remove-Item Env:AAC_BACKEND
```
