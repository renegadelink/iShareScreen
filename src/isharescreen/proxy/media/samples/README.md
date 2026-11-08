# HEVC hardware probe frames

These small Annex-B streams are generated test inputs, not configuration values.
Each contains VPS, SPS, PPS and a single 64x64 8-bit IDR picture:

- `hevc444-idr.hevc`: HEVC 4:4:4, generated with libx265 from a YUV444P frame.
- `hevc420-idr.hevc`: HEVC 4:2:0, generated with libx265 from a black YUV420P
  frame (Y=16, U=V=128).

The 4:2:0 input uses a one-frame encode with these x265 parameters:

```text
log-level=error:repeat-headers=1:keyint=1:no-info=1:pools=none:frame-threads=1
```

`hwcaps.py` loads the packaged streams using `importlib.resources` and tries
decoding them through the same hardware backends used for the live stream.
Software fallback is disabled so an ordinary CPU decode cannot pass the probe.
The result is cached for the process lifetime.

The tests verify that the 4:2:0 stream decodes to the expected dimensions and
pixel format. These probes check basic decoder operation; they do not benchmark
throughput or prove support at every stream resolution.
