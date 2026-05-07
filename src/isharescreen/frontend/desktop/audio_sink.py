"""PCM playback sink for the desktop frontend.

The proxy's audio RX thread decodes Apple's PT=101 AAC-ELD-SBR into
float32 stereo @ 48 kHz and pushes each chunk through
`Session.set_audio_callback`. This module receives those chunks on
the proxy thread and hands them to the OS sound device on a separate
PortAudio realtime callback thread, with a small jitter buffer in
between.

Two threads, no shared state with the render loop:

  proxy audio rx thread          PortAudio callback thread
  ─────────────────────          ──────────────────────────
  feed(pcm) ──┐                  cb(outdata, frames) ── drains queue
              ▼                                          (drops chunks
       deque [(t, pcm), ...] ◀──────────────────────── older than 100 ms,
              ▲                                          fills silence on
              └── stop() drops + closes stream            empty)

Sync model: wall-clock master with a fixed-target jitter buffer. The
sound card consumes samples at its own steady rate; we never resample
or stretch. If the audio decoder lags, the callback writes silence for
that block (≈5 ms) and continues. If the audio decoder bursts (e.g.
after a network hiccup), chunks older than `JITTER_TARGET_S +
SLACK_LATE_S` are dropped at the head of the queue rather than queued
for late playback. Net: a stalled audio path NEVER blocks the render
thread, and a long audio backlog NEVER produces a delayed flood.

Latency budget on macOS CoreAudio @ blocksize=240, latency='low':
~10 ms output buffer + 40 ms jitter target = ~50 ms steady-state
end-to-end. Windows WASAPI shared mode adds ~5-10 ms on top; Linux
ALSA-via-PipeWire is in the same range.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np


log = logging.getLogger(__name__)


# Wall-clock target: callback aims to keep ≈40 ms of audio queued
# ahead, so a single 5 ms PortAudio block can underrun without an
# audible glitch. Lower → tighter sync, more dropouts; higher → safer
# audio, more lip-sync slip.
JITTER_TARGET_S = 0.040
# Anything older than this since `feed()` enqueued it is dropped at the
# head of the queue. Picked so a single Wi-Fi reorder of ≤60 ms never
# triggers the drop, but a ≥100 ms stall drops the backlog instead of
# replaying it half a second behind the video.
SLACK_LATE_S = 0.060
# PortAudio block size in frames. 240 @ 48 kHz = 5 ms. Smaller blocks
# = lower output latency at the cost of more callback wake-ups.
DEFAULT_BLOCKSIZE = 240
# Max chunks queued before we start dropping at the tail in `feed()`.
# Sized at 50 × ~10 ms ≈ 500 ms — far above JITTER_TARGET_S, so the
# normal callback-side drop path handles steady-state pressure;
# this is just a hard upper bound to stop runaway memory growth if
# the playback callback dies.
MAX_QUEUE_CHUNKS = 50


class AudioSink:
    """sounddevice OutputStream + thread-safe jitter buffer.

    Lifecycle: construct → `start()` → `feed()` from any thread →
    `stop()` once. `feed()` and the PortAudio callback are both safe
    to call concurrently; the render thread doesn't touch this class
    at all.

    Construct lazily — instantiating `AudioSink` opens the OS sound
    device, which can fail (no default device, all devices in
    exclusive use, etc). Callers should catch the exception and
    proceed silently if audio playback isn't critical.
    """

    def __init__(
        self,
        sample_rate: int = 48_000,
        channels: int = 2,
        blocksize: int = DEFAULT_BLOCKSIZE,
        jitter_target_s: float = JITTER_TARGET_S,
        slack_late_s: float = SLACK_LATE_S,
    ) -> None:
        # Import here so a missing/broken sounddevice install fails at
        # AudioSink() construction (caught by frontend) rather than at
        # module import time (which would break `iss --headless` too).
        import sounddevice as sd

        self._sd = sd
        self._sr = sample_rate
        self._ch = channels
        self._jitter_s = jitter_target_s
        self._slack_late_s = slack_late_s

        self._queue: deque[tuple[float, np.ndarray]] = deque()
        self._lock = threading.Lock()
        # In-flight chunk being drained across multiple callbacks.
        self._head: Optional[np.ndarray] = None
        self._head_off = 0
        # Counters for end-of-session diagnostic only — incremented
        # inside the lock alongside the queue mutation, so they're
        # consistent without their own atomics.
        self._feed_count = 0
        self._frames_played = 0
        self._frames_silence = 0

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=blocksize,
            latency="low",
            callback=self._cb,
        )
        self._started = False
        self._closed = False

    # -- public API --------------------------------------------------

    def start(self) -> None:
        """Open the device + begin pulling. Idempotent."""
        if not self._started and not self._closed:
            self._stream.start()
            self._started = True
            log.info(
                "audio sink started: %d Hz × %d ch, %.0f ms jitter target",
                self._sr, self._ch, self._jitter_s * 1000,
            )

    def feed(self, pcm: np.ndarray) -> None:
        """Push one decoded PCM chunk. Called from the proxy's audio
        RX thread; never blocks the render thread.

        Accepts `(N, 2)` float32 stereo. Anything else is reshaped if
        possible, dropped if not.
        """
        if pcm is None or pcm.size == 0 or self._closed:
            return
        # Be liberal in what we accept: a (N,) mono array gets
        # duplicated, a (N, 1) array too. Anything else is dropped.
        if pcm.ndim == 1:
            pcm = np.repeat(pcm[:, None], self._ch, axis=1)
        elif pcm.ndim == 2 and pcm.shape[1] == 1 and self._ch == 2:
            pcm = np.repeat(pcm, 2, axis=1)
        elif pcm.ndim != 2 or pcm.shape[1] != self._ch:
            return
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32, copy=False)
        if not pcm.flags["C_CONTIGUOUS"]:
            pcm = np.ascontiguousarray(pcm)

        now = time.monotonic()
        with self._lock:
            if len(self._queue) >= MAX_QUEUE_CHUNKS:
                self._queue.popleft()
            self._queue.append((now, pcm))
            self._feed_count += 1

    def stop(self) -> None:
        """Stop playback + close the device. Idempotent + safe in
        teardown after `start()` was never called."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._started:
                self._stream.stop()
            self._stream.close()
        except Exception as e:
            log.debug("audio sink stop: %s", e)
        with self._lock:
            self._queue.clear()
            self._head = None
            self._head_off = 0
            feed = self._feed_count
            played = self._frames_played
            silence = self._frames_silence
        total = played + silence
        ratio = (silence / total) if total else 0.0
        log.info(
            "audio sink stopped: %d feeds, %d frames played (%d silence, %.1f%%)",
            feed, played, silence, ratio * 100.0,
        )

    # -- PortAudio callback (RT thread) ------------------------------

    def _cb(self, outdata: np.ndarray, frames: int, _time_info, status) -> None:
        """Pull `frames` samples into `outdata`. Runs on PortAudio's
        own high-priority thread; we re-acquire the GIL only because
        `sounddevice` uses a Python callback, so keep the body to
        numpy slicing + a deque popleft. No allocations, no logging
        on the steady path."""
        if status:
            # Underflow / overflow flag from the OS; usually fine to
            # ignore but log at DEBUG so it's available when someone
            # actively chases a glitch.
            log.debug("portaudio status: %s", status)

        # Drop chunks older than the jitter target + late-slack. This
        # keeps a network or thread stall from playing ancient audio
        # half a second behind the video once it recovers.
        cutoff = time.monotonic() - (self._jitter_s + self._slack_late_s)
        with self._lock:
            while self._queue and self._queue[0][0] < cutoff:
                self._queue.popleft()

        idx = 0
        while idx < frames:
            if self._head is None:
                with self._lock:
                    if not self._queue:
                        break
                    _, self._head = self._queue.popleft()
                    self._head_off = 0
            avail = len(self._head) - self._head_off
            need = frames - idx
            n = avail if avail < need else need
            outdata[idx:idx + n] = self._head[self._head_off:self._head_off + n]
            idx += n
            self._head_off += n
            if self._head_off >= len(self._head):
                self._head = None
                self._head_off = 0

        if idx < frames:
            outdata[idx:].fill(0.0)
            self._frames_silence += frames - idx
        self._frames_played += idx


def make_audio_sink() -> Optional[AudioSink]:
    """Open an `AudioSink` with sensible defaults, or return `None` if
    the OS doesn't have a usable sound device. Callers should treat
    `None` as "audio playback off, video continues" — every other
    audio failure (decoder unavailable, sound API not installed) is
    already handled non-fatally elsewhere in the stack.
    """
    sink: Optional[AudioSink] = None
    try:
        sink = AudioSink()
        sink.start()
        return sink
    except ImportError as e:
        log.warning(
            "sounddevice not installed; audio playback disabled (%s)", e,
        )
    except Exception as e:
        log.warning("audio sink init failed; audio playback disabled: %s", e)
    # If `AudioSink()` constructed the OutputStream but `start()`
    # raised, close the device — otherwise we leak it until GC.
    if sink is not None:
        sink.stop()
    return None


__all__ = ["AudioSink", "make_audio_sink"]
