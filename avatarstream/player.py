"""Plays video frames at a steady 25 fps and keeps their audio in step."""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

from .engine import VIDEO_FPS, VideoFrame

log = logging.getLogger(__name__)


class Stats:
    """Rolling numbers for the status page and the console."""

    def __init__(self, window: int = 125):
        self._lat: deque[float] = deque(maxlen=window)
        self._times: deque[float] = deque(maxlen=window)
        self.speaking = False
        self.level_db = -200.0
        self.frames = 0
        self.stalls = 0
        self.extra: dict = {}
        self._lock = threading.Lock()

    def on_frame(self, frame: VideoFrame, now: float) -> None:
        with self._lock:
            self.frames += 1
            self.speaking = frame.speaking
            self._times.append(now)
            t = frame.t_capture
            if t is not None:
                self._lat.append(now - t)

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._lat)
            times = list(self._times)
            out = {
                "frames": self.frames,
                "speaking": self.speaking,
                "level_db": round(self.level_db, 1),
                "stalls": self.stalls,
                "fps": round((len(times) - 1) / (times[-1] - times[0]), 1) if len(times) > 1 and times[-1] > times[0] else 0.0,
                "latency_ms": round(1000 * sum(lat) / len(lat)) if lat else None,
                "latency_p95_ms": round(1000 * lat[int(0.95 * (len(lat) - 1))]) if lat else None,
            }
            out.update(self.extra)
        # The audio OBS takes straight from the voice changer needs this much delay.
        out["obs_sync_offset_ms"] = out["latency_ms"]
        return out


class Player:
    """Takes frames from the engine and hands them to the sinks one every 40 ms.

    The engine produces frames in bursts of one batch; the player smooths them
    out. If it is given an audio output, it plays each frame's audio at the
    moment the frame is shown, so sound and picture leave together.
    """

    def __init__(self, sinks: list, audio_out=None, stats: Optional[Stats] = None,
                 fps: int = VIDEO_FPS, prebuffer: int = 2):
        self.sinks = sinks
        self.audio_out = audio_out
        self.stats = stats or Stats()
        self.period = 1.0 / fps
        self.prebuffer = prebuffer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run(self, frames: "queue.Queue[VideoFrame]") -> None:
        next_t: Optional[float] = None
        while not self._stop.is_set():
            if next_t is None:
                # (Re)start only once a few frames are queued, so the first batch plays smoothly.
                if frames.qsize() < self.prebuffer:
                    time.sleep(0.005)
                    continue
                next_t = time.monotonic()
            try:
                frame = frames.get(timeout=max(0.0, next_t - time.monotonic()) + 0.2)
            except queue.Empty:
                self.stats.stalls += 1
                next_t = None
                continue
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            now = time.monotonic()
            for sink in self.sinks:
                try:
                    sink.send(frame)
                except Exception:
                    log.exception("输出 %s 出错", type(sink).__name__)
            if self.audio_out is not None:
                self.audio_out.write(frame.audio_src())
            self.stats.on_frame(frame, now)
            next_t += self.period
            if now - next_t > 0.25:  # fell far behind: resync instead of rushing
                next_t = now

    def start(self, frames: "queue.Queue[VideoFrame]") -> None:
        self._thread = threading.Thread(target=self.run, args=(frames,), name="player", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
