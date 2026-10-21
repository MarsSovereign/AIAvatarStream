"""The lip-sync engine: audio frames in, composited video frames out."""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .audio_io import AudioFrame, AudioSource
from .avatar import Avatar
from .compositor import Compositor
from .features import MelFeaturizer
from .vad import EnergyGate

log = logging.getLogger(__name__)

VIDEO_FPS = 25


@dataclass
class VideoFrame:
    image: np.ndarray  # BGR uint8. Treat as read-only: idle frames are shared with the avatar.
    audio: list[AudioFrame]  # the 40 ms of audio that belongs to this frame
    speaking: bool
    index: int

    @property
    def t_capture(self) -> Optional[float]:
        times = [a.t_capture for a in self.audio if a.t_capture is not None]
        return min(times) if times else None

    def audio_src(self) -> np.ndarray:
        return np.concatenate([a.pcm_src for a in self.audio])


class LipSyncEngine:
    """Synchronous core. `process` takes 2*batch_size audio frames and returns batch_size video frames.

    Batches where nobody is speaking skip the model and show the idle loop.
    """

    def __init__(
        self,
        model,
        avatar: Avatar,
        src_sr: int,
        batch_size: int = 8,
        gate: Optional[EnergyGate] = None,
        compositor: Optional[Compositor] = None,
    ):
        self.model = model
        self.avatar = avatar
        self.batch_size = batch_size
        self.gate = gate or EnergyGate()
        self.compositor = compositor or Compositor()
        self.featurizer = MelFeaturizer(batch_size, src_sr)
        self._index = 0
        self.infer_seconds = 0.0
        self.infer_frames = 0

    @property
    def frames_per_step(self) -> int:
        return self.featurizer.frames_per_step

    def warm_up(self) -> float:
        """Prepares the audio features and the model so the first sentence is not delayed. Returns seconds per batch."""
        self.featurizer.warm_up()
        return self.model.warm_up(self.batch_size)

    def process(self, frames: list[AudioFrame]) -> list[VideoFrame]:
        for f in frames:
            if f.speech is None:
                f.speech = self.gate(f.pcm)
        items = self.featurizer.step(frames)
        idxs = [self.avatar.mirror_index(self._index + i) for i in range(len(items))]
        speaking = [any(a.speech for a in pair) for _, pair in items]
        self._index += len(items)

        if not any(speaking):
            return [
                VideoFrame(self.avatar.frames[idx], pair, False, idx) for idx, (_, pair) in zip(idxs, items)
            ]

        t = time.perf_counter()
        faces = np.stack([self.avatar.faces[i] for i in idxs])
        mels = np.stack([mel for mel, _ in items])
        preds = self.model.infer(faces, mels)
        self.infer_seconds += time.perf_counter() - t
        self.infer_frames += len(items)

        out = []
        for idx, pred, (_, pair), spk in zip(idxs, preds, items, speaking):
            image = self.compositor.paste(self.avatar.frames[idx], pred, self.avatar.coords[idx])
            out.append(VideoFrame(image, pair, spk, idx))
        return out

    @property
    def infer_fps(self) -> float:
        return self.infer_frames / self.infer_seconds if self.infer_seconds else 0.0


class EngineThread:
    """Pulls audio from a live source, runs the engine, pushes video frames to a queue."""

    def __init__(self, engine: LipSyncEngine, source: AudioSource, out_queue: "queue.Queue[VideoFrame]",
                 on_error: Optional[Callable[[BaseException], None]] = None):
        self.engine = engine
        self.source = source
        self.out_queue = out_queue
        self.on_error = on_error
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lipsync-engine", daemon=True)
        self.dropped_video = 0

    def _run(self) -> None:
        need = self.engine.frames_per_step
        batch: list[AudioFrame] = []
        try:
            while not self._stop.is_set():
                f = self.source.get(timeout=0.1)
                if f is None:
                    continue
                batch.append(f)
                if len(batch) < need:
                    continue
                for vf in self.engine.process(batch):
                    try:
                        self.out_queue.put_nowait(vf)
                    except queue.Full:
                        # Output is not being consumed fast enough: drop the oldest frame.
                        try:
                            self.out_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.dropped_video += 1
                        self.out_queue.put_nowait(vf)
                batch = []
        except BaseException as e:  # surface crashes instead of freezing the stream
            log.exception("口型引擎出错")
            if self.on_error:
                self.on_error(e)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
