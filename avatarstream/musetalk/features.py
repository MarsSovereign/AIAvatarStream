"""Streaming Whisper features for MuseTalk.

MuseTalk's offline pipeline encodes a whole clip, pads 4 zero features on the
left, and gives video frame i the 10 features starting at 2*i: 80 ms before the
frame to 120 ms after it, 5 layers each, flattened to 50x384. Here the same
window is cut from Whisper features of a short sliding buffer instead.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..audio_io import AudioFrame, silent_frame

WINDOW_BEFORE = 4  # features (20 ms each) before a frame's first audio feature
WINDOW = 10


class WhisperFeaturizer:
    """Same contract as MelFeaturizer: 2*B audio frames in, B (features, audio pair) items out."""

    def __init__(self, model, batch_size: int, src_sr: int, left: int = 20, right: int = 10):
        if left < WINDOW_BEFORE or right < WINDOW - WINDOW_BEFORE - 2:
            raise ValueError("context too short for the MuseTalk audio window")
        self.model = model
        self.batch_size = batch_size
        self.left = left
        self.right = right
        self._context: list[np.ndarray] = [np.zeros(320, np.float32) for _ in range(left + right)]
        self._pending: deque[AudioFrame] = deque(silent_frame(src_sr) for _ in range(right))

    @property
    def frames_per_step(self) -> int:
        return 2 * self.batch_size

    def warm_up(self) -> None:
        self.model.audio_features(np.zeros((self.left + self.right + self.frames_per_step) * 320, np.float32))

    def step(self, frames: list[AudioFrame]) -> list[tuple[np.ndarray, list[AudioFrame]]]:
        if len(frames) != self.frames_per_step:
            raise ValueError(f"expected {self.frames_per_step} audio frames, got {len(frames)}")
        self._context.extend(f.pcm for f in frames)
        self._pending.extend(frames)

        feats = self.model.audio_features(np.concatenate(self._context))  # (n_frames, 5, 384)
        out = []
        for j in range(self.batch_size):
            # The frame's audio starts at context index left + 2j (output lags input by `right` frames).
            start = self.left + 2 * j - WINDOW_BEFORE
            window = feats[start: start + WINDOW]
            out.append((window.reshape(-1, window.shape[-1]), [self._pending.popleft(), self._pending.popleft()]))

        self._context = self._context[-(self.left + self.right):]
        return out
