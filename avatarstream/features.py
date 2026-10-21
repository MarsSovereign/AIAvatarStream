"""Turns a stream of 20 ms audio frames into Wav2Lip mel windows.

This is the feature step of LipASR in LiveTalking (Apache-2.0), rewritten so it
can be driven and tested without threads. The windowing must stay identical to
what the model was trained with.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .audio_io import AudioFrame, silent_frame
from .wav2lip import audio as w2l_audio

MEL_STEP = 16  # mel frames per window (200 ms)
MEL_PER_AUDIO_FRAME = 80.0 / 50.0  # mel frames per 20 ms audio frame


class MelFeaturizer:
    """Consumes 2*B audio frames per step and returns B (mel window, audio pair) items.

    `left` and `right` audio frames of context surround each step. The audio
    paired with each video frame lags the input by `right` frames, because the
    window for a frame needs that much audio after it.
    """

    def __init__(self, batch_size: int, src_sr: int, left: int = 10, right: int = 10):
        self.batch_size = batch_size
        self.left = left
        self.right = right
        self._context: list[np.ndarray] = [np.zeros_like(silent_frame(src_sr).pcm) for _ in range(left + right)]
        self._pending: deque[AudioFrame] = deque(silent_frame(src_sr) for _ in range(right))

    @property
    def frames_per_step(self) -> int:
        return 2 * self.batch_size

    def warm_up(self) -> None:
        """librosa compiles its numba kernels on first use, which can take seconds; do it before going live."""
        n = (self.left + self.right + self.frames_per_step) * len(self._context[0])
        w2l_audio.melspectrogram(np.zeros(n, np.float32))

    def step(self, frames: list[AudioFrame]) -> list[tuple[np.ndarray, list[AudioFrame]]]:
        if len(frames) != self.frames_per_step:
            raise ValueError(f"expected {self.frames_per_step} audio frames, got {len(frames)}")
        self._context.extend(f.pcm for f in frames)
        self._pending.extend(frames)

        mel = w2l_audio.melspectrogram(np.concatenate(self._context))
        start = self.left * MEL_PER_AUDIO_FRAME
        mult = 2 * MEL_PER_AUDIO_FRAME
        n_mel = mel.shape[1]
        out = []
        for i in range(self.batch_size):
            idx = int(start + i * mult)
            window = mel[:, n_mel - MEL_STEP:] if idx + MEL_STEP > n_mel else mel[:, idx: idx + MEL_STEP]
            pair = [self._pending.popleft(), self._pending.popleft()]
            out.append((window, pair))

        self._context = self._context[-(self.left + self.right):]
        return out
