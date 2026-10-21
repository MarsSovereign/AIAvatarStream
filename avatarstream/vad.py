"""Decides which 20 ms frames count as speech.

A live microphone never delivers true silence, so the engine needs a gate:
frames below the threshold let the avatar fall back to its idle loop and skip
model inference entirely.
"""
from __future__ import annotations

import numpy as np

from .audio_io import FRAME_MS


def level_db(pcm: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64)))) if len(pcm) else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


class EnergyGate:
    """RMS threshold with a hangover, so short pauses inside a sentence stay "speech"."""

    def __init__(self, threshold_db: float = -45.0, hangover_ms: int = 300):
        self.threshold_db = threshold_db
        self.hangover_frames = max(0, hangover_ms // FRAME_MS)
        self._hang = 0
        self.last_db = -200.0

    def __call__(self, pcm: np.ndarray) -> bool:
        self.last_db = level_db(pcm)
        if self.last_db >= self.threshold_db:
            self._hang = self.hangover_frames
            return True
        if self._hang > 0:
            self._hang -= 1
            return True
        return False
