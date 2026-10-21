"""Pastes the generated face back into the full frame."""
from __future__ import annotations

import cv2
import numpy as np


class Compositor:
    """Blends the model output into the original frame.

    blend="lower" only replaces the mouth and chin and keeps the original,
    sharper eyes; blend="full" replaces the whole face box like LiveTalking.
    `feather` softens the box edges so no rectangle shows around the face.
    """

    def __init__(self, blend: str = "lower", feather: float = 0.08):
        if blend not in ("lower", "full"):
            raise ValueError("blend must be 'lower' or 'full'")
        self.blend = blend
        self.feather = feather
        self._masks: dict[tuple[int, int], np.ndarray] = {}

    def _mask(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._masks:
            ramp = max(1.0, self.feather * min(h, w))
            ys = np.arange(h, dtype=np.float32)[:, None]
            xs = np.arange(w, dtype=np.float32)[None, :]
            edge = np.minimum(np.minimum(xs, w - 1 - xs), np.minimum(ys, h - 1 - ys))
            m = np.clip(edge / ramp, 0, 1)
            if self.blend == "lower":
                # Fade in from just below the eyes to the nose.
                top, bottom = 0.45 * h, 0.6 * h
                m = m * np.clip((ys - top) / (bottom - top), 0, 1)
            self._masks[key] = m[..., None]
        return self._masks[key]

    def paste(self, frame: np.ndarray, face: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        y1, y2, x1, x2 = box
        h, w = y2 - y1, x2 - x1
        resized = cv2.resize(face, (w, h), interpolation=cv2.INTER_LINEAR)
        out = frame.copy()
        if self.feather <= 0 and self.blend == "full":
            out[y1:y2, x1:x2] = resized
            return out
        m = self._mask(h, w)
        roi = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = (resized.astype(np.float32) * m + roi * (1.0 - m)).astype(np.uint8)
        return out
