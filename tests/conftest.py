import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from avatarstream.avatar import FACE_SIZE, Avatar  # noqa: E402

MODEL_PATH = os.environ.get("AVATARSTREAM_MODEL", os.path.join(ROOT, "models", "wav2lip.pth"))
AVATAR_PATH = os.environ.get("AVATARSTREAM_AVATAR", os.path.join(ROOT, "data", "wav2lip_avatar_glass_man"))

needs_model = pytest.mark.skipif(
    not (os.path.exists(MODEL_PATH) and os.path.isdir(AVATAR_PATH)),
    reason="需要 models/wav2lip.pth 和 data/wav2lip_avatar_glass_man（python -m avatarstream download）",
)


class FakeModel:
    """Stands in for Wav2Lip: paints the lower half of each face with a level taken from the mel window."""

    def __init__(self):
        self.calls = 0

    def infer(self, faces, mels):
        self.calls += 1
        out = faces.copy()
        for i, mel in enumerate(mels):
            out[i, FACE_SIZE // 2:] = int(np.clip((mel.mean() + 4) * 30, 0, 255))
        return out

    def warm_up(self, batch_size):
        return 0.001


def make_synthetic_avatar(n=10, w=200, h=240, box=(40, 200, 30, 170)):
    rng = np.random.default_rng(0)
    frames, faces = [], []
    for i in range(n):
        f = np.full((h, w, 3), 40 + i, np.uint8)
        f[box[0]:box[1], box[2]:box[3]] = rng.integers(0, 255, (box[1] - box[0], box[3] - box[2], 3), dtype=np.uint8)
        frames.append(f)
        faces.append(np.ascontiguousarray(
            np.resize(f[box[0]:box[1], box[2]:box[3]], (FACE_SIZE, FACE_SIZE, 3)).astype(np.uint8)))
    return Avatar(frames, faces, [box] * n)


def speech_like(seconds, sr, start=0.0, stop=None, seed=1):
    """Silence, then a voiced-sounding signal (harmonics with a syllable-rate envelope), then silence."""
    t = np.arange(int(seconds * sr)) / sr
    stop = seconds if stop is None else stop
    rng = np.random.default_rng(seed)
    voice = sum(np.sin(2 * np.pi * f0 * t) / k for k, f0 in enumerate((140, 280, 420, 700, 1100), 1))
    env = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) * ((t >= start) & (t < stop))
    x = 0.3 * voice * env + 0.003 * rng.standard_normal(len(t))
    return x.astype(np.float32)


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def synthetic_avatar():
    return make_synthetic_avatar()
