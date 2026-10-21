"""MuseTalk with the real weights (slow on CPU: a few minutes). Skipped without models/musetalk."""
import json
import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from avatarstream.audio_io import FrameAssembler
from avatarstream.avatar import load_avatar
from avatarstream.engine import LipSyncEngine
from avatarstream.musetalk.model import missing_files

from conftest import AVATAR_PATH, ROOT
from test_musetalk_features import offline_chunks

MUSETALK_DIR = os.environ.get("AVATARSTREAM_MUSETALK", os.path.join(ROOT, "models", "musetalk"))
SPEECH = os.path.join(os.path.dirname(__file__), "data", "speech_16k.wav")  # speech from 0.64 s to 3.2 s

pytestmark = pytest.mark.skipif(
    bool(missing_files(MUSETALK_DIR)) or not os.path.isdir(AVATAR_PATH),
    reason="需要 MuseTalk 模型（download --musetalk）和示例形象",
)


@pytest.fixture(scope="module")
def model():
    import torch

    from avatarstream.musetalk.model import MuseTalkModel

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    return MuseTalkModel(MUSETALK_DIR, "cpu")


@pytest.fixture(scope="module")
def small_avatar(model, tmp_path_factory):
    """The first 6 frames of the sample avatar, with MuseTalk data prepared."""
    from avatarstream.musetalk.avatar import prepare

    d = tmp_path_factory.mktemp("avatar")
    for sub in ("full_imgs", "face_imgs"):
        os.makedirs(d / sub)
        for i in range(6):
            shutil.copy(os.path.join(AVATAR_PATH, sub, f"{i:08d}.png"), d / sub / f"{i:08d}.png")
    coords = load_avatar(AVATAR_PATH, max_frames=6).coords
    (d / "coords.json").write_text(json.dumps([list(c) for c in coords]))
    n = prepare(str(d), model, os.path.join(MUSETALK_DIR, "face-parse-bisent", "79999_iter.pth"), "cpu")
    assert n == 6
    return str(d)


def test_prepare_finds_the_lower_face(small_avatar):
    from avatarstream.musetalk.avatar import load

    avatar = load_avatar(small_avatar)
    data = load(small_avatar, avatar)
    assert tuple(data.latents.shape) == (6, 8, 32, 32)
    for (x1, y1, x2, y2), (wy1, wy2, wx1, wx2), mask in zip(data.boxes, avatar.coords, data.masks):
        # Same face as the Wav2Lip detector found: chin within 60 px, box inside the face's width.
        assert abs(y2 - wy2) < 60 and x1 >= wx1 - 30 and x2 <= wx2 + 30
        assert y1 < (wy1 + wy2) / 2 < y2
        assert mask.max() == 255 and mask[: mask.shape[0] // 3].max() == 0


def test_streaming_features_line_up_with_offline(model):
    audio, sr = sf.read(SPEECH, dtype="float32")
    frames = FrameAssembler(sr).push(audio)
    frames = frames[: len(frames) // 16 * 16]
    from avatarstream.musetalk.features import WhisperFeaturizer

    feat = WhisperFeaturizer(model, 8, sr)
    got = []
    for s in range(0, len(frames), 16):
        got += [w for w, _ in feat.step(frames[s:s + 16])]
    stream = np.concatenate([np.zeros(3200, np.float32)] + [f.pcm for f in frames])
    full = model.audio_features(stream)

    def similarity(shift):
        padded = np.concatenate([np.zeros_like(full[:4]), full, np.zeros_like(full[:30])])
        out = []
        for v in range(24, 80):  # video frames that carry speech
            e = padded[2 * v + shift: 2 * v + shift + 10][:, 1:4]
            g = got[v].reshape(10, 5, 384)[:, 1:4]
            out.append(((g * e).sum(-1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(e, axis=-1) + 1e-9)).mean())
        return float(np.mean(out))

    scores = {s: similarity(s) for s in (-2, -1, 0, 1, 2)}
    assert max(scores, key=scores.get) == 0, scores  # not off by even one 20 ms step
    assert scores[0] > 0.9
    assert offline_chunks(full, 1)[0].shape == (50, 384)


def test_mouth_moves_with_speech_and_eyes_stay(model, small_avatar):
    from avatarstream.musetalk import MuseTalkBackend
    from avatarstream.musetalk.avatar import load

    avatar = load_avatar(small_avatar)
    data = load(small_avatar, avatar)
    audio, sr = sf.read(SPEECH, dtype="float32")
    frames = FrameAssembler(sr).push(audio[int(1.18 * sr): int(2.9 * sr)])  # speech only, from the first word
    backend = MuseTalkBackend(model, avatar, data, sr, batch_size=2)
    eng = LipSyncEngine(None, avatar, sr, 2, backend=backend)
    out = []
    for s in range(0, 40, 4):  # 10 steps; the first 5 video frames carry the 200 ms start-up delay
        out += eng.process(frames[s:s + 4])
    talk = [f for f in out if f.speaking]
    assert len(talk) >= 10

    x1, y1, x2, y2 = data.boxes[0]
    xs, ys = data.crop_boxes[0][:2]
    mask_top = ys + np.argmax(data.masks[0].max(axis=1) > 0)
    mouth = (slice(y1 + int(0.7 * (y2 - y1)), y2 - 10), slice(x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4))
    changes = []
    for f in talk:
        orig = avatar.frames[f.index]
        assert np.array_equal(f.image[:mask_top], orig[:mask_top])  # everything above the mask, bit for bit
        changes.append(np.abs(f.image[mouth].astype(float) - orig[mouth]).mean())
    assert np.mean(changes) > 3  # the mouth is redrawn
    mouths = np.array([f.image[mouth] for f in talk], np.float32)
    assert np.abs(np.diff(mouths, axis=0)).mean() > 1.0  # and it moves from frame to frame
