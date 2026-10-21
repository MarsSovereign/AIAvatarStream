"""Checks with the real Wav2Lip weights and a real avatar (slow on CPU: about a minute)."""
import json
import os
import subprocess

import numpy as np
import pytest
import soundfile as sf

from avatarstream import cli
from avatarstream.audio_io import FrameAssembler
from avatarstream.avatar import load_avatar
from avatarstream.compositor import Compositor
from avatarstream.engine import LipSyncEngine
from avatarstream.model import LipSyncModel

from conftest import AVATAR_PATH, MODEL_PATH, needs_model

pytestmark = needs_model

SPEECH = os.path.join(os.path.dirname(__file__), "data", "speech_16k.wav")  # 0.64 s silence, speech, 0.64 s silence


@pytest.fixture(scope="module")
def engine_parts():
    return LipSyncModel(MODEL_PATH, "cpu"), load_avatar(AVATAR_PATH, max_frames=40)


def test_mouth_moves_only_while_talking(engine_parts):
    model, avatar = engine_parts
    audio, sr = sf.read(SPEECH, dtype="float32")
    eng = LipSyncEngine(model, avatar, sr, batch_size=8, compositor=Compositor("lower"))
    frames = FrameAssembler(sr).push(audio)
    out = []
    for s in range(0, len(frames) - 15, 16):
        out += eng.process(frames[s:s + 16])

    # Speech starts at 0.64 s: the first two batches are pure silence and must skip the model.
    # (A batch that mixes silence and speech is run through the model as a whole.)
    for f in out[:16]:
        assert not f.speaking and f.image is avatar.frames[f.index]
    talk = [f for f in out if f.speaking]
    assert len(talk) > 40

    y1, y2, x1, x2 = avatar.coords[0]
    h = y2 - y1
    eyes = (slice(y1 + h // 10, y1 + int(0.4 * h)), slice(x1, x2))
    mouth = (slice(y1 + int(0.65 * h), y2 - h // 20), slice(x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4))
    for f in talk:
        assert np.array_equal(f.image[eyes], avatar.frames[f.index][eyes])  # lower blend keeps the eyes

    gen = np.array([f.image[mouth] for f in talk], np.float32)
    idle = np.array([avatar.frames[f.index][mouth] for f in talk], np.float32)
    gen_motion = np.abs(np.diff(gen, axis=0)).mean()
    idle_motion = np.abs(np.diff(idle, axis=0)).mean()
    # The redrawn mouth moves clearly more than the idle video does on its own.
    assert gen_motion > 1.2 * idle_motion, (gen_motion, idle_motion)
    assert np.abs(gen - idle).mean(axis=(1, 2, 3)).std() > 0.6


def test_render_command_keeps_audio_and_video_the_same_length(tmp_path):
    out = tmp_path / "out.mp4"
    cli.main(["render", "--model", MODEL_PATH, "--avatar", AVATAR_PATH, "--device", "cpu",
              "--audio", SPEECH, "--out", str(out)])
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                                      "-of", "json", str(out)], capture_output=True, text=True).stdout)
    d = {s["codec_type"]: float(s["duration"]) for s in info["streams"]}
    assert abs(d["video"] - d["audio"]) < 0.05
    assert d["video"] >= 3.84
