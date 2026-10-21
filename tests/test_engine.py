import numpy as np

from avatarstream.audio_io import FrameAssembler
from avatarstream.compositor import Compositor
from avatarstream.engine import LipSyncEngine
from avatarstream.vad import EnergyGate

from conftest import speech_like


def _frames(sr, seconds, start, stop):
    return FrameAssembler(sr).push(speech_like(seconds, sr, start, stop), 100.0)


def test_silence_skips_model_and_shows_idle_loop(fake_model, synthetic_avatar):
    eng = LipSyncEngine(fake_model, synthetic_avatar, 48000, batch_size=4)
    frames = _frames(48000, 0.8, 10, 10)  # all silence
    out = []
    for s in range(0, len(frames), 8):
        out += eng.process(frames[s:s + 8])
    assert fake_model.calls == 0
    assert len(out) == 20
    idx = [vf.index for vf in out]
    assert idx[:12] == list(range(10)) + [9, 8]  # forward, then mirrored back
    for vf in out:
        assert vf.image is synthetic_avatar.frames[vf.index]
        assert not vf.speaking


def test_speech_runs_model_and_only_touches_the_face_box(fake_model, synthetic_avatar):
    eng = LipSyncEngine(fake_model, synthetic_avatar, 16000, batch_size=4,
                        gate=EnergyGate(-40), compositor=Compositor("full", feather=0))
    frames = _frames(16000, 1.6, 0.0, 1.6)
    out = []
    for s in range(0, len(frames), 8):
        out += eng.process(frames[s:s + 8])
    assert fake_model.calls > 0
    spoken = [vf for vf in out if vf.speaking]
    assert spoken
    y1, y2, x1, x2 = synthetic_avatar.coords[0]
    for vf in spoken:
        orig = synthetic_avatar.frames[vf.index]
        outside = np.ones(orig.shape[:2], bool)
        outside[y1:y2, x1:x2] = False
        assert np.array_equal(vf.image[outside], orig[outside])
        assert not np.array_equal(vf.image[y1:y2, x1:x2], orig[y1:y2, x1:x2])


def test_lower_blend_keeps_the_eyes():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    face = np.zeros((256, 256, 3), np.uint8)
    out = Compositor("lower", feather=0.05).paste(frame, face, (0, 256, 0, 256))
    assert np.array_equal(out[:110], frame[:110])  # top 43 %: untouched
    assert out[160:235, 20:236].max() == 0  # mouth area, inside the feathered edge: fully replaced
    assert np.array_equal(out[256:], frame[256:])
