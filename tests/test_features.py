import numpy as np

from avatarstream.audio_io import AudioFrame
from avatarstream.features import MelFeaturizer
from avatarstream.wav2lip import audio


def reference_lipasr(chunks, batch_size, left=10, right=10):
    """The windowing loop of LipASR.run_step in LiveTalking, kept verbatim as the reference."""
    frames = [np.zeros(320, np.float32)] * (left + right)
    results = []
    for s in range(0, len(chunks), batch_size * 2):
        frames = frames + chunks[s:s + batch_size * 2]
        inputs = np.concatenate(frames)
        mel = audio.melspectrogram(inputs)
        lft = max(0, left * 80 / 50)
        mel_idx_multiplier = 80. * 2 / 50
        mel_step_size = 16
        i = 0
        while i < (len(frames) - left - right) / 2:
            start_idx = int(lft + i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                results.append(mel[:, len(mel[0]) - mel_step_size:])
            else:
                results.append(mel[:, start_idx: start_idx + mel_step_size])
            i += 1
        frames = frames[-(left + right):]
    return results


def test_matches_livetalking_windowing():
    rng = np.random.default_rng(3)
    b = 4
    chunks = [rng.standard_normal(320).astype(np.float32) * 0.2 for _ in range(2 * b * 5)]
    expected = reference_lipasr(chunks, b)
    feat = MelFeaturizer(b, 16000)
    got = []
    for s in range(0, len(chunks), 2 * b):
        got += [m for m, _ in feat.step([AudioFrame(c, c, None) for c in chunks[s:s + 2 * b]])]
    assert len(got) == len(expected) == 5 * b
    for g, e in zip(got, expected):
        assert g.shape == (80, 16)
        np.testing.assert_allclose(g, e, atol=1e-5)


def test_audio_pairs_lag_by_right_context():
    b, right = 3, 10
    feat = MelFeaturizer(b, 48000, right=right)
    frames = [AudioFrame(np.zeros(320, np.float32), np.full(960, i, np.float32), float(i)) for i in range(2 * b * 4)]
    pairs = []
    for s in range(0, len(frames), 2 * b):
        pairs += [p for _, p in feat.step(frames[s:s + 2 * b])]
    flat = [a for p in pairs for a in p]
    assert all(a.t_capture is None for a in flat[:right])  # warm-up padding
    assert [a.t_capture for a in flat[right:]] == [float(i) for i in range(len(frames) - right)]
