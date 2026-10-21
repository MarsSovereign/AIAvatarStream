"""MuseTalk audio windows, checked against MuseTalk's offline indexing (no model files needed)."""
import numpy as np
import pytest

from avatarstream.audio_io import AudioFrame
from avatarstream.musetalk.features import WhisperFeaturizer
from avatarstream.musetalk.avatar import blend_mask, box_from_landmarks, crop_box, paste


class LocalFeatures:
    """Stands in for Whisper with a per-frame feature: row n = value of audio frame n, times (layer + 1)."""

    def audio_features(self, pcm):
        n = len(pcm) // 320
        vals = pcm[: n * 320].reshape(n, 320)[:, 0]
        layers = np.arange(1, 6, dtype=np.float32)[None, :, None]
        return np.broadcast_to(vals[:, None, None] * layers, (n, 5, 384)).astype(np.float32)


def offline_chunks(features, num_frames, left_pad=2, right_pad=2):
    """get_whisper_chunk from MuseTalk 1.5 at 25 fps, on precomputed features."""
    per_frame = 2 * (left_pad + right_pad + 1)
    padded = np.concatenate([np.zeros_like(features[: 2 * left_pad]), features,
                             np.zeros_like(features[: 2 * 3 * right_pad])])
    return [padded[2 * i: 2 * i + per_frame].reshape(-1, 384) for i in range(num_frames)]


@pytest.mark.parametrize("batch_size", [2, 5])
def test_streaming_windows_match_offline_indexing(batch_size):
    right = 10
    n_audio = 2 * batch_size * 6
    values = np.arange(1, n_audio + 1, dtype=np.float32)
    frames = [AudioFrame(np.full(320, v, np.float32), np.full(960, v, np.float32), None) for v in values]

    feat = WhisperFeaturizer(LocalFeatures(), batch_size, 48000, left=20, right=right)
    got = []
    for s in range(0, n_audio, 2 * batch_size):
        got += feat.step(frames[s:s + 2 * batch_size])

    # Offline view of the same stream: `right` frames of silence first (the output delay), then the audio.
    stream = np.concatenate([np.zeros(right, np.float32), values])
    expected = offline_chunks(LocalFeatures().audio_features(np.repeat(stream, 320)), len(got))
    for v, ((window, pair), exp) in enumerate(zip(got, expected)):
        assert window.shape == (50, 384)
        np.testing.assert_array_equal(window, exp, err_msg=f"video frame {v}")
        # and the audio played with that frame is the one the window is centred on
        first = 2 * v - right
        assert [a.pcm[0] for a in pair] == [stream[right + first] if first >= 0 else 0,
                                            stream[right + first + 1] if first + 1 >= 0 else 0]


def test_context_must_cover_the_window():
    with pytest.raises(ValueError):
        WhisperFeaturizer(LocalFeatures(), 4, 16000, left=2)
    with pytest.raises(ValueError):
        WhisperFeaturizer(LocalFeatures(), 4, 16000, right=3)


def test_box_follows_musetalk_rule():
    lm = np.zeros((68, 2))
    lm[:, 0] = np.linspace(100, 300, 68)
    lm[:, 1] = 200
    lm[29, 1] = 250  # nose
    lm[8, 1] = 400   # chin
    x1, y1, x2, y2 = box_from_landmarks(lm, height=1000, extra_margin=10)
    assert (x1, x2) == (100, 300)
    assert y1 == 100  # as far above the nose as the chin is below it: 250 - (400 - 250)
    assert y2 == 410
    assert crop_box((100, 100, 300, 410)) == [200 - 232, 255 - 232, 200 + 232, 255 + 232]


def test_mask_and_paste_only_touch_the_lower_face():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (600, 500, 3), dtype=np.uint8)
    box = (150, 150, 350, 400)
    cbox = crop_box(box)
    mask = blend_mask(np.full((512, 512), 255, np.uint8), box, cbox)
    assert mask.shape == (cbox[3] - cbox[1], cbox[2] - cbox[0])
    blur = int(0.1 * mask.shape[0] // 2 * 2) + 1
    top = mask.shape[0] // 2 - blur // 2 - 1  # the lower half, softened upwards by half the blur kernel
    assert mask[:top].max() == 0 and mask[top + blur:].max() == 255

    out = paste(frame, np.zeros((256, 256, 3), np.uint8), box, mask, cbox)
    top_of_mask = cbox[1] + top
    assert np.array_equal(out[:top_of_mask], frame[:top_of_mask])  # eyes untouched, bit for bit
    assert np.array_equal(out[:, :150], frame[:, :150]) and np.array_equal(out[400:], frame[400:])
    assert out[330:380, 200:300].max() < 40  # mouth area replaced by the (black) generated face
