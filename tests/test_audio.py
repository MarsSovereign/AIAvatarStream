import numpy as np
import pytest
import soundfile as sf

from avatarstream.audio_io import MODEL_FRAME, FrameAssembler, Resampler, iter_file_frames, read_audio_file
from avatarstream.vad import EnergyGate, level_db


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_resampler_keeps_frame_count_and_pitch(sr):
    rs = Resampler(sr)
    t = np.arange(sr * 2) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    n = sr // 50
    out = [rs.process(x[i:i + n]) for i in range(0, len(x), n)]
    assert all(len(o) == MODEL_FRAME for o in out)
    y = np.concatenate(out)[MODEL_FRAME * 5:]  # skip start-up
    spectrum = np.abs(np.fft.rfft(y))
    peak_hz = np.argmax(spectrum) * 16000 / len(y)
    assert abs(peak_hz - 440) < 3
    assert 0.45 < np.max(np.abs(y)) < 0.55


def test_assembler_is_independent_of_input_block_size():
    sr = 48000
    x = np.random.default_rng(0).standard_normal(sr).astype(np.float32) * 0.1
    whole = FrameAssembler(sr).push(x)
    pieces = []
    asm = FrameAssembler(sr)
    i = 0
    for size in [17, 960, 1500, 333, 4096] * 50:
        pieces += asm.push(x[i:i + size])
        i += size
        if i >= len(x):
            break
    assert len(whole) == len(pieces) == 50
    for a, b in zip(whole, pieces):
        np.testing.assert_array_equal(a.pcm_src, b.pcm_src)
        np.testing.assert_allclose(a.pcm, b.pcm, atol=1e-6)


def test_bad_sample_rate_is_rejected():
    with pytest.raises(ValueError):
        Resampler(11025)  # 11025 / 50 is not a whole number


def test_odd_file_rates_are_converted(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(p, np.zeros(11025, np.float32), 11025)
    data, sr = read_audio_file(str(p))
    assert sr == 48000 and abs(len(data) - 48000) <= 2
    assert len(list(iter_file_frames(str(p)))) == 50


def test_energy_gate_with_hangover():
    gate = EnergyGate(threshold_db=-40, hangover_ms=100)
    loud = np.full(MODEL_FRAME, 0.1, np.float32)  # -20 dB
    quiet = np.full(MODEL_FRAME, 0.001, np.float32)  # -60 dB
    assert level_db(loud) == pytest.approx(-20, abs=0.1)
    assert gate(quiet) is False
    assert gate(loud) is True
    assert [gate(quiet) for _ in range(7)] == [True] * 5 + [False] * 2
