"""The live pipeline end to end, in real time, with a stand-in model."""
import json
import queue
import socket
import time
import urllib.request

import cv2
import numpy as np
import soundfile as sf

from avatarstream.audio_io import FrameAssembler, FileSource
from avatarstream.engine import EngineThread, LipSyncEngine
from avatarstream.player import Player, Stats
from avatarstream.sinks import MJPEGSink

from conftest import speech_like

B = 4
RIGHT = 10


class Collect:
    def __init__(self):
        self.frames = []

    def send(self, frame):
        self.frames.append(frame)

    def close(self):
        pass


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_realtime_pipeline_keeps_pace_and_audio_in_step(tmp_path, fake_model, synthetic_avatar):
    sr = 48000
    audio = speech_like(2.0, sr, 0.3, 1.6)
    path = tmp_path / "in.wav"
    sf.write(path, audio, sr, subtype="FLOAT")

    source = FileSource(str(path))
    engine = LipSyncEngine(fake_model, synthetic_avatar, sr, batch_size=B)
    frames = queue.Queue(maxsize=64)
    stats = Stats()
    sink = Collect()
    port = _free_port()
    mjpeg = MJPEGSink("127.0.0.1", port, stats=stats.snapshot)
    player = Player([sink, mjpeg], None, stats)
    worker = EngineThread(engine, source, frames)
    engine.warm_up()

    mjpeg.start()
    player.start(frames)
    worker.start()
    source.start()
    try:
        time.sleep(3.2)
        snap = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=5).read())
        jpg = urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot.jpg", timeout=5).read()
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5).read().decode()
        stream = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream.mjpg", timeout=5)
        head = stream.read(200)
        stream.close()
    finally:
        source.stop()
        worker.stop()
        player.stop()
        mjpeg.close()

    # Steady 25 fps out.
    assert 23.5 <= snap["fps"] <= 26.5
    assert 0.25 < snap["latency_ms"] / 1000 < 0.8
    assert snap["obs_sync_offset_ms"] == snap["latency_ms"]
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == synthetic_avatar.frames[0].shape
    assert "同步偏移" in page
    assert b"--frame" in head and b"image/jpeg" in head

    # Every sample of the input comes out, in order, after the R-frame warm-up.
    out_audio = np.concatenate([f.audio_src() for f in sink.frames])
    n_pad = RIGHT * sr // 50
    assert np.all(out_audio[:n_pad] == 0)
    expected = np.concatenate([f.pcm_src for f in FrameAssembler(sr).push(audio)])
    got = out_audio[n_pad:n_pad + len(expected)]
    assert len(got) == len(expected)
    np.testing.assert_array_equal(got, expected)

    # Mouth moves while talking, idle loop afterwards.
    speaking = [f.speaking for f in sink.frames]
    assert any(speaking) and not speaking[-1]
    assert fake_model.calls > 0
