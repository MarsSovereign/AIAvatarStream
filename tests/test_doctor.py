"""The diagnostics report and the ffmpeg fallback."""
import shutil

import numpy as np

from avatarstream import cli
from avatarstream.sinks import recorder


def test_doctor_writes_a_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cli.main(["doctor"])
    out = capsys.readouterr().out
    for section in ("[系统]", "[依赖]", "[显卡]", "[录像]", "[模型]", "[形象]", "[音频]", "[配置]"):
        assert section in out
    report = (tmp_path / "output" / "diagnostics.txt").read_text(encoding="utf-8")
    assert "AIAvatarStream" in report and "Wav2Lip 模型" in report


def test_recording_uses_bundled_ffmpeg_when_none_on_path(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    path = recorder.find_ffmpeg()
    assert path and "imageio_ffmpeg" in path

    class Frame:
        image = np.zeros((64, 48, 3), np.uint8)

        def audio_src(self):
            return np.zeros(1920, np.float32)

    rec = recorder.RecorderSink(str(tmp_path / "o.mp4"), 48, 64, 48000)
    for _ in range(10):
        rec.send(Frame())
    rec.close()
    assert (tmp_path / "o.mp4").stat().st_size > 1000
    assert not (tmp_path / "o.wav").exists()  # audio was muxed in, not left beside the video
