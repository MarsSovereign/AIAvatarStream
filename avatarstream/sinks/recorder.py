"""Writes frames and their audio to an .mp4 file."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

log = logging.getLogger(__name__)


class RecorderSink:
    def __init__(self, path: str, width: int, height: int, audio_sr: int, fps: int = 25):
        self.path = path
        self.audio_sr = audio_sr
        self._tmpdir = tempfile.mkdtemp(prefix="avatarstream-")
        self._video_path = os.path.join(self._tmpdir, "video.mp4")
        self._writer = cv2.VideoWriter(self._video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError("无法创建视频文件")
        self._audio: list[np.ndarray] = []
        self.frames = 0

    def send(self, frame) -> None:
        self._writer.write(frame.image)
        self._audio.append(frame.audio_src())
        self.frames += 1

    def close(self) -> None:
        import soundfile as sf

        self._writer.release()
        wav_path = os.path.join(self._tmpdir, "audio.wav")
        audio = np.concatenate(self._audio) if self._audio else np.zeros(0, np.float32)
        sf.write(wav_path, audio, self.audio_sr)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", self._video_path, "-i", wav_path,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
                   "-c:a", "aac", "-b:a", "160k", "-shortest", self.path]
            subprocess.run(cmd, check=True)
        else:
            log.warning("没找到 ffmpeg：视频和音频分开保存")
            shutil.copy(self._video_path, self.path)
            shutil.copy(wav_path, os.path.splitext(self.path)[0] + ".wav")
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        log.info("已保存：%s（%d 帧）", self.path, self.frames)
