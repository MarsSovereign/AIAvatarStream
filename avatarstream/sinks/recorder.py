"""Writes frames and their audio to an .mp4 file.

Frames are piped straight into ffmpeg (H.264). OpenCV's own video writer is
only a fallback when ffmpeg is missing, since its codecs differ between builds.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

log = logging.getLogger(__name__)


def find_ffmpeg() -> str | None:
    """ffmpeg on PATH, or the copy bundled with the imageio-ffmpeg package (Windows has none by default)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class RecorderSink:
    def __init__(self, path: str, width: int, height: int, audio_sr: int, fps: int = 25):
        self.path = path
        self.audio_sr = audio_sr
        self.size = (width, height)
        self._tmpdir = tempfile.mkdtemp(prefix="avatarstream-")
        self._video_path = os.path.join(self._tmpdir, "video.mp4")
        self._audio: list[np.ndarray] = []
        self.frames = 0
        self._ffmpeg = find_ffmpeg()
        self._proc = None
        self._writer = None
        if self._ffmpeg:
            self._proc = subprocess.Popen(
                [self._ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "20", "-pix_fmt", "yuv420p", self._video_path],
                stdin=subprocess.PIPE,
            )
        else:
            log.warning("没找到 ffmpeg：改用 OpenCV 编码，视频和音频分开保存")
            self._writer = cv2.VideoWriter(self._video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError("无法创建视频文件")

    def send(self, frame) -> None:
        image = frame.image
        if image.shape[1::-1] != self.size:
            image = cv2.resize(image, self.size)
        if self._proc is not None:
            self._proc.stdin.write(np.ascontiguousarray(image).tobytes())
        else:
            self._writer.write(image)
        self._audio.append(frame.audio_src())
        self.frames += 1

    def close(self) -> None:
        import soundfile as sf

        wav_path = os.path.join(self._tmpdir, "audio.wav")
        audio = np.concatenate(self._audio) if self._audio else np.zeros(0, np.float32)
        sf.write(wav_path, audio, self.audio_sr)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if self._proc is not None:
            self._proc.stdin.close()
            if self._proc.wait() != 0:
                raise RuntimeError("ffmpeg 编码失败")
            subprocess.run([self._ffmpeg, "-y", "-loglevel", "error", "-i", self._video_path, "-i", wav_path,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", self.path], check=True)
        else:
            self._writer.release()
            shutil.copy(self._video_path, self.path)
            shutil.copy(wav_path, os.path.splitext(self.path)[0] + ".wav")
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        log.info("已保存：%s（%d 帧）", self.path, self.frames)
