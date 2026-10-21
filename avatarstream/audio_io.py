"""Audio input and output for the live pipeline.

The lip-sync model reads 16 kHz mono PCM in 20 ms frames (320 samples). Every
frame produced here also carries the same 20 ms at the input device's own
sample rate, so the player can send full-quality audio out in step with the
video instead of the 16 kHz copy the model sees.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from math import gcd
from typing import Iterator, Optional

import numpy as np

log = logging.getLogger(__name__)

MODEL_SR = 16000
FRAME_MS = 20
FRAMES_PER_SECOND = 1000 // FRAME_MS  # 50
MODEL_FRAME = MODEL_SR // FRAMES_PER_SECOND  # 320 samples


@dataclass
class AudioFrame:
    """20 ms of mono audio."""

    pcm: np.ndarray  # float32, MODEL_FRAME samples at 16 kHz, fed to the model
    pcm_src: np.ndarray  # float32, the same 20 ms at the source sample rate, for playback
    t_capture: Optional[float] = None  # time.monotonic() when it arrived; None for padding
    speech: Optional[bool] = field(default=None)  # set by the voice gate


def silent_frame(src_sr: int) -> AudioFrame:
    return AudioFrame(
        pcm=np.zeros(MODEL_FRAME, np.float32),
        pcm_src=np.zeros(src_sr // FRAMES_PER_SECOND, np.float32),
        t_capture=None,
        speech=False,
    )


def check_sample_rate(sr: int) -> None:
    if sr % FRAMES_PER_SECOND:
        raise ValueError(f"采样率 {sr} 不能被 50 整除，无法切成 20 毫秒的帧。请改用 16000、44100 或 48000。")


class Resampler:
    """Streaming resampler from the source rate to 16 kHz.

    Takes one 20 ms block at the source rate and always returns exactly
    MODEL_FRAME samples, so frame counts on both sides stay locked together.
    """

    def __init__(self, src_sr: int):
        check_sample_rate(src_sr)
        self.src_sr = src_sr
        self._stream = None
        self._up = self._down = 1
        if src_sr != MODEL_SR:
            try:
                import soxr

                self._stream = soxr.ResampleStream(src_sr, MODEL_SR, 1, dtype="float32", quality="HQ")
            except ImportError:
                g = gcd(MODEL_SR, src_sr)
                self._up, self._down = MODEL_SR // g, src_sr // g
                log.warning("未安装 soxr，改用逐块重采样，音质略差。建议 pip install soxr")
        # One frame of priming absorbs the resampler's start-up delay.
        self._buf = np.zeros(MODEL_FRAME, np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, np.float32)
        if self.src_sr == MODEL_SR:
            return block.copy()
        if self._stream is not None:
            out = self._stream.resample_chunk(block)
        else:
            from scipy.signal import resample_poly

            out = resample_poly(block, self._up, self._down).astype(np.float32)
        self._buf = np.concatenate([self._buf, out])
        if len(self._buf) < MODEL_FRAME:
            self._buf = np.concatenate([np.zeros(MODEL_FRAME - len(self._buf), np.float32), self._buf])
        elif len(self._buf) > 3 * MODEL_FRAME:
            # Never let the buffer grow into extra latency.
            self._buf = self._buf[-2 * MODEL_FRAME:]
        res, self._buf = self._buf[:MODEL_FRAME], self._buf[MODEL_FRAME:]
        return res


class FrameAssembler:
    """Cuts arbitrary-sized input blocks into AudioFrames."""

    def __init__(self, src_sr: int):
        check_sample_rate(src_sr)
        self.src_sr = src_sr
        self.src_frame = src_sr // FRAMES_PER_SECOND
        self._resampler = Resampler(src_sr)
        self._pending = np.zeros(0, np.float32)

    def push(self, samples: np.ndarray, t_capture: Optional[float] = None) -> list[AudioFrame]:
        samples = np.asarray(samples, np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        self._pending = np.concatenate([self._pending, samples])
        frames = []
        while len(self._pending) >= self.src_frame:
            block, self._pending = self._pending[: self.src_frame], self._pending[self.src_frame:]
            frames.append(AudioFrame(self._resampler.process(block), block.copy(), t_capture))
        return frames


class AudioSource:
    """Base class: something that produces AudioFrames in real time."""

    src_sr: int = MODEL_SR

    def __init__(self, max_queue_seconds: float = 2.0):
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=int(max_queue_seconds * FRAMES_PER_SECOND))
        self.dropped = 0

    def _put(self, frame: AudioFrame) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # The engine fell behind. Drop the oldest audio to keep latency bounded.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            self._queue.put_nowait(frame)

    def get(self, timeout: float = 0.1) -> Optional[AudioFrame]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass


def _resolve_device(device, kind: str):
    """Accepts a device index, an exact name or a unique substring of a name."""
    import sounddevice as sd

    if device is None or device == "":
        return None
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        return int(device)
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        i for i, d in enumerate(sd.query_devices()) if device.lower() in d["name"].lower() and d[channels_key] > 0
    ]
    if not matches:
        what = "输入" if kind == "input" else "输出"
        raise ValueError(f"找不到名字里包含“{device}”的{what}设备。运行 python -m avatarstream devices 查看列表。")
    return matches[0]


class MicSource(AudioSource):
    """Live input from a microphone or a virtual audio cable (the voice changer's output)."""

    def __init__(self, device=None, samplerate: Optional[int] = None, **kw):
        super().__init__(**kw)
        import sounddevice as sd

        self._sd = sd
        self.device = _resolve_device(device, "input")
        info = sd.query_devices(self.device, "input")
        self.src_sr = int(samplerate or info["default_samplerate"])
        self.device_name = info["name"]
        self._assembler = FrameAssembler(self.src_sr)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # runs on the audio thread
        if status:
            log.debug("input status: %s", status)
        now = time.monotonic()
        for f in self._assembler.push(indata[:, 0] if indata.shape[1] == 1 else indata, now):
            self._put(f)

    def start(self) -> None:
        self._stream = self._sd.InputStream(
            device=self.device,
            samplerate=self.src_sr,
            channels=1,
            dtype="float32",
            blocksize=self.src_sr // FRAMES_PER_SECOND,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        log.info("音频输入：%s，%d Hz", self.device_name, self.src_sr)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def read_audio_file(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr % FRAMES_PER_SECOND:
        # Rare rates (e.g. 22050) get converted to 48 kHz once up front.
        from scipy.signal import resample_poly

        g = gcd(48000, sr)
        data = resample_poly(data, 48000 // g, sr // g).astype(np.float32)
        sr = 48000
    return data, sr


def iter_file_frames(path: str) -> Iterator[AudioFrame]:
    """All frames of a file, as fast as possible (offline rendering)."""
    data, sr = read_audio_file(path)
    assembler = FrameAssembler(sr)
    n = sr // FRAMES_PER_SECOND
    if len(data) % n:
        data = np.concatenate([data, np.zeros(n - len(data) % n, np.float32)])
    yield from assembler.push(data, None)


class FileSource(AudioSource):
    """Plays a file into the pipeline in real time, as if it were a microphone.

    Useful for rehearsing without a mic and for testing. After the file ends it
    keeps sending silence (or loops), so the avatar returns to its idle loop.
    """

    def __init__(self, path: str, loop: bool = False, **kw):
        super().__init__(**kw)
        self.path = path
        self.loop = loop
        data, self.src_sr = read_audio_file(path)
        self._frames = list(FrameAssembler(self.src_sr).push(data))
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.finished = threading.Event()

    def _run(self) -> None:
        period = 1.0 / FRAMES_PER_SECOND
        t_next = time.monotonic()
        i = 0
        while not self._stop.is_set():
            if i < len(self._frames):
                f = self._frames[i]
                frame = AudioFrame(f.pcm, f.pcm_src, None)
                i += 1
            else:
                self.finished.set()
                if self.loop:
                    i = 0
                    continue
                frame = silent_frame(self.src_sr)
                frame.speech = None
            t_next += period
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            frame.t_capture = time.monotonic()
            self._put(frame)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="file-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


class AudioPlayer:
    """Sends audio to an output device (for example a second virtual cable that OBS listens to)."""

    def __init__(self, device, samplerate: int, max_buffer_seconds: float = 0.5):
        import sounddevice as sd

        self.device = _resolve_device(device, "output")
        self.samplerate = samplerate
        self._max = int(max_buffer_seconds * samplerate)
        self._buf = np.zeros(0, np.float32)
        self._lock = threading.Lock()
        self.underruns = 0
        self._stream = sd.OutputStream(
            device=self.device,
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self.device_name = sd.query_devices(self.device, "output")["name"]

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            n = min(frames, len(self._buf))
            outdata[:n, 0] = self._buf[:n]
            self._buf = self._buf[n:]
        if n < frames:
            outdata[n:, 0] = 0
            self.underruns += 1

    def write(self, samples: np.ndarray) -> None:
        with self._lock:
            self._buf = np.concatenate([self._buf, samples.astype(np.float32)])
            if len(self._buf) > self._max:
                self._buf = self._buf[-self._max:]

    def start(self) -> None:
        self._stream.start()
        log.info("音频输出：%s，%d Hz", self.device_name, self.samplerate)

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


def describe_devices() -> str:
    import sounddevice as sd

    lines = []
    for i, d in enumerate(sd.query_devices()):
        kinds = []
        if d["max_input_channels"] > 0:
            kinds.append("输入")
        if d["max_output_channels"] > 0:
            kinds.append("输出")
        lines.append(f"[{i:>2}] {'/'.join(kinds):<5} {int(d['default_samplerate']):>6} Hz  {d['name']}")
    return "\n".join(lines)
