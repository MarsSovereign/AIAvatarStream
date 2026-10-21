"""Sends frames to a virtual camera (OBS Virtual Camera on Windows), via pyvirtualcam."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class VirtualCamSink:
    def __init__(self, width: int, height: int, fps: int = 25, device: str | None = None):
        try:
            import pyvirtualcam
        except ImportError as e:
            raise RuntimeError("需要先 pip install pyvirtualcam，并装好 OBS（自带虚拟摄像头）") from e
        self._cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, device=device)
        log.info("虚拟摄像头：%s，%dx%d", self._cam.device, width, height)

    def send(self, frame) -> None:
        self._cam.send(frame.image)

    def close(self) -> None:
        self._cam.close()
