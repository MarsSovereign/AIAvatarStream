"""Wav2Lip model loading and batched inference."""
from __future__ import annotations

import logging
import time

import numpy as np
import torch

from .wav2lip.models import Wav2Lip

log = logging.getLogger(__name__)


def pick_device(prefer: str | None = None) -> str:
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_checkpoint(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older checkpoints may hold objects weights_only refuses
        log.warning("以兼容模式加载模型文件。请只使用可信来源的模型。")
        return torch.load(path, map_location="cpu", weights_only=False)


class LipSyncModel:
    def __init__(self, path: str, device: str | None = None, fp16: bool = False):
        self.device = pick_device(device)
        self.fp16 = fp16 and self.device.startswith("cuda")
        ckpt = _load_checkpoint(path)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net = Wav2Lip()
        net.load_state_dict(state)
        net = net.to(self.device).eval()
        if self.fp16:
            net = net.half()
        self.net = net
        self.dtype = torch.float16 if self.fp16 else torch.float32
        log.info("模型已加载：%s，设备 %s%s", path, self.device, "（FP16）" if self.fp16 else "")

    @torch.no_grad()
    def infer(self, faces: np.ndarray, mels: np.ndarray) -> np.ndarray:
        """faces: (B, 256, 256, 3) uint8 BGR; mels: (B, 80, 16). Returns (B, 256, 256, 3) uint8."""
        faces = np.asarray(faces)
        masked = faces.copy()
        masked[:, faces.shape[1] // 2:] = 0
        img = np.concatenate((masked, faces), axis=3).astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(0, 3, 1, 2))).to(self.device, self.dtype)
        mel_t = torch.from_numpy(np.asarray(mels, np.float32)[:, None]).to(self.device, self.dtype)
        pred = self.net(mel_t, img_t)
        pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        return np.clip(pred, 0, 255).astype(np.uint8)

    def warm_up(self, batch_size: int) -> float:
        faces = np.zeros((batch_size, 256, 256, 3), np.uint8)
        mels = np.zeros((batch_size, 80, 16), np.float32)
        self.infer(faces, mels)
        t = time.perf_counter()
        self.infer(faces, mels)
        return time.perf_counter() - t
