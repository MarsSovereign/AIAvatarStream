"""MuseTalk 1.5 inference: Whisper audio features + UNet in the SD VAE latent space.

Follows TMElyralab/MuseTalk (MIT): one UNet step (timestep 0) predicts the
latent of the lower half of the face from the masked face latent, the
reference face latent and 50x384 Whisper features per video frame.

Expected files under `model_dir` (python -m avatarstream download --musetalk):

    musetalkV15/unet.pth, musetalkV15/musetalk.json
    sd-vae/            stabilityai/sd-vae-ft-mse
    whisper/           openai/whisper-tiny
    face-parse-bisent/79999_iter.pth
"""
from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np
import torch

from ..model import pick_device

log = logging.getLogger(__name__)

FACE = 256
WHISPER_SR = 16000
FEATURES_PER_SECOND = 50  # Whisper encoder output rate: one feature per 20 ms
LAYERS = 5  # embedding output + 4 encoder layers of whisper-tiny
WIDTH = 384


def missing_files(model_dir: str) -> list[str]:
    need = [
        "musetalkV15/unet.pth", "musetalkV15/musetalk.json",
        "sd-vae/config.json", "whisper/config.json", "whisper/preprocessor_config.json",
        "face-parse-bisent/79999_iter.pth",
    ]
    out = [p for p in need if not os.path.exists(os.path.join(model_dir, p))]
    if not any(os.path.exists(os.path.join(model_dir, "sd-vae", f))
               for f in ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin")):
        out.append("sd-vae/diffusion_pytorch_model.safetensors")
    if not any(os.path.exists(os.path.join(model_dir, "whisper", f)) for f in ("model.safetensors", "pytorch_model.bin")):
        out.append("whisper/model.safetensors")
    return out


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int = WIDTH, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :].to(x.dtype)


class MuseTalkModel:
    def __init__(self, model_dir: str = "models/musetalk", device: str | None = None, fp16: bool = True):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import WhisperFeatureExtractor, WhisperModel

        missing = missing_files(model_dir)
        if missing:
            raise FileNotFoundError(f"MuseTalk 模型不全，缺少：{', '.join(missing)}。"
                                    "运行 python -m avatarstream download --musetalk")
        self.device = pick_device(device)
        self.dtype = torch.float16 if fp16 and self.device.startswith("cuda") else torch.float32
        self.model_dir = model_dir

        with open(os.path.join(model_dir, "musetalkV15", "musetalk.json"), encoding="utf-8") as f:
            unet = UNet2DConditionModel(**json.load(f))
        weights = torch.load(os.path.join(model_dir, "musetalkV15", "unet.pth"), map_location="cpu", weights_only=True)
        unet.load_state_dict(weights)
        del weights
        self.unet = unet.to(self.device, self.dtype).eval()
        self.vae = AutoencoderKL.from_pretrained(os.path.join(model_dir, "sd-vae")).to(self.device, self.dtype).eval()
        self.scaling = self.vae.config.scaling_factor
        whisper_dir = os.path.join(model_dir, "whisper")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_dir)
        self.whisper = WhisperModel.from_pretrained(whisper_dir).encoder.to(self.device, self.dtype).eval()
        self.pe = PositionalEncoding().to(self.device)
        self.timesteps = torch.tensor([0], device=self.device)
        log.info("MuseTalk 已加载：设备 %s，%s", self.device, "FP16" if self.dtype == torch.float16 else "FP32")

    # --- audio ------------------------------------------------------------------
    @torch.no_grad()
    def audio_features(self, pcm16k: np.ndarray) -> np.ndarray:
        """Whisper hidden states for up to 30 s of 16 kHz audio: (n, 5, 384), one row per 20 ms."""
        n = int(math.floor(len(pcm16k) / WHISPER_SR * FEATURES_PER_SECOND))
        if len(pcm16k) > 30 * WHISPER_SR:
            raise ValueError("一次最多处理 30 秒音频")
        mel = self.feature_extractor(pcm16k, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
        hidden = self.whisper(mel.to(self.device, self.dtype), output_hidden_states=True).hidden_states
        feats = torch.stack(hidden, dim=2)[0, :n]  # (n, 5, 384)
        return feats.float().cpu().numpy()

    # --- faces ------------------------------------------------------------------
    @torch.no_grad()
    def encode_faces(self, faces_bgr: np.ndarray) -> torch.Tensor:
        """256x256 face crops -> UNet input latents (N, 8, 32, 32): masked face latent + reference latent."""
        rgb = np.ascontiguousarray(faces_bgr[..., ::-1]).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(self.device, self.dtype)
        masked = x.clone()
        masked[:, :, FACE // 2:, :] = 0  # the model redraws the lower half
        x, masked = x * 2 - 1, masked * 2 - 1
        # MuseTalk samples from the posterior; its variance is negligible, so the mean keeps prep deterministic.
        enc = lambda t: self.vae.encode(t).latent_dist.mode() * self.scaling  # noqa: E731
        return torch.cat([enc(masked), enc(x)], dim=1).float().cpu()

    @torch.no_grad()
    def generate(self, latents: torch.Tensor, features: np.ndarray) -> np.ndarray:
        """latents (B, 8, 32, 32), features (B, 50, 384) -> faces (B, 256, 256, 3) BGR uint8."""
        audio = self.pe(torch.from_numpy(np.asarray(features, np.float32)).to(self.device, self.dtype))
        pred = self.unet(latents.to(self.device, self.dtype), self.timesteps, encoder_hidden_states=audio).sample
        image = self.vae.decode(pred / self.scaling).sample
        image = ((image.float() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        return image.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]

    def warm_up(self, batch_size: int) -> float:
        latents = torch.zeros(batch_size, 8, 32, 32)
        feats = np.zeros((batch_size, 50, WIDTH), np.float32)
        self.audio_features(np.zeros(WHISPER_SR, np.float32))
        self.generate(latents, feats)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        self.audio_features(np.zeros(WHISPER_SR, np.float32))
        self.generate(latents, feats)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter() - t
