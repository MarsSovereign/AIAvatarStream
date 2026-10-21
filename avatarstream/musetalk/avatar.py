"""MuseTalk data for an avatar, stored next to its frames in <avatar>/musetalk/.

    musetalk/latents.pt      (N, 8, 32, 32) float16 UNet input latents
    musetalk/boxes.json      face box per frame (x1, y1, x2, y2), from landmarks like MuseTalk 1.5
    musetalk/crop_boxes.json square region around the face that the mask covers
    musetalk/masks/*.png     blending mask per frame (face parsing, lower half, blurred)
    musetalk/info.json       settings used, so stale data can be detected

`prepare` builds it from full_imgs/ once; streaming only needs the UNet and the VAE decoder.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from ..avatar import Avatar, _read_all, _sorted_images
from .model import FACE

log = logging.getLogger(__name__)

SUBDIR = "musetalk"
VERSION = 1


@dataclass
class MuseTalkAvatar:
    latents: torch.Tensor
    boxes: list[tuple[int, int, int, int]]
    crop_boxes: list[tuple[int, int, int, int]]
    masks: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.boxes)


# --- face boxes -------------------------------------------------------------

def box_from_landmarks(lm: np.ndarray, height: int, bbox_shift: int = 0, extra_margin: int = 10):
    """MuseTalk 1.5's crop: landmark width; from as far above the nose (point 29) as the chin is below it, to the chin."""
    lm = np.asarray(lm, np.float64)
    half_y = lm[29, 1] + bbox_shift
    chin_y = lm[:, 1].max()
    upper = max(0.0, half_y - (chin_y - half_y))
    x1, x2 = lm[:, 0].min(), lm[:, 0].max()
    y2 = min(chin_y + extra_margin, height)
    return [x1, upper, x2, y2]


def crop_box(box, expand: float = 1.5):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    s = int(max(x2 - x1, y2 - y1) // 2 * expand)
    return [cx - s, cy - s, cx + s, cy + s]


def _crop_padded(frame: np.ndarray, box) -> np.ndarray:
    """Crop that may extend past the image; outside pixels are black (as PIL's crop does)."""
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    out = np.zeros((y2 - y1, x2 - x1, 3), frame.dtype)
    sx1, sy1, sx2, sy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1: sy2 - y1, sx1 - x1: sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def blend_mask(parsing512: np.ndarray, box, cbox, upper_boundary_ratio: float = 0.5) -> np.ndarray:
    """MuseTalk's get_image_prepare_material: parsing limited to the face box, lower half only, blurred."""
    x1, y1, x2, y2 = box
    xs, ys, xe, ye = cbox
    size = (xe - xs, ye - ys)
    parsing = cv2.resize(parsing512, size, interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((size[1], size[0]), np.uint8)
    mask[y1 - ys: y2 - ys, x1 - xs: x2 - xs] = parsing[y1 - ys: y2 - ys, x1 - xs: x2 - xs]
    mask[: int(size[1] * upper_boundary_ratio)] = 0
    k = int(0.1 * size[1] // 2 * 2) + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _smooth(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    out = boxes.copy()
    for i in range(len(boxes)):
        lo = min(i, max(0, len(boxes) - window))
        out[i] = boxes[lo: lo + window].mean(axis=0)
    return out


def detect_landmarks(frames: list[np.ndarray], device: str) -> list:
    import face_alignment

    kind = getattr(face_alignment.LandmarksType, "TWO_D", None) or face_alignment.LandmarksType._2D
    fa = face_alignment.FaceAlignment(kind, flip_input=False, device=device)
    out = []
    for f in frames:
        found = fa.get_landmarks_from_image(f[..., ::-1].copy())
        if not found:
            out.append(None)
            continue
        out.append(max(found, key=lambda lm: np.ptp(lm[:, 0]) * np.ptp(lm[:, 1])))
    return out


def prepare(avatar_dir: str, model, parsing_weights: str, device: str | None = None, bbox_shift: int = 0,
            extra_margin: int = 10, smooth: bool = True, landmarks=None) -> int:
    """Adds MuseTalk data to an existing avatar directory. `landmarks` may be given to skip detection."""
    from .face_parsing import FaceParsing

    device = device or model.device
    frames = _read_all(_sorted_images(os.path.join(avatar_dir, "full_imgs")))
    h, w = frames[0].shape[:2]
    landmarks = detect_landmarks(frames, device) if landmarks is None else landmarks
    found = [i for i, lm in enumerate(landmarks) if lm is not None]
    if len(found) < 0.8 * len(frames):
        raise ValueError(f"{len(frames) - len(found)}/{len(frames)} 帧没检测到人脸关键点，请换一段正脸更清晰的视频")
    raw = np.array([box_from_landmarks(landmarks[i], h, bbox_shift, extra_margin) for i in found])
    idx = np.arange(len(frames))
    boxes = np.stack([np.interp(idx, found, raw[:, k]) for k in range(4)], axis=1)
    if smooth:
        boxes = _smooth(boxes)
    boxes = [(int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))) for x1, y1, x2, y2 in boxes]

    faces = np.stack([cv2.resize(f[y1:y2, x1:x2], (FACE, FACE), interpolation=cv2.INTER_LANCZOS4)
                      for f, (x1, y1, x2, y2) in zip(frames, boxes)])
    latents = torch.cat([model.encode_faces(faces[i:i + 16]) for i in range(0, len(faces), 16)])

    parser = FaceParsing(parsing_weights, device)
    out_dir = os.path.join(avatar_dir, SUBDIR)
    os.makedirs(os.path.join(out_dir, "masks"), exist_ok=True)
    crop_boxes = []
    for i, (f, box) in enumerate(zip(frames, boxes)):
        cbox = crop_box(box)
        mask = blend_mask(parser(_crop_padded(f, cbox)), box, cbox)
        cv2.imwrite(os.path.join(out_dir, "masks", f"{i:08d}.png"), mask)
        crop_boxes.append(cbox)

    torch.save(latents.half(), os.path.join(out_dir, "latents.pt"))
    for name, data in (("boxes.json", boxes), ("crop_boxes.json", crop_boxes)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump([list(map(int, b)) for b in data], fh)
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": VERSION, "frames": len(frames), "bbox_shift": bbox_shift,
                   "extra_margin": extra_margin, "size": [w, h]}, fh)
    log.info("MuseTalk 形象数据已生成：%s（%d 帧）", out_dir, len(frames))
    return len(frames)


def has_musetalk_data(avatar_dir: str) -> bool:
    return os.path.exists(os.path.join(avatar_dir, SUBDIR, "info.json"))


def load(avatar_dir: str, avatar: Avatar) -> MuseTalkAvatar:
    d = os.path.join(avatar_dir, SUBDIR)
    if not has_musetalk_data(avatar_dir):
        raise FileNotFoundError(f"{avatar_dir} 还没有 MuseTalk 数据。"
                                f"先运行：python -m avatarstream prepare-musetalk --avatar {avatar_dir}")
    with open(os.path.join(d, "info.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    if info.get("frames") != len(avatar) or info.get("size") != list(avatar.size):
        raise ValueError("MuseTalk 数据和形象的帧不匹配（形象改过？）。请重新运行 prepare-musetalk。")
    latents = torch.load(os.path.join(d, "latents.pt"), map_location="cpu", weights_only=True)
    with open(os.path.join(d, "boxes.json"), encoding="utf-8") as fh:
        boxes = [tuple(b) for b in json.load(fh)]
    with open(os.path.join(d, "crop_boxes.json"), encoding="utf-8") as fh:
        crop_boxes = [tuple(b) for b in json.load(fh)]
    masks = [m[..., 0] if m.ndim == 3 else m for m in _read_all(_sorted_images(os.path.join(d, "masks")))]
    return MuseTalkAvatar(latents, boxes, crop_boxes, masks)


def paste(frame: np.ndarray, face: np.ndarray, box, mask: np.ndarray, cbox) -> np.ndarray:
    """Pastes a generated 256x256 face into the frame through the precomputed mask."""
    x1, y1, x2, y2 = box
    xs, ys = cbox[0], cbox[1]
    resized = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    m = mask[y1 - ys: y2 - ys, x1 - xs: x2 - xs, None].astype(np.float32) / 255.0
    out = frame.copy()
    roi = out[y1:y2, x1:x2].astype(np.float32)
    out[y1:y2, x1:x2] = (resized.astype(np.float32) * m + roi * (1 - m) + 0.5).astype(np.uint8)
    return out
