"""Avatar loading.

An avatar directory holds the idle video as frames plus a face crop per frame:

    full_imgs/00000000.png ...   full frames
    face_imgs/00000000.png ...   face crops, 256x256
    coords.json (or coords.pkl)  face box (y1, y2, x1, x2) per frame

This is the same layout LiveTalking and AIAvatar use, so their avatars work as is.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

FACE_SIZE = 256


@dataclass
class Avatar:
    frames: list[np.ndarray]
    faces: list[np.ndarray]
    coords: list[tuple[int, int, int, int]]

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h

    def mirror_index(self, i: int) -> int:
        """Plays the idle loop forward, then backward, so there is no jump at the seam."""
        n = len(self.frames)
        turn, res = divmod(i, n)
        return res if turn % 2 == 0 else n - res - 1


def _sorted_images(folder: str) -> list[str]:
    files = glob.glob(os.path.join(folder, "*.[jpJP][pnPN]*[gG]"))
    return sorted(files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def _read_all(paths: list[str]) -> list[np.ndarray]:
    with ThreadPoolExecutor(max_workers=8) as ex:
        imgs = list(ex.map(cv2.imread, paths))
    missing = [p for p, im in zip(paths, imgs) if im is None]
    if missing:
        raise IOError(f"无法读取图片：{missing[0]} 等 {len(missing)} 张")
    return imgs


def load_coords(path: str) -> list[tuple[int, int, int, int]]:
    json_path = os.path.join(path, "coords.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return [tuple(int(v) for v in c) for c in json.load(f)]
    pkl_path = os.path.join(path, "coords.pkl")
    if os.path.exists(pkl_path):
        # pickle can run code when loaded: only use avatars from sources you trust.
        with open(pkl_path, "rb") as f:
            return [tuple(int(v) for v in c) for c in pickle.load(f)]
    raise FileNotFoundError(f"{path} 下没有 coords.json 或 coords.pkl")


def load_avatar(path: str, max_frames: int | None = None) -> Avatar:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"形象目录不存在：{path}")
    full = _sorted_images(os.path.join(path, "full_imgs"))
    face = _sorted_images(os.path.join(path, "face_imgs"))
    coords = load_coords(path)
    n = min(len(full), len(face), len(coords))
    if n == 0:
        raise ValueError(f"{path} 里没有可用的帧")
    if not (len(full) == len(face) == len(coords)):
        log.warning("帧数不一致（完整帧 %d，人脸 %d，坐标 %d），按 %d 帧使用", len(full), len(face), len(coords), n)
    if max_frames:
        n = min(n, max_frames)
    frames, faces = _read_all(full[:n]), _read_all(face[:n])
    for i, f in enumerate(faces):
        if f.shape[:2] != (FACE_SIZE, FACE_SIZE):
            faces[i] = cv2.resize(f, (FACE_SIZE, FACE_SIZE))
    log.info("已加载形象 %s：%d 帧，%dx%d", os.path.basename(path.rstrip("/\\")), n, *frames[0].shape[1::-1])
    return Avatar(frames, faces, list(coords[:n]))
