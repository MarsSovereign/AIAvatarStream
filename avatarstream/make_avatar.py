"""Builds an avatar directory from a video of a person with their mouth closed.

Face detection uses S3FD (the detector Wav2Lip was trained with) by default.
`--detector haar` uses OpenCV's built-in detector, which needs no download but
is less accurate.
"""
from __future__ import annotations

import json
import logging
import os
import pickle

import cv2
import numpy as np

from .avatar import FACE_SIZE

log = logging.getLogger(__name__)


def read_video(path: str, max_seconds: float | None = None, resize_height: int | None = None) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"打不开视频：{path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    limit = int(max_seconds * fps) if max_seconds else None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or (limit is not None and len(frames) >= limit):
            break
        if resize_height and frame.shape[0] > resize_height:
            scale = resize_height / frame.shape[0]
            frame = cv2.resize(frame, (int(round(frame.shape[1] * scale / 2) * 2), resize_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("视频里没有读到任何帧")
    return frames, fps


def _detect_s3fd(frames: list[np.ndarray], batch_size: int, device: str) -> list:
    from .wav2lip import face_detection

    det = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    rects = []
    for i in range(0, len(frames), batch_size):
        rects.extend(det.get_detections_for_batch(np.array(frames[i:i + batch_size])))
    return [None if r is None else tuple(int(v) for v in r[:4]) for r in rects]


def _detect_haar(frames: list[np.ndarray]) -> list:
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("当前 OpenCV 版本没有 Haar 检测器，请安装 opencv-python<5，或改用 --detector s3fd")
    cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    rects = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
        if len(faces) == 0:
            rects.append(None)
            continue
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        # Haar boxes stop around the upper lip; extend down so the chin is inside, like S3FD.
        rects.append((int(x), int(y), int(x + w), int(y + h * 1.12)))
    return rects


def _fill_missing(rects: list) -> list:
    found = [i for i, r in enumerate(rects) if r is not None]
    if not found:
        raise ValueError("整段视频都没检测到人脸。请换一段正脸、光线充足的视频。")
    missing = len(rects) - len(found)
    if missing > 0.2 * len(rects):
        raise ValueError(f"{missing}/{len(rects)} 帧没检测到人脸，太多了。请换一段正脸更清晰的视频。")
    if missing:
        log.warning("%d 帧没检测到人脸，用相邻帧补上", missing)
    arr = np.array([rects[i] for i in found], dtype=np.float64)
    idx = np.arange(len(rects))
    filled = np.stack([np.interp(idx, found, arr[:, k]) for k in range(4)], axis=1)
    return [tuple(r) for r in filled]


def _smooth(boxes: np.ndarray, window: int = 5) -> np.ndarray:
    out = boxes.copy()
    for i in range(len(boxes)):
        lo = min(i, max(0, len(boxes) - window))
        out[i] = boxes[lo: lo + window].mean(axis=0)
    return out


def make_avatar(video: str, out_dir: str, detector: str = "s3fd", pads=(0, 10, 0, 0),
                max_seconds: float | None = 30, resize_height: int | None = None, smooth: bool = True,
                device: str | None = None, batch_size: int = 16) -> int:
    frames, fps = read_video(video, max_seconds, resize_height)
    log.info("读取 %d 帧（%.1f 秒，%dx%d）", len(frames), len(frames) / fps, frames[0].shape[1], frames[0].shape[0])
    if detector == "s3fd":
        import torch

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        rects = _detect_s3fd(frames, batch_size, device)
    elif detector == "haar":
        rects = _detect_haar(frames)
    else:
        raise ValueError("detector 只能是 s3fd 或 haar")

    boxes = np.array(_fill_missing(rects))
    if smooth:
        boxes = _smooth(boxes)
    pad_top, pad_bottom, pad_left, pad_right = pads
    h, w = frames[0].shape[:2]
    coords = []
    for x1, y1, x2, y2 in boxes:
        y1 = int(max(0, y1 - pad_top))
        y2 = int(min(h, y2 + pad_bottom))
        x1 = int(max(0, x1 - pad_left))
        x2 = int(min(w, x2 + pad_right))
        coords.append((y1, y2, x1, x2))

    full_dir, face_dir = os.path.join(out_dir, "full_imgs"), os.path.join(out_dir, "face_imgs")
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)
    for i, (frame, (y1, y2, x1, x2)) in enumerate(zip(frames, coords)):
        cv2.imwrite(os.path.join(full_dir, f"{i:08d}.png"), frame)
        cv2.imwrite(os.path.join(face_dir, f"{i:08d}.png"), cv2.resize(frame[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE)))
    with open(os.path.join(out_dir, "coords.json"), "w", encoding="utf-8") as f:
        json.dump(coords, f)
    with open(os.path.join(out_dir, "coords.pkl"), "wb") as f:  # for LiveTalking / AIAvatar
        pickle.dump(coords, f)
    log.info("形象已生成：%s（%d 帧）", out_dir, len(frames))
    return len(frames)
