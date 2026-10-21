"""Face parsing (BiSeNet) used to build MuseTalk's blending masks.

model.py and resnet.py come from MuseTalk (MIT, Copyright (c) 2024 Tencent Music
Entertainment Group), which took them from zllrunning/face-parsing.PyTorch (MIT).
The "jaw" mask logic below follows MuseTalk 1.5's FaceParsing.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import BiSeNet

# BiSeNet labels: 1 skin, 10 nose, 11 mouth, 12 upper lip, 13 lower lip
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _jaw_kernel() -> np.ndarray:
    cone_height, tail_height = 21, 12
    total = cone_height + tail_height
    kernel = np.zeros((total, total), np.uint8)
    center = total // 2
    for row in range(cone_height // 2, cone_height):
        width = 2 * (row - cone_height // 2) + 1
        kernel[row, center - width // 2: center + width // 2 + 1] = 1
    base = int(kernel[cone_height - 1].sum())
    for row in range(cone_height, total):
        kernel[row, max(0, center - base // 2): min(total, center + base // 2 + 1)] = 1
    return kernel


class FaceParsing:
    def __init__(self, weights: str, device: str = "cpu", left_cheek_width: int = 90, right_cheek_width: int = 90):
        self.device = device
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.net = net.to(device).eval()
        self.kernel = _jaw_kernel()
        self.cheek_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 3))
        self.cheek_mask = np.zeros((512, 512), np.uint8)
        cv2.rectangle(self.cheek_mask, (0, 0), (256 - left_cheek_width, 512), 255, -1)
        cv2.rectangle(self.cheek_mask, (256 + right_cheek_width, 0), (512, 512), 255, -1)

    @torch.no_grad()
    def __call__(self, image_bgr: np.ndarray, mode: str = "jaw") -> np.ndarray:
        """Returns a 512x512 uint8 mask (255 = region the generated face may replace)."""
        rgb = cv2.cvtColor(cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device)
        parsing = self.net(x)[0].squeeze(0).argmax(0).cpu().numpy()
        if mode == "jaw":
            face = (parsing == 1).astype(np.uint8) * 255
            dilated = cv2.dilate(face, self.kernel, iterations=1)
            eroded = cv2.erode(dilated, self.cheek_kernel, iterations=2)
            face = cv2.bitwise_or(cv2.bitwise_and(eroded, self.cheek_mask),
                                  cv2.bitwise_and(dilated, cv2.bitwise_not(self.cheek_mask)))
            keep = ((face == 255) & (parsing != 10)) | np.isin(parsing, [11, 12, 13])
        elif mode == "neck":
            keep = np.isin(parsing, [1, 11, 12, 13, 14])
        else:
            keep = np.isin(parsing, [1, 11, 12, 13])
        return keep.astype(np.uint8) * 255
