"""MuseTalk 1.5 backend (https://github.com/TMElyralab/MuseTalk, MIT)."""
from __future__ import annotations

import numpy as np

from ..avatar import Avatar
from .avatar import MuseTalkAvatar, paste
from .features import WhisperFeaturizer


class MuseTalkBackend:
    name = "musetalk"

    def __init__(self, model, avatar: Avatar, data: MuseTalkAvatar, src_sr: int, batch_size: int):
        if len(data) != len(avatar):
            raise ValueError("MuseTalk 数据和形象的帧数不一致，请重新运行 prepare-musetalk")
        self.model = model
        self.avatar = avatar
        self.data = data
        self.batch_size = batch_size
        self.featurizer = WhisperFeaturizer(model, batch_size, src_sr)
        self._latents = data.latents.to(model.device, model.dtype)

    def warm_up(self) -> float:
        self.featurizer.warm_up()
        return self.model.warm_up(self.batch_size)

    def render(self, idxs: list[int], feats: list) -> list[np.ndarray]:
        faces = self.model.generate(self._latents[idxs], np.stack(feats))
        d = self.data
        return [paste(self.avatar.frames[i], f, d.boxes[i], d.masks[i], d.crop_boxes[i]) for i, f in zip(idxs, faces)]
