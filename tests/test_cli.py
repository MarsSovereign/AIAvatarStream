"""The `live` command end to end with a stand-in model, plus avatar creation."""
import json
import os
import socket
import subprocess

import cv2
import numpy as np
import pytest
import soundfile as sf

import avatarstream.model
from avatarstream import cli
from avatarstream.avatar import load_avatar
from avatarstream.make_avatar import make_avatar

from conftest import AVATAR_PATH, FakeModel, make_synthetic_avatar, needs_model, speech_like


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _save_avatar(avatar, folder):
    os.makedirs(folder / "full_imgs")
    os.makedirs(folder / "face_imgs")
    for i, (f, face) in enumerate(zip(avatar.frames, avatar.faces)):
        cv2.imwrite(str(folder / "full_imgs" / f"{i:08d}.png"), f)
        cv2.imwrite(str(folder / "face_imgs" / f"{i:08d}.png"), face)
    (folder / "coords.json").write_text(json.dumps(avatar.coords))


def _probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def test_avatar_round_trip(tmp_path):
    av = make_synthetic_avatar()
    _save_avatar(av, tmp_path / "av")
    loaded = load_avatar(str(tmp_path / "av"))
    assert len(loaded) == len(av)
    assert loaded.coords == [tuple(c) for c in av.coords]
    np.testing.assert_array_equal(loaded.frames[3], av.frames[3])


def test_live_command_with_file_input_and_recording(tmp_path, monkeypatch):
    monkeypatch.setattr(avatarstream.model, "LipSyncModel", lambda *a, **k: FakeModel())
    _save_avatar(make_synthetic_avatar(), tmp_path / "av")
    model = tmp_path / "fake.pth"
    model.write_bytes(b"")
    wav = tmp_path / "in.wav"
    sf.write(wav, speech_like(2.0, 44100, 0.2, 1.5), 44100)
    out = tmp_path / "live.mp4"

    cli.main(["live", "--avatar", str(tmp_path / "av"), "--model", str(model), "--input-file", str(wav),
              "--batch-size", "4", "--port", str(_free_port()), "--duration", "3", "--record", str(out)])

    streams = _probe(out)
    assert 2.4 < streams["video"] < 3.2
    assert abs(streams["audio"] - streams["video"]) < 0.1


@needs_model
def test_make_avatar_finds_the_same_face(tmp_path):
    ref = load_avatar(AVATAR_PATH, max_frames=40)
    video = tmp_path / "idle.mp4"
    h, w = ref.frames[0].shape[:2]
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    for f in ref.frames:
        vw.write(f)
    vw.release()

    def iou(a, b):
        ay1, ay2, ax1, ax2 = a
        by1, by2, bx1, bx2 = b
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        return inter / ((ay2 - ay1) * (ax2 - ax1) + (by2 - by1) * (bx2 - bx1) - inter)

    for detector, min_iou in (("haar", 0.6), ("s3fd", 0.8)):
        out = tmp_path / detector
        try:
            n = make_avatar(str(video), str(out), detector=detector, device="cpu")
        except Exception as e:  # s3fd weights come from the internet on first use
            if detector == "s3fd" and "urlopen" in repr(e).lower():
                pytest.skip("S3FD 权重下载失败")
            raise
        made = load_avatar(str(out))
        assert n == len(made) == len(ref)
        scores = [iou(a, b) for a, b in zip(made.coords, ref.coords)]
        assert np.mean(scores) > min_iou, (detector, np.mean(scores))
