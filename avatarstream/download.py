"""Downloads the Wav2Lip model and the sample avatars published with AIAvatar."""
from __future__ import annotations

import logging
import os
import shutil
import zipfile

log = logging.getLogger(__name__)

REPO = "shibing624/ai-avatar-wav2lip"
MODEL_FILE = "wav2lip.pth"
AVATARS = {
    "glass_man": "wav2lip_avatar_glass_man",
    "female_model": "wav2lip_avatar_female_model",
    "long_hair_girl": "wav2lip_avatar_long_hair_girl",
}


# MuseTalk 1.5 and the models it builds on: (repo, file in repo, path under models/musetalk)
MUSETALK_FILES = [
    ("TMElyralab/MuseTalk", "musetalkV15/musetalk.json", "musetalkV15/musetalk.json"),
    ("TMElyralab/MuseTalk", "musetalkV15/unet.pth", "musetalkV15/unet.pth"),
    ("stabilityai/sd-vae-ft-mse", "config.json", "sd-vae/config.json"),
    ("stabilityai/sd-vae-ft-mse", "diffusion_pytorch_model.safetensors", "sd-vae/diffusion_pytorch_model.safetensors"),
    ("openai/whisper-tiny", "config.json", "whisper/config.json"),
    ("openai/whisper-tiny", "preprocessor_config.json", "whisper/preprocessor_config.json"),
    ("openai/whisper-tiny", "model.safetensors", "whisper/model.safetensors"),
    ("ManyOtherFunctions/face-parse-bisent", "79999_iter.pth", "face-parse-bisent/79999_iter.pth"),
]


def hf_url(filename: str, repo: str = REPO) -> str:
    # Set HF_ENDPOINT=https://hf-mirror.com if huggingface.co is not reachable.
    base = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{base}/{repo}/resolve/main/{filename}"


def _fetch(url: str, dest: str) -> None:
    import requests
    from tqdm import tqdm

    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)


def download_model(models_dir: str = "models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    dest = os.path.join(models_dir, MODEL_FILE)
    if os.path.exists(dest):
        log.info("模型已存在：%s", dest)
    else:
        _fetch(hf_url(MODEL_FILE), dest)
    return dest


def download_musetalk(target_dir: str = "models/musetalk") -> str:
    for repo, filename, rel in MUSETALK_FILES:
        dest = os.path.join(target_dir, rel)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _fetch(hf_url(filename, repo), dest)
    log.info("MuseTalk 模型已就绪：%s", target_dir)
    return target_dir


def download_avatar(name: str, data_dir: str = "data") -> str:
    folder = AVATARS.get(name, name)
    target = os.path.join(data_dir, folder)
    if os.path.isdir(target):
        log.info("形象已存在：%s", target)
        return target
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, folder + ".zip")
    _fetch(hf_url(folder + ".zip"), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.startswith("__MACOSX") or ".." in member or member.startswith("/"):
                continue
            z.extract(member, data_dir)
    os.remove(zip_path)
    shutil.rmtree(os.path.join(data_dir, "__MACOSX"), ignore_errors=True)
    return target
