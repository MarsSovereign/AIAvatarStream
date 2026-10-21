"""Checks the installation and writes a report that can be sent along with a problem."""
from __future__ import annotations

import glob
import importlib
import os
import platform
import sys
from typing import Callable

OK, WARN, FAIL = "正常", "注意", "问题"

Check = tuple[str, str, str]  # (status, item, detail)


def _python() -> list[Check]:
    v = sys.version_info
    status = OK if (3, 10) <= (v.major, v.minor) <= (3, 12) else WARN
    detail = f"{platform.python_version()}，{platform.system()} {platform.release()}"
    if status == WARN:
        detail += "（推荐 3.10 或 3.11）"
    return [(status, "Python", detail)]


def _packages() -> list[Check]:
    out = []
    for name, module, required in (
        ("PyTorch", "torch", True), ("OpenCV", "cv2", True), ("librosa", "librosa", True),
        ("sounddevice", "sounddevice", True), ("soxr", "soxr", True), ("aiohttp", "aiohttp", True),
        ("diffusers（MuseTalk）", "diffusers", False), ("transformers（MuseTalk）", "transformers", False),
        ("face-alignment（MuseTalk）", "face_alignment", False), ("pyvirtualcam（虚拟摄像头）", "pyvirtualcam", False),
    ):
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "已安装")
            status = OK
            if module == "cv2" and str(version).startswith("5"):
                status, version = WARN, f"{version}（需要 4.x：pip install \"opencv-python<5\"）"
            out.append((status, name, str(version)))
        except Exception as e:  # import errors, missing DLLs, missing PortAudio...
            out.append((FAIL if required else WARN, name, f"不可用：{type(e).__name__}: {e}"))
    return out


def _gpus() -> list[Check]:
    try:
        import torch
    except Exception:
        return [(FAIL, "显卡", "PyTorch 不可用")]
    if not torch.cuda.is_available():
        build = torch.version.cuda or "CPU 版"
        detail = f"CUDA 不可用（PyTorch {torch.__version__}，{build}）。装的可能是 CPU 版 PyTorch，见 README 安装步骤"
        return [(FAIL, "显卡", detail)]
    out = [(OK, "CUDA", f"PyTorch {torch.__version__}，CUDA {torch.version.cuda}")]
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append((OK, f"显卡 cuda:{i}", f"{p.name}，{p.total_memory / 1024**3:.0f} GB"))
    return out


def _ffmpeg() -> list[Check]:
    from .sinks.recorder import find_ffmpeg

    path = find_ffmpeg()
    return [(OK, "ffmpeg", path) if path else (WARN, "ffmpeg", "没找到：录像和 render 会没有声音。pip install imageio-ffmpeg")]


def _models(root: str) -> list[Check]:
    out = []
    w2l = os.path.join(root, "models", "wav2lip.pth")
    out.append((OK, "Wav2Lip 模型", w2l) if os.path.exists(w2l) else
               (WARN, "Wav2Lip 模型", "没有：python -m avatarstream download"))
    from .musetalk.model import missing_files

    missing = missing_files(os.path.join(root, "models", "musetalk"))
    out.append((OK, "MuseTalk 模型", "齐全") if not missing else
               (WARN, "MuseTalk 模型", f"缺 {len(missing)} 个文件：python -m avatarstream download --musetalk"))
    return out


def _avatars(root: str) -> list[Check]:
    from .musetalk.avatar import has_musetalk_data

    dirs = [d for d in sorted(glob.glob(os.path.join(root, "data", "*"))) if os.path.isdir(os.path.join(d, "full_imgs"))]
    if not dirs:
        return [(WARN, "形象", "data 目录下没有形象：python -m avatarstream download")]
    out = []
    for d in dirs:
        n = len(glob.glob(os.path.join(d, "full_imgs", "*.png"))) + len(glob.glob(os.path.join(d, "full_imgs", "*.jpg")))
        mt = "有 MuseTalk 数据" if has_musetalk_data(d) else "没有 MuseTalk 数据（需要时运行 prepare-musetalk）"
        out.append((OK, f"形象 {os.path.basename(d)}", f"{n} 帧，{mt}"))
    return out


def _audio() -> list[Check]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception as e:
        return [(FAIL, "音频设备", f"无法读取：{e}")]
    inputs = [d["name"] for d in devices if d["max_input_channels"] > 0]
    cables = [n for n in inputs if "cable" in n.lower() or "voicemeeter" in n.lower()]
    out = [(OK, "音频输入设备", f"{len(inputs)} 个")]
    out.append((OK, "虚拟声卡", "、".join(sorted(set(cables)))) if cables else
               (WARN, "虚拟声卡", "没找到 CABLE / Voicemeeter。装好 VB-CABLE 后要重启电脑"))
    return out


def _gpu_config(root: str) -> list[Check]:
    from .gpus import load_config

    cfg = load_config(os.path.join(root, "gpu_config.json"))
    if not cfg:
        return [(OK, "显卡分配", "没有配置（单卡不需要；多卡运行 setup_gpus.bat）")]
    parts = [f"{t}={(e or {}).get('name', 'CPU')}" for t, e in cfg.get("tasks", {}).items()]
    return [(OK, "显卡分配", f"{cfg.get('kinds')}；" + "，".join(parts))]


CHECKS: list[tuple[str, Callable[[str], list[Check]]]] = [
    ("系统", lambda root: _python()),
    ("依赖", lambda root: _packages()),
    ("显卡", lambda root: _gpus()),
    ("录像", lambda root: _ffmpeg()),
    ("模型", _models),
    ("形象", _avatars),
    ("音频", lambda root: _audio()),
    ("配置", _gpu_config),
]


def run(root: str = ".") -> tuple[str, int]:
    """Returns the report text and the number of problems found."""
    from . import __version__

    lines = [f"AIAvatarStream {__version__} 环境检查", ""]
    problems = 0
    for section, check in CHECKS:
        lines.append(f"[{section}]")
        try:
            results = check(root)
        except Exception as e:
            results = [(FAIL, section, f"检查时出错：{type(e).__name__}: {e}")]
        for status, item, detail in results:
            problems += status == FAIL
            lines.append(f"  {status}  {item}：{detail}")
        lines.append("")
    lines.append("没有发现问题。" if problems == 0 else f"发现 {problems} 个问题，见上面标“{FAIL}”的行。")
    return "\n".join(lines), problems
