"""Choosing which GPU runs which task.

Two tasks compete for GPUs during a stream: lip sync (this program) and the
voice changer (an external program). Each task has a rough demand level; the
heaviest task goes to the fastest card, and the user can override every choice.

GPUs are stored by UUID, not by index: CUDA numbers cards fastest-first by
default while nvidia-smi and most tools number them by PCI slot, so "GPU 1"
can mean different cards in different programs. A UUID is the same everywhere,
and CUDA_VISIBLE_DEVICES accepts it directly.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

log = logging.getLogger(__name__)

CONFIG_FILE = "gpu_config.json"

TASKS = {
    "lipsync": "口型生成（本程序）",
    "voice_changer": "实时变声（外部程序）",
}

# Rough relative load of each task at live-stream settings (1 = light, 5 = heavy).
DEMAND = {
    "lipsync": {"wav2lip": 2, "musetalk": 5},
    "voice_changer": {"rvc": 2, "seed-vc": 5, "other": 3},
}


@dataclass
class GPU:
    index: int  # torch index in this process
    name: str
    uuid: str  # "GPU-xxxxxxxx-...", as nvidia-smi and CUDA_VISIBLE_DEVICES write it
    memory_gb: float
    sm_count: int
    tflops: Optional[float] = None  # measured FP16 matmul throughput

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"

    @property
    def speed(self) -> float:
        """For ranking: measured throughput when available, otherwise SM count."""
        return self.tflops if self.tflops is not None else float(self.sm_count)

    def label(self) -> str:
        perf = f"，实测 {self.tflops:.0f} TFLOPS" if self.tflops is not None else ""
        return f"{self.name}（{self.memory_gb:.0f} GB 显存{perf}）"


def _uuid_of(props, index: int) -> str:
    raw = getattr(props, "uuid", None)
    if raw is None:
        return f"index-{index}"
    s = str(raw)
    return s if s.startswith("GPU-") else f"GPU-{s}"


def list_gpus() -> list[GPU]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append(GPU(i, p.name, _uuid_of(p, i), p.total_memory / 1024**3, p.multi_processor_count))
    return gpus


def benchmark(gpu: GPU, seconds: float = 0.6) -> float:
    """Measures FP16 matrix-multiply throughput in TFLOPS, a fair proxy for both tasks."""
    import torch

    dev = torch.device(gpu.device)
    n = 4096
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize(dev)
    count, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        a @ b
        count += 1
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    gpu.tflops = 2 * n**3 * count / elapsed / 1e12
    return gpu.tflops


def recommend(gpus: list[GPU], kinds: dict[str, str]) -> dict[str, Optional[GPU]]:
    """Heaviest task on the fastest card, next heaviest on the next card, and so on.

    `kinds` names what each task runs, e.g. {"lipsync": "wav2lip", "voice_changer": "rvc"}.
    With fewer cards than tasks, the lightest tasks share the slowest card.
    On equal demand, lip sync gets the faster card: it must finish each batch
    inside a fixed time budget, while voice changers buffer their audio.
    """
    if not gpus:
        return {t: None for t in TASKS}
    order = sorted(TASKS, key=lambda t: (-DEMAND[t].get(kinds.get(t, ""), 3), t != "lipsync"))
    ranked = sorted(gpus, key=lambda g: (-g.speed, -g.memory_gb, g.index))
    return {task: ranked[min(i, len(ranked) - 1)] for i, task in enumerate(order)}


# --- config file ---------------------------------------------------------------

def load_config(path: str = CONFIG_FILE) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_config(assignment: dict[str, Optional[GPU]], kinds: dict[str, str], path: str = CONFIG_FILE) -> None:
    data = {
        "version": 1,
        "kinds": kinds,
        "tasks": {t: (asdict(g) if g else None) for t, g in assignment.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_device(task: str, requested: Optional[str], gpus: Optional[list[GPU]] = None,
                   path: str = CONFIG_FILE) -> str:
    """The device a task should use: an explicit --device wins, then the config file, then auto."""
    if requested and requested != "auto":
        return requested
    entry = load_config(path).get("tasks", {}).get(task)
    if not entry:
        return "auto"
    gpus = list_gpus() if gpus is None else gpus
    for g in gpus:
        if g.uuid == entry["uuid"]:
            return g.device
    log.warning("配置里给%s分配的显卡（%s）现在找不到了，改为自动选择。运行 python -m avatarstream gpus --setup 重新分配。",
                TASKS.get(task, task), entry.get("name"))
    return "auto"


def write_launcher(target_script: str, gpu: GPU, out_path: str) -> str:
    """Writes a .bat that starts an external program pinned to one GPU."""
    target = os.path.abspath(target_script)
    ext = os.path.splitext(target)[1].lower()
    if ext not in (".bat", ".cmd", ".exe"):
        raise ValueError("变声器启动程序要是 .bat、.cmd 或 .exe。Python 脚本请先给它写一个 .bat 再填进来。")
    lines = [
        "@echo off",
        "chcp 65001 >nul",
        f"rem 由 python -m avatarstream gpus --setup 生成：把下面的程序固定在 {gpu.name} 上运行",
        "set CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"set CUDA_VISIBLE_DEVICES={gpu.uuid}",
        f'cd /d "{os.path.dirname(target)}"',
        f'call "{target}"' if ext in (".bat", ".cmd") else f'"{target}"',
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("\r\n".join(lines) + "\r\n")
    return out_path
