"""Command line: python -m avatarstream <command>"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time

log = logging.getLogger("avatarstream")

RIGHT_CONTEXT_FRAMES = 10  # see MelFeaturizer


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--avatar", default="data/wav2lip_avatar_glass_man", help="形象目录")
    p.add_argument("--engine", choices=["wav2lip", "musetalk"], default="wav2lip",
                   help="口型模型：wav2lip 轻快；musetalk 嘴部更清晰、可商用，但更吃显卡")
    p.add_argument("--model", default="models/wav2lip.pth", help="Wav2Lip 模型文件")
    p.add_argument("--musetalk-dir", default="models/musetalk", help="MuseTalk 模型目录")
    p.add_argument("--device", default="auto",
                   help="auto 表示按 gpu_config.json 的分配（没有就自动选）；也可以直接写 cuda:1 / cpu")
    p.add_argument("--fp16", action="store_true", help="N 卡上用半精度，更快、更省显存")
    p.add_argument("--batch-size", type=int, default=8,
                   help="每批生成的帧数。越小延迟越低，越大显卡越省力（直播建议 4 到 8）")
    p.add_argument("--blend", choices=["lower", "full"], default="lower",
                   help="lower 只替换嘴和下巴，眼睛保持原画质；full 替换整张脸")
    p.add_argument("--feather", type=float, default=0.08, help="脸框边缘羽化比例，0 表示不羽化")
    p.add_argument("--vad-db", type=float, default=-45.0,
                   help="音量高于这个值才算在说话（dBFS）。底噪大就调高，比如 -38")
    p.add_argument("--hangover-ms", type=int, default=300, help="说话停顿多久以内仍保持嘴型驱动")


def _build_engine(args, src_sr: int):
    from .avatar import load_avatar
    from .compositor import Compositor
    from .engine import LipSyncEngine
    from .model import LipSyncModel
    from .vad import EnergyGate

    if not os.path.exists(args.avatar):
        sys.exit(f"找不到 {args.avatar}。先运行：python -m avatarstream download")
    gate = EnergyGate(args.vad_db, args.hangover_ms)
    if args.engine == "musetalk":
        from .musetalk import MuseTalkBackend
        from .musetalk.avatar import load as load_musetalk_data
        from .musetalk.model import MuseTalkModel, missing_files

        if missing_files(args.musetalk_dir):
            sys.exit("MuseTalk 模型还没下载。先运行：python -m avatarstream download --musetalk")
        avatar = load_avatar(args.avatar)
        try:
            data = load_musetalk_data(args.avatar, avatar)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(str(e))
        model = MuseTalkModel(args.musetalk_dir, _lipsync_device(args))
        backend = MuseTalkBackend(model, avatar, data, src_sr, args.batch_size)
        return LipSyncEngine(None, avatar, src_sr, args.batch_size, gate, backend=backend), model, avatar
    if not os.path.exists(args.model):
        sys.exit(f"找不到 {args.model}。先运行：python -m avatarstream download")
    model = LipSyncModel(args.model, _lipsync_device(args), args.fp16)
    avatar = load_avatar(args.avatar)
    engine = LipSyncEngine(model, avatar, src_sr, args.batch_size, gate, Compositor(args.blend, args.feather))
    return engine, model, avatar


def _lipsync_device(args) -> str:
    from .gpus import resolve_device

    device = resolve_device("lipsync", args.device)
    if device.startswith("cuda"):
        import torch

        idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        log.info("口型生成使用显卡 %s：%s", device, torch.cuda.get_device_name(idx))
    return device


def cmd_doctor(args) -> None:
    from .doctor import run

    report, problems = run(".")
    print(report)
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "diagnostics.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n报告已保存到 {path}，遇到问题时把它和报错一起发出来。")


def cmd_devices(args) -> None:
    from .audio_io import describe_devices

    print(describe_devices())


def cmd_download(args) -> None:
    from .download import AVATARS, download_avatar, download_model, download_musetalk

    download_model(args.models_dir)
    if args.musetalk:
        download_musetalk(os.path.join(args.models_dir, "musetalk"))
    names = list(AVATARS) if args.avatar == "all" else [args.avatar]
    for n in names:
        print("形象：", download_avatar(n, args.data_dir))


def cmd_make_avatar(args) -> None:
    from .make_avatar import make_avatar

    from .gpus import resolve_device

    device = resolve_device("lipsync", args.device)
    make_avatar(args.video, args.out, args.detector, tuple(args.pads), args.max_seconds,
                args.resize_height, not args.no_smooth, device=None if device == "auto" else device)


def cmd_prepare_musetalk(args) -> None:
    from .gpus import resolve_device
    from .musetalk.avatar import prepare
    from .musetalk.model import MuseTalkModel, missing_files

    if missing_files(args.musetalk_dir):
        sys.exit("MuseTalk 模型还没下载。先运行：python -m avatarstream download --musetalk")
    device = resolve_device("lipsync", args.device)
    model = MuseTalkModel(args.musetalk_dir, None if device == "auto" else device)
    prepare(args.avatar, model, os.path.join(args.musetalk_dir, "face-parse-bisent", "79999_iter.pth"),
            bbox_shift=args.bbox_shift, smooth=not args.no_smooth)


def cmd_render(args) -> None:
    from .audio_io import iter_file_frames, read_audio_file, silent_frame
    from .sinks import RecorderSink

    _, sr = read_audio_file(args.audio)
    engine, _, avatar = _build_engine(args, sr)
    w, h = avatar.size
    rec = RecorderSink(args.out, w, h, sr)
    frames = list(iter_file_frames(args.audio))
    # Flush the right-hand context so the last words get their frames too.
    frames += [silent_frame(sr) for _ in range(RIGHT_CONTEXT_FRAMES)]
    step = engine.frames_per_step
    if len(frames) % step:
        frames += [silent_frame(sr) for _ in range(step - len(frames) % step)]
    t = time.perf_counter()
    for i in range(0, len(frames), step):
        for vf in engine.process(frames[i:i + step]):
            rec.send(vf)
    rec.close()
    log.info("渲染用时 %.1f 秒，推理 %.1f 帧/秒", time.perf_counter() - t, engine.infer_fps)


def cmd_live(args) -> None:
    from .audio_io import AudioPlayer, FileSource, MicSource
    from .engine import VIDEO_FPS, EngineThread
    from .player import Player, Stats
    from .sinks import MJPEGSink, RecorderSink, VirtualCamSink

    if args.input_file:
        source = FileSource(args.input_file, loop=args.loop)
    else:
        source = MicSource(args.input_device, args.input_samplerate)
    engine, model, avatar = _build_engine(args, source.src_sr)

    per_batch = engine.warm_up()
    budget = args.batch_size / VIDEO_FPS
    est = (2 * args.batch_size + RIGHT_CONTEXT_FRAMES) * 0.02 + per_batch + args.prebuffer / VIDEO_FPS
    log.info("每批 %d 帧推理 %.0f ms（实时上限 %.0f ms），预计延迟约 %.0f ms",
             args.batch_size, per_batch * 1000, budget * 1000, est * 1000)
    if per_batch > budget:
        log.warning("这台机器跑不动实时：一批推理比一批画面的时长还长。可以把口型放到单独的强卡上（gpus --setup）、"
                    "调大 --batch-size，Wav2Lip 加 --fp16，或者换成更轻的 --engine wav2lip。")

    stats = Stats()
    sinks = []
    mjpeg = MJPEGSink(args.host, args.port, stats=stats.snapshot)
    sinks.append(mjpeg)
    w, h = avatar.size
    if args.virtualcam:
        sinks.append(VirtualCamSink(w, h, VIDEO_FPS))
    if args.record:
        sinks.append(RecorderSink(args.record, w, h, source.src_sr))
    audio_out = AudioPlayer(args.audio_out_device, source.src_sr) if args.audio_out_device else None

    frames: queue.Queue = queue.Queue(maxsize=max(4 * args.batch_size, 32))
    failed = threading.Event()
    worker = EngineThread(engine, source, frames, on_error=lambda e: failed.set())
    player = Player(sinks, audio_out, stats, VIDEO_FPS, args.prebuffer)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    mjpeg.start()
    if audio_out:
        audio_out.start()
    player.start(frames)
    worker.start()
    source.start()
    log.info("开始运行，按 Ctrl+C 停止")
    t_end = time.monotonic() + args.duration if args.duration else None
    last_print = 0.0
    try:
        while not stop.is_set() and not failed.is_set():
            if t_end and time.monotonic() >= t_end:
                break
            stats.level_db = engine.gate.last_db
            stats.extra.update(infer_fps=round(engine.infer_fps, 1), dropped_audio=source.dropped,
                               dropped_video=worker.dropped_video)
            if time.monotonic() - last_print > 5:
                s = stats.snapshot()
                log.info("帧率 %s｜延迟 %s ms（P95 %s）｜音量 %s dB｜%s",
                         s["fps"], s["latency_ms"], s["latency_p95_ms"], s["level_db"], "说话中" if s["speaking"] else "待机")
                last_print = time.monotonic()
            time.sleep(0.1)
    finally:
        source.stop()
        worker.stop()
        player.stop()
        if audio_out:
            audio_out.stop()
        for s in sinks:
            s.close()
    if failed.is_set():
        sys.exit(1)


from .gpus import DEMAND as DEMAND_KINDS  # noqa: E402  (no torch import at module load)

LAUNCHER_PATH = os.path.join("scripts", "start_voice_changer.bat")


def _ask(prompt: str, default: str = "") -> str:
    reply = input(f"{prompt}{f' [回车 = {default}]' if default else ''}：").strip()
    return reply or default


def _run_lipsync_test(args, found, engine: str) -> dict:
    if engine == "musetalk":
        from .musetalk.model import missing_files

        if missing_files(args.musetalk_dir):
            sys.exit("MuseTalk 模型还没下载。先运行：python -m avatarstream download --musetalk")
        path = args.musetalk_dir
    else:
        if not os.path.exists(args.model):
            sys.exit(f"找不到 {args.model}，先运行 python -m avatarstream download")
        path = args.model
    print(f"\n在每张卡上实测 {engine}：")
    return _test_lipsync(found, path, args.batch_size, engine)


def _test_lipsync(gpus, model_path: str, batch_size: int, engine: str = "wav2lip") -> dict:
    from .engine import VIDEO_FPS

    budget = batch_size / VIDEO_FPS
    results = {}
    for g in gpus:
        if engine == "musetalk":
            from .musetalk.model import MuseTalkModel

            m = MuseTalkModel(model_path, g.device)
        else:
            from .model import LipSyncModel

            m = LipSyncModel(model_path, g.device, fp16=True)
        per_batch = min(m.warm_up(batch_size) for _ in range(3))
        results[g.uuid] = per_batch / budget
        print(f"  [{gpus.index(g)}] {g.name}：每批 {batch_size} 帧 {per_batch * 1000:.0f} ms，"
              f"占实时预算 {per_batch / budget:.0%}")
        del m
    return results


def cmd_gpus(args) -> None:
    from . import gpus as G

    found = G.list_gpus()
    if not found:
        print("没有检测到 NVIDIA 显卡（或者装的是 CPU 版 PyTorch）。口型只能用 CPU 跑，速度不够直播。")
        return
    if not args.no_benchmark:
        print("正在测试每张显卡的性能……")
        for g in found:
            G.benchmark(g)
    print("\n检测到的显卡：")
    for i, g in enumerate(found):
        print(f"  [{i}] {g.label()}")

    config = G.load_config()
    kinds = dict(config.get("kinds") or {"lipsync": "wav2lip", "voice_changer": "rvc"})
    if args.lipsync_engine:
        kinds["lipsync"] = args.lipsync_engine
    if args.voice_changer:
        kinds["voice_changer"] = args.voice_changer

    if not (args.setup or args.auto or args.assign):
        if args.test_lipsync:
            _run_lipsync_test(args, found, kinds["lipsync"])
        current = config.get("tasks") or {}
        print("\n当前分配：" if current else "\n还没有分配。运行 python -m avatarstream gpus --setup 开始设置。")
        for task, entry in current.items():
            print(f"  {G.TASKS[task]}：{entry['name'] if entry else 'CPU'}")
        return

    if args.setup and not args.lipsync_engine:
        engines = list(DEMAND_KINDS["lipsync"])
        choice = _ask("\n口型用哪个模型？1 = Wav2Lip（轻），2 = MuseTalk（重，画质好）",
                      str(engines.index(kinds.get("lipsync", "wav2lip")) + 1))
        kinds["lipsync"] = engines[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(engines) else "wav2lip"
    if args.setup and not args.voice_changer:
        names = list(DEMAND_KINDS["voice_changer"])
        choice = _ask("\n你用哪个变声器？1 = RVC，2 = Seed-VC，3 = 其他",
                      str(names.index(kinds.get("voice_changer", "rvc")) + 1))
        kinds["voice_changer"] = names[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(names) else "rvc"

    load = _run_lipsync_test(args, found, kinds["lipsync"]) if args.test_lipsync else {}
    plan = G.recommend(found, kinds)
    lip = plan["lipsync"]
    if lip is not None and load.get(lip.uuid, 0) > 0.7:
        # The recommended card is too slow for lip sync: move it to the fastest card that keeps up.
        fast_enough = [g for g in found if load.get(g.uuid, 1) <= 0.7]
        if fast_enough:
            best = max(fast_enough, key=lambda g: g.speed)
            other = plan["voice_changer"]
            plan["lipsync"], plan["voice_changer"] = best, (lip if other is best else other)
            print(f"\n实测发现 {lip.name} 跑口型太吃力，已改为 {best.name}。")
    print("\n推荐方案（负载重的任务放在更快的卡上）：")
    for task, g in plan.items():
        level = DEMAND_KINDS[task].get(kinds.get(task, ""), 3)
        print(f"  {G.TASKS[task]}（负载 {level}/5）→ [{found.index(g)}] {g.name}")

    if args.assign:
        for item in args.assign:
            task, _, num = item.partition("=")
            if task not in G.TASKS or not num.isdigit() or int(num) >= len(found):
                sys.exit(f"看不懂“{item}”。格式是 任务=编号，任务可选：{', '.join(G.TASKS)}")
            plan[task] = found[int(num)]
    elif args.setup:
        for task in G.TASKS:
            num = _ask(f"{G.TASKS[task]}用哪张卡？", str(found.index(plan[task])))
            if num.isdigit() and int(num) < len(found):
                plan[task] = found[int(num)]
            else:
                print(f"  没有编号 {num}，保持推荐的 [{found.index(plan[task])}]")

    if len(found) > 1 and plan["lipsync"] is plan["voice_changer"]:
        print("\n提醒：口型和变声分到了同一张卡上，两边会互相抢资源。有意为之的话可以忽略。")
    for task, g in plan.items():
        if load.get(g.uuid, 0) > 0.7 and task == "lipsync":
            print(f"提醒：{g.name} 跑口型占了实时预算的 {load[g.uuid]:.0%}，直播时可能卡顿。")

    G.save_config(plan, kinds)
    print(f"\n已保存到 {G.CONFIG_FILE}：")
    for task, g in plan.items():
        print(f"  {G.TASKS[task]} → {g.name}")

    script = args.voice_changer_script
    if args.setup and not script:
        script = _ask("变声器的启动脚本路径（例如 D:\\RVC\\go-realtime-gui.bat，直接回车跳过）")
    if script:
        G.write_launcher(script, plan["voice_changer"], LAUNCHER_PATH)
        print(f"已生成 {LAUNCHER_PATH}：以后用它启动变声器，变声器就只会用 {plan['voice_changer'].name}。")
    else:
        print(f"变声器那边：启动它之前设置 CUDA_VISIBLE_DEVICES={plan['voice_changer'].uuid}，"
              "或者重新运行 gpus --setup 并填上它的启动脚本，自动生成。")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m avatarstream", description="真人驱动的数字人直播：你说话，数字人实时对口型")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("devices", help="列出音频设备")
    s.set_defaults(func=cmd_devices)

    s = sub.add_parser("doctor", help="检查安装情况，生成诊断报告")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("download", help="下载模型和示例形象")
    s.add_argument("--avatar", default="glass_man", help="glass_man / female_model / long_hair_girl / all")
    s.add_argument("--models-dir", default="models")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--musetalk", action="store_true", help="同时下载 MuseTalk 模型（约 4 GB）")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("prepare-musetalk", help="给形象生成 MuseTalk 需要的数据（每个形象做一次）")
    s.add_argument("--avatar", required=True, help="形象目录，例如 data/my_avatar")
    s.add_argument("--musetalk-dir", default="models/musetalk")
    s.add_argument("--device", default="auto")
    s.add_argument("--bbox-shift", type=int, default=0, help="嘴部区域上下微调（像素）。张嘴不够就调成正数")
    s.add_argument("--no-smooth", action="store_true", help="不对人脸框做时间平滑")
    s.set_defaults(func=cmd_prepare_musetalk)

    s = sub.add_parser("make-avatar", help="用一段闭嘴视频生成自己的形象")
    s.add_argument("--video", required=True)
    s.add_argument("--out", required=True, help="输出目录，例如 data/my_avatar")
    s.add_argument("--detector", choices=["s3fd", "haar"], default="s3fd")
    s.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0], metavar=("上", "下", "左", "右"),
                   help="人脸框外扩像素，下边至少要包住下巴")
    s.add_argument("--max-seconds", type=float, default=30)
    s.add_argument("--resize-height", type=int, default=None, help="视频太大时缩到这个高度，例如 1280")
    s.add_argument("--no-smooth", action="store_true")
    s.add_argument("--device", default="auto", help="人脸检测用的设备，默认跟口型生成用同一张卡")
    s.set_defaults(func=cmd_make_avatar)

    s = sub.add_parser("gpus", help="查看显卡，给口型和变声分配显卡")
    s.add_argument("--setup", action="store_true", help="一步步问你，每项任务用哪张卡")
    s.add_argument("--auto", action="store_true", help="直接采用推荐方案并保存")
    s.add_argument("--assign", nargs="+", metavar="任务=编号",
                   help="手动指定，例如 lipsync=1 voice_changer=0（编号见列表）")
    s.add_argument("--lipsync-engine", choices=list(DEMAND_KINDS["lipsync"]), default=None,
                   help="口型模型，决定它对显卡的要求")
    s.add_argument("--voice-changer", choices=list(DEMAND_KINDS["voice_changer"]), default=None,
                   help="你用的变声器，决定它对显卡的要求")
    s.add_argument("--voice-changer-script", default=None,
                   help="变声器的启动脚本（例如 RVC 的 go-realtime-gui.bat），会生成一个锁定显卡的启动脚本")
    s.add_argument("--test-lipsync", action="store_true", help="在每张卡上实测口型模型的速度")
    s.add_argument("--model", default="models/wav2lip.pth")
    s.add_argument("--musetalk-dir", default="models/musetalk")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--no-benchmark", action="store_true", help="不跑性能测试，只按硬件参数排序")
    s.set_defaults(func=cmd_gpus)

    s = sub.add_parser("render", help="把一段音频离线生成口型视频（用来检查效果）")
    _add_engine_args(s)
    s.add_argument("--audio", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_render)

    s = sub.add_parser("live", help="开始直播")
    _add_engine_args(s)
    s.add_argument("--input-device", default=None, help="输入设备编号或名字片段，例如 \"CABLE Output\"")
    s.add_argument("--input-samplerate", type=int, default=None)
    s.add_argument("--input-file", default=None, help="用音频文件代替麦克风（彩排、测试用）")
    s.add_argument("--loop", action="store_true", help="--input-file 循环播放")
    s.add_argument("--audio-out-device", default=None,
                   help="把和画面对齐的声音送到这个输出设备，例如第二条虚拟声卡 \"CABLE-A Input\"")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8010)
    s.add_argument("--virtualcam", action="store_true", help="同时输出到虚拟摄像头")
    s.add_argument("--record", default=None, help="同时录成 mp4")
    s.add_argument("--prebuffer", type=int, default=2, help="播放前缓冲的帧数，网络或显卡抖动大就加大")
    s.add_argument("--duration", type=float, default=None, help="运行多少秒后自动退出（测试用）")
    s.set_defaults(func=cmd_live)
    return p


def main(argv=None) -> None:
    # Number GPUs by PCI slot, like nvidia-smi and the voice-changer launcher do.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
