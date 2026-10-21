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
    p.add_argument("--model", default="models/wav2lip.pth", help="Wav2Lip 模型文件")
    p.add_argument("--device", default="auto", help="auto / cuda / cuda:1（第二张显卡）/ cpu / mps")
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

    for path, hint in ((args.model, "python -m avatarstream download"), (args.avatar, "python -m avatarstream download")):
        if not os.path.exists(path):
            sys.exit(f"找不到 {path}。先运行：{hint}")
    model = LipSyncModel(args.model, args.device, args.fp16)
    avatar = load_avatar(args.avatar)
    gate = EnergyGate(args.vad_db, args.hangover_ms)
    engine = LipSyncEngine(model, avatar, src_sr, args.batch_size, gate, Compositor(args.blend, args.feather))
    return engine, model, avatar


def cmd_devices(args) -> None:
    from .audio_io import describe_devices

    print(describe_devices())


def cmd_download(args) -> None:
    from .download import AVATARS, download_avatar, download_model

    download_model(args.models_dir)
    names = list(AVATARS) if args.avatar == "all" else [args.avatar]
    for n in names:
        print("形象：", download_avatar(n, args.data_dir))


def cmd_make_avatar(args) -> None:
    from .make_avatar import make_avatar

    make_avatar(args.video, args.out, args.detector, tuple(args.pads), args.max_seconds,
                args.resize_height, not args.no_smooth)


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
        log.warning("这台机器跑不动实时：一批推理比一批画面的时长还长。换 N 卡、加 --fp16，或调大 --batch-size。")

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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m avatarstream", description="真人驱动的数字人直播：你说话，数字人实时对口型")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("devices", help="列出音频设备")
    s.set_defaults(func=cmd_devices)

    s = sub.add_parser("download", help="下载模型和示例形象")
    s.add_argument("--avatar", default="glass_man", help="glass_man / female_model / long_hair_girl / all")
    s.add_argument("--models-dir", default="models")
    s.add_argument("--data-dir", default="data")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("make-avatar", help="用一段闭嘴视频生成自己的形象")
    s.add_argument("--video", required=True)
    s.add_argument("--out", required=True, help="输出目录，例如 data/my_avatar")
    s.add_argument("--detector", choices=["s3fd", "haar"], default="s3fd")
    s.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0], metavar=("上", "下", "左", "右"),
                   help="人脸框外扩像素，下边至少要包住下巴")
    s.add_argument("--max-seconds", type=float, default=30)
    s.add_argument("--resize-height", type=int, default=None, help="视频太大时缩到这个高度，例如 1280")
    s.add_argument("--no-smooth", action="store_true")
    s.set_defaults(func=cmd_make_avatar)

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
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
