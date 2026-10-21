# AIAvatarStream

真人驱动的数字人直播：**你对着麦克风说话，变声器把声音换成数字人的声音，数字人实时跟着对口型，OBS 推到直播间。**

```
你的麦克风 ──► 变声器（RVC / Seed-VC）──► 虚拟声卡
                                          │
                     ┌────────────────────┴───────────────────┐
                     ▼                                        ▼
              AIAvatarStream                         OBS 直接采集变声后的声音
  语音检测 → 口型（MuseTalk / Wav2Lip）→ 合成画面  （设好同步偏移，和口型对齐）
                     │
                     ▼
     OBS 浏览器源 / 虚拟摄像头  ──►  直播伴侣 / 推流
```

画面里始终是数字人形象，声音始终是数字人的音色，但说话的人是你本人。这符合抖音等平台对数字人直播“真人驱动、实时互动”的要求（详见[合规](#合规与授权)）。

## 项目状态

**代码已完成，还没有在 Windows + 显卡上实测。** 第一次使用前，请先按 **[Windows 首次验收](docs/Windows首次验收.md)** 走一遍。

| | 状态 | 怎么验证的 |
|---|---|---|
| 口型生成正确（MuseTalk、Wav2Lip） | ✅ 已验证 | 在 Linux 服务器上用 CPU 跑真实模型：说话时嘴在动，停下来闭嘴，眼睛和背景不变 |
| 音画没有错位 | ✅ 已验证 | 每帧的音频窗口和官方实现逐帧比对一致 |
| 画质：MuseTalk 明显好于 Wav2Lip | ✅ 已验证 | 同一段语音两种模型对比，MuseTalk 嘴部清晰、能看到牙齿 |
| 整条流程能跑通 | ✅ 已验证 | 用音频文件代替麦克风，跑完输入、口型、25 帧输出、预览、状态页、录像 |
| 显卡分配逻辑 | ✅ 已验证 | 用模拟的强卡和弱卡测试 |
| **显卡上实时跑到 25 帧/秒** | ⏳ 待验证 | 只能在有 N 卡的电脑上测，是能否直播的前提 |
| **Windows 安装和脚本** | ⏳ 待验证 | 所有 .bat 脚本还没在 Windows 上运行过 |
| **麦克风、虚拟声卡、RVC、OBS** | ⏳ 待验证 | 服务器上没有这些设备 |
| **实际延迟、音画同步、长时间稳定** | ⏳ 待验证 | 需要实际试播 |

## 文档

| 文档 | 内容 |
|---|---|
| [Windows 首次验收](docs/Windows首次验收.md) | 第一次使用：6 步验收清单，每一步都有通过标准，以及出问题时该发什么 |
| [直播搭建指南](docs/直播搭建指南.md) | 虚拟声卡、变声器、多显卡、OBS、推流，按步骤搭好整套直播 |
| [形象制作指南](docs/形象制作指南.md) | 怎么准备待机视频、生成自己的数字人形象、调整嘴部区域 |

## 它做什么，不做什么

| 做 | 不做 |
|---|---|
| 把实时音频变成对上口型的数字人画面 | 不自动回复、不接大模型：内容全部由你本人说 |
| 不说话时播放形象的待机视频，不占显卡 | 不内置变声：用成熟的 RVC 或 Seed-VC，通过虚拟声卡接进来 |
| 输出给 OBS：浏览器源、虚拟摄像头，或录成 mp4 | 不直接推流：推流交给 OBS 和直播伴侣 |
| 状态页实时显示延迟、音量和 OBS 该设的同步偏移 | |
| 用一段闭嘴视频生成你自己的数字人形象 | |

## 口型模型：MuseTalk 还是 Wav2Lip

| | MuseTalk 1.5（推荐） | Wav2Lip |
|---|---|---|
| 嘴部画质 | 清晰，牙齿和唇形细节好（在 SD VAE 隐空间重绘下半张脸） | 偏糊 |
| 显卡要求 | 高：官方数据 V100 上 30 帧/秒以上 | 低：RTX 3060 约 60 帧/秒 |
| 模型大小 | 约 4 GB | 215 MB |
| 商用 | 代码 MIT，官方声明模型可用于任何用途，包括商用 | 原作者声明禁止商用（见[合规](#合规与授权)） |
| 已知缺点 | 胡子、唇色还原不完全；逐帧生成，偶有轻微抖动 | 嘴部糊，牙齿细节差 |

两个模型共用同一套直播流程，启动时用 `--engine musetalk` 或 `--engine wav2lip` 切换。

## 硬件要求

- Windows 10/11，NVIDIA 显卡。实时需要每秒生成 25 帧。
- **MuseTalk**：建议口型单独用一张强卡（RTX 4070 / 3090 或更好，12 GB 显存以上），变声器放另一张卡，见[多张显卡](#多张显卡)。只有一张卡的话，先用 `gpus --test-lipsync` 实测能不能跟上。
- **Wav2Lip**：RTX 3060 就够，变声器和口型可以挤在一张 12 GB 的卡上。
- 一个虚拟声卡，比如免费的 [VB-CABLE](https://vb-audio.com/Cable/)。
- 没有 N 卡也能装、能离线生成视频，但跑不动实时（CPU 大约 2 帧/秒）。

## 安装

### Windows 一键安装

1. 装 [Python 3.10 或 3.11](https://www.python.org/downloads/)，安装时勾选 “Add python.exe to PATH”。
2. 下载本项目，双击 `scripts\install.bat`。
   - 它会创建 `.venv`、装 GPU 版 PyTorch 和依赖，再下载 Wav2Lip 和 MuseTalk 模型（共约 4.3 GB）和示例形象，最后给示例形象生成 MuseTalk 数据。
   - RTX 50 系显卡先用记事本打开 `install.bat`，把 `cu124` 改成 `cu128`。
   - 国内下载 HuggingFace 慢的话，先在命令行执行 `set HF_ENDPOINT=https://hf-mirror.com` 再运行。
3. 双击 `scripts\doctor.bat` 检查安装情况，结果同时保存在 `output\diagnostics.txt`。
4. 双击 `scripts\test_render.bat`：用自带的测试语音分别生成 MuseTalk 和 Wav2Lip 两个版本，放在 `output\` 里，可以对比画质，也能看出显卡速度。

接下来按 [Windows 首次验收](docs/Windows首次验收.md) 继续。

### 手动安装（任何系统）

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS / Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按显卡选
pip install -r requirements.txt -r requirements-optional.txt -r requirements-musetalk.txt
python -m avatarstream download --musetalk          # 模型和示例形象；--avatar all 下载全部三个
python -m avatarstream prepare-musetalk --avatar data/wav2lip_avatar_glass_man
```

## 开播

完整的分步说明（变声器、虚拟声卡、OBS、抖音直播伴侣、排查问题）在 **[docs/直播搭建指南.md](docs/直播搭建指南.md)**。简要流程：

1. 变声器：输入选你的麦克风，输出选 `CABLE Input`。
2. 运行 `scripts\list_devices.bat`，确认能看到 `CABLE Output`。
3. 双击 `scripts\start_live.bat`，会自动打开状态页 `http://127.0.0.1:8010/status`。对着麦克风说话，看预览里的嘴动不动。
4. OBS：
   - 画面：添加“浏览器”源，URL 填 `http://127.0.0.1:8010/`，宽高填形象的分辨率。
   - 声音：添加“音频输入采集”，设备选 `CABLE Output`。在“高级音频属性”里把它的**同步偏移**设成状态页显示的数值（毫秒）。
5. OBS 推流到直播伴侣（或 OBS 直接推流）。开播前先用 OBS 录一段，检查口型和声音是否对齐。

## 命令速查

```bash
python -m avatarstream doctor                        # 检查安装情况，生成诊断报告
python -m avatarstream devices                       # 列出音频设备
python -m avatarstream gpus --setup --test-lipsync   # 多显卡：给口型和变声分配显卡
python -m avatarstream download --avatar all          # 下载模型和三个示例形象
python -m avatarstream download --musetalk            # 加上 MuseTalk 模型（约 4 GB）
python -m avatarstream make-avatar --video me.mp4 --out data/my_avatar
python -m avatarstream prepare-musetalk --avatar data/my_avatar   # 每个形象做一次
python -m avatarstream render --engine musetalk --audio a.wav --out out.mp4   # 离线生成，检查效果
python -m avatarstream live --engine musetalk --input-device "CABLE Output"   # 开始直播
```

`live` 的常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--engine` | `wav2lip` | 口型模型：`musetalk` 或 `wav2lip` |
| `--input-device` | 系统默认 | 变声器输出到的设备，写编号或名字片段 |
| `--device` | `auto` | 用哪张卡。`auto` 按 `gpu_config.json` 的分配；也可以写 `cuda:1` 临时指定 |
| `--avatar` | `data/wav2lip_avatar_glass_man` | 形象目录 |
| `--batch-size` | 8 | 每批生成多少帧。越小延迟越低，越大越省显卡。直播建议 4 到 8 |
| `--fp16` | 关 | Wav2Lip 在 N 卡上用半精度，更快更省显存。MuseTalk 在 N 卡上总是用半精度 |
| `--vad-db` | -45 | 音量高于它才算说话。没说话嘴也在动，就调高到 -40 或 -38 |
| `--blend` | `lower` | 仅 Wav2Lip：`lower` 只换嘴和下巴，眼睛保持原画质；`full` 换整张脸。MuseTalk 总是用人脸分割出的下半张脸 |
| `--virtualcam` | 关 | 同时输出到 OBS 虚拟摄像头（需要 `pyvirtualcam`） |
| `--audio-out-device` | 无 | 把和画面严格对齐的声音送到另一个输出设备，OBS 采集它就不用设偏移 |
| `--input-file` | 无 | 用音频文件代替麦克风，彩排用，加 `--loop` 循环 |
| `--record` | 无 | 同时录一份 mp4 |
| `--port` | 8010 | 预览和状态页端口 |

## 多张显卡

变声器和口型各用一张卡，两边就不会互相抢资源，延迟也更稳定。项目会按任务负载推荐分配方案，**每一项你都可以改**。

双击 `scripts\setup_gpus.bat`（或运行 `python -m avatarstream gpus --setup --test-lipsync`），它会：

1. 列出所有 N 卡，测出每张卡的实际算力。
2. 在每张卡上实测口型模型的速度，算出占实时预算的百分比。
3. 问你用的是哪个变声器，按负载给出推荐：**负载重的任务放在快的卡上**。
4. 逐项问你“口型用哪张卡、变声用哪张卡”，直接回车就是采用推荐，也可以输入其他编号。
5. 保存到 `gpu_config.json`。以后 `live`、`render`、`make-avatar` 都会自动使用口型那张卡。
6. 你填上变声器的启动脚本（比如 RVC 的 `go-realtime-gui.bat`）后，它会生成 `scripts\start_voice_changer.bat`，以后用这个脚本启动变声器，变声器就只会用分给它的那张卡。

各任务的负载参考（1 轻，5 重）：

| 任务 | 选项 | 负载 |
|---|---|---|
| 口型 | Wav2Lip | 2 |
| 口型 | MuseTalk | 5 |
| 变声 | RVC | 2 |
| 变声 | Seed-VC | 5 |

两项负载相同时，口型优先用强卡，因为每批画面都必须在固定时间内算完；变声器自己有音频缓冲，抗抖动能力更强。所以：

- Wav2Lip 加 RVC：**口型用强卡，RVC 用弱卡**。RVC 对显卡要求不高，弱卡就够。
- Wav2Lip 加 Seed-VC：Seed-VC 更重，它用强卡，口型用弱卡。
- 如果实测发现某张卡跑口型超过实时预算的 70%，推荐方案会自动把口型换到能跑得动的卡上。

不想用问答方式的话：

```bash
python -m avatarstream gpus                                   # 查看显卡和当前分配
python -m avatarstream gpus --auto --voice-changer rvc        # 直接采用推荐方案
python -m avatarstream gpus --assign lipsync=1 voice_changer=0 --voice-changer-script D:\RVC\go-realtime-gui.bat
python -m avatarstream live --device cuda:0                   # 临时指定，不改配置
```

显卡在配置里按硬件 UUID 记录，不按编号。不同程序给显卡排序的规则不一样（CUDA 默认把最快的卡排第一，nvidia-smi 按插槽排），按编号记录可能会把任务分错卡。生成的变声器启动脚本里也直接写 UUID，所以两边指的一定是同一张卡。

## 延迟

延迟主要来自三块：

- **凑一批**：要攒够 `2 × batch-size` 个 20 毫秒的音频帧才开始生成。
- **右侧上下文**：模型生成每一帧都要看到它之后 200 毫秒的声音。
- **推理时间**：取决于显卡。

预计延迟 ≈ (2 × batch-size + 10) × 20 ms + 一批推理时间 + 80 ms。`batch-size 6` 在中高端显卡上大约 0.5 到 0.6 秒。启动时会打印估算值，状态页显示实测值。

观众看不到你本人，所以只要声音和画面对齐，整体晚半秒没有影响。对齐方式二选一：

- **偏移法（简单）**：OBS 直接采集变声器输出，同步偏移设成状态页的“平均延迟”。
- **转发法（严格对齐）**：加 `--audio-out-device "第二条虚拟声卡"`，程序把每一帧画面对应的声音在同一时刻送出去，OBS 采集那条声卡即可。需要两条虚拟声卡（VB-CABLE A+B，或 Voicemeeter）。

## 做自己的形象

详细要求和技巧见 **[形象制作指南](docs/形象制作指南.md)**。简要流程：

1. 准备一段 **10 到 30 秒、正脸、全程闭嘴、动作轻微**的待机视频。推荐用图生视频工具生成一个不存在的虚拟人，没有肖像权问题。
2. 生成形象：`python -m avatarstream make-avatar --video idle.mp4 --out data/my_avatar --resize-height 1280`
3. 生成 MuseTalk 数据（每个形象一次）：`python -m avatarstream prepare-musetalk --avatar data/my_avatar`
4. 用 `render` 生成一段检查效果，满意后把 `start_live.bat` 里的 `AVATAR` 改成它。

## 合规与授权

- **形象和声音必须有授权。** 不要复刻明星或任何真实他人的长相和声音，否则侵犯肖像权和声音权益。推荐 AI 生成的虚拟人，配你自己训练或有授权的音色。
- **平台规则（以平台最新公告为准）。** 抖音对虚拟人直播的要求是：使用者实名认证、由真人驱动并实时互动、不允许完全由 AI 驱动，直播间要显著标识数字人。本项目的用法是真人说话驱动，但标识、实名这些需要你在平台侧完成。视频号等平台的规则可能更严，开播前请核实。
- **AI 生成内容标识。** 《人工智能生成合成内容标识办法》自 2025 年 9 月 1 日起施行，直播画面里请加上“数字人 / AI 生成”的提示。
- **模型权重。** MuseTalk：代码 MIT 协议，官方声明训练好的模型可用于任何用途，包括商用。它依赖的 sd-vae-ft-mse（MIT）、whisper-tiny（MIT）、人脸分割 BiSeNet（MIT）、face-alignment（BSD-3）各有自己的协议，商用前请自行确认一遍。Wav2Lip：`wav2lip.pth` 来自 [shibing624/ai-avatar-wav2lip](https://huggingface.co/shibing624/ai-avatar-wav2lip)，标注为 Apache-2.0。但 Wav2Lip 原作者声明，用 LRS2 数据集训练的官方模型禁止商用，这个 256 版本用什么数据训练的没有公开说明。**做带货等商业用途，建议用 MuseTalk。**

## 开发与测试

```bash
pip install -r requirements-dev.txt
pytest                       # 没有模型时，真实模型相关的测试自动跳过
ruff check .
```

测试覆盖的内容：

- **音频**：重采样后音高和音量不变；不管音频块怎么切，输出都一致。
- **语音检测**：门限和拖尾时长。
- **特征**：mel 窗口和 LiveTalking 的原始实现逐值比对一致。
- **口型引擎**：静音时跳过模型、显示待机帧；说话时只改动脸框内的像素；`lower` 模式下眼睛区域不变。
- **实时流水线**（用替身模型）：稳定输出 25 帧/秒；输入的每个音频采样都按顺序、无缺漏地随画面输出；预览、状态页和 MJPEG 流可以访问。
- **`live` 命令**：完整跑一遍并录成 mp4，检查音画时长一致。
- **显卡分配**（用模拟的强卡和弱卡）：负载重的任务分到快的卡；配置按 UUID 找卡，显卡编号变了也能找对；交互式设置里可以逐项改掉推荐；变声器启动脚本里写入了正确的 UUID；实测口型跑不动的卡会被自动避开。
- **MuseTalk 音频窗口**：流式切出的每一帧 50×384 特征，和 MuseTalk 官方离线实现的取法逐值一致（用可追踪的替身特征验证）；用真实 Whisper 模型时，和离线特征在零偏移处最相似（相似度 0.9 以上），错开一帧就明显变差，说明没有一帧的错位。
- **MuseTalk 真实模型**（CPU 上几分钟）：生成的人脸框和 Wav2Lip 检测到的是同一张脸；说话时嘴部被重绘且逐帧在动；遮罩以上的部分（眼睛等）与原画逐像素一致。
- **诊断与录像**：`doctor` 生成完整报告；电脑上没装 ffmpeg 时，录像改用 Python 包自带的 ffmpeg，音画照样合在一起。
- **形象生成**：S3FD 和 Haar 检测出的人脸框，与示例形象自带的人脸框重合度分别超过 0.8 和 0.6。
- **真实模型**：用真人语音驱动时，嘴部的运动明显大于待机视频本身的运动；离线渲染的音画时长一致。

**没有自动测试的部分**：MuseTalk 和 Wav2Lip 在 GPU 上的实时性能和多卡的真实运行（分配逻辑已用模拟显卡测过）、Windows 脚本、真实麦克风和虚拟声卡、虚拟摄像头输出。这些需要在 Windows 电脑上验证，见 [Windows 首次验收](docs/Windows首次验收.md)。

## 致谢与许可

Apache License 2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

- MuseTalk 推理、人脸框规则、嘴部融合和人脸分割模型来自 [MuseTalk](https://github.com/TMElyralab/MuseTalk)（MIT，Copyright (c) 2024 Tencent Music Entertainment Group）；人脸分割网络最初来自 [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)（MIT）。
- Wav2Lip 口型推理和音频分帧改编自 [LiveTalking](https://github.com/lipku/LiveTalking) 和 [AIAvatar](https://github.com/shibing624/AIAvatar)（均为 Apache-2.0）。
- 模型结构来自 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)，人脸检测来自 [face-alignment](https://github.com/1adrianb/face-alignment)。
- 测试语音片段取自 [LibriSpeech](https://www.openslr.org/12)（CC BY 4.0）。
