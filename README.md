# AIAvatarStream

真人驱动的数字人直播：**你对着麦克风说话，变声器把声音换成数字人的声音，数字人实时跟着对口型，OBS 推到直播间。**

```
你的麦克风 ──► 变声器（RVC / Seed-VC）──► 虚拟声卡
                                          │
                     ┌────────────────────┴───────────────────┐
                     ▼                                        ▼
              AIAvatarStream                         OBS 直接采集变声后的声音
      语音检测 → Wav2Lip 口型 → 合成画面           （设好同步偏移，和口型对齐）
                     │
                     ▼
     OBS 浏览器源 / 虚拟摄像头  ──►  直播伴侣 / 推流
```

画面里始终是数字人形象，声音始终是数字人的音色，但说话的人是你本人。这符合抖音等平台对数字人直播“真人驱动、实时互动”的要求（详见[合规](#合规与授权)）。

## 它做什么，不做什么

| 做 | 不做 |
|---|---|
| 把实时音频变成对上口型的数字人画面 | 不自动回复、不接大模型：内容全部由你本人说 |
| 不说话时播放形象的待机视频，不占显卡 | 不内置变声：用成熟的 RVC 或 Seed-VC，通过虚拟声卡接进来 |
| 输出给 OBS：浏览器源、虚拟摄像头，或录成 mp4 | 不直接推流：推流交给 OBS 和直播伴侣 |
| 状态页实时显示延迟、音量和 OBS 该设的同步偏移 | |
| 用一段闭嘴视频生成你自己的数字人形象 | |

## 硬件要求

- Windows 10/11，NVIDIA 显卡。口型模型本身在 RTX 3060 上约 60 帧/秒，实时只需要 25 帧/秒。
- 变声器和口型要**同时**跑在一张卡上，建议 **12 GB 显存起步**，RTX 4060 Ti 16G、4070 或更好会更稳。
- 一个虚拟声卡，比如免费的 [VB-CABLE](https://vb-audio.com/Cable/)。
- 没有 N 卡也能装、能离线生成视频，但跑不动实时（CPU 大约 2 帧/秒）。

## 安装

### Windows 一键安装

1. 装 [Python 3.10 或 3.11](https://www.python.org/downloads/)，安装时勾选 “Add python.exe to PATH”。
2. 下载本项目，双击 `scripts\install.bat`。
   - 它会创建 `.venv`、装 GPU 版 PyTorch 和依赖，再下载模型（215 MB）和示例形象。
   - RTX 50 系显卡先用记事本打开 `install.bat`，把 `cu124` 改成 `cu128`。
   - 国内下载 HuggingFace 慢的话，先在命令行执行 `set HF_ENDPOINT=https://hf-mirror.com` 再运行。
3. 双击 `scripts\test_render.bat`：用自带的测试语音生成 `output\test_render.mp4`，看看口型效果和速度。

### 手动安装（任何系统）

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS / Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # 按显卡选
pip install -r requirements.txt -r requirements-optional.txt
python -m avatarstream download   # 模型和示例形象；--avatar all 下载全部三个
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
python -m avatarstream devices                       # 列出音频设备
python -m avatarstream download --avatar all          # 下载模型和三个示例形象
python -m avatarstream make-avatar --video me.mp4 --out data/my_avatar
python -m avatarstream render --audio a.wav --out out.mp4         # 离线生成，检查效果
python -m avatarstream live --input-device "CABLE Output" --fp16  # 开始直播
```

`live` 的常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input-device` | 系统默认 | 变声器输出到的设备，写编号或名字片段 |
| `--avatar` | `data/wav2lip_avatar_glass_man` | 形象目录 |
| `--batch-size` | 8 | 每批生成多少帧。越小延迟越低，越大越省显卡。直播建议 4 到 8 |
| `--fp16` | 关 | N 卡半精度，更快更省显存，推荐打开 |
| `--vad-db` | -45 | 音量高于它才算说话。没说话嘴也在动，就调高到 -40 或 -38 |
| `--blend` | `lower` | `lower` 只换嘴和下巴，眼睛保持原画质；`full` 换整张脸 |
| `--virtualcam` | 关 | 同时输出到 OBS 虚拟摄像头（需要 `pyvirtualcam`） |
| `--audio-out-device` | 无 | 把和画面严格对齐的声音送到另一个输出设备，OBS 采集它就不用设偏移 |
| `--input-file` | 无 | 用音频文件代替麦克风，彩排用，加 `--loop` 循环 |
| `--record` | 无 | 同时录一份 mp4 |
| `--port` | 8010 | 预览和状态页端口 |

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

1. 拍或生成一段 **10 到 30 秒、正脸、闭着嘴、有轻微自然动作**的视频。眨眼、轻微点头都可以，不要张嘴说话。
   - 最省事也最安全：用图生视频工具（可灵、MiniMax、Wan 等）生成一个**不存在的虚拟人**，没有肖像权问题。
2. 生成形象：
   ```bash
   python -m avatarstream make-avatar --video idle.mp4 --out data/my_avatar --resize-height 1280
   ```
   - 默认用 S3FD 检测人脸（第一次会自动下载 85 MB 权重），`--detector haar` 不用下载但不如它准。
   - 下巴被切掉的话，加大下边距：`--pads 0 20 0 0`。
3. 开播时加 `--avatar data/my_avatar`。

待机画面会正放、倒放循环，所以视频首尾不需要刻意衔接。

## 合规与授权

- **形象和声音必须有授权。** 不要复刻明星或任何真实他人的长相和声音，否则侵犯肖像权和声音权益。推荐 AI 生成的虚拟人，配你自己训练或有授权的音色。
- **平台规则（以平台最新公告为准）。** 抖音对虚拟人直播的要求是：使用者实名认证、由真人驱动并实时互动、不允许完全由 AI 驱动，直播间要显著标识数字人。本项目的用法是真人说话驱动，但标识、实名这些需要你在平台侧完成。视频号等平台的规则可能更严，开播前请核实。
- **AI 生成内容标识。** 《人工智能生成合成内容标识办法》自 2025 年 9 月 1 日起施行，直播画面里请加上“数字人 / AI 生成”的提示。
- **模型权重。** `wav2lip.pth` 来自 [shibing624/ai-avatar-wav2lip](https://huggingface.co/shibing624/ai-avatar-wav2lip)，标注为 Apache-2.0。但 Wav2Lip 原作者声明，用 LRS2 数据集训练的官方模型禁止商用，这个 256 版本用什么数据训练的没有公开说明。**带货等商业用途前，请自行评估或换用授权明确的模型。**

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
- **形象生成**：S3FD 和 Haar 检测出的人脸框，与示例形象自带的人脸框重合度分别超过 0.8 和 0.6。
- **真实模型**：用真人语音驱动时，嘴部的运动明显大于待机视频本身的运动；离线渲染的音画时长一致。

**没有自动测试的部分**：GPU 上的实时性能、真实麦克风和虚拟声卡、虚拟摄像头输出。这些需要在你的 Windows 电脑上验证，步骤见[直播搭建指南](docs/直播搭建指南.md)里的“开播前检查”。

## 致谢与许可

Apache License 2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。

- 口型推理和音频分帧改编自 [LiveTalking](https://github.com/lipku/LiveTalking) 和 [AIAvatar](https://github.com/shibing624/AIAvatar)（均为 Apache-2.0）。
- 模型结构来自 [Wav2Lip](https://github.com/Rudrabha/Wav2Lip)，人脸检测来自 [face-alignment](https://github.com/1adrianb/face-alignment)。
- 测试语音片段取自 [LibriSpeech](https://www.openslr.org/12)（CC BY 4.0）。
