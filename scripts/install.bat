@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

rem ---- 按显卡改这一行：RTX 50 系用 cu128，30/40 系用 cu124 ----
set TORCH_INDEX=https://download.pytorch.org/whl/cu124

if not exist .venv (
    echo [1/5] 创建虚拟环境 .venv
    python -m venv .venv || goto :error
)
echo [2/5] 安装 PyTorch（GPU 版）
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url %TORCH_INDEX% || goto :error
echo [3/5] 安装其他依赖
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-optional.txt -r requirements-musetalk.txt || goto :error
echo [4/5] 下载 Wav2Lip、MuseTalk 模型（约 4.3 GB）和示例形象（国内网络可先 set HF_ENDPOINT=https://hf-mirror.com）
.venv\Scripts\python -m avatarstream download --musetalk || goto :error
echo [5/5] 给示例形象生成 MuseTalk 数据
.venv\Scripts\python -m avatarstream prepare-musetalk --avatar data\wav2lip_avatar_glass_man || goto :error
.venv\Scripts\python -c "import torch; print('CUDA 可用:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo 安装完成。下一步：有多张显卡就运行 scripts\setup_gpus.bat 分配显卡；然后运行 scripts\list_devices.bat 找到虚拟声卡的名字。
pause
exit /b 0

:error
echo 安装失败，请把上面的报错发给我。
pause
exit /b 1
