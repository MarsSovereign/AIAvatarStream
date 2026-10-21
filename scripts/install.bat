@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0\.."

rem ---- 按显卡改这一行：RTX 50 系用 cu128，30/40 系用 cu124 ----
set TORCH_INDEX=https://download.pytorch.org/whl/cu124

if not exist .venv (
    echo [1/4] 创建虚拟环境 .venv
    python -m venv .venv || goto :error
)
echo [2/4] 安装 PyTorch（GPU 版）
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch torchvision --index-url %TORCH_INDEX% || goto :error
echo [3/4] 安装其他依赖
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-optional.txt || goto :error
echo [4/4] 下载模型和示例形象（国内网络可先 set HF_ENDPOINT=https://hf-mirror.com）
.venv\Scripts\python -m avatarstream download || goto :error
.venv\Scripts\python -c "import torch; print('CUDA 可用:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo.
echo 安装完成。下一步：运行 scripts\list_devices.bat 找到虚拟声卡的名字。
pause
exit /b 0

:error
echo 安装失败，请把上面的报错发给我。
pause
exit /b 1
