@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
rem 测试每张显卡，按任务负载推荐分配方案，再让你逐项确认或修改
.venv\Scripts\python -m avatarstream gpus --setup --test-lipsync
pause
