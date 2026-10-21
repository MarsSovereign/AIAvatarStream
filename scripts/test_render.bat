@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
rem 用自带的测试语音生成一段口型视频，检查画质和显卡速度
if not exist output mkdir output
.venv\Scripts\python -m avatarstream render --audio tests\data\speech_16k.wav --out output\test_render.mp4 --fp16
start "" output\test_render.mp4
pause
