@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
rem 用自带的测试语音分别用 MuseTalk 和 Wav2Lip 生成口型视频，对比画质和显卡速度
if not exist output mkdir output
.venv\Scripts\python -m avatarstream render --engine musetalk --audio tests\data\speech_16k.wav --out output\test_musetalk.mp4
.venv\Scripts\python -m avatarstream render --engine wav2lip --audio tests\data\speech_16k.wav --out output\test_wav2lip.mp4 --fp16
start "" output
pause
