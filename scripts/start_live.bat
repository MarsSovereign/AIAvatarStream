@echo off
chcp 65001 >nul
cd /d "%~dp0\.."

rem ================== 按需修改 ==================
rem 变声器输出到的虚拟声卡（运行 list_devices.bat 查看名字）
set INPUT_DEVICE=CABLE Output
rem 形象目录
set AVATAR=data\wav2lip_avatar_glass_man
rem 每批帧数：越小延迟越低。显卡吃力（状态页卡顿次数在涨）就改成 8
set BATCH=6
rem 底噪大、嘴巴在没说话时乱动，就把这个值调高，例如 -38
set VAD_DB=-45
rem 额外参数，例如 --virtualcam 或 --audio-out-device "CABLE-A Input"
set EXTRA=
rem =============================================

start "" http://127.0.0.1:8010/status
.venv\Scripts\python -m avatarstream live --input-device "%INPUT_DEVICE%" --avatar "%AVATAR%" --batch-size %BATCH% --vad-db %VAD_DB% --fp16 %EXTRA%
pause
