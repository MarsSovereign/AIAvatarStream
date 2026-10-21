@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
rem 检查安装情况，报告保存在 output\diagnostics.txt
.venv\Scripts\python -m avatarstream doctor
pause
