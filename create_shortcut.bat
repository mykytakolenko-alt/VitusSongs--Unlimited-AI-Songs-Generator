@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File create_shortcut.ps1
echo Desktop shortcut created.
pause
