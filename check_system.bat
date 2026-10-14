@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" ( echo Please run install.bat first. & pause & exit /b 1 )
".venv\Scripts\python.exe" app\probe.py
set YUE2_KIT=%~dp0
set PYTHONPATH=%~dp0src
".venv\Scripts\python.exe" -m yue2.cli doctor
pause
