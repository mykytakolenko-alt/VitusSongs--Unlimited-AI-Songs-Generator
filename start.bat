@echo off
cd /d "%~dp0"
title VitusSongs (console)
if not exist ".venv\Scripts\python.exe" (
  echo Please run install.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app\server.py
if errorlevel 1 pause
