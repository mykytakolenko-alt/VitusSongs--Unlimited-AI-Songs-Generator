@echo off
setlocal
cd /d "%~dp0"
title VitusSongs - Setup
echo ============================================================
echo   VitusSongs - setup (run once)
echo ============================================================
echo.

rem ---------- 1. Python 3.12
set "PYEXE="
py -3.12 --version >nul 2>nul && set "PYEXE=py -3.12"
if not defined PYEXE goto nopython

rem ---------- 2. GPU detection
powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name -join '; '" > "%TEMP%\yue_gpus.txt" 2>nul
set "GPUS="
set /p GPUS=<"%TEMP%\yue_gpus.txt"
set "MODE=cpu"
set "GFX=none"
for /f "tokens=1,2" %%a in ('powershell -NoProfile -ExecutionPolicy Bypass -File app\detect_gpu.ps1') do (set "MODE=%%a" & set "GFX=%%b")

rem command line override:  install.bat cpu | cuda | amd gfx1032
if /i "%~1"=="cpu" set "MODE=cpu"
if /i "%~1"=="cuda" set "MODE=cuda"
if /i "%~1"=="amd" (set "MODE=amd" & if not "%~2"=="" set "GFX=%~2")
if /i "%~1"=="rocm6000" (set "MODE=amd" & set "GFX=gfx1032")
if /i "%~1"=="rocm7000" (set "MODE=amd" & set "GFX=gfx1100")
if /i "%~1"=="rocm9000" (set "MODE=amd" & set "GFX=gfx1201")

echo Detected: %GPUS%
echo Install mode: %MODE%   GPU target: %GFX%
echo.
echo   cuda  - NVIDIA GeForce / RTX
echo   amd   - AMD Radeon (ROCm for Windows, experimental; target such as gfx1032 = RX 6600)
echo   cpu   - processor only (works everywhere, very slow)
echo.
set "OVR="
set /p OVR=Press Enter to accept "%MODE%", or type cuda / amd / cpu: 
if not "%OVR%"=="" set "MODE=%OVR%"
if /i "%MODE%"=="amd" if /i "%GFX%"=="none" set /p GFX=Enter your GPU target (RX 6600 = gfx1032, RX 7900 = gfx1100, Radeon VII = gfx906): 
echo.

rem ---------- 3. Environment
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PYEXE% -m venv .venv
  if errorlevel 1 goto fail
)
set "VPY=.venv\Scripts\python.exe"
set "PIP_DEFAULT_TIMEOUT=60"
set "PIP_RETRIES=10"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
"%VPY%" -m pip install --upgrade pip

rem ---------- 4. PyTorch (remove any old build first, otherwise pip keeps it)
echo Removing any previously installed PyTorch / DirectML packages...
"%VPY%" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul
echo Installing PyTorch for mode %MODE% (several GB, this takes a while)...
if /i "%MODE%"=="cuda" goto t_cuda
if /i "%MODE%"=="amd" goto t_amd
goto t_cpu

:t_cuda
"%VPY%" -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && goto t_done
"%VPY%" -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126 && goto t_done
goto t_fallback

:t_amd
echo Installing PyTorch dependencies from PyPI...
"%VPY%" -m pip install numpy sympy networkx jinja2 fsspec filelock typing-extensions setuptools
echo.
echo Trying AMD multi-arch ROCm index for %GFX% ...
"%VPY%" -m pip install --no-cache-dir --pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ "torch[device-%GFX%]"
call :check_hip && goto t_done
echo.
echo [!] Not a ROCm build yet. Trying the older gfx103X index (RX 6000 only)...
echo %GFX%| findstr /b /c:"gfx103" >nul
if errorlevel 1 goto t_fallback
"%VPY%" -m pip uninstall -y torch >nul 2>nul
"%VPY%" -m pip install --no-cache-dir --pre --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/ torch
call :check_hip && goto t_done
goto t_fallback

:t_fallback
echo.
echo [!] The GPU build of PyTorch could not be installed (AMD package indexes change often).
echo     Installing the CPU version so the app still works. Run diagnose.bat and send diagnose.txt.
"%VPY%" -m pip uninstall -y torch >nul 2>nul
:t_cpu
"%VPY%" -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail

:t_done
echo.
echo Installing the remaining libraries...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Accelerator check:
"%VPY%" app\probe.py
"%VPY%" -c "import torch;print('torch',torch.__version__,'| HIP:',torch.version.hip,'| CUDA:',torch.version.cuda)"
echo.
echo If it says "backend": "cpu" although you have a GPU - update the GPU driver, reboot,
echo then run diagnose.bat and send me diagnose.txt.

powershell -NoProfile -ExecutionPolicy Bypass -File create_shortcut.ps1 >nul 2>nul
echo A "VitusSongs" shortcut was placed on your Desktop (opens the app in its own window).

echo.
set "DL="
set /p DL=Download the models now (about 8-10 GB)? [Y/N] (you can also do it later in the app): 
if /i "%DL%"=="Y" "%VPY%" app\download_models.py

echo.
echo ============================================================
echo   Done! Open VitusSongs from the Desktop shortcut or start.bat
echo ============================================================
pause
exit /b 0

:check_hip
"%VPY%" -c "import sys,torch;sys.exit(0 if torch.version.hip else 1)" >nul 2>nul
exit /b %errorlevel%

:nopython
echo Python 3.12 not found. Trying to install it with winget...
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
echo.
echo CLOSE this window and run install.bat again.
echo If that failed, install Python 3.12 manually: https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH" and "py launcher")
pause
exit /b 1

:fail
echo.
echo [ERROR] Setup aborted. Copy the text above and send it for analysis.
pause
exit /b 1
