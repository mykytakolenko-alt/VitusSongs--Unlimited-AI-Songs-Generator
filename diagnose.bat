@echo off
cd /d "%~dp0"
set "OUT=%~dp0diagnose.txt"
echo VitusSongs diagnostics > "%OUT%"
echo ---- Windows / GPU >> "%OUT%"
powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption; Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' | driver ' + $_.DriverVersion + ' | VRAM(bytes) ' + $_.AdapterRAM }; 'RAM GB: ' + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1); (Get-CimInstance Win32_Processor).Name" >> "%OUT%" 2>&1
echo ---- Python / packages >> "%OUT%"
if not exist ".venv\Scripts\python.exe" ( echo no .venv found >> "%OUT%" & goto done )
".venv\Scripts\python.exe" --version >> "%OUT%" 2>&1
".venv\Scripts\python.exe" -m pip list 2>nul | findstr /i "torch rocm amd transformers accelerate numpy" >> "%OUT%"
echo ---- PyTorch probe >> "%OUT%"
".venv\Scripts\python.exe" app\probe.py >> "%OUT%" 2>&1
".venv\Scripts\python.exe" -c "import torch;print('torch',torch.__version__,'HIP',torch.version.hip,'cuda_available',torch.cuda.is_available())" >> "%OUT%" 2>&1
echo ---- AMD index reachability >> "%OUT%"
".venv\Scripts\python.exe" -m pip index versions torch --pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ >> "%OUT%" 2>&1
echo ---- Models >> "%OUT%"
dir /b models\YuE2-3B models\YuE2-Vae >> "%OUT%" 2>&1
echo ---- Last server log >> "%OUT%"
if exist app\server.log powershell -NoProfile -Command "Get-Content app\server.log -Tail 40" >> "%OUT%" 2>&1
:done
echo.
echo Done. Please send the file diagnose.txt (in this folder) back.
pause
