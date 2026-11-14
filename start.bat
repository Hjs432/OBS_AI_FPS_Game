@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem Repair PATH: this machine's PATH is missing System32, which breaks
rem fltmc / powershell / timeout / where and makes the window close instantly.
set "PATH=%SystemRoot%\System32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%"

rem ============================================================
rem  Watermelon Aimbot launcher
rem    start.bat         start config UI + run.py
rem    start.bat config  start config UI only
rem ============================================================

rem Admin check: never self-elevate (a failed UAC makes the window vanish).
fltmc >nul 2>&1
if errorlevel 1 (
    echo [WARN] Not running as administrator. Starting with normal privileges.
    echo        If mouse/keys do not work in game, right-click start.bat
    echo        and choose "Run as administrator".
    echo.
)

set "PY=%~dp0venv\Scripts\python.exe"
if exist "%PY%" goto :have_python

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found and venv is missing.
    echo         Install Python 3.12 and create venv first.
    pause
    exit /b 1
)
set "PY=python"

:have_python

if not exist "%~dp0webui\server.py" (
    echo [ERROR] webui\server.py not found.
    pause
    exit /b 1
)
if not exist "%~dp0run.py" (
    echo [ERROR] run.py not found.
    pause
    exit /b 1
)

echo ============================================================
echo   Watermelon Aimbot launcher
echo   Python: %PY%
echo ============================================================

echo [1/2] Starting config UI: http://127.0.0.1:8765
start "Watermelon Config UI" /min "%PY%" "%~dp0webui\server.py" --open
timeout /t 2 /nobreak >nul

if /i "%~1"=="config" (
    echo.
    echo Config UI started. Close the minimized "Watermelon Config UI" window to stop it.
    timeout /t 4 /nobreak >nul
    exit /b 0
)

echo [2/2] Starting run.py
echo ------------------------------------------------------------
"%PY%" -u "%~dp0run.py"
set "RC=%ERRORLEVEL%"

echo.
echo ------------------------------------------------------------
if not "%RC%"=="0" echo [WARN] run.py exited with code %RC%
echo Program exited. Config UI is still running in the background.
pause
