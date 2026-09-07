@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
call "%SCRIPT_DIR%\.venv\Scripts\activate.bat"
set "GRADIO_ANALYTICS_ENABLED=False"
set "NO_PROXY=localhost,127.0.0.1,::1,%NO_PROXY%"
echo Launching the program for the first time may require waiting up to 20 seconds.
runtime\python.exe -I webui.py --pycmd runtime\python.exe --port 7897
pause
