@echo off
chcp 65001 >nul
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
call "%SCRIPT_DIR%\.venv\Scripts\activate.bat"
echo Launching the program for the first time may require waiting up to 20 seconds.
python realtime_gui.py
pause
