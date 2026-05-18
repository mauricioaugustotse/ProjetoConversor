@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%yt_clip_gui.ps1"

where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%"
) else (
    pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%"
)

endlocal
