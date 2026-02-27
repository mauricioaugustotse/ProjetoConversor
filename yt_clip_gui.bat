@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%yt_clip_gui.ps1"

where pwsh >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%"
) else (
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1_FILE%"
)

endlocal
