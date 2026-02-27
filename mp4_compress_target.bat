@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1_FILE=%SCRIPT_DIR%mp4_compress_target.ps1"

set "INPUT_ARG="
if not "%~1"=="" (
    set "INPUT_ARG=%~1"
)

where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    if defined INPUT_ARG (
        powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%" -InputFile "%INPUT_ARG%"
    ) else (
        powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%"
    )
) else (
    if defined INPUT_ARG (
        pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%" -InputFile "%INPUT_ARG%"
    ) else (
        pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Sta -File "%PS1_FILE%"
    )
)

endlocal
