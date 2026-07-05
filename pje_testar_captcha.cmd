@echo off
REM Testa se a busca por nome na Consulta Publica Unificada do TSE passa no
REM hCaptcha usando uma copia do seu perfil real do Chrome. Fecha o Chrome (com
REM aviso), copia o perfil, relanca com porta de depuracao e roda a sonda.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0pje_testar_captcha.ps1" %*
echo.
pause
