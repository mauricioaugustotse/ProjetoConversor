# Testa se a Consulta Publica Unificada do TSE libera a busca por nome usando uma
# COPIA do seu perfil real do Chrome (cookies/login/historico dao reputacao ao
# hCaptcha invisivel). Fecha o Chrome, copia o perfil, relanca com porta de
# depuracao na pagina da consulta e roda a sonda de teste.
#
# ATENCAO: este script FECHA todas as janelas do Chrome. Salve seu trabalho antes.
# Uso:  powershell -ExecutionPolicy Bypass -File pje_testar_captcha.ps1  ["Nome"]

param([string]$Nome = "Jair Bolsonaro")
$ErrorActionPreference = "Stop"
$proj   = "C:\Users\mauri\ProjetoConversor"
$ud     = "$env:LOCALAPPDATA\Google\Chrome\User Data"
$copia  = "$proj\pje_work\chrome_real_profile"
$url    = "https://consultaunificadapje.tse.jus.br/"
$chrome = @(
  "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
  "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
  "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) { Write-Host "Chrome nao encontrado."; exit 1 }

Write-Host ""
Write-Host "=== Teste do hCaptcha com seu perfil real ===" -ForegroundColor Cyan
Write-Host "Vou FECHAR o Chrome agora. Salve seu trabalho." -ForegroundColor Yellow
Read-Host "Tecle ENTER para continuar (ou feche esta janela para cancelar)"

Write-Host "Fechando o Chrome..."
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host "Copiando seu perfil (sem caches; ~alguns segundos)..."
New-Item -ItemType Directory -Force -Path "$copia\Default" | Out-Null
Copy-Item "$ud\Local State" "$copia\Local State" -Force -ErrorAction SilentlyContinue
# copia o perfil Default excluindo pastas de cache pesadas
$xd = @("Cache","Code Cache","GPUCache","GraphiteDawnCache","DawnCache","Service Worker","Component Cache","Crashpad")
robocopy "$ud\Default" "$copia\Default" /E /XD $xd /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null

Write-Host "Lancando Chrome (perfil real + porta 9222)..."
Start-Process $chrome -ArgumentList `
  "--remote-debugging-port=9222", "--user-data-dir=`"$copia`"", `
  "--no-first-run", "--no-default-browser-check", $url
Start-Sleep -Seconds 6

Write-Host "Rodando a sonda de teste..." -ForegroundColor Cyan
Write-Host ""
python "$proj\pje_work\testar_busca.py" $Nome
Write-Host ""
Write-Host "Fim. Se deu SUCESSO, me avise que eu construo o pesquisador em cima disso." -ForegroundColor Green
