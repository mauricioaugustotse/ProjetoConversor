$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Aponta direto para o pythonw (sem console, sem flash de cmd).
$pythonw = "$env:LOCALAPPDATA\Programs\Python\Python313\pythonw.exe"
$gui = Join-Path $scriptDir "insta_baixar_gui.py"
$icone = Join-Path $scriptDir "icones\insta_baixar.ico"

if (-not (Test-Path $pythonw)) {
    throw "pythonw nao encontrado: $pythonw"
}
if (-not (Test-Path $gui)) {
    throw "GUI nao encontrada: $gui"
}
if (-not (Test-Path $icone)) {
    throw "Icone nao encontrado: $icone (rode: py _gerar_icones.py)"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Baixar do Instagram.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "`"$gui`""
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description = "Baixa videos do Instagram (link colado) para a pasta Downloads via yt-dlp"
$shortcut.IconLocation = "$icone,0"
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
