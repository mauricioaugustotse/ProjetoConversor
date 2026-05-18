$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $scriptDir "yt_clip_gui.bat"

if (-not (Test-Path $launcher)) {
    throw "Launcher nao encontrado: $launcher"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Recortar YouTube para Tamanho Alvo.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description = "Recorta videos do YouTube e comprime para tamanho alvo"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,70"
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
