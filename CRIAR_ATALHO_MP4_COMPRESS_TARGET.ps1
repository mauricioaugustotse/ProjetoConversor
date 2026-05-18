$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$launcher = Join-Path $scriptDir "mp4_compress_target.bat"

if (-not (Test-Path $launcher)) {
    throw "Launcher nao encontrado: $launcher"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Reduzir MP4 para Tamanho Alvo.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcher
$shortcut.WorkingDirectory = $scriptDir
$shortcut.Description = "Reduz videos MP4 para quantidade de MB alvo"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,70"
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
