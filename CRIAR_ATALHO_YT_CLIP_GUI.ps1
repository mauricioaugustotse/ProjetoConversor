$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# Lancador oculto (vbs): abre a GUI sem janela de console.
$launcher = Join-Path $scriptDir "yt_clip_gui_oculto.vbs"

if (-not (Test-Path $launcher)) {
    throw "Launcher nao encontrado: $launcher"
}

$shell = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktop "Recortar YouTube.lnk"
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "$env:SystemRoot\System32\wscript.exe"
$shortcut.Arguments = "`"$launcher`""
$shortcut.WorkingDirectory = $scriptDir
$shortcut.WindowStyle = 7
$shortcut.Description = "Recorta videos do YouTube e comprime para tamanho alvo"
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,115"
$shortcut.Save()

Write-Host "Atalho criado em: $shortcutPath"
