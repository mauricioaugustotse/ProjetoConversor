Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

param(
    [string]$VideoPath
)

function Resolve-BrowserPath {
    $candidates = @(
        "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "C:\Program Files\Google\Chrome\Application\chrome.exe",
        "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "Microsoft Edge ou Google Chrome nao encontrado."
}

function Select-VideoPath {
    Add-Type -AssemblyName System.Windows.Forms

    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Selecionar video"
    $dialog.InitialDirectory = [Environment]::GetFolderPath("UserProfile") + "\Downloads"
    $dialog.Filter = "Videos|*.mp4;*.mkv;*.mov;*.webm;*.wmv;*.avi|Todos os arquivos|*.*"
    $dialog.Multiselect = $false

    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        exit 0
    }

    return $dialog.FileName
}

if ([string]::IsNullOrWhiteSpace($VideoPath)) {
    $VideoPath = Select-VideoPath
}

$VideoPath = $VideoPath.Trim().Trim('"')
if (-not (Test-Path -LiteralPath $VideoPath)) {
    throw "Arquivo nao encontrado: $VideoPath"
}

$browserPath = Resolve-BrowserPath
$resolvedPath = (Resolve-Path -LiteralPath $VideoPath).Path
$fileUri = ([System.Uri]::new($resolvedPath)).AbsoluteUri

Start-Process -FilePath $browserPath -ArgumentList @($fileUri)
