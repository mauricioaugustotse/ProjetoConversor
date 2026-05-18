param(
    [Parameter(Mandatory = $true)][string]$YtDlpPath,
    [switch]$UsePythonModule,
    [Parameter(Mandatory = $true)][string]$FfmpegPath,
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Start,
    [Parameter(Mandatory = $true)][string]$End,
    [Parameter(Mandatory = $true)][string]$OutputDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-StreamingProcess {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ExecutablePath
    $psi.Arguments = Join-Arguments -Args $Arguments
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $false
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    if (-not $process.Start()) {
        throw ("Could not start command: {0}" -f (Split-Path -Leaf $ExecutablePath))
    }

    while ($true) {
        $line = $process.StandardOutput.ReadLine()
        if ($null -eq $line) {
            break
        }

        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $lines.Add($line)
            [Console]::Out.WriteLine($line)
        }
    }

    $process.WaitForExit()

    if ($process.ExitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $process.ExitCode, (Split-Path -Leaf $ExecutablePath))
    }

    return $lines.ToArray()
}

function Escape-Argument {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value)

    if ($Value.Length -eq 0) {
        return '""'
    }

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }

    return $Value
}

function Join-Arguments {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Args)
    return ($Args | ForEach-Object { Escape-Argument -Value $_ }) -join " "
}

if (-not (Test-Path -LiteralPath $YtDlpPath)) {
    throw "yt-dlp executable was not found: $YtDlpPath"
}

if (-not (Test-Path -LiteralPath $FfmpegPath)) {
    throw "ffmpeg executable was not found: $FfmpegPath"
}

if (-not (Test-Path -LiteralPath $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$tempDir = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_full_{0}" -f ([System.Guid]::NewGuid().ToString("N")))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    Write-Output "Downloading full MP4 streams to a temporary folder before cutting locally."
    Write-Output "Temporary folder: $tempDir"

    $ytDlpArgs = @()
    if ($UsePythonModule) {
        $ytDlpArgs += @("-m", "yt_dlp")
    }

    $ytDlpArgs += @(
        "--newline",
        "--no-playlist",
        "--force-overwrites",
        "--replace-in-metadata", "title", "\s*[|｜]\s*.+$", "",
        "--replace-in-metadata", "title", "\s*[：:]\s*", " - ",
        "--paths", $tempDir,
        "--windows-filenames",
        "--ffmpeg-location", $FfmpegPath,
        "--print", "after_move:filepath",
        "-o", "%(title)s.%(ext)s",
        "-f", "bv*[height<=720][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4][vcodec^=avc1][acodec!=none]/18/bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1]",
        "--merge-output-format", "mp4",
        $Url
    )

    $downloadLines = Invoke-StreamingProcess -ExecutablePath $YtDlpPath -Arguments $ytDlpArgs

    $sourcePath = $downloadLines |
        Where-Object { ($_ -match '\.mp4$') -and (Test-Path -LiteralPath $_) } |
        Select-Object -Last 1

    if ([string]::IsNullOrWhiteSpace($sourcePath)) {
        $sourcePath = Get-ChildItem -LiteralPath $tempDir -File -Filter "*.mp4" |
            Sort-Object -Property LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }

    if ([string]::IsNullOrWhiteSpace($sourcePath) -or -not (Test-Path -LiteralPath $sourcePath)) {
        throw "Could not locate the temporary MP4 downloaded by yt-dlp."
    }

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($sourcePath)
    $finalPath = Join-Path -Path $OutputDir -ChildPath ($baseName + ".mp4")

    Write-Output "Cutting locally with ffmpeg from $Start to $End."

    $ffmpegArgs = @(
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-nostdin",
        "-y",
        "-ss", $Start,
        "-to", $End,
        "-i", $sourcePath,
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-sn",
        "-dn",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        $finalPath
    )

    Invoke-StreamingProcess -ExecutablePath $FfmpegPath -Arguments $ffmpegArgs | Out-Null

    if (-not (Test-Path -LiteralPath $finalPath)) {
        throw "ffmpeg finished but the final MP4 was not found: $finalPath"
    }

    Write-Output "Final MP4 created:"
    Write-Output $finalPath
}
finally {
    try {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }
    catch {
        Write-Output "Warning: could not remove temporary folder: $tempDir"
    }
}
