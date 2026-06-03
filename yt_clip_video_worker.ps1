param(
    [Parameter(Mandatory = $true)][string]$YtDlpPath,
    [switch]$UsePythonModule,
    [Parameter(Mandatory = $true)][string]$FfmpegPath,
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Start,
    [Parameter(Mandatory = $true)][string]$End,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [AllowEmptyString()][string]$ClipSuffix = "",
    [AllowEmptyString()][string]$ProgressLogFile = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-ProgressLine {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Line)

    [Console]::Out.WriteLine($Line)

    if ([string]::IsNullOrWhiteSpace($ProgressLogFile)) {
        return
    }

    try {
        [System.IO.File]::AppendAllText($ProgressLogFile, $Line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    catch {
        # Keep the worker running even if progress logging fails.
    }
}

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
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    if (-not $process.Start()) {
        throw ("Could not start command: {0}" -f (Split-Path -Leaf $ExecutablePath))
    }

    $stderrTask = $process.StandardError.ReadToEndAsync()

    while ($true) {
        $line = $process.StandardOutput.ReadLine()
        if ($null -eq $line) {
            break
        }

        if (-not [string]::IsNullOrWhiteSpace($line)) {
            $lines.Add($line)
            Write-ProgressLine -Line $line
        }
    }

    $process.WaitForExit()

    $stderrText = $stderrTask.Result
    if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
        foreach ($line in ($stderrText -split "\r?\n")) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $lines.Add($line)
                Write-ProgressLine -Line $line
            }
        }
    }

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

function Convert-TimecodeToSeconds {
    param([Parameter(Mandatory = $true)][string]$Value)

    $parts = $Value.Trim().Split(":")
    if ($parts.Count -ne 3) {
        throw "Invalid timecode received by worker: $Value"
    }

    $hours = [double]::Parse($parts[0], [System.Globalization.CultureInfo]::InvariantCulture)
    $minutes = [double]::Parse($parts[1], [System.Globalization.CultureInfo]::InvariantCulture)
    $seconds = [double]::Parse($parts[2], [System.Globalization.CultureInfo]::InvariantCulture)
    return ($hours * 3600) + ($minutes * 60) + $seconds
}

function Format-SecondsForFfmpeg {
    param([Parameter(Mandatory = $true)][double]$TotalSeconds)

    $rounded = [Math]::Round($TotalSeconds, 3)
    $hours = [int][Math]::Floor($rounded / 3600)
    $minutes = [int][Math]::Floor(($rounded % 3600) / 60)
    $seconds = $rounded - ($hours * 3600) - ($minutes * 60)

    return [string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        "{0:00}:{1:00}:{2:00.###}",
        $hours,
        $minutes,
        $seconds
    )
}

function Get-QueryParameter {
    param(
        [Parameter(Mandatory = $true)][System.Uri]$Uri,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if ([string]::IsNullOrWhiteSpace($Uri.Query)) {
        return $null
    }

    foreach ($part in $Uri.Query.TrimStart("?").Split("&")) {
        if ([string]::IsNullOrWhiteSpace($part)) {
            continue
        }

        $pair = $part -split "=", 2
        $key = [System.Uri]::UnescapeDataString($pair[0].Replace("+", " "))
        if ($key -ine $Name) {
            continue
        }

        if ($pair.Count -lt 2) {
            return ""
        }

        return [System.Uri]::UnescapeDataString($pair[1].Replace("+", " "))
    }

    return $null
}

function Normalize-YouTubeUrl {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$InputText)

    $raw = $InputText.Trim()
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "URL cannot be empty."
    }

    [System.Uri]$uri = $null
    $isUri = [System.Uri]::TryCreate($raw, [System.UriKind]::Absolute, [ref]$uri)
    if (-not $isUri -or (@("http", "https") -notcontains $uri.Scheme.ToLowerInvariant())) {
        throw "Please provide a valid URL, for example https://www.youtube.com/watch?v=... or https://youtu.be/..."
    }

    $urlHost = $uri.Host.ToLowerInvariant()
    $isShortHost = ($urlHost -eq "youtu.be") -or ($urlHost -eq "www.youtu.be")
    $isYoutubeHost = ($urlHost -eq "youtube.com") -or $urlHost.EndsWith(".youtube.com")

    if (-not ($isShortHost -or $isYoutubeHost)) {
        throw "Please provide a YouTube video URL."
    }

    $videoId = $null
    $pathParts = @($uri.AbsolutePath.Trim("/").Split("/") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

    if ($isShortHost) {
        if ($pathParts.Count -gt 0) {
            $videoId = $pathParts[0]
        }
    }
    elseif ($pathParts.Count -gt 0) {
        switch ($pathParts[0].ToLowerInvariant()) {
            "watch" {
                $videoId = Get-QueryParameter -Uri $uri -Name "v"
                break
            }
            { @("shorts", "embed", "v", "live") -contains $_ } {
                if ($pathParts.Count -gt 1) {
                    $videoId = $pathParts[1]
                }
                break
            }
        }
    }

    if ($videoId -notmatch '^[A-Za-z0-9_-]{11}$') {
        throw "Please provide a direct YouTube video URL (watch?v=..., youtu.be/..., /shorts/...). Channel, playlist, and home URLs are not supported."
    }

    return "https://www.youtube.com/watch?v={0}" -f [System.Uri]::EscapeDataString($videoId)
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

try {
    $Url = Normalize-YouTubeUrl -InputText $Url
}
catch {
    Write-ProgressLine -Line ("ERROR: {0}" -f $_.Exception.Message)
    exit 1
}

$tempDir = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_window_{0}" -f ([System.Guid]::NewGuid().ToString("N")))
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    Write-ProgressLine -Line "Downloading a synced source MP4 window to a temporary folder before cutting locally."
    Write-ProgressLine -Line "Temporary folder: $tempDir"

    $startSeconds = Convert-TimecodeToSeconds -Value $Start
    $endSeconds = Convert-TimecodeToSeconds -Value $End
    $durationSeconds = $endSeconds - $startSeconds
    if ($durationSeconds -le 0) {
        throw "Invalid cut duration calculated from $Start to $End."
    }

    $paddingSeconds = 30.0
    $windowStartSeconds = [Math]::Max(0, $startSeconds - $paddingSeconds)
    $windowEndSeconds = $endSeconds + $paddingSeconds
    $localStartSeconds = $startSeconds - $windowStartSeconds

    $windowStartText = Format-SecondsForFfmpeg -TotalSeconds $windowStartSeconds
    $windowEndText = Format-SecondsForFfmpeg -TotalSeconds $windowEndSeconds
    $localStartText = Format-SecondsForFfmpeg -TotalSeconds $localStartSeconds
    $durationText = Format-SecondsForFfmpeg -TotalSeconds $durationSeconds
    $downloadSection = "*{0}-{1}" -f $windowStartText, $windowEndText

    Write-ProgressLine -Line "Downloading only a small source window instead of the full video."
    Write-ProgressLine -Line ("Requested clip: {0} to {1} ({2})." -f $Start, $End, $durationText)
    Write-ProgressLine -Line ("Source window: {0} to {1}; final cut starts {2} inside that window." -f $windowStartText, $windowEndText, $localStartText)

    $ytDlpArgs = @()
    if ($UsePythonModule) {
        $ytDlpArgs += @("-m", "yt_dlp")
    }

    $ytDlpArgs += @(
        "--newline",
        "--no-playlist",
        "--force-overwrites",
        "--download-sections", $downloadSection,
        "--force-keyframes-at-cuts",
        "--replace-in-metadata", "title", "\s*[|｜]\s*.+$", "",
        "--replace-in-metadata", "title", "\s*[：:]\s*", " - ",
        "--replace-in-metadata", "title", "[/\\]+", "-",
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
    $safeClipSuffix = $ClipSuffix.Trim()
    $finalFileName = if ([string]::IsNullOrWhiteSpace($safeClipSuffix)) {
        $baseName + ".mp4"
    }
    else {
        "{0} [{1}].mp4" -f $baseName, $safeClipSuffix
    }
    $finalPath = Join-Path -Path $OutputDir -ChildPath $finalFileName

    Write-ProgressLine -Line "Cutting locally with ffmpeg from the downloaded source window."
    Write-ProgressLine -Line ("Precise local cut: {0} for {1}." -f $localStartText, $durationText)
    Write-ProgressLine -Line "Using timestamp reset to keep audio and video synchronized."
    Write-ProgressLine -Line "Normalizing audio loudness without changing audio/video timing."

    $ffmpegArgs = @(
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-nostdin",
        "-y",
        "-i", $sourcePath,
        "-ss", $localStartText,
        "-t", $durationText,
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-sn",
        "-dn",
        "-vf", "setpts=PTS-STARTPTS",
        "-af", "aresample=async=1:first_pts=0,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-b:a", "160k",
        "-movflags", "+faststart",
        "-avoid_negative_ts", "make_zero",
        $finalPath
    )

    Invoke-StreamingProcess -ExecutablePath $FfmpegPath -Arguments $ffmpegArgs | Out-Null

    if (-not (Test-Path -LiteralPath $finalPath)) {
        throw "ffmpeg finished but the final MP4 was not found: $finalPath"
    }

    Write-ProgressLine -Line "Final MP4 created:"
    Write-ProgressLine -Line $finalPath
}
catch {
    Write-ProgressLine -Line ("ERROR: {0}" -f $_.Exception.Message)
    exit 1
}
finally {
    try {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }
    catch {
        Write-ProgressLine -Line "Warning: could not remove temporary folder: $tempDir"
    }
}
