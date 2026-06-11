param(
    [Parameter(Mandatory = $true)][string]$YtDlpPath,
    [switch]$UsePythonModule,
    [Parameter(Mandatory = $true)][string]$FfmpegPath,
    [Parameter(Mandatory = $true)][string]$Url,
    # Lista de trechos no formato canonico "HH:MM:SS.mmm-HH:MM:SS.mmm" separados por ";".
    [AllowEmptyString()][string]$Segments = "",
    # Compatibilidade com chamadas antigas de trecho unico.
    [AllowEmptyString()][string]$Start = "",
    [AllowEmptyString()][string]$End = "",
    [switch]$JoinSegments,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [AllowEmptyString()][string]$ProgressLogFile = "",
    [int]$MaxParallel = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:CrossfadeSeconds = 0.5

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

function Invoke-StreamingProcess {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments,
        [AllowEmptyString()][string]$LinePrefix = ""
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
            Write-ProgressLine -Line ($LinePrefix + $line)
        }
    }

    $process.WaitForExit()

    $stderrText = $stderrTask.Result
    if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
        foreach ($line in ($stderrText -split "\r?\n")) {
            if (-not [string]::IsNullOrWhiteSpace($line)) {
                $lines.Add($line)
                Write-ProgressLine -Line ($LinePrefix + $line)
            }
        }
    }

    if ($process.ExitCode -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $process.ExitCode, (Split-Path -Leaf $ExecutablePath))
    }

    return $lines.ToArray()
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

function Format-SecondsForFileName {
    param([Parameter(Mandatory = $true)][double]$TotalSeconds)

    $rounded = [Math]::Round($TotalSeconds)
    $hours = [int][Math]::Floor($rounded / 3600)
    $minutes = [int][Math]::Floor(($rounded % 3600) / 60)
    $seconds = [int]($rounded % 60)

    if ($hours -gt 0) {
        return ("{0:00}h{1:00}m{2:00}s" -f $hours, $minutes, $seconds)
    }

    return ("{0:00}m{1:00}s" -f $minutes, $seconds)
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

function Build-YtDlpSectionArgs {
    param(
        [Parameter(Mandatory = $true)][string]$DownloadSection,
        [Parameter(Mandatory = $true)][string]$TargetDir
    )

    $argsList = @()
    if ($UsePythonModule) {
        $argsList += @("-m", "yt_dlp")
    }

    $argsList += @(
        "--newline",
        "--no-playlist",
        "--force-overwrites",
        "--download-sections", $DownloadSection,
        "--force-keyframes-at-cuts",
        "--replace-in-metadata", "title", "\s*[|｜]\s*.+$", "",
        "--replace-in-metadata", "title", "\s*[：:]\s*", " - ",
        "--replace-in-metadata", "title", "[/\\]+", "-",
        "--paths", $TargetDir,
        "--windows-filenames",
        "--ffmpeg-location", $FfmpegPath,
        "--print", "after_move:filepath",
        "-o", "%(title)s.%(ext)s",
        "-f", "bv*[height<=720][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4][vcodec^=avc1][acodec!=none]/18/bv*[ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[ext=mp4][vcodec^=avc1]",
        "--merge-output-format", "mp4",
        $Url
    )

    return $argsList
}

function Find-DownloadedMp4 {
    param(
        [Parameter(Mandatory = $true)][string[]]$OutputLines,
        [Parameter(Mandatory = $true)][string]$TargetDir
    )

    $sourcePath = $OutputLines |
        Where-Object { ($_ -match '\.mp4$') -and (Test-Path -LiteralPath $_) } |
        Select-Object -Last 1

    if ([string]::IsNullOrWhiteSpace($sourcePath)) {
        $sourcePath = Get-ChildItem -LiteralPath $TargetDir -File -Filter "*.mp4" -ErrorAction SilentlyContinue |
            Sort-Object -Property LastWriteTime -Descending |
            Select-Object -First 1 -ExpandProperty FullName
    }

    return $sourcePath
}

function Invoke-FfmpegCut {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][double]$LocalStartSeconds,
        [Parameter(Mandatory = $true)][double]$DurationSeconds,
        [Parameter(Mandatory = $true)][string]$TargetPath,
        [AllowEmptyString()][string]$LinePrefix = ""
    )

    $localStartText = Format-SecondsForFfmpeg -TotalSeconds $LocalStartSeconds
    $durationText = Format-SecondsForFfmpeg -TotalSeconds $DurationSeconds

    $ffmpegArgs = @(
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-nostdin",
        "-y",
        "-i", $SourcePath,
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
        $TargetPath
    )

    Invoke-StreamingProcess -ExecutablePath $FfmpegPath -Arguments $ffmpegArgs -LinePrefix $LinePrefix | Out-Null

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        throw "ffmpeg finished but the MP4 was not found: $TargetPath"
    }
}

function Join-ClipsWithCrossfade {
    param(
        [Parameter(Mandatory = $true)][object[]]$Clips,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    if ($Clips.Count -lt 2) {
        throw "Join requires at least 2 clips."
    }

    $fade = $script:CrossfadeSeconds
    $ffmpegArgs = @(
        "-hide_banner",
        "-loglevel", "error",
        "-progress", "pipe:1",
        "-nostats",
        "-nostdin",
        "-y"
    )

    foreach ($clip in $Clips) {
        $ffmpegArgs += @("-i", $clip.Path)
    }

    # Encadeia xfade (video) e acrossfade (audio): a cada transicao o offset e a
    # duracao acumulada menos o tempo de fade.
    $videoChain = New-Object System.Collections.Generic.List[string]
    $audioChain = New-Object System.Collections.Generic.List[string]
    $accumulated = [double]$Clips[0].Duration
    $prevV = "[0:v]"
    $prevA = "[0:a]"

    for ($i = 1; $i -lt $Clips.Count; $i++) {
        $offset = [Math]::Max(0.0, $accumulated - $fade)
        $offsetText = $offset.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
        $fadeText = $fade.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
        $outV = "[vx$i]"
        $outA = "[ax$i]"
        $videoChain.Add(("{0}[{1}:v]xfade=transition=fade:duration={2}:offset={3}{4}" -f $prevV, $i, $fadeText, $offsetText, $outV))
        $audioChain.Add(("{0}[{1}:a]acrossfade=d={2}{3}" -f $prevA, $i, $fadeText, $outA))
        $prevV = $outV
        $prevA = $outA
        $accumulated = $accumulated + ([double]$Clips[$i].Duration) - $fade
    }

    $filterComplex = (($videoChain + $audioChain) -join ";")

    $ffmpegArgs += @(
        "-filter_complex", $filterComplex,
        "-map", $prevV.Trim('[', ']') ,
        "-map", $prevA.Trim('[', ']'),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "48000",
        "-b:a", "160k",
        "-movflags", "+faststart",
        $TargetPath
    )

    # -map espera o nome do pad entre colchetes.
    for ($i = 0; $i -lt $ffmpegArgs.Count; $i++) {
        if ($ffmpegArgs[$i] -eq "-map" -and ($i + 1) -lt $ffmpegArgs.Count -and $ffmpegArgs[$i + 1] -notmatch '^\[') {
            $ffmpegArgs[$i + 1] = "[" + $ffmpegArgs[$i + 1] + "]"
        }
    }

    Write-ProgressLine -Line ("Juntando {0} trechos com transicao suave de {1}s..." -f $Clips.Count, $fade)
    Invoke-StreamingProcess -ExecutablePath $FfmpegPath -Arguments $ffmpegArgs | Out-Null

    if (-not (Test-Path -LiteralPath $TargetPath)) {
        throw "ffmpeg finished but the joined MP4 was not found: $TargetPath"
    }
}

# --- Validacao de entrada -----------------------------------------------------

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

$segmentTexts = @()
if (-not [string]::IsNullOrWhiteSpace($Segments)) {
    $segmentTexts = @($Segments.Split(";") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
elseif (-not [string]::IsNullOrWhiteSpace($Start) -and -not [string]::IsNullOrWhiteSpace($End)) {
    $segmentTexts = @("{0}-{1}" -f $Start, $End)
}

if ($segmentTexts.Count -eq 0) {
    Write-ProgressLine -Line "ERROR: no segments provided (use -Segments or -Start/-End)."
    exit 1
}

$specs = @()
$index = 0
foreach ($text in $segmentTexts) {
    $index++
    $bounds = $text.Trim().Split("-")
    if ($bounds.Count -ne 2) {
        Write-ProgressLine -Line ("ERROR: invalid segment spec: {0}" -f $text)
        exit 1
    }

    $startSeconds = Convert-TimecodeToSeconds -Value $bounds[0]
    $endSeconds = Convert-TimecodeToSeconds -Value $bounds[1]
    $duration = $endSeconds - $startSeconds
    if ($duration -le 0) {
        Write-ProgressLine -Line ("ERROR: segment {0} has non-positive duration ({1})." -f $index, $text)
        exit 1
    }

    $paddingSeconds = 30.0
    $windowStart = [Math]::Max(0, $startSeconds - $paddingSeconds)

    $specs += [PSCustomObject]@{
        Index         = $index
        StartSeconds  = $startSeconds
        EndSeconds    = $endSeconds
        Duration      = $duration
        WindowStart   = $windowStart
        WindowEnd     = $endSeconds + $paddingSeconds
        LocalStart    = $startSeconds - $windowStart
        Label         = "{0}-{1}" -f (Format-SecondsForFileName -TotalSeconds $startSeconds), (Format-SecondsForFileName -TotalSeconds $endSeconds)
        TempDir       = $null
        StdoutFile    = $null
        Process       = $null
        SourcePath    = $null
        ClipPath      = $null
        Done          = $false
    }
}

$total = $specs.Count
$joinRequested = [bool]$JoinSegments
if ($joinRequested -and $total -lt 2) {
    Write-ProgressLine -Line "Apenas 1 trecho valido: a juncao foi ignorada."
    $joinRequested = $false
}

$effectiveParallel = [Math]::Max(1, [Math]::Min($MaxParallel, $total))
Write-ProgressLine -Line ("Processando {0} trecho(s); downloads em paralelo: ate {1}." -f $total, $effectiveParallel)

$rootTempDir = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_multi_{0}" -f ([System.Guid]::NewGuid().ToString("N")))
New-Item -ItemType Directory -Path $rootTempDir -Force | Out-Null

try {
    # --- Fase 1: downloads das janelas (paralelo limitado) --------------------
    $pending = New-Object System.Collections.Generic.Queue[object]
    foreach ($spec in $specs) {
        $pending.Enqueue($spec)
    }
    $active = New-Object System.Collections.Generic.List[object]
    $completedDownloads = 0

    while ($pending.Count -gt 0 -or $active.Count -gt 0) {
        while ($pending.Count -gt 0 -and $active.Count -lt $effectiveParallel) {
            $spec = $pending.Dequeue()
            $spec.TempDir = Join-Path -Path $rootTempDir -ChildPath ("seg{0:00}" -f $spec.Index)
            New-Item -ItemType Directory -Path $spec.TempDir -Force | Out-Null
            $spec.StdoutFile = Join-Path -Path $spec.TempDir -ChildPath "yt_dlp_output.log"

            $section = "*{0}-{1}" -f (Format-SecondsForFfmpeg -TotalSeconds $spec.WindowStart), (Format-SecondsForFfmpeg -TotalSeconds $spec.WindowEnd)
            $argText = Join-Arguments -Args (Build-YtDlpSectionArgs -DownloadSection $section -TargetDir $spec.TempDir)

            Write-ProgressLine -Line ("[trecho {0}/{1}] Baixando janela {2} ate {3}..." -f $spec.Index, $total, (Format-SecondsForFfmpeg -TotalSeconds $spec.WindowStart), (Format-SecondsForFfmpeg -TotalSeconds $spec.WindowEnd))

            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $YtDlpPath
            $psi.Arguments = $argText
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true

            $process = New-Object System.Diagnostics.Process
            $process.StartInfo = $psi
            if (-not $process.Start()) {
                throw ("[trecho {0}] Could not start yt-dlp." -f $spec.Index)
            }

            # Stdout/stderr de cada processo vao para buffers proprios (evita entrelacar logs).
            $spec | Add-Member -NotePropertyName StdoutTask -NotePropertyValue ($process.StandardOutput.ReadToEndAsync()) -Force
            $spec | Add-Member -NotePropertyName StderrTask -NotePropertyValue ($process.StandardError.ReadToEndAsync()) -Force
            $spec.Process = $process
            $active.Add($spec)
        }

        Start-Sleep -Milliseconds 400

        for ($i = $active.Count - 1; $i -ge 0; $i--) {
            $spec = $active[$i]
            if (-not $spec.Process.HasExited) {
                continue
            }

            $exitCode = $spec.Process.ExitCode
            $stdoutText = $spec.StdoutTask.Result
            $stderrText = $spec.StderrTask.Result
            try {
                [System.IO.File]::WriteAllText($spec.StdoutFile, $stdoutText + [Environment]::NewLine + $stderrText, [System.Text.Encoding]::UTF8)
            }
            catch {
                # Log file is best-effort.
            }

            if ($exitCode -ne 0) {
                $tail = (($stderrText -split "\r?\n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) | Select-Object -Last 3) -join " | "
                throw ("[trecho {0}/{1}] yt-dlp falhou (exit {2}): {3}" -f $spec.Index, $total, $exitCode, $tail)
            }

            $outputLines = @(($stdoutText + "`n" + $stderrText) -split "\r?\n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            $spec.SourcePath = Find-DownloadedMp4 -OutputLines $outputLines -TargetDir $spec.TempDir
            if ([string]::IsNullOrWhiteSpace($spec.SourcePath) -or -not (Test-Path -LiteralPath $spec.SourcePath)) {
                throw ("[trecho {0}/{1}] MP4 temporario nao encontrado apos o download." -f $spec.Index, $total)
            }

            $active.RemoveAt($i)
            $completedDownloads++
            Write-ProgressLine -Line ("[trecho {0}/{1}] Download concluido." -f $spec.Index, $total)
            Write-ProgressLine -Line ("ETAPA {0}/{1}" -f $completedDownloads, ($total * 2))
        }
    }

    # --- Fase 2: cortes precisos (sequencial; ffmpeg e rapido) -----------------
    $clips = @()
    $baseName = $null
    foreach ($spec in ($specs | Sort-Object -Property Index)) {
        if ($null -eq $baseName) {
            $baseName = [System.IO.Path]::GetFileNameWithoutExtension($spec.SourcePath)
        }

        $clipFileName = if ($joinRequested) {
            "clip_{0:00}.mp4" -f $spec.Index
        }
        elseif ($total -gt 1) {
            "{0} [{1:00} - {2}].mp4" -f $baseName, $spec.Index, $spec.Label
        }
        else {
            "{0} [{1}].mp4" -f $baseName, $spec.Label
        }

        $clipTargetDir = if ($joinRequested) { $rootTempDir } else { $OutputDir }
        $clipPath = Join-Path -Path $clipTargetDir -ChildPath $clipFileName

        Write-ProgressLine -Line ("[trecho {0}/{1}] Cortando {2} (duracao {3})..." -f $spec.Index, $total, $spec.Label, (Format-SecondsForFfmpeg -TotalSeconds $spec.Duration))
        Invoke-FfmpegCut `
            -SourcePath $spec.SourcePath `
            -LocalStartSeconds $spec.LocalStart `
            -DurationSeconds $spec.Duration `
            -TargetPath $clipPath `
            -LinePrefix ("[trecho {0}/{1}] " -f $spec.Index, $total)

        $spec.ClipPath = $clipPath
        $clips += [PSCustomObject]@{ Path = $clipPath; Duration = $spec.Duration }

        if (-not $joinRequested) {
            Write-ProgressLine -Line ("[trecho {0}/{1}] Arquivo final criado:" -f $spec.Index, $total)
            Write-ProgressLine -Line $clipPath
        }
        Write-ProgressLine -Line ("ETAPA {0}/{1}" -f ($total + $spec.Index), ($total * 2))
    }

    # --- Fase 3 (opcional): juncao com crossfade -------------------------------
    if ($joinRequested) {
        $joinedName = "{0} [montagem {1} trechos].mp4" -f $baseName, $total
        $joinedPath = Join-Path -Path $OutputDir -ChildPath $joinedName
        Join-ClipsWithCrossfade -Clips $clips -TargetPath $joinedPath
        Write-ProgressLine -Line "Arquivo final criado:"
        Write-ProgressLine -Line $joinedPath
    }

    Write-ProgressLine -Line ("Concluido: {0} trecho(s) processado(s)." -f $total)
}
catch {
    Write-ProgressLine -Line ("ERROR: {0}" -f $_.Exception.Message)
    exit 1
}
finally {
    foreach ($spec in $specs) {
        if ($null -ne $spec.Process -and -not $spec.Process.HasExited) {
            try {
                Stop-Process -Id $spec.Process.Id -Force -ErrorAction SilentlyContinue
            }
            catch {
                # Ignore cleanup races.
            }
        }
    }

    try {
        if (Test-Path -LiteralPath $rootTempDir) {
            Remove-Item -LiteralPath $rootTempDir -Recurse -Force
        }
    }
    catch {
        Write-ProgressLine -Line "Warning: could not remove temporary folder: $rootTempDir"
    }
}
