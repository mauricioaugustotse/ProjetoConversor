param(
    [string]$InputFile,
    [double]$TargetSizeMB,
    [Nullable[int]]$MaxHeight,
    [Nullable[double]]$Fps,
    [string]$OutputFile,
    [switch]$NoGui,
    [switch]$NoPrompt,
    [string]$InputFileB64,
    [string]$OutputFileB64,
    [string]$ResultFile
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:activeProcess = $null
$script:sessionLogFile = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("mp4_compress_gui_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Add-ToProcessPathIfExists {
    param([Parameter(Mandatory = $true)][string]$PathEntry)

    if ([string]::IsNullOrWhiteSpace($PathEntry)) {
        return
    }

    if (-not (Test-Path -LiteralPath $PathEntry)) {
        return
    }

    $existingParts = @()
    if (-not [string]::IsNullOrWhiteSpace($env:Path)) {
        $existingParts = $env:Path.Split(';') | ForEach-Object { $_.Trim() }
    }

    if ($existingParts -contains $PathEntry) {
        return
    }

    if ([string]::IsNullOrWhiteSpace($env:Path)) {
        $env:Path = $PathEntry
    }
    else {
        $env:Path += ";" + $PathEntry
    }
}

function Ensure-CommonToolPaths {
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    $programData = [Environment]::GetFolderPath("CommonApplicationData")
    $userProfile = [Environment]::GetFolderPath("UserProfile")

    Add-ToProcessPathIfExists -PathEntry (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links")
    Add-ToProcessPathIfExists -PathEntry (Join-Path -Path $programData -ChildPath "chocolatey\bin")
    Add-ToProcessPathIfExists -PathEntry (Join-Path -Path $userProfile -ChildPath "scoop\shims")
}

function Resolve-CommandPathOrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$CommandName,
        [Parameter(Mandatory = $true)][string[]]$CandidatePaths,
        [Parameter(Mandatory = $true)][string]$ErrorHint
    )

    Ensure-CommonToolPaths

    $command = Get-Command $CommandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
        return $command.Source
    }

    foreach ($candidate in $CandidatePaths) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw $ErrorHint
}

function Resolve-CommandPathOrNull {
    param(
        [Parameter(Mandatory = $true)][string]$CommandName,
        [Parameter(Mandatory = $true)][string[]]$CandidatePaths
    )

    Ensure-CommonToolPaths

    $command = Get-Command $CommandName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
        return $command.Source
    }

    foreach ($candidate in $CandidatePaths) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    return $null
}

function Parse-PositiveDouble {
    param(
        [Parameter(Mandatory = $true)][string]$Raw,
        [Parameter(Mandatory = $true)][string]$FieldName
    )

    $clean = $Raw.Trim().Replace(',', '.')
    [double]$value = 0

    $ok = [double]::TryParse(
        $clean,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$value
    )

    if (-not $ok -or $value -le 0) {
        throw "$FieldName deve ser um numero positivo."
    }

    return $value
}

function Decode-Base64Utf8OrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$FieldName
    )

    try {
        $bytes = [Convert]::FromBase64String($Value)
        return [System.Text.Encoding]::UTF8.GetString($bytes)
    }
    catch {
        throw "Parametro invalido em $FieldName."
    }
}

function Parse-NonNegativeIntOrNull {
    param(
        [Parameter(Mandatory = $true)][string]$Raw,
        [Parameter(Mandatory = $true)][string]$FieldName,
        [Parameter(Mandatory = $true)][int]$DefaultValue
    )

    $clean = $Raw.Trim()

    if ([string]::IsNullOrWhiteSpace($clean)) {
        return [Nullable[int]]$DefaultValue
    }

    if ($clean -notmatch '^\d+$') {
        throw "$FieldName deve ser um numero inteiro."
    }

    $value = [int]$clean
    if ($value -eq 0) {
        return [Nullable[int]]$null
    }

    return [Nullable[int]]$value
}

function Parse-NonNegativeDoubleOrNull {
    param(
        [Parameter(Mandatory = $true)][string]$Raw,
        [Parameter(Mandatory = $true)][string]$FieldName,
        [Parameter(Mandatory = $true)][double]$DefaultValue
    )

    $clean = $Raw.Trim()

    if ([string]::IsNullOrWhiteSpace($clean)) {
        return [Nullable[double]]$DefaultValue
    }

    [double]$value = 0
    $ok = [double]::TryParse(
        $clean.Replace(',', '.'),
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$value
    )

    if (-not $ok -or $value -lt 0) {
        throw "$FieldName deve ser um numero maior ou igual a 0."
    }

    if ($value -eq 0) {
        return [Nullable[double]]$null
    }

    return [Nullable[double]]$value
}

function Normalize-MaxHeight {
    param([AllowNull()]$Value)

    if ($null -eq $Value) {
        return [Nullable[int]]$null
    }

    [int]$height = 0
    try {
        $height = [int]$Value
    }
    catch {
        throw "Altura maxima invalida."
    }

    if ($height -le 0) {
        return [Nullable[int]]$null
    }

    return [Nullable[int]]$height
}

function Normalize-Fps {
    param([AllowNull()]$Value)

    if ($null -eq $Value) {
        return [Nullable[double]]$null
    }

    [double]$fpsNumber = 0
    $fpsText = $Value.ToString().Trim().Replace(',', '.')

    $ok = [double]::TryParse(
        $fpsText,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$fpsNumber
    )

    if (-not $ok) {
        throw "FPS invalido."
    }

    if ($fpsNumber -le 0) {
        return [Nullable[double]]$null
    }

    return [Nullable[double]]$fpsNumber
}

function Write-LogLine {
    param(
        [AllowNull()][scriptblock]$Log,
        [AllowEmptyString()][string]$Line
    )

    if ($null -ne $Log) {
        & $Log $Line
    }
    else {
        if ([string]::IsNullOrEmpty($Line)) {
            Write-Host ""
        }
        else {
            Write-Host $Line
        }
    }
}

function Append-SessionLog {
    param([AllowEmptyString()][string]$Line)

    try {
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        [System.IO.File]::AppendAllText(
            $script:sessionLogFile,
            ("[{0}] {1}{2}" -f $stamp, $Line, [Environment]::NewLine),
            [System.Text.Encoding]::UTF8
        )
    }
    catch {
        # Ignore session log write failures.
    }
}

function Quote-ForDisplay {
    param([Parameter(Mandatory = $true)][string]$Text)

    if ($Text -match '[\s"]') {
        return '"' + ($Text -replace '"', '\\"') + '"'
    }

    return $Text
}

function Escape-Argument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\\"') + '"'
    }

    return $Value
}

function Join-Arguments {
    param([Parameter(Mandatory = $true)][string[]]$Args)

    return ($Args | ForEach-Object { Escape-Argument -Value $_ }) -join " "
}

function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $argumentText = Join-Arguments -Args $Arguments
    $tmpOut = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("mp4_native_out_{0}.log" -f [Guid]::NewGuid().ToString("N"))
    $tmpErr = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("mp4_native_err_{0}.log" -f [Guid]::NewGuid().ToString("N"))

    try {
        $process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $argumentText `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $tmpOut `
            -RedirectStandardError $tmpErr

        $lines = @()
        if (Test-Path -LiteralPath $tmpOut) {
            $lines += Get-Content -LiteralPath $tmpOut -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $tmpErr) {
            $lines += Get-Content -LiteralPath $tmpErr -ErrorAction SilentlyContinue
        }

        $cleanLines = @(
            $lines |
            ForEach-Object { if ($null -ne $_) { $_.ToString().TrimEnd("`r") } } |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        )

        return @{
            ExitCode = [int]$process.ExitCode
            Lines    = $cleanLines
        }
    }
    finally {
        Remove-Item -LiteralPath $tmpOut -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $tmpErr -Force -ErrorAction SilentlyContinue
    }
}

if (-not [string]::IsNullOrWhiteSpace($InputFileB64)) {
    $InputFile = Decode-Base64Utf8OrThrow -Value $InputFileB64 -FieldName "InputFileB64"
}

if (-not [string]::IsNullOrWhiteSpace($OutputFileB64)) {
    $OutputFile = Decode-Base64Utf8OrThrow -Value $OutputFileB64 -FieldName "OutputFileB64"
}

function Invoke-ExternalOrThrow {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Step,
        [AllowNull()][scriptblock]$Log
    )

    Write-LogLine -Log $Log -Line ""
    Write-LogLine -Log $Log -Line ("[{0}]" -f $Step)
    Write-LogLine -Log $Log -Line ((Quote-ForDisplay -Text $Executable) + " " + (($Arguments | ForEach-Object { Quote-ForDisplay -Text $_ }) -join " "))

    $result = Invoke-NativeCapture -Executable $Executable -Arguments $Arguments

    foreach ($line in $result.Lines) {
        Write-LogLine -Log $Log -Line $line
    }

    if ($result.ExitCode -ne 0) {
        throw ("Falha em '{0}' (exit code {1})." -f $Step, $result.ExitCode)
    }
}

function Get-MediaProbeLinesViaFfmpeg {
    param(
        [Parameter(Mandatory = $true)][string]$FfmpegPath,
        [Parameter(Mandatory = $true)][string]$FilePath
    )

    $result = Invoke-NativeCapture -Executable $FfmpegPath -Arguments @("-hide_banner", "-i", $FilePath)
    return @($result.Lines)
}

function Get-DurationViaShellMetadataOrNull {
    param([Parameter(Mandatory = $true)][string]$FilePath)

    try {
        $folderPath = Split-Path -Path $FilePath -Parent
        $fileName = Split-Path -Path $FilePath -Leaf

        $shell = New-Object -ComObject Shell.Application
        $folder = $shell.Namespace($folderPath)
        if ($null -eq $folder) {
            return $null
        }

        $item = $folder.ParseName($fileName)
        if ($null -eq $item) {
            return $null
        }

        $duration100ns = $item.ExtendedProperty("System.Media.Duration")
        if ($null -eq $duration100ns) {
            return $null
        }

        [double]$rawValue = 0
        $ok = [double]::TryParse(
            $duration100ns.ToString(),
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$rawValue
        )

        if (-not $ok -or $rawValue -le 0) {
            return $null
        }

        $seconds = $rawValue / 10000000.0
        if ($seconds -le 0) {
            return $null
        }

        return $seconds
    }
    catch {
        return $null
    }
}

function Get-FileDurationSeconds {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$FfmpegPath,
        [AllowNull()][string]$FfprobePath
    )

    if (-not [string]::IsNullOrWhiteSpace($FfprobePath)) {
        $probeResult = Invoke-NativeCapture -Executable $FfprobePath -Arguments @("-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", "--", $FilePath)
        $probeLines = @($probeResult.Lines | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

        foreach ($line in $probeLines) {
            [double]$durationFromProbe = 0
            $ok = [double]::TryParse(
                $line,
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$durationFromProbe
            )

            if ($ok -and $durationFromProbe -gt 0) {
                return $durationFromProbe
            }
        }
    }

    $ffmpegProbeLines = Get-MediaProbeLinesViaFfmpeg -FfmpegPath $FfmpegPath -FilePath $FilePath
    $durationLine = $ffmpegProbeLines | Where-Object { $_ -match 'Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)' } | Select-Object -First 1

    if ($null -ne $durationLine) {
        $match = [regex]::Match($durationLine, 'Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)')
        if ($match.Success) {
            [int]$hours = [int]$match.Groups[1].Value
            [int]$minutes = [int]$match.Groups[2].Value
            [double]$seconds = 0

            $okSeconds = [double]::TryParse(
                $match.Groups[3].Value,
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$seconds
            )

            if ($okSeconds) {
                $durationTotal = ($hours * 3600) + ($minutes * 60) + $seconds
                if ($durationTotal -gt 0) {
                    return $durationTotal
                }
            }
        }
    }

    # Final fallback: decode to null and read out_time_ms from ffmpeg progress output.
    $progressResult = Invoke-NativeCapture -Executable $FfmpegPath -Arguments @("-v", "error", "-i", $FilePath, "-map", "0:v:0", "-f", "null", "-", "-progress", "pipe:1", "-nostats")
    $progressLines = @($progressResult.Lines | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

    $outTimeLine = $progressLines | Where-Object { $_ -like 'out_time_ms=*' } | Select-Object -Last 1
    if ($null -ne $outTimeLine) {
        $rawValue = ($outTimeLine -replace '^out_time_ms=', '').Trim()
        [long]$outTimeMs = 0
        if ([long]::TryParse($rawValue, [ref]$outTimeMs) -and $outTimeMs -gt 0) {
            return ($outTimeMs / 1000000.0)
        }
    }

    $shellDuration = Get-DurationViaShellMetadataOrNull -FilePath $FilePath
    if ($null -ne $shellDuration -and $shellDuration -gt 0) {
        return [double]$shellDuration
    }

    throw "Nao foi possivel obter a duracao do arquivo de entrada."
}

function Test-InputHasAudio {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$FfmpegPath,
        [AllowNull()][string]$FfprobePath
    )

    if (-not [string]::IsNullOrWhiteSpace($FfprobePath)) {
        $probeResult = Invoke-NativeCapture -Executable $FfprobePath -Arguments @("-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", "--", $FilePath)
        $probeLines = @($probeResult.Lines | ForEach-Object { $_.Trim() } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })

        if ($probeLines.Count -gt 0) {
            return $true
        }
    }

    $ffmpegProbeLines = Get-MediaProbeLinesViaFfmpeg -FfmpegPath $FfmpegPath -FilePath $FilePath
    $audioLine = $ffmpegProbeLines | Where-Object { $_ -match 'Stream #.*Audio:' } | Select-Object -First 1
    return ($null -ne $audioLine)
}

function Get-AudioBitrateKbpsOrZero {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string]$FfmpegPath,
        [AllowNull()][string]$FfprobePath
    )

    if (-not [string]::IsNullOrWhiteSpace($FfprobePath)) {
        $probeResult = Invoke-NativeCapture -Executable $FfprobePath -Arguments @(
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=bit_rate",
            "-of", "default=nk=1:nw=1",
            "--", $FilePath
        )

        foreach ($line in $probeResult.Lines) {
            [double]$bitRateBps = 0
            $ok = [double]::TryParse(
                $line.Trim(),
                [System.Globalization.NumberStyles]::Float,
                [System.Globalization.CultureInfo]::InvariantCulture,
                [ref]$bitRateBps
            )

            if ($ok -and $bitRateBps -gt 0) {
                return ($bitRateBps / 1000.0)
            }
        }
    }

    $ffmpegProbeLines = Get-MediaProbeLinesViaFfmpeg -FfmpegPath $FfmpegPath -FilePath $FilePath
    $audioLine = $ffmpegProbeLines | Where-Object { $_ -match 'Audio:.*?(\d+)\s*kb/s' } | Select-Object -First 1
    if ($null -ne $audioLine) {
        $match = [regex]::Match($audioLine, 'Audio:.*?(\d+)\s*kb/s')
        if ($match.Success) {
            [double]$audioKbps = 0
            if ([double]::TryParse(
                    $match.Groups[1].Value,
                    [System.Globalization.NumberStyles]::Float,
                    [System.Globalization.CultureInfo]::InvariantCulture,
                    [ref]$audioKbps
                )) {
                return $audioKbps
            }
        }
    }

    return 0.0
}

function Get-MinimumTargetEstimateInfo {
    param([Parameter(Mandatory = $true)][string]$FilePath)

    if (-not (Test-Path -LiteralPath $FilePath)) {
        throw "Arquivo nao encontrado para estimativa."
    }

    $tools = Resolve-ToolPaths
    $durationSec = Get-FileDurationSeconds -FilePath $FilePath -FfmpegPath $tools.Ffmpeg -FfprobePath $tools.Ffprobe
    $audioKbps = Get-AudioBitrateKbpsOrZero -FilePath $FilePath -FfmpegPath $tools.Ffmpeg -FfprobePath $tools.Ffprobe

    $minVideoKbps = 80.0
    $containerOverheadMb = 0.2
    $audioMb = ($durationSec * $audioKbps) / (8.0 * 1024.0)
    $videoFloorMb = ($durationSec * $minVideoKbps) / (8.0 * 1024.0)
    $minimumMb = $audioMb + $videoFloorMb + $containerOverheadMb
    $recommendedMb = $minimumMb + 0.5

    return @{
        DurationSec = $durationSec
        AudioKbps = $audioKbps
        MinimumMb = $minimumMb
        RecommendedMb = $recommendedMb
    }
}

function Get-DefaultOutputName {
    param(
        [Parameter(Mandatory = $true)][string]$InputPath,
        [Parameter(Mandatory = $true)][double]$TargetMb
    )

    $dir = Split-Path -Path $InputPath -Parent
    $name = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
    $targetLabel = $TargetMb.ToString("0.##", [System.Globalization.CultureInfo]::InvariantCulture)

    return (Join-Path -Path $dir -ChildPath ("{0}_compressed_{1}MB.mp4" -f $name, $targetLabel))
}

function Resolve-ToolPaths {
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    $programData = [Environment]::GetFolderPath("CommonApplicationData")
    $userProfile = [Environment]::GetFolderPath("UserProfile")

    $ffmpegCandidates = @(
        (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links\ffmpeg.exe"),
        (Join-Path -Path $programData -ChildPath "chocolatey\bin\ffmpeg.exe"),
        (Join-Path -Path $userProfile -ChildPath "scoop\shims\ffmpeg.exe")
    )

    $ffprobeCandidates = @(
        (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links\ffprobe.exe"),
        (Join-Path -Path $programData -ChildPath "chocolatey\bin\ffprobe.exe"),
        (Join-Path -Path $userProfile -ChildPath "scoop\shims\ffprobe.exe")
    )

    return @{
        Ffmpeg = Resolve-CommandPathOrThrow -CommandName "ffmpeg" -CandidatePaths $ffmpegCandidates -ErrorHint "ffmpeg nao encontrado. Instale com: winget install Gyan.FFmpeg"
        Ffprobe = Resolve-CommandPathOrNull -CommandName "ffprobe" -CandidatePaths $ffprobeCandidates
    }
}

function Invoke-TargetCompression {
    param(
        [Parameter(Mandatory = $true)][string]$InputPath,
        [Parameter(Mandatory = $true)][double]$TargetMb,
        [Nullable[int]]$MaxHeightValue,
        [Nullable[double]]$FpsValue,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [AllowNull()][scriptblock]$Log
    )

    if (-not (Test-Path -LiteralPath $InputPath)) {
        throw "Arquivo nao encontrado: $InputPath"
    }

    if ($TargetMb -le 0) {
        throw "Tamanho alvo deve ser maior que zero."
    }

    $MaxHeightValue = Normalize-MaxHeight -Value $MaxHeightValue
    $FpsValue = Normalize-Fps -Value $FpsValue

    $outputDir = Split-Path -Path $OutputPath -Parent
    if (-not [string]::IsNullOrWhiteSpace($outputDir) -and -not (Test-Path -LiteralPath $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }

    $tools = Resolve-ToolPaths
    $ffmpegPath = $tools.Ffmpeg
    $ffprobePath = $tools.Ffprobe

    $tempRoot = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("mp4_target_{0}" -f [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null

    try {
        if ([string]::IsNullOrWhiteSpace($ffprobePath)) {
            Write-LogLine -Log $Log -Line "ffprobe nao encontrado ou sem resposta valida. Usando fallback via ffmpeg para inspecao."
        }

        $durationSec = Get-FileDurationSeconds -FilePath $InputPath -FfmpegPath $ffmpegPath -FfprobePath $ffprobePath
        $inputBytes = (Get-Item -LiteralPath $InputPath).Length
        $targetBytes = [int64][Math]::Floor($TargetMb * 1024 * 1024)

        $hasAudio = Test-InputHasAudio -FilePath $InputPath -FfmpegPath $ffmpegPath -FfprobePath $ffprobePath
        $audioBytes = 0L
        $tempAudio = Join-Path -Path $tempRoot -ChildPath "audio_only.mkv"

        if ($hasAudio) {
            Invoke-ExternalOrThrow -Executable $ffmpegPath -Arguments @(
                "-y",
                "-i", $InputPath,
                "-map", "0:a",
                "-vn",
                "-c:a", "copy",
                $tempAudio
            ) -Step "Extracao do audio sem perda" -Log $Log

            $audioBytes = (Get-Item -LiteralPath $tempAudio).Length
        }

        $overheadBytes = [int64][Math]::Ceiling($targetBytes * 0.015) + 131072
        $videoBudgetBytes = $targetBytes - $audioBytes - $overheadBytes
        $minTargetAudioOnlyBytes = $audioBytes + $overheadBytes

        if ($videoBudgetBytes -le 0) {
            $audioMb = $audioBytes / 1MB
            $minMb = $minTargetAudioOnlyBytes / 1MB
            throw ("Tamanho alvo muito baixo. So o audio ocupa cerca de {0:N2} MB. Use alvo minimo de ~{1:N2} MB mantendo audio sem perda." -f $audioMb, $minMb)
        }

        $initialVideoBitrateK = [int][Math]::Floor(($videoBudgetBytes * 8.0) / $durationSec / 1000.0)
        if ($initialVideoBitrateK -lt 80) {
            $minVideoBytes = [int64][Math]::Ceiling(($durationSec * 80.0 * 1000.0) / 8.0)
            $suggestedMinMb = ($audioBytes + $overheadBytes + $minVideoBytes) / 1MB
            throw ("Bitrate de video calculado ficou muito baixo (<80 kbps). Use alvo de pelo menos ~{0:N2} MB, ou reduza a duracao." -f $suggestedMinMb)
        }

        $filters = @()

        if ($null -ne $MaxHeightValue) {
            $maxHeightNumber = [int]$MaxHeightValue
            $filters += ("scale=-2:'min(ih,{0})'" -f $maxHeightNumber)
        }

        if ($null -ne $FpsValue) {
            $fpsNumber = [double]$FpsValue
            $fpsText = $fpsNumber.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
            $filters += ("fps={0}" -f $fpsText)
        }

        $videoFilter = $null
        if ($filters.Count -gt 0) {
            $videoFilter = $filters -join ","
        }

        $nullDevice = if ([Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT) { "NUL" } else { "/dev/null" }
        $passLog = Join-Path -Path $tempRoot -ChildPath "ffmpeg2pass"
        $tempVideo = Join-Path -Path $tempRoot -ChildPath "video_only.mp4"

        $currentBitrateK = $initialVideoBitrateK
        $lastOutputBytes = 0L
        $finalOutputPath = $OutputPath

        Write-LogLine -Log $Log -Line ""
        Write-LogLine -Log $Log -Line "Resumo"
        Write-LogLine -Log $Log -Line ("- Entrada: {0}" -f $InputPath)
        Write-LogLine -Log $Log -Line ("- Tamanho atual: {0:N2} MB" -f ($inputBytes / 1MB))
        Write-LogLine -Log $Log -Line ("- Tamanho alvo: {0:N2} MB" -f ($targetBytes / 1MB))

        if ($null -ne $MaxHeightValue) {
            Write-LogLine -Log $Log -Line ("- Altura maxima: {0}px" -f ([int]$MaxHeightValue))
        }
        else {
            Write-LogLine -Log $Log -Line "- Altura maxima: original"
        }

        if ($null -ne $FpsValue) {
            $fpsDisplay = ([double]$FpsValue).ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
            Write-LogLine -Log $Log -Line ("- FPS: {0}" -f $fpsDisplay)
        }
        else {
            Write-LogLine -Log $Log -Line "- FPS: original"
        }

        Write-LogLine -Log $Log -Line "- Audio: copia sem reencode (qualidade preservada)"

        for ($attempt = 1; $attempt -le 3; $attempt++) {
            Write-LogLine -Log $Log -Line ""
            Write-LogLine -Log $Log -Line ("Tentativa {0}/3 com bitrate de video: {1} kbps" -f $attempt, $currentBitrateK)

            if (Test-Path -LiteralPath $tempVideo) {
                Remove-Item -LiteralPath $tempVideo -Force -ErrorAction SilentlyContinue
            }

            $pass1Args = @(
                "-y",
                "-i", $InputPath,
                "-map", "0:v:0",
                "-an",
                "-c:v", "libx264",
                "-preset", "slow",
                "-b:v", ("{0}k" -f $currentBitrateK),
                "-pass", "1",
                "-passlogfile", $passLog,
                "-pix_fmt", "yuv420p"
            )

            if (-not [string]::IsNullOrWhiteSpace($videoFilter)) {
                $pass1Args += @("-vf", $videoFilter)
            }

            $pass1Args += @("-f", "mp4", $nullDevice)

            Invoke-ExternalOrThrow -Executable $ffmpegPath -Arguments $pass1Args -Step ("Encode video pass 1 (tentativa {0})" -f $attempt) -Log $Log

            $pass2Args = @(
                "-y",
                "-i", $InputPath,
                "-map", "0:v:0",
                "-an",
                "-c:v", "libx264",
                "-preset", "slow",
                "-b:v", ("{0}k" -f $currentBitrateK),
                "-pass", "2",
                "-passlogfile", $passLog,
                "-pix_fmt", "yuv420p"
            )

            if (-not [string]::IsNullOrWhiteSpace($videoFilter)) {
                $pass2Args += @("-vf", $videoFilter)
            }

            $pass2Args += $tempVideo

            Invoke-ExternalOrThrow -Executable $ffmpegPath -Arguments $pass2Args -Step ("Encode video pass 2 (tentativa {0})" -f $attempt) -Log $Log

            $muxArgs = @(
                "-y",
                "-i", $tempVideo,
                "-i", $InputPath,
                "-map", "0:v:0",
                "-map", "1:a?",
                "-map_metadata", "1",
                "-map_chapters", "1",
                "-c:v", "copy",
                "-c:a", "copy"
            )

            if ([System.IO.Path]::GetExtension($finalOutputPath).ToLowerInvariant() -eq ".mp4") {
                $muxArgs += @("-movflags", "+faststart")
            }

            if (Test-Path -LiteralPath $finalOutputPath) {
                Remove-Item -LiteralPath $finalOutputPath -Force -ErrorAction SilentlyContinue
            }

            try {
                Invoke-ExternalOrThrow -Executable $ffmpegPath -Arguments ($muxArgs + @($finalOutputPath)) -Step ("Mux final (tentativa {0})" -f $attempt) -Log $Log
            }
            catch {
                if ($hasAudio -and [System.IO.Path]::GetExtension($finalOutputPath).ToLowerInvariant() -eq ".mp4") {
                    $fallbackOutput = [System.IO.Path]::ChangeExtension($finalOutputPath, ".mkv")

                    Write-LogLine -Log $Log -Line ""
                    Write-LogLine -Log $Log -Line "Audio original nao compativel com container MP4. Salvando em MKV para manter audio sem perda."

                    if (Test-Path -LiteralPath $fallbackOutput) {
                        Remove-Item -LiteralPath $fallbackOutput -Force -ErrorAction SilentlyContinue
                    }

                    $fallbackMuxArgs = @(
                        "-y",
                        "-i", $tempVideo,
                        "-i", $InputPath,
                        "-map", "0:v:0",
                        "-map", "1:a?",
                        "-map_metadata", "1",
                        "-map_chapters", "1",
                        "-c:v", "copy",
                        "-c:a", "copy",
                        $fallbackOutput
                    )

                    Invoke-ExternalOrThrow -Executable $ffmpegPath -Arguments $fallbackMuxArgs -Step ("Mux final em MKV (tentativa {0})" -f $attempt) -Log $Log
                    $finalOutputPath = $fallbackOutput
                }
                else {
                    throw
                }
            }

            $lastOutputBytes = (Get-Item -LiteralPath $finalOutputPath).Length
            $ratio = $lastOutputBytes / [double]$targetBytes

            Write-LogLine -Log $Log -Line ("Tamanho obtido: {0:N2} MB" -f ($lastOutputBytes / 1MB))

            if ($ratio -ge 0.92 -and $ratio -le 1.02) {
                break
            }

            if ($attempt -lt 3) {
                $factor = $targetBytes / [double]$lastOutputBytes
                if ($ratio -gt 1.02) {
                    $factor *= 0.98
                }
                else {
                    $factor *= 1.02
                }

                $newBitrate = [int][Math]::Floor($currentBitrateK * $factor)
                if ($newBitrate -lt 80) {
                    $newBitrate = 80
                }

                if ($newBitrate -eq $currentBitrateK) {
                    break
                }

                $currentBitrateK = $newBitrate
            }
        }

        Write-LogLine -Log $Log -Line ""
        Write-LogLine -Log $Log -Line "Concluido"
        Write-LogLine -Log $Log -Line ("- Arquivo final: {0}" -f $finalOutputPath)
        Write-LogLine -Log $Log -Line ("- Tamanho final: {0:N2} MB" -f ($lastOutputBytes / 1MB))
        Write-LogLine -Log $Log -Line ("- Alvo: {0:N2} MB" -f ($targetBytes / 1MB))

        return @{
            OutputPath = $finalOutputPath
            FinalBytes = $lastOutputBytes
            TargetBytes = $targetBytes
        }
    }
    finally {
        if (Test-Path -LiteralPath $tempRoot) {
            Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Resolve-RunnerHostPath {
    $currentProcess = Get-Process -Id $PID -ErrorAction SilentlyContinue
    if ($null -ne $currentProcess -and -not [string]::IsNullOrWhiteSpace($currentProcess.Path) -and (Test-Path -LiteralPath $currentProcess.Path)) {
        return $currentProcess.Path
    }

    $pwsh = Get-Command pwsh -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $pwsh -and -not [string]::IsNullOrWhiteSpace($pwsh.Source)) {
        return $pwsh.Source
    }

    $ps = Get-Command powershell -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $ps -and -not [string]::IsNullOrWhiteSpace($ps.Source)) {
        return $ps.Source
    }

    throw "Nao foi possivel localizar PowerShell para executar a compressao."
}

function Append-UiLog {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.TextBox]$Target,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Line
    )

    $writer = {
        param([System.Windows.Forms.TextBox]$Box, [string]$Text)
        $Box.AppendText($Text + [Environment]::NewLine)
        $Box.SelectionStart = $Box.TextLength
        $Box.ScrollToCaret()
    }

    try {
        Append-SessionLog -Line $Line

        if ($Target.IsDisposed) {
            return
        }

        if ($Target.InvokeRequired) {
            $Target.BeginInvoke($writer, $Target, $Line) | Out-Null
        }
        else {
            & $writer $Target $Line
        }
    }
    catch {
        # Ignore UI logging failures while the window is closing/disposed.
    }
}

function Set-UiLabelText {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.Label]$Target,
        [Parameter(Mandatory = $true)][string]$Text
    )

    $setter = {
        param([System.Windows.Forms.Label]$Label, [string]$Value)
        $Label.Text = $Value
    }

    try {
        if ($Target.IsDisposed) {
            return
        }

        if ($Target.InvokeRequired) {
            $Target.BeginInvoke($setter, $Target, $Text) | Out-Null
        }
        else {
            & $setter $Target $Text
        }
    }
    catch {
        # Ignore UI state updates when form is closing.
    }
}

function Set-UiButtonEnabled {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$Target,
        [Parameter(Mandatory = $true)][bool]$Enabled
    )

    $setter = {
        param([System.Windows.Forms.Button]$Button, [bool]$State)
        $Button.Enabled = $State
    }

    try {
        if ($Target.IsDisposed) {
            return
        }

        if ($Target.InvokeRequired) {
            $Target.BeginInvoke($setter, $Target, $Enabled) | Out-Null
        }
        else {
            & $setter $Target $Enabled
        }
    }
    catch {
        # Ignore UI state updates when form is closing.
    }
}

function Get-DefaultTargetFromInputOrFallback {
    param([double]$Fallback)

    if ($Fallback -gt 0) {
        return $Fallback
    }

    return 5.0
}

function Start-GuiCompressionProcess {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptPath,
        [Parameter(Mandatory = $true)][string]$InputPath,
        [Parameter(Mandatory = $true)][double]$TargetMb,
        [Nullable[int]]$MaxHeightValue,
        [Nullable[double]]$FpsValue,
        [Parameter(Mandatory = $true)][string]$OutputPath,
        [Parameter(Mandatory = $true)][System.Windows.Forms.TextBox]$LogBox,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Label]$StatusLabel,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$StartButton,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$CancelButton
    )

    if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
        throw "Ja existe uma compressao em andamento."
    }

    $hostPath = Resolve-RunnerHostPath
    $targetText = $TargetMb.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
    $heightText = if ($null -eq $MaxHeightValue) { "0" } else { ([int]$MaxHeightValue).ToString() }
    $fpsText = if ($null -eq $FpsValue) { "0" } else { ([double]$FpsValue).ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture) }
    $inputPathB64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($InputPath))
    $outputPathB64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($OutputPath))
    $resultFilePath = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("mp4_compress_result_{0}.json" -f [Guid]::NewGuid().ToString("N"))

    $args = @(
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", $ScriptPath,
        "-NoGui",
        "-NoPrompt",
        "-InputFileB64", $inputPathB64,
        "-TargetSizeMB", $targetText,
        "-MaxHeight", $heightText,
        "-Fps", $fpsText,
        "-OutputFileB64", $outputPathB64,
        "-ResultFile", $resultFilePath
    )

    $argumentText = Join-Arguments -Args $args

    Append-UiLog -Target $LogBox -Line ""
    Append-UiLog -Target $LogBox -Line ("> " + (Quote-ForDisplay -Text $hostPath) + " " + $argumentText)
    Append-UiLog -Target $LogBox -Line ""

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $hostPath
    $psi.Arguments = $argumentText
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $process.EnableRaisingEvents = $true

    $process.add_OutputDataReceived({
            param($sender, $eventArgs)
            try {
                if (-not [string]::IsNullOrWhiteSpace($eventArgs.Data)) {
                    Append-UiLog -Target $LogBox -Line $eventArgs.Data
                    try {
                        [Console]::WriteLine($eventArgs.Data)
                    }
                    catch {
                        # Ignore console write errors.
                    }
                }
            }
            catch {
                # Ignore callback errors to avoid closing the GUI host.
            }
        })

    $process.add_ErrorDataReceived({
            param($sender, $eventArgs)
            try {
                if (-not [string]::IsNullOrWhiteSpace($eventArgs.Data)) {
                    Append-UiLog -Target $LogBox -Line $eventArgs.Data
                    try {
                        [Console]::WriteLine($eventArgs.Data)
                    }
                    catch {
                        # Ignore console write errors.
                    }
                }
            }
            catch {
                # Ignore callback errors to avoid closing the GUI host.
            }
        })

    $process.add_Exited({
            param($sender, $eventArgs)
            try {
                $exitCode = 1
                try {
                    $exitCode = $sender.ExitCode
                }
                catch {
                    # Keep non-zero code.
                }

                $script:activeProcess = $null
                Set-UiButtonEnabled -Target $StartButton -Enabled $true
                Set-UiButtonEnabled -Target $CancelButton -Enabled $false

                if ($exitCode -eq 0) {
                    Set-UiLabelText -Target $StatusLabel -Text "Concluido com sucesso."
                    Append-UiLog -Target $LogBox -Line ""
                    Append-UiLog -Target $LogBox -Line "Processo finalizado com sucesso."
                    $effectiveOutput = $OutputPath
                    if (Test-Path -LiteralPath $resultFilePath) {
                        try {
                            $resultJson = Get-Content -LiteralPath $resultFilePath -Raw -Encoding UTF8
                            $resultObj = $resultJson | ConvertFrom-Json
                            if ($null -ne $resultObj -and -not [string]::IsNullOrWhiteSpace($resultObj.outputPath)) {
                                $effectiveOutput = [string]$resultObj.outputPath
                                Append-UiLog -Target $LogBox -Line ("Arquivo final salvo em: {0}" -f $effectiveOutput)
                            }
                        }
                        catch {
                            Append-UiLog -Target $LogBox -Line "Nao foi possivel ler o arquivo de resultado final."
                        }
                        finally {
                            Remove-Item -LiteralPath $resultFilePath -Force -ErrorAction SilentlyContinue
                        }
                    }
                    else {
                        Append-UiLog -Target $LogBox -Line ("Arquivo final esperado em: {0}" -f $OutputPath)
                    }

                    if (-not (Test-Path -LiteralPath $effectiveOutput)) {
                        $mkvFallback = [System.IO.Path]::ChangeExtension($effectiveOutput, ".mkv")
                        if (Test-Path -LiteralPath $mkvFallback) {
                            Append-UiLog -Target $LogBox -Line ("Arquivo final salvo em (fallback): {0}" -f $mkvFallback)
                        }
                        else {
                            Set-UiLabelText -Target $StatusLabel -Text "Terminou sem arquivo final."
                            Append-UiLog -Target $LogBox -Line "ATENCAO: processo terminou sem arquivo de saida detectado."
                            Append-UiLog -Target $LogBox -Line ("Log de sessao: {0}" -f $script:sessionLogFile)
                        }
                    }
                }
                else {
                    Set-UiLabelText -Target $StatusLabel -Text "Falhou. Veja o log abaixo."
                    Append-UiLog -Target $LogBox -Line ""
                    Append-UiLog -Target $LogBox -Line "Processo terminou com erro."
                    Append-UiLog -Target $LogBox -Line ("Log de sessao: {0}" -f $script:sessionLogFile)
                    if (Test-Path -LiteralPath $resultFilePath) {
                        Remove-Item -LiteralPath $resultFilePath -Force -ErrorAction SilentlyContinue
                    }
                }
            }
            catch {
                # Ignore callback errors to avoid closing the GUI host.
            }
        })

    $started = $process.Start()
    if (-not $started) {
        throw "Nao foi possivel iniciar o processo de compressao."
    }

    $script:activeProcess = $process
    Set-UiButtonEnabled -Target $StartButton -Enabled $false
    Set-UiButtonEnabled -Target $CancelButton -Enabled $true
    Set-UiLabelText -Target $StatusLabel -Text "Comprimindo..."

    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
}

function Start-GuiMode {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "MP4 Compressor por Tamanho"
    $form.StartPosition = "CenterScreen"
    $form.Size = New-Object System.Drawing.Size(980, 700)
    $form.MinimumSize = New-Object System.Drawing.Size(980, 700)
    $form.MaximizeBox = $true

    $lblInput = New-Object System.Windows.Forms.Label
    $lblInput.Text = "Arquivo MP4 de entrada:"
    $lblInput.Location = New-Object System.Drawing.Point(15, 16)
    $lblInput.AutoSize = $true

    $txtInput = New-Object System.Windows.Forms.TextBox
    $txtInput.Location = New-Object System.Drawing.Point(15, 38)
    $txtInput.Size = New-Object System.Drawing.Size(840, 24)
    $txtInput.Anchor = "Top, Left, Right"

    if (-not [string]::IsNullOrWhiteSpace($InputFile)) {
        $txtInput.Text = $InputFile
    }

    $btnInput = New-Object System.Windows.Forms.Button
    $btnInput.Text = "Selecionar..."
    $btnInput.Location = New-Object System.Drawing.Point(865, 36)
    $btnInput.Size = New-Object System.Drawing.Size(90, 28)
    $btnInput.Anchor = "Top, Right"

    $lblTarget = New-Object System.Windows.Forms.Label
    $lblTarget.Text = "Tamanho alvo (MB):"
    $lblTarget.Location = New-Object System.Drawing.Point(15, 78)
    $lblTarget.AutoSize = $true

    $txtTarget = New-Object System.Windows.Forms.TextBox
    $txtTarget.Location = New-Object System.Drawing.Point(15, 100)
    $txtTarget.Size = New-Object System.Drawing.Size(150, 24)

    $targetDefault = Get-DefaultTargetFromInputOrFallback -Fallback $TargetSizeMB
    $txtTarget.Text = $targetDefault.ToString("0.##", [System.Globalization.CultureInfo]::InvariantCulture)

    $lblHeight = New-Object System.Windows.Forms.Label
    $lblHeight.Text = "Altura maxima (px):"
    $lblHeight.Location = New-Object System.Drawing.Point(190, 78)
    $lblHeight.AutoSize = $true

    $txtHeight = New-Object System.Windows.Forms.TextBox
    $txtHeight.Location = New-Object System.Drawing.Point(190, 100)
    $txtHeight.Size = New-Object System.Drawing.Size(150, 24)
    $normalizedInitialHeight = Normalize-MaxHeight -Value $MaxHeight
    if ($null -eq $normalizedInitialHeight) {
        $txtHeight.Text = "720"
    }
    else {
        $txtHeight.Text = $normalizedInitialHeight.ToString()
    }

    $lblFps = New-Object System.Windows.Forms.Label
    $lblFps.Text = "FPS:"
    $lblFps.Location = New-Object System.Drawing.Point(365, 78)
    $lblFps.AutoSize = $true

    $txtFps = New-Object System.Windows.Forms.TextBox
    $txtFps.Location = New-Object System.Drawing.Point(365, 100)
    $txtFps.Size = New-Object System.Drawing.Size(120, 24)
    $normalizedInitialFps = Normalize-Fps -Value $Fps
    if ($null -eq $normalizedInitialFps) {
        $txtFps.Text = "24"
    }
    else {
        $txtFps.Text = $normalizedInitialFps.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
    }

    $lblMinEstimate = New-Object System.Windows.Forms.Label
    $lblMinEstimate.Text = "Minimo estimado: selecione um arquivo."
    $lblMinEstimate.Location = New-Object System.Drawing.Point(15, 132)
    $lblMinEstimate.AutoSize = $true

    $lblHints = New-Object System.Windows.Forms.Label
    $lblHints.Text = "Dica: use 0 em Altura/FPS para manter original. Audio sempre e copiado sem reencode."
    $lblHints.Location = New-Object System.Drawing.Point(15, 152)
    $lblHints.AutoSize = $true

    $lblOutput = New-Object System.Windows.Forms.Label
    $lblOutput.Text = "Arquivo de saida (MP4 ou MKV):"
    $lblOutput.Location = New-Object System.Drawing.Point(15, 180)
    $lblOutput.AutoSize = $true

    $txtOutput = New-Object System.Windows.Forms.TextBox
    $txtOutput.Location = New-Object System.Drawing.Point(15, 202)
    $txtOutput.Size = New-Object System.Drawing.Size(725, 24)
    $txtOutput.Anchor = "Top, Left, Right"

    $btnSuggest = New-Object System.Windows.Forms.Button
    $btnSuggest.Text = "Sugerir"
    $btnSuggest.Location = New-Object System.Drawing.Point(750, 200)
    $btnSuggest.Size = New-Object System.Drawing.Size(95, 28)
    $btnSuggest.Anchor = "Top, Right"

    $btnOutput = New-Object System.Windows.Forms.Button
    $btnOutput.Text = "Salvar como..."
    $btnOutput.Location = New-Object System.Drawing.Point(855, 200)
    $btnOutput.Size = New-Object System.Drawing.Size(100, 28)
    $btnOutput.Anchor = "Top, Right"

    $btnStart = New-Object System.Windows.Forms.Button
    $btnStart.Text = "Comprimir"
    $btnStart.Location = New-Object System.Drawing.Point(15, 244)
    $btnStart.Size = New-Object System.Drawing.Size(130, 34)

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = "Cancelar"
    $btnCancel.Location = New-Object System.Drawing.Point(155, 244)
    $btnCancel.Size = New-Object System.Drawing.Size(120, 34)
    $btnCancel.Enabled = $false

    $lblStatus = New-Object System.Windows.Forms.Label
    $lblStatus.Text = "Pronto."
    $lblStatus.Location = New-Object System.Drawing.Point(15, 288)
    $lblStatus.AutoSize = $true

    $txtLog = New-Object System.Windows.Forms.TextBox
    $txtLog.Location = New-Object System.Drawing.Point(15, 314)
    $txtLog.Size = New-Object System.Drawing.Size(940, 330)
    $txtLog.Multiline = $true
    $txtLog.ScrollBars = "Vertical"
    $txtLog.ReadOnly = $true
    $txtLog.Anchor = "Top, Bottom, Left, Right"

    $form.Controls.AddRange(@(
            $lblInput,
            $txtInput,
            $btnInput,
            $lblTarget,
            $txtTarget,
            $lblHeight,
            $txtHeight,
            $lblFps,
            $txtFps,
            $lblMinEstimate,
            $lblHints,
            $lblOutput,
            $txtOutput,
            $btnSuggest,
            $btnOutput,
            $btnStart,
            $btnCancel,
            $lblStatus,
            $txtLog
        ))

    $setSuggestedOutput = {
        try {
            $inputPath = $txtInput.Text.Trim().Trim('"')
            if ([string]::IsNullOrWhiteSpace($inputPath)) {
                return
            }

            $targetMbParsed = Parse-PositiveDouble -Raw $txtTarget.Text -FieldName "Tamanho alvo"
            $txtOutput.Text = Get-DefaultOutputName -InputPath $inputPath -TargetMb $targetMbParsed
        }
        catch {
            # Ignore temporary invalid user typing.
        }
    }

    $updateMinimumEstimate = {
        try {
            $inputPath = $txtInput.Text.Trim().Trim('"')
            if ([string]::IsNullOrWhiteSpace($inputPath) -or -not (Test-Path -LiteralPath $inputPath)) {
                $lblMinEstimate.Text = "Minimo estimado: selecione um arquivo."
                return
            }

            $lblMinEstimate.Text = "Minimo estimado: calculando..."
            $form.UseWaitCursor = $true
            $form.Refresh()

            $estimate = Get-MinimumTargetEstimateInfo -FilePath $inputPath
            $audioText = if ([double]$estimate.AudioKbps -gt 0) {
                ("{0:0.#} kbps" -f [double]$estimate.AudioKbps)
            }
            else {
                "desconhecido/sem audio"
            }

            $lblMinEstimate.Text = ("Minimo estimado: ~{0:N2} MB | Recomendado >= {1:N2} MB | Audio: {2}" -f ([double]$estimate.MinimumMb), ([double]$estimate.RecommendedMb), $audioText)
        }
        catch {
            $lblMinEstimate.Text = "Minimo estimado: nao foi possivel calcular."
            Append-UiLog -Target $txtLog -Line ("Estimativa minima indisponivel: {0}" -f $_.Exception.Message)
        }
        finally {
            $form.UseWaitCursor = $false
        }
    }

    if ([string]::IsNullOrWhiteSpace($txtOutput.Text) -and -not [string]::IsNullOrWhiteSpace($txtInput.Text)) {
        & $setSuggestedOutput
    }

    if (-not [string]::IsNullOrWhiteSpace($txtInput.Text)) {
        & $updateMinimumEstimate
    }

    Append-UiLog -Target $txtLog -Line "Interface iniciada."
    Append-UiLog -Target $txtLog -Line ("Log de sessao: {0}" -f $script:sessionLogFile)

    $btnInput.Add_Click({
            $dialog = New-Object System.Windows.Forms.OpenFileDialog
            $dialog.Filter = "Video MP4|*.mp4|Video files|*.mp4;*.mkv;*.mov;*.webm|All files|*.*"
            $dialog.Multiselect = $false

            if (-not [string]::IsNullOrWhiteSpace($txtInput.Text) -and (Test-Path -LiteralPath $txtInput.Text.Trim().Trim('"'))) {
                $dialog.FileName = $txtInput.Text.Trim().Trim('"')
            }

            $result = $dialog.ShowDialog()
            if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
                $txtInput.Text = $dialog.FileName
                & $setSuggestedOutput
                & $updateMinimumEstimate
            }
        })

    $txtInput.Add_Leave({
            & $setSuggestedOutput
            & $updateMinimumEstimate
        })

    $btnSuggest.Add_Click({
            & $setSuggestedOutput
            & $updateMinimumEstimate
        })

    $btnOutput.Add_Click({
            $dialog = New-Object System.Windows.Forms.SaveFileDialog
            $dialog.Filter = "MP4 file|*.mp4|MKV file|*.mkv|All files|*.*"
            $dialog.OverwritePrompt = $true

            if (-not [string]::IsNullOrWhiteSpace($txtOutput.Text)) {
                $dialog.FileName = $txtOutput.Text
            }
            elseif (-not [string]::IsNullOrWhiteSpace($txtInput.Text)) {
                try {
                    $inputPath = $txtInput.Text.Trim().Trim('"')
                    $targetMbParsed = Parse-PositiveDouble -Raw $txtTarget.Text -FieldName "Tamanho alvo"
                    $dialog.FileName = Get-DefaultOutputName -InputPath $inputPath -TargetMb $targetMbParsed
                }
                catch {
                    # Ignore and use default dialog behavior.
                }
            }

            $result = $dialog.ShowDialog()
            if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
                $txtOutput.Text = $dialog.FileName
            }
        })

    $btnStart.Add_Click({
            try {
                $inputPath = $txtInput.Text.Trim().Trim('"')
                if ([string]::IsNullOrWhiteSpace($inputPath)) {
                    throw "Informe o arquivo de entrada."
                }

                if (-not (Test-Path -LiteralPath $inputPath)) {
                    throw "Arquivo nao encontrado: $inputPath"
                }

                $targetMbParsed = Parse-PositiveDouble -Raw $txtTarget.Text -FieldName "Tamanho alvo"
                $heightParsed = Parse-NonNegativeIntOrNull -Raw $txtHeight.Text -FieldName "Altura maxima" -DefaultValue 720
                $fpsParsed = Parse-NonNegativeDoubleOrNull -Raw $txtFps.Text -FieldName "FPS" -DefaultValue 24

                $heightParsed = Normalize-MaxHeight -Value $heightParsed
                $fpsParsed = Normalize-Fps -Value $fpsParsed

                $outputPath = $txtOutput.Text.Trim().Trim('"')
                if ([string]::IsNullOrWhiteSpace($outputPath)) {
                    $outputPath = Get-DefaultOutputName -InputPath $inputPath -TargetMb $targetMbParsed
                    $txtOutput.Text = $outputPath
                }

                Append-UiLog -Target $txtLog -Line ("Saida configurada para: {0}" -f $outputPath)

                Start-GuiCompressionProcess `
                    -ScriptPath $PSCommandPath `
                    -InputPath $inputPath `
                    -TargetMb $targetMbParsed `
                    -MaxHeightValue $heightParsed `
                    -FpsValue $fpsParsed `
                    -OutputPath $outputPath `
                    -LogBox $txtLog `
                    -StatusLabel $lblStatus `
                    -StartButton $btnStart `
                    -CancelButton $btnCancel
            }
            catch {
                [System.Windows.Forms.MessageBox]::Show(
                    $_.Exception.Message,
                    "Erro de validacao",
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Error
                ) | Out-Null
            }
        })

    $btnCancel.Add_Click({
            if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
                try {
                    $script:activeProcess.Kill()
                    Append-UiLog -Target $txtLog -Line "Processo cancelado pelo usuario."
                    Set-UiLabelText -Target $lblStatus -Text "Cancelado."
                    Set-UiButtonEnabled -Target $btnStart -Enabled $true
                    Set-UiButtonEnabled -Target $btnCancel -Enabled $false
                }
                catch {
                    [System.Windows.Forms.MessageBox]::Show(
                        "Nao foi possivel cancelar: $($_.Exception.Message)",
                        "Aviso",
                        [System.Windows.Forms.MessageBoxButtons]::OK,
                        [System.Windows.Forms.MessageBoxIcon]::Warning
                    ) | Out-Null
                }
            }
        })

    $form.Add_FormClosing({
            if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
                $confirm = [System.Windows.Forms.MessageBox]::Show(
                    "Existe compressao em andamento. Fechar e cancelar?",
                    "Compressor MP4",
                    [System.Windows.Forms.MessageBoxButtons]::YesNo,
                    [System.Windows.Forms.MessageBoxIcon]::Question
                )

                if ($confirm -eq [System.Windows.Forms.DialogResult]::No) {
                    $_.Cancel = $true
                    return
                }

                try {
                    $script:activeProcess.Kill()
                }
                catch {
                    # Ignore close-time errors.
                }
            }
        })

    [void]$form.ShowDialog()
}

function Run-NoGuiMode {
    Write-Host "NoGui iniciado."

    if ([string]::IsNullOrWhiteSpace($InputFile)) {
        if ($NoPrompt) {
            throw "InputFile nao informado para execucao NoGui."
        }
        else {
            $InputFile = Read-Host "Arquivo de entrada (mp4)"
        }
    }

    if ([string]::IsNullOrWhiteSpace($InputFile)) {
        throw "Arquivo de entrada nao informado."
    }

    $InputFile = $InputFile.Trim().Trim('"')

    if ($TargetSizeMB -le 0) {
        if ($NoPrompt) {
            throw "TargetSizeMB nao informado para execucao NoGui."
        }
        else {
            $targetRaw = Read-Host "Tamanho alvo em MB [2.5]"
            if ([string]::IsNullOrWhiteSpace($targetRaw)) {
                $TargetSizeMB = 2.5
            }
            else {
                $TargetSizeMB = Parse-PositiveDouble -Raw $targetRaw -FieldName "Tamanho alvo"
            }
        }
    }

    if ($null -eq $MaxHeight -or $MaxHeight -lt 0) {
        if ($NoPrompt) {
            throw "MaxHeight nao informado para execucao NoGui."
        }
        else {
            $heightRaw = Read-Host "Altura maxima (px) [720] (0 = manter original)"
            $MaxHeight = Parse-NonNegativeIntOrNull -Raw $heightRaw -FieldName "Altura maxima" -DefaultValue 720
        }
    }

    if ($null -eq $Fps -or $Fps -lt 0) {
        if ($NoPrompt) {
            throw "Fps nao informado para execucao NoGui."
        }
        else {
            $fpsRaw = Read-Host "FPS de saida [24] (0 = manter original)"
            $Fps = Parse-NonNegativeDoubleOrNull -Raw $fpsRaw -FieldName "FPS" -DefaultValue 24
        }
    }

    $MaxHeight = Normalize-MaxHeight -Value $MaxHeight
    $Fps = Normalize-Fps -Value $Fps

    if ([string]::IsNullOrWhiteSpace($OutputFile)) {
        if ($NoPrompt) {
            throw "OutputFile nao informado para execucao NoGui."
        }
        else {
            $defaultOut = Get-DefaultOutputName -InputPath $InputFile -TargetMb $TargetSizeMB
            $outputRaw = Read-Host ("Arquivo de saida [{0}]" -f $defaultOut)

            if ([string]::IsNullOrWhiteSpace($outputRaw)) {
                $OutputFile = $defaultOut
            }
            else {
                $OutputFile = $outputRaw.Trim().Trim('"')
            }
        }
    }

    $result = Invoke-TargetCompression -InputPath $InputFile -TargetMb $TargetSizeMB -MaxHeightValue $MaxHeight -FpsValue $Fps -OutputPath $OutputFile

    if (-not [string]::IsNullOrWhiteSpace($ResultFile)) {
        try {
            $payload = @{
                outputPath = $result.OutputPath
                finalBytes = $result.FinalBytes
                targetBytes = $result.TargetBytes
            }

            $json = $payload | ConvertTo-Json -Compress
            Set-Content -LiteralPath $ResultFile -Value $json -Encoding UTF8 -Force
        }
        catch {
            # Ignore result-file write errors; compression already completed.
        }
    }
}

try {
    if ($NoGui) {
        Run-NoGuiMode
    }
    else {
        Start-GuiMode
    }
}
catch {
    if ($NoGui) {
        Write-Error $_.Exception.Message
        exit 1
    }

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $_.Exception.Message,
            "Erro",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    }
    catch {
        Write-Error $_.Exception.Message
    }

    exit 1
}
