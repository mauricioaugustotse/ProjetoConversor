Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:activeProcess = $null
$script:downloadLogBox = $null
$script:downloadStatusLabel = $null
$script:downloadButton = $null
$script:cancelButton = $null
$script:downloadProgressBar = $null
$script:downloadTimer = $null
$script:downloadFinished = $false
$script:downloadExpectedPath = $null
$script:downloadActualPath = $null
$script:downloadPollFile = $null
$script:downloadPollPosition = 0
$script:sessionLogFile = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_gui_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Parse-IntegerPart {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$PartName
    )

    $clean = $Value.Trim()
    if ($clean -notmatch '^\d+$') {
        throw "Invalid $PartName value: '$Value'."
    }

    return [int]$clean
}

function Parse-DoublePart {
    param(
        [Parameter(Mandatory = $true)][string]$Value,
        [Parameter(Mandatory = $true)][string]$PartName
    )

    $clean = $Value.Trim().Replace(',', '.')
    [double]$number = 0

    $ok = [double]::TryParse(
        $clean,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )

    if (-not $ok) {
        throw "Invalid $PartName value: '$Value'."
    }

    if ($number -lt 0) {
        throw "$PartName cannot be negative."
    }

    return $number
}

function Convert-TimeInputToSeconds {
    param([Parameter(Mandatory = $true)][string]$InputText)

    if ([string]::IsNullOrWhiteSpace($InputText)) {
        throw "Time field is empty."
    }

    $parts = $InputText.Trim().Split(':')

    switch ($parts.Count) {
        1 {
            return Parse-DoublePart -Value $parts[0] -PartName "seconds"
        }
        2 {
            $minutes = Parse-IntegerPart -Value $parts[0] -PartName "minutes"
            $seconds = Parse-DoublePart -Value $parts[1] -PartName "seconds"

            if ($seconds -ge 60) {
                throw "For MM:SS, seconds must be lower than 60."
            }

            return ($minutes * 60) + $seconds
        }
        3 {
            $hours = Parse-IntegerPart -Value $parts[0] -PartName "hours"
            $minutes = Parse-IntegerPart -Value $parts[1] -PartName "minutes"
            $seconds = Parse-DoublePart -Value $parts[2] -PartName "seconds"

            if ($minutes -ge 60) {
                throw "For HH:MM:SS, minutes must be lower than 60."
            }

            if ($seconds -ge 60) {
                throw "For HH:MM:SS, seconds must be lower than 60."
            }

            return ($hours * 3600) + ($minutes * 60) + $seconds
        }
        default {
            throw "Invalid time format. Use HH:MM:SS, MM:SS or SS."
        }
    }
}

function Format-SecondsForYtDlp {
    param([Parameter(Mandatory = $true)][double]$TotalSeconds)

    if ($TotalSeconds -lt 0) {
        throw "Time cannot be negative."
    }

    $rounded = [Math]::Round($TotalSeconds, 3)
    $hours = [int][Math]::Floor($rounded / 3600)
    $minutes = [int][Math]::Floor(($rounded % 3600) / 60)
    $seconds = $rounded - ($hours * 3600) - ($minutes * 60)

    if ($seconds -ge 59.9995) {
        $seconds = 0
        $minutes += 1
    }

    if ($minutes -ge 60) {
        $minutes = 0
        $hours += 1
    }

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

function Convert-TitleToOutputBaseName {
    param([Parameter(Mandatory = $true)][string]$Title)

    $clean = $Title.Trim()
    $clean = $clean -replace '\s*[|｜]\s*.+$', ''
    $clean = $clean -replace '\s*[：:]\s*', ' - '
    $clean = $clean -replace '[<>"/\\?*\x00-\x1F]', ' '
    $clean = $clean -replace '\s+', ' '
    $clean = $clean -replace '\s+-\s+', ' - '
    $clean = $clean.Trim().TrimEnd('.', ' ')

    if ([string]::IsNullOrWhiteSpace($clean)) {
        return "youtube_clip"
    }

    return $clean
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
    if (($urlHost -eq "youtu.be") -or ($urlHost -eq "www.youtu.be")) {
        $path = $uri.AbsolutePath.Trim("/")
        if ([string]::IsNullOrWhiteSpace($path)) {
            throw "Short YouTube URL does not contain a video id."
        }

        $videoId = $path.Split("/")[0]
        $normalized = "https://www.youtube.com/watch?v={0}" -f [System.Uri]::EscapeDataString($videoId)

        $query = $uri.Query
        if (-not [string]::IsNullOrWhiteSpace($query)) {
            $query = $query.TrimStart("?")
            if (-not [string]::IsNullOrWhiteSpace($query)) {
                $normalized = "{0}&{1}" -f $normalized, $query
            }
        }

        return $normalized
    }

    return $raw
}

function Invoke-ToolCaptureSingleLine {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments,
        [int]$TimeoutMilliseconds = 30000
    )

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
        throw "Nao foi possivel iniciar comando auxiliar."
    }

    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()

    if (-not $process.WaitForExit($TimeoutMilliseconds)) {
        try {
            $process.Kill()
        }
        catch {
            # Ignore cleanup errors.
        }
        throw "Tempo esgotado ao consultar titulo do YouTube."
    }

    $stdout = $stdoutTask.Result
    $stderr = $stderrTask.Result

    if ($process.ExitCode -ne 0) {
        $message = if ([string]::IsNullOrWhiteSpace($stderr)) { $stdout } else { $stderr }
        throw ("Falha ao consultar titulo do YouTube: {0}" -f $message.Trim())
    }

    $line = ($stdout -split "\r?\n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "yt-dlp nao retornou titulo do YouTube."
    }

    return $line.Trim()
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

function Append-Log {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.TextBox]$Target,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Line
    )

    $consoleLine = if ([string]::IsNullOrEmpty($Line)) {
        ""
    }
    else {
        "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Line
    }

    $fileLine = if ([string]::IsNullOrEmpty($Line)) {
        ""
    }
    else {
        "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Line
    }

    try {
        [System.IO.File]::AppendAllText($script:sessionLogFile, $fileLine + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
    }
    catch {
        # Ignore disk logging errors.
    }

    if (-not [string]::IsNullOrEmpty($consoleLine)) {
        try {
            [Console]::WriteLine($consoleLine)
        }
        catch {
            # Ignore console output errors.
        }
    }

    $writer = {
        param([System.Windows.Forms.TextBox]$Box, [string]$Text)
        $Box.AppendText($Text + [Environment]::NewLine)
        $Box.SelectionStart = $Box.TextLength
        $Box.ScrollToCaret()
    }

    try {
        if (-not $Target.IsDisposed) {
            if ($Target.InvokeRequired) {
                $Target.BeginInvoke($writer, $Target, $Line) | Out-Null
            }
            else {
                & $writer $Target $Line
            }
        }
    }
    catch {
        # Ignore UI logging errors while form is closing/disposed.
    }
}

function Set-StatusText {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.Label]$Target,
        [Parameter(Mandatory = $true)][string]$Text
    )

    $setter = {
        param([System.Windows.Forms.Label]$Label, [string]$Message)
        $Label.Text = $Message
    }

    if ($Target.InvokeRequired) {
        $Target.BeginInvoke($setter, $Target, $Text) | Out-Null
    }
    else {
        & $setter $Target $Text
    }
}

function Set-ButtonState {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$Target,
        [Parameter(Mandatory = $true)][bool]$Enabled
    )

    $setter = {
        param([System.Windows.Forms.Button]$Button, [bool]$State)
        $Button.Enabled = $State
    }

    if ($Target.InvokeRequired) {
        $Target.BeginInvoke($setter, $Target, $Enabled) | Out-Null
    }
    else {
        & $setter $Target $Enabled
    }
}

function Set-ProgressMarquee {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.ProgressBar]$Target,
        [Parameter(Mandatory = $true)][bool]$Enabled
    )

    $setter = {
        param([System.Windows.Forms.ProgressBar]$Bar, [bool]$State)
        if ($State) {
            $Bar.Style = [System.Windows.Forms.ProgressBarStyle]::Marquee
            $Bar.MarqueeAnimationSpeed = 30
            $Bar.Value = 0
        }
        else {
            $Bar.MarqueeAnimationSpeed = 0
            $Bar.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous
        }
    }

    if ($Target.InvokeRequired) {
        $Target.BeginInvoke($setter, $Target, $Enabled) | Out-Null
    }
    else {
        & $setter $Target $Enabled
    }
}

function Set-ProgressPercent {
    param(
        [Parameter(Mandatory = $true)][System.Windows.Forms.ProgressBar]$Target,
        [Parameter(Mandatory = $true)][int]$Value
    )

    $clamped = [Math]::Max(0, [Math]::Min(100, $Value))

    $setter = {
        param([System.Windows.Forms.ProgressBar]$Bar, [int]$Percent)
        if ($Bar.Style -ne [System.Windows.Forms.ProgressBarStyle]::Continuous) {
            $Bar.MarqueeAnimationSpeed = 0
            $Bar.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous
        }
        $Bar.Value = $Percent
    }

    if ($Target.InvokeRequired) {
        $Target.BeginInvoke($setter, $Target, $clamped) | Out-Null
    }
    else {
        & $setter $Target $clamped
    }
}

function Complete-YtDlpDownload {
    param([Parameter(Mandatory = $true)][int]$ExitCode)

    if ($script:downloadFinished) {
        return
    }

    $script:downloadFinished = $true

    if ($null -ne $script:downloadTimer) {
        try {
            $script:downloadTimer.Stop()
            $script:downloadTimer.Dispose()
        }
        catch {
            # Ignore timer cleanup errors.
        }
        $script:downloadTimer = $null
    }

    $script:activeProcess = $null
    $script:downloadPollFile = $null
    $script:downloadPollPosition = 0

    Set-ButtonState -Target $script:downloadButton -Enabled $true
    Set-ButtonState -Target $script:cancelButton -Enabled $false

    if ($ExitCode -eq 0) {
        if ($null -ne $script:downloadProgressBar) {
            Set-ProgressPercent -Target $script:downloadProgressBar -Value 100
        }

        Set-StatusText -Target $script:downloadStatusLabel -Text "Done. Clip downloaded successfully."
        Append-Log -Target $script:downloadLogBox -Line ""
        Append-Log -Target $script:downloadLogBox -Line "Download finished successfully."

        if (-not [string]::IsNullOrWhiteSpace($script:downloadActualPath) -and (Test-Path -LiteralPath $script:downloadActualPath)) {
            Append-Log -Target $script:downloadLogBox -Line ("Arquivo final encontrado: {0}" -f $script:downloadActualPath)
        }
        elseif (-not [string]::IsNullOrWhiteSpace($script:downloadExpectedPath)) {
            if (Test-Path -LiteralPath $script:downloadExpectedPath) {
                Append-Log -Target $script:downloadLogBox -Line ("Arquivo final encontrado: {0}" -f $script:downloadExpectedPath)
            }
            else {
                Append-Log -Target $script:downloadLogBox -Line ("ATENCAO: caminho esperado nao encontrado: {0}" -f $script:downloadExpectedPath)
                Append-Log -Target $script:downloadLogBox -Line "Verifique mensagens acima para saber se o yt-dlp salvou com outro nome."
            }
        }

        Append-Log -Target $script:downloadLogBox -Line "Window kept open. You can close it when finished reviewing the log."
    }
    else {
        if ($null -ne $script:downloadProgressBar) {
            Set-ProgressMarquee -Target $script:downloadProgressBar -Enabled $false
        }

        Set-StatusText -Target $script:downloadStatusLabel -Text "Failed. Check log for details."
        Append-Log -Target $script:downloadLogBox -Line ""
        Append-Log -Target $script:downloadLogBox -Line ("Download failed with exit code {0}." -f $ExitCode)
        if (-not [string]::IsNullOrWhiteSpace($script:downloadExpectedPath)) {
            Append-Log -Target $script:downloadLogBox -Line ("Arquivo esperado seria: {0}" -f $script:downloadExpectedPath)
        }
        Append-Log -Target $script:downloadLogBox -Line "Window kept open so you can review the error above."
    }
}

function Handle-DownloadOutputLine {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }

    $cleanLine = $Line.Trim()
    Append-Log -Target $script:downloadLogBox -Line $cleanLine

    if ($cleanLine -match '\[download\]\s+([0-9]+(?:\.[0-9]+)?)%') {
        Set-ProgressPercent -Target $script:downloadProgressBar -Value ([int][Math]::Round([double]::Parse($matches[1], [System.Globalization.CultureInfo]::InvariantCulture)))
    }
    elseif ($cleanLine -like "Cutting locally*") {
        Set-ProgressMarquee -Target $script:downloadProgressBar -Enabled $true
    }
    elseif ($cleanLine -eq "progress=end") {
        Set-ProgressPercent -Target $script:downloadProgressBar -Value 100
    }

    if (($cleanLine -match '\.(mp4|mp3|mkv|webm)$') -and (Test-Path -LiteralPath $cleanLine)) {
        $script:downloadActualPath = $cleanLine
    }
}

function Read-PolledDownloadLog {
    if ([string]::IsNullOrWhiteSpace($script:downloadPollFile)) {
        return
    }

    if (-not (Test-Path -LiteralPath $script:downloadPollFile)) {
        return
    }

    $stream = $null
    $reader = $null
    try {
        $stream = [System.IO.File]::Open(
            $script:downloadPollFile,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        [void]$stream.Seek($script:downloadPollPosition, [System.IO.SeekOrigin]::Begin)
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8, $true, 4096, $true)
        $text = $reader.ReadToEnd()
        $script:downloadPollPosition = $stream.Position
    }
    catch {
        return
    }
    finally {
        if ($null -ne $reader) {
            $reader.Dispose()
        }
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }

    if ([string]::IsNullOrWhiteSpace($text)) {
        return
    }

    foreach ($line in ($text -split "\r?\n")) {
        Handle-DownloadOutputLine -Line $line
    }
}

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

    foreach ($pythonScriptsDir in (Get-PythonScriptsDirectories)) {
        Add-ToProcessPathIfExists -PathEntry $pythonScriptsDir
    }
}

function Get-PythonInstallRoots {
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    $appData = [Environment]::GetFolderPath("ApplicationData")

    $roots = @()
    $pythonInstallRoot = Join-Path -Path $localAppData -ChildPath "Programs\Python"
    if (Test-Path -LiteralPath $pythonInstallRoot) {
        $roots += Get-ChildItem -LiteralPath $pythonInstallRoot -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
            Sort-Object -Property Name -Descending |
            ForEach-Object { $_.FullName }
    }

    $pythonRoamingRoot = Join-Path -Path $appData -ChildPath "Python"
    if (Test-Path -LiteralPath $pythonRoamingRoot) {
        $roots += Get-ChildItem -LiteralPath $pythonRoamingRoot -Directory -Filter "Python*" -ErrorAction SilentlyContinue |
            Sort-Object -Property Name -Descending |
            ForEach-Object { $_.FullName }
    }

    return @($roots | Select-Object -Unique)
}

function Get-PythonScriptsDirectories {
    $dirs = @()

    foreach ($root in (Get-PythonInstallRoots)) {
        $scriptsDir = Join-Path -Path $root -ChildPath "Scripts"
        if (Test-Path -LiteralPath $scriptsDir) {
            $dirs += $scriptsDir
        }
    }

    return @($dirs | Select-Object -Unique)
}

function Get-PythonToolCandidates {
    param([Parameter(Mandatory = $true)][string]$ExecutableName)

    $candidates = @()

    foreach ($scriptsDir in (Get-PythonScriptsDirectories)) {
        $candidates += Join-Path -Path $scriptsDir -ChildPath $ExecutableName
    }

    return @($candidates | Select-Object -Unique)
}

function Resolve-PythonYtDlpPathOrNull {
    foreach ($root in (Get-PythonInstallRoots)) {
        $pythonExe = Join-Path -Path $root -ChildPath "python.exe"
        $moduleDir = Join-Path -Path $root -ChildPath "Lib\site-packages\yt_dlp"

        if ((Test-Path -LiteralPath $pythonExe) -and (Test-Path -LiteralPath $moduleDir)) {
            return $pythonExe
        }
    }

    return $null
}

function Get-ImageioFfmpegCandidates {
    $candidates = @()

    foreach ($root in (Get-PythonInstallRoots)) {
        $binariesDir = Join-Path -Path $root -ChildPath "Lib\site-packages\imageio_ffmpeg\binaries"
        if (Test-Path -LiteralPath $binariesDir) {
            $candidates += Get-ChildItem -LiteralPath $binariesDir -File -Filter "ffmpeg*.exe" -ErrorAction SilentlyContinue |
                ForEach-Object { $_.FullName }
        }
    }

    return @($candidates | Select-Object -Unique)
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

function Start-YtDlpDownload {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string[]]$Arguments,
        [Parameter(Mandatory = $true)][System.Windows.Forms.TextBox]$LogBox,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Label]$StatusLabel,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$DownloadButton,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$CancelButton,
        [Parameter(Mandatory = $true)][System.Windows.Forms.ProgressBar]$ProgressBar,
        [AllowEmptyString()][string]$ExpectedOutputPath = "",
        [AllowEmptyString()][string]$PollLogFile = ""
    )

    if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
        throw "A download is already running."
    }

    $argumentText = Join-Arguments -Args $Arguments
    $executableName = Split-Path -Leaf $ExecutablePath
    Append-Log -Target $LogBox -Line ""
    Append-Log -Target $LogBox -Line ("> {0} {1}" -f $executableName, $argumentText)
    Append-Log -Target $LogBox -Line "Starting download. Keep this window open to follow progress."
    if (-not [string]::IsNullOrWhiteSpace($ExpectedOutputPath)) {
        Append-Log -Target $LogBox -Line ("Arquivo final esperado: {0}" -f $ExpectedOutputPath)
    }
    Append-Log -Target $LogBox -Line ""

    $usePollLog = -not [string]::IsNullOrWhiteSpace($PollLogFile)
    if ($usePollLog) {
        try {
            [System.IO.File]::WriteAllText($PollLogFile, "", [System.Text.Encoding]::UTF8)
        }
        catch {
            throw "Could not create process log file: $PollLogFile"
        }
        Append-Log -Target $LogBox -Line ("Process progress log: {0}" -f $PollLogFile)
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ExecutablePath
    $psi.Arguments = $argumentText
    $psi.RedirectStandardOutput = -not $usePollLog
    $psi.RedirectStandardError = -not $usePollLog
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $script:downloadLogBox = $LogBox
    $script:downloadStatusLabel = $StatusLabel
    $script:downloadButton = $DownloadButton
    $script:cancelButton = $CancelButton
    $script:downloadProgressBar = $ProgressBar
    $script:downloadExpectedPath = $ExpectedOutputPath
    $script:downloadActualPath = $null
    $script:downloadFinished = $false
    $script:downloadPollFile = if ($usePollLog) { $PollLogFile } else { $null }
    $script:downloadPollPosition = 0

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $process.EnableRaisingEvents = $true

    if (-not $usePollLog) {
        $process.add_OutputDataReceived({
            param($sender, $eventArgs)
            try {
                if (-not [string]::IsNullOrWhiteSpace($eventArgs.Data)) {
                    Handle-DownloadOutputLine -Line $eventArgs.Data
                }
            }
            catch {
                # Ignore callback errors to avoid crashing the UI host.
            }
        })

        $process.add_ErrorDataReceived({
            param($sender, $eventArgs)
            try {
                if (-not [string]::IsNullOrWhiteSpace($eventArgs.Data)) {
                    Handle-DownloadOutputLine -Line $eventArgs.Data
                }
            }
            catch {
                # Ignore callback errors to avoid crashing the UI host.
            }
        })
    }

    $started = $process.Start()
    if (-not $started) {
        throw "Could not start yt-dlp process."
    }

    $script:activeProcess = $process
    Set-ButtonState -Target $DownloadButton -Enabled $false
    Set-ButtonState -Target $CancelButton -Enabled $true
    Set-ProgressMarquee -Target $ProgressBar -Enabled $true
    Set-StatusText -Target $StatusLabel -Text "Downloading clip... keep this window open to follow progress."

    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 1000
    $timer.Add_Tick({
            try {
                Read-PolledDownloadLog
                if ($script:activeProcess -and $script:activeProcess.HasExited) {
                    Read-PolledDownloadLog
                    $exitCode = $script:activeProcess.ExitCode
                    Complete-YtDlpDownload -ExitCode $exitCode
                }
            }
            catch {
                # Ignore timer polling errors.
            }
        })
    $script:downloadTimer = $timer
    $timer.Start()

    if (-not $usePollLog) {
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
    }
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "YouTube Clip Downloader"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(860, 620)
$form.MinimumSize = New-Object System.Drawing.Size(860, 620)
$form.MaximizeBox = $true

$lblUrl = New-Object System.Windows.Forms.Label
$lblUrl.Text = "YouTube URL:"
$lblUrl.Location = New-Object System.Drawing.Point(15, 18)
$lblUrl.AutoSize = $true

$txtUrl = New-Object System.Windows.Forms.TextBox
$txtUrl.Location = New-Object System.Drawing.Point(15, 40)
$txtUrl.Size = New-Object System.Drawing.Size(810, 24)
$txtUrl.Anchor = "Top, Left, Right"

$lblStart = New-Object System.Windows.Forms.Label
$lblStart.Text = "Start (HH:MM:SS or MM:SS):"
$lblStart.Location = New-Object System.Drawing.Point(15, 78)
$lblStart.AutoSize = $true

$txtStart = New-Object System.Windows.Forms.TextBox
$txtStart.Location = New-Object System.Drawing.Point(15, 100)
$txtStart.Size = New-Object System.Drawing.Size(180, 24)

$lblEnd = New-Object System.Windows.Forms.Label
$lblEnd.Text = "End (HH:MM:SS or MM:SS):"
$lblEnd.Location = New-Object System.Drawing.Point(220, 78)
$lblEnd.AutoSize = $true

$txtEnd = New-Object System.Windows.Forms.TextBox
$txtEnd.Location = New-Object System.Drawing.Point(220, 100)
$txtEnd.Size = New-Object System.Drawing.Size(180, 24)

$lblMode = New-Object System.Windows.Forms.Label
$lblMode.Text = "Output:"
$lblMode.Location = New-Object System.Drawing.Point(430, 78)
$lblMode.AutoSize = $true

$cmbMode = New-Object System.Windows.Forms.ComboBox
$cmbMode.Location = New-Object System.Drawing.Point(430, 100)
$cmbMode.Size = New-Object System.Drawing.Size(230, 24)
$cmbMode.DropDownStyle = "DropDownList"
[void]$cmbMode.Items.Add("Video MP4 (image + audio)")
[void]$cmbMode.Items.Add("Audio MP3 only")
$cmbMode.SelectedIndex = 0

$lblOutput = New-Object System.Windows.Forms.Label
$lblOutput.Text = "Output folder:"
$lblOutput.Location = New-Object System.Drawing.Point(15, 140)
$lblOutput.AutoSize = $true

$txtOutput = New-Object System.Windows.Forms.TextBox
$txtOutput.Location = New-Object System.Drawing.Point(15, 162)
$txtOutput.Size = New-Object System.Drawing.Size(730, 24)
$txtOutput.Anchor = "Top, Left, Right"

$defaultOut = Join-Path -Path $env:USERPROFILE -ChildPath "Downloads"
$txtOutput.Text = $defaultOut

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Text = "Browse..."
$btnBrowse.Location = New-Object System.Drawing.Point(755, 160)
$btnBrowse.Size = New-Object System.Drawing.Size(70, 28)
$btnBrowse.Anchor = "Top, Right"

$lblHint = New-Object System.Windows.Forms.Label
$lblHint.Text = "Examples: 34 (seconds), 01:10, 00:34:00"
$lblHint.Location = New-Object System.Drawing.Point(15, 198)
$lblHint.AutoSize = $true

$btnDownload = New-Object System.Windows.Forms.Button
$btnDownload.Text = "Download clip"
$btnDownload.Location = New-Object System.Drawing.Point(15, 228)
$btnDownload.Size = New-Object System.Drawing.Size(140, 34)

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = "Cancel"
$btnCancel.Location = New-Object System.Drawing.Point(165, 228)
$btnCancel.Size = New-Object System.Drawing.Size(110, 34)
$btnCancel.Enabled = $false

$lblStatus = New-Object System.Windows.Forms.Label
$lblStatus.Text = "Ready."
$lblStatus.Location = New-Object System.Drawing.Point(15, 274)
$lblStatus.AutoSize = $true

$progressDownload = New-Object System.Windows.Forms.ProgressBar
$progressDownload.Location = New-Object System.Drawing.Point(15, 300)
$progressDownload.Size = New-Object System.Drawing.Size(810, 18)
$progressDownload.Anchor = "Top, Left, Right"
$progressDownload.Minimum = 0
$progressDownload.Maximum = 100
$progressDownload.Value = 0
$progressDownload.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous

$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Location = New-Object System.Drawing.Point(15, 328)
$txtLog.Size = New-Object System.Drawing.Size(810, 242)
$txtLog.Multiline = $true
$txtLog.ScrollBars = "Vertical"
$txtLog.ReadOnly = $true
$txtLog.Anchor = "Top, Bottom, Left, Right"

$form.Controls.AddRange(@(
        $lblUrl,
        $txtUrl,
        $lblStart,
        $txtStart,
        $lblEnd,
        $txtEnd,
        $lblMode,
        $cmbMode,
        $lblOutput,
        $txtOutput,
        $btnBrowse,
        $lblHint,
        $btnDownload,
        $btnCancel,
        $lblStatus,
        $progressDownload,
        $txtLog
    ))

Append-Log -Target $txtLog -Line "GUI started."
Append-Log -Target $txtLog -Line ("Session log file: " + $script:sessionLogFile)

$btnBrowse.Add_Click({
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.SelectedPath = $txtOutput.Text
        $result = $dialog.ShowDialog()
        if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
            $txtOutput.Text = $dialog.SelectedPath
        }
    })

$btnCancel.Add_Click({
        if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
            try {
                $script:activeProcess.Kill()
                if ($null -ne $script:downloadTimer) {
                    $script:downloadTimer.Stop()
                    $script:downloadTimer.Dispose()
                    $script:downloadTimer = $null
                }
                $script:activeProcess = $null
                $script:downloadFinished = $true
                Set-ButtonState -Target $btnDownload -Enabled $true
                Set-ButtonState -Target $btnCancel -Enabled $false
                Set-ProgressMarquee -Target $progressDownload -Enabled $false
                Append-Log -Target $txtLog -Line "Process cancelled by user."
                Set-StatusText -Target $lblStatus -Text "Cancelled."
            }
            catch {
                [System.Windows.Forms.MessageBox]::Show(
                    "Could not cancel the process: $($_.Exception.Message)",
                    "yt clip",
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Warning
                ) | Out-Null
            }
        }
    })

$btnDownload.Add_Click({
        try {
            $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
            $programData = [Environment]::GetFolderPath("CommonApplicationData")
            $userProfile = [Environment]::GetFolderPath("UserProfile")

            $ytDlpCandidates = @(
                (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links\yt-dlp.exe"),
                (Join-Path -Path $programData -ChildPath "chocolatey\bin\yt-dlp.exe"),
                (Join-Path -Path $userProfile -ChildPath "scoop\shims\yt-dlp.exe")
            ) + (Get-PythonToolCandidates -ExecutableName "yt-dlp.exe")

            $ffmpegCandidates = @(
                (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links\ffmpeg.exe"),
                (Join-Path -Path $programData -ChildPath "chocolatey\bin\ffmpeg.exe"),
                (Join-Path -Path $userProfile -ChildPath "scoop\shims\ffmpeg.exe")
            ) + (Get-ImageioFfmpegCandidates)

            $ytDlpPrefixArgs = @()
            $pythonYtDlpPath = Resolve-PythonYtDlpPathOrNull
            if (-not [string]::IsNullOrWhiteSpace($pythonYtDlpPath)) {
                $ytDlpPath = $pythonYtDlpPath
                $ytDlpPrefixArgs = @("-m", "yt_dlp")
            }
            else {
                $ytDlpPath = Resolve-CommandPathOrThrow `
                    -CommandName "yt-dlp" `
                    -CandidatePaths $ytDlpCandidates `
                    -ErrorHint "yt-dlp not found. Install with: winget install yt-dlp.yt-dlp"
            }

            $ffmpegPath = Resolve-CommandPathOrThrow `
                    -CommandName "ffmpeg" `
                    -CandidatePaths $ffmpegCandidates `
                    -ErrorHint "ffmpeg not found. Install with: winget install Gyan.FFmpeg"

            $url = Normalize-YouTubeUrl -InputText $txtUrl.Text

            $startSeconds = Convert-TimeInputToSeconds -InputText $txtStart.Text
            $endSeconds = Convert-TimeInputToSeconds -InputText $txtEnd.Text

            if ($endSeconds -le $startSeconds) {
                throw "End time must be greater than start time."
            }

            $outputDir = $txtOutput.Text.Trim()
            if ([string]::IsNullOrWhiteSpace($outputDir)) {
                throw "Output folder cannot be empty."
            }

            if (-not (Test-Path -LiteralPath $outputDir)) {
                New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
            }

            $startText = Format-SecondsForYtDlp -TotalSeconds $startSeconds
            $endText = Format-SecondsForYtDlp -TotalSeconds $endSeconds
            $section = "*$startText-$endText"
            $clipSuffix = "{0}-{1}" -f (Format-SecondsForFileName -TotalSeconds $startSeconds), (Format-SecondsForFileName -TotalSeconds $endSeconds)
            $expectedOutputPath = ""

            if ($cmbMode.SelectedIndex -eq 0) {
                $workerPath = Join-Path -Path $PSScriptRoot -ChildPath "yt_clip_video_worker.ps1"
                if (-not (Test-Path -LiteralPath $workerPath)) {
                    throw "Video worker script not found: $workerPath"
                }
                $workerLogFile = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_worker_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

                $powershellPath = Join-Path -Path $PSHOME -ChildPath "powershell.exe"
                if (-not (Test-Path -LiteralPath $powershellPath)) {
                    $powershellPath = "powershell.exe"
                }

                $workerArgs = @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $workerPath,
                    "-YtDlpPath", $ytDlpPath,
                    "-FfmpegPath", $ffmpegPath,
                    "-Url", $url,
                    "-Start", $startText,
                    "-End", $endText,
                    "-OutputDir", $outputDir,
                    "-ClipSuffix", $clipSuffix,
                    "-ProgressLogFile", $workerLogFile
                )

                if ($ytDlpPrefixArgs.Count -gt 0) {
                    $workerArgs += "-UsePythonModule"
                }

                Start-YtDlpDownload `
                    -ExecutablePath $powershellPath `
                    -Arguments $workerArgs `
                    -LogBox $txtLog `
                    -StatusLabel $lblStatus `
                    -DownloadButton $btnDownload `
                    -CancelButton $btnCancel `
                    -ProgressBar $progressDownload `
                    -ExpectedOutputPath $expectedOutputPath `
                    -PollLogFile $workerLogFile
            }
            else {
                $ytDlpArgs = @(
                    "--newline",
                    "--no-playlist",
                    "--force-overwrites",
                    "--replace-in-metadata", "title", "\s*[|｜]\s*.+$", "",
                    "--replace-in-metadata", "title", "\s*[：:]\s*", " - ",
                    "--replace-in-metadata", "title", "[/\\]+", "-",
                    "--download-sections", $section,
                    "--paths", $outputDir,
                    "--windows-filenames",
                    "--ffmpeg-location", $ffmpegPath,
                    "--print", "after_move:filepath",
                    "-o", ("%(title)s [{0}].%(ext)s" -f $clipSuffix),
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "0"
                )

                $ytDlpArgs += $url

                Start-YtDlpDownload `
                    -ExecutablePath $ytDlpPath `
                    -Arguments ($ytDlpPrefixArgs + $ytDlpArgs) `
                    -LogBox $txtLog `
                    -StatusLabel $lblStatus `
                    -DownloadButton $btnDownload `
                    -CancelButton $btnCancel `
                    -ProgressBar $progressDownload `
                    -ExpectedOutputPath $expectedOutputPath
            }
        }
        catch {
            Append-Log -Target $txtLog -Line ("ERROR: {0}" -f $_.Exception.Message)
            Set-StatusText -Target $lblStatus -Text "Error. Check log for details."
            Set-ButtonState -Target $btnDownload -Enabled $true
            Set-ButtonState -Target $btnCancel -Enabled $false
            Set-ProgressMarquee -Target $progressDownload -Enabled $false
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                "Validation error",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })

$form.Add_FormClosing({
        if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
            [System.Windows.Forms.MessageBox]::Show(
                "A download is running. Use Cancel before closing this window.",
                "yt clip",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            $_.Cancel = $true
            return
        }
    })

[void]$form.ShowDialog()
