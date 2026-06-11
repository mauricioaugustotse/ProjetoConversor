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

# ---------------------------------------------------------------------------
# Parsing tolerante de tempos e trechos
# ---------------------------------------------------------------------------

function Parse-FlexibleTimeToSeconds {
    # Aceita SS, MM:SS, HH:MM:SS e variantes "sujas": digitos colados (1:0056 =
    # 1:00:56, 813 apos ':' = 8:13), zeros ausentes (01:8:13) e fracoes (1:02.5).
    param([Parameter(Mandatory = $true)][string]$Text)

    $clean = $Text.Trim() -replace '[hH]', ':' -replace '[mM]', ':' -replace '[sS]$', ''
    $clean = $clean -replace '[^\d:.,]', ''
    $clean = $clean.Trim(':')
    if ([string]::IsNullOrWhiteSpace($clean)) {
        throw "tempo vazio"
    }

    $rawParts = @($clean.Split(':') | Where-Object { $_ -ne '' })
    if ($rawParts.Count -eq 0) {
        throw "tempo vazio"
    }

    $parts = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $rawParts.Count; $i++) {
        $piece = $rawParts[$i]
        $intPart = $piece
        $fracSuffix = ""
        if ($piece -match '^(\d+)[\.,](\d+)$') {
            $intPart = $Matches[1]
            $fracSuffix = "." + $Matches[2]
        }
        elseif ($piece -notmatch '^\d+$') {
            throw ("valor invalido: '{0}'" -f $piece)
        }

        if ($i -gt 0 -and $intPart.Length -eq 4) {
            # "0056" depois de ':' = minutos+segundos colados.
            $parts.Add($intPart.Substring(0, 2))
            $parts.Add($intPart.Substring(2) + $fracSuffix)
        }
        elseif ($i -gt 0 -and $intPart.Length -eq 3) {
            # "813" depois de ':' = M+SS colados.
            $parts.Add($intPart.Substring(0, 1))
            $parts.Add($intPart.Substring(1) + $fracSuffix)
        }
        else {
            $parts.Add($intPart + $fracSuffix)
        }
    }

    if ($parts.Count -gt 3) {
        throw ("formato de tempo invalido: '{0}'" -f $Text.Trim())
    }

    $values = @()
    for ($i = 0; $i -lt $parts.Count; $i++) {
        [double]$number = 0
        $ok = [double]::TryParse(
            $parts[$i],
            [System.Globalization.NumberStyles]::Float,
            [System.Globalization.CultureInfo]::InvariantCulture,
            [ref]$number
        )
        if (-not $ok -or $number -lt 0) {
            throw ("valor invalido: '{0}'" -f $parts[$i])
        }
        $values += $number
    }

    switch ($values.Count) {
        1 { return $values[0] }
        2 {
            if ($values[1] -ge 60) { throw ("segundos devem ser < 60 em '{0}'" -f $Text.Trim()) }
            return ($values[0] * 60) + $values[1]
        }
        3 {
            if ($values[1] -ge 60) { throw ("minutos devem ser < 60 em '{0}'" -f $Text.Trim()) }
            if ($values[2] -ge 60) { throw ("segundos devem ser < 60 em '{0}'" -f $Text.Trim()) }
            return ($values[0] * 3600) + ($values[1] * 60) + $values[2]
        }
    }
}

function Parse-SegmentLine {
    # Aceita "inicio - fim", "inicio a fim", "inicio-fim", travessoes etc.
    param([Parameter(Mandatory = $true)][string]$Line)

    $norm = $Line.Trim()
    $norm = $norm -replace '[–—]', '-'
    $norm = $norm -replace '(?i)\s+(?:a|à|ate|até)\s+', ' - '

    $pieces = @($norm -split '-' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
    if ($pieces.Count -ne 2) {
        throw "use o formato 'inicio - fim' (ex.: 1:07:12 - 1:07:31)"
    }

    $startSeconds = Parse-FlexibleTimeToSeconds -Text $pieces[0]
    $endSeconds = Parse-FlexibleTimeToSeconds -Text $pieces[1]

    if ($endSeconds -le $startSeconds) {
        throw ("fim ({0}) deve ser maior que o inicio ({1})" -f $pieces[1], $pieces[0])
    }

    return [PSCustomObject]@{
        StartSeconds = $startSeconds
        EndSeconds   = $endSeconds
        Original     = $Line.Trim()
    }
}

function Parse-SegmentsText {
    # Retorna @{ Segments = lista; Errors = lista de "linha N: motivo" }.
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text)

    $segments = @()
    $errors = @()
    $lineNumber = 0
    foreach ($line in ($Text -split "\r?\n")) {
        $lineNumber++
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        try {
            $segments += Parse-SegmentLine -Line $line
        }
        catch {
            $errors += ("linha {0} ('{1}'): {2}" -f $lineNumber, $line.Trim(), $_.Exception.Message)
        }
    }

    return [PSCustomObject]@{ Segments = $segments; Errors = $errors }
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

# ---------------------------------------------------------------------------
# URL e utilidades de processo (inalteradas em essencia)
# ---------------------------------------------------------------------------

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
        throw "Informe a URL do video."
    }

    [System.Uri]$uri = $null
    $isUri = [System.Uri]::TryCreate($raw, [System.UriKind]::Absolute, [ref]$uri)
    if (-not $isUri -or (@("http", "https") -notcontains $uri.Scheme.ToLowerInvariant())) {
        throw "Informe uma URL valida, por exemplo https://www.youtube.com/watch?v=... ou https://youtu.be/..."
    }

    $urlHost = $uri.Host.ToLowerInvariant()
    $isShortHost = ($urlHost -eq "youtu.be") -or ($urlHost -eq "www.youtu.be")
    $isYoutubeHost = ($urlHost -eq "youtube.com") -or $urlHost.EndsWith(".youtube.com")

    if (-not ($isShortHost -or $isYoutubeHost)) {
        throw "Informe uma URL de video do YouTube."
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
        throw "Informe a URL direta de um video (watch?v=..., youtu.be/..., /shorts/...). URLs de canal, playlist ou home nao sao suportadas."
    }

    return "https://www.youtube.com/watch?v={0}" -f [System.Uri]::EscapeDataString($videoId)
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

        Set-StatusText -Target $script:downloadStatusLabel -Text "Concluido com sucesso."
        Append-Log -Target $script:downloadLogBox -Line ""
        Append-Log -Target $script:downloadLogBox -Line "Processo finalizado com sucesso."

        if (-not [string]::IsNullOrWhiteSpace($script:downloadActualPath) -and (Test-Path -LiteralPath $script:downloadActualPath)) {
            Append-Log -Target $script:downloadLogBox -Line ("Ultimo arquivo gerado: {0}" -f $script:downloadActualPath)
        }

        Append-Log -Target $script:downloadLogBox -Line "A janela permanece aberta para revisar o log."
    }
    else {
        if ($null -ne $script:downloadProgressBar) {
            Set-ProgressMarquee -Target $script:downloadProgressBar -Enabled $false
        }

        Set-StatusText -Target $script:downloadStatusLabel -Text "Falhou. Veja o log para detalhes."
        Append-Log -Target $script:downloadLogBox -Line ""
        Append-Log -Target $script:downloadLogBox -Line ("Processo falhou com codigo {0}." -f $ExitCode)
        Append-Log -Target $script:downloadLogBox -Line "A janela permanece aberta para revisar o erro acima."
    }
}

function Handle-DownloadOutputLine {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Line)

    if ([string]::IsNullOrWhiteSpace($Line)) {
        return
    }

    $cleanLine = $Line.Trim()

    # Metricas internas do ffmpeg (-progress) nao agregam nada ao usuario.
    if ($cleanLine -match '^(?:\[trecho \d+/\d+\] )?(?:frame=|fps=|stream_\d|bitrate=|total_size=|out_time|dup_frames=|drop_frames=|speed=|progress=continue)') {
        return
    }

    Append-Log -Target $script:downloadLogBox -Line $cleanLine

    if ($cleanLine -match '^ETAPA\s+(\d+)/(\d+)$') {
        $done = [int]$matches[1]
        $totalSteps = [Math]::Max(1, [int]$matches[2])
        Set-ProgressPercent -Target $script:downloadProgressBar -Value ([int][Math]::Floor(($done * 100.0) / $totalSteps))
    }
    elseif ($cleanLine -match '\[download\]\s+([0-9]+(?:\.[0-9]+)?)%') {
        Set-ProgressPercent -Target $script:downloadProgressBar -Value ([int][Math]::Round([double]::Parse($matches[1], [System.Globalization.CultureInfo]::InvariantCulture)))
    }
    elseif ($cleanLine -like "Juntando*") {
        Set-ProgressMarquee -Target $script:downloadProgressBar -Enabled $true
    }
    elseif ($cleanLine -eq "progress=end") {
        # ffmpeg terminou uma etapa; a barra avanca pelas linhas ETAPA.
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

function Stop-ChildProcessTree {
    param([Parameter(Mandatory = $true)][int]$ParentProcessId)

    $children = @(
        Get-CimInstance Win32_Process -Filter ("ParentProcessId = {0}" -f $ParentProcessId) -ErrorAction SilentlyContinue
    )

    foreach ($child in $children) {
        $childId = [int]$child.ProcessId
        Stop-ChildProcessTree -ParentProcessId $childId

        try {
            Stop-Process -Id $childId -Force -ErrorAction SilentlyContinue
        }
        catch {
            # Ignore cancellation races when a child process exits on its own.
        }
    }
}

function Stop-DownloadProcess {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process)

    if ($Process.HasExited) {
        return
    }

    Stop-ChildProcessTree -ParentProcessId $Process.Id

    try {
        if (-not $Process.WaitForExit(5000)) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        # Ignore cancellation races when the worker exits while cleanup is running.
    }
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
        throw "Ja existe um processamento em andamento."
    }

    $argumentText = Join-Arguments -Args $Arguments
    $executableName = Split-Path -Leaf $ExecutablePath
    Append-Log -Target $LogBox -Line ""
    Append-Log -Target $LogBox -Line ("> {0} {1}" -f $executableName, $argumentText)
    Append-Log -Target $LogBox -Line "Iniciando. Mantenha a janela aberta para acompanhar o progresso."
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
        Append-Log -Target $LogBox -Line ("Log de progresso do processo: {0}" -f $PollLogFile)
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
        throw "Could not start process."
    }

    $script:activeProcess = $process
    Set-ButtonState -Target $DownloadButton -Enabled $false
    Set-ButtonState -Target $CancelButton -Enabled $true
    Set-ProgressMarquee -Target $ProgressBar -Enabled $true
    Set-StatusText -Target $StatusLabel -Text "Processando... acompanhe o progresso pelo log."

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

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

$form = New-Object System.Windows.Forms.Form
$form.Text = "Recortar YouTube — multi-trechos"
$form.StartPosition = "CenterScreen"
$form.Size = New-Object System.Drawing.Size(900, 760)
$form.MinimumSize = New-Object System.Drawing.Size(900, 700)
$form.MaximizeBox = $true

$grpSource = New-Object System.Windows.Forms.GroupBox
$grpSource.Text = "Video"
$grpSource.Location = New-Object System.Drawing.Point(12, 10)
$grpSource.Size = New-Object System.Drawing.Size(860, 58)
$grpSource.Anchor = "Top, Left, Right"

$lblUrl = New-Object System.Windows.Forms.Label
$lblUrl.Text = "URL:"
$lblUrl.Location = New-Object System.Drawing.Point(12, 25)
$lblUrl.AutoSize = $true

$txtUrl = New-Object System.Windows.Forms.TextBox
$txtUrl.Location = New-Object System.Drawing.Point(50, 22)
$txtUrl.Size = New-Object System.Drawing.Size(795, 24)
$txtUrl.Anchor = "Top, Left, Right"

$grpSource.Controls.AddRange(@($lblUrl, $txtUrl))

$grpSegments = New-Object System.Windows.Forms.GroupBox
$grpSegments.Text = "Trechos (um por linha) — aceita 1:07:12 - 1:07:31  |  1:20:26 a 1:20:53  |  1:0056 (digitos colados)"
$grpSegments.Location = New-Object System.Drawing.Point(12, 74)
$grpSegments.Size = New-Object System.Drawing.Size(860, 190)
$grpSegments.Anchor = "Top, Left, Right"

$txtSegments = New-Object System.Windows.Forms.TextBox
$txtSegments.Location = New-Object System.Drawing.Point(12, 22)
$txtSegments.Size = New-Object System.Drawing.Size(833, 130)
$txtSegments.Multiline = $true
$txtSegments.ScrollBars = "Vertical"
$txtSegments.AcceptsReturn = $true
$txtSegments.Font = New-Object System.Drawing.Font("Consolas", 10)
$txtSegments.Anchor = "Top, Left, Right"

$lblSegmentsInfo = New-Object System.Windows.Forms.Label
$lblSegmentsInfo.Text = "Nenhum trecho informado."
$lblSegmentsInfo.Location = New-Object System.Drawing.Point(12, 158)
$lblSegmentsInfo.Size = New-Object System.Drawing.Size(833, 24)
$lblSegmentsInfo.Anchor = "Top, Left, Right"

$grpSegments.Controls.AddRange(@($txtSegments, $lblSegmentsInfo))

$grpOutput = New-Object System.Windows.Forms.GroupBox
$grpOutput.Text = "Saida"
$grpOutput.Location = New-Object System.Drawing.Point(12, 270)
$grpOutput.Size = New-Object System.Drawing.Size(860, 118)
$grpOutput.Anchor = "Top, Left, Right"

$lblMode = New-Object System.Windows.Forms.Label
$lblMode.Text = "Formato:"
$lblMode.Location = New-Object System.Drawing.Point(12, 27)
$lblMode.AutoSize = $true

$cmbMode = New-Object System.Windows.Forms.ComboBox
$cmbMode.Location = New-Object System.Drawing.Point(75, 24)
$cmbMode.Size = New-Object System.Drawing.Size(230, 24)
$cmbMode.DropDownStyle = "DropDownList"
[void]$cmbMode.Items.Add("Video MP4 (imagem + audio)")
[void]$cmbMode.Items.Add("Somente audio MP3")
$cmbMode.SelectedIndex = 0

$chkJoin = New-Object System.Windows.Forms.CheckBox
$chkJoin.Text = "Juntar todos os trechos em um so video (transicao suave de 0,5s)"
$chkJoin.Location = New-Object System.Drawing.Point(330, 25)
$chkJoin.Size = New-Object System.Drawing.Size(515, 24)
$chkJoin.Anchor = "Top, Left, Right"

$lblOutput = New-Object System.Windows.Forms.Label
$lblOutput.Text = "Pasta de saida:"
$lblOutput.Location = New-Object System.Drawing.Point(12, 60)
$lblOutput.AutoSize = $true

$txtOutput = New-Object System.Windows.Forms.TextBox
$txtOutput.Location = New-Object System.Drawing.Point(12, 80)
$txtOutput.Size = New-Object System.Drawing.Size(745, 24)
$txtOutput.Anchor = "Top, Left, Right"
$txtOutput.Text = (Join-Path -Path $env:USERPROFILE -ChildPath "Downloads")

$btnBrowse = New-Object System.Windows.Forms.Button
$btnBrowse.Text = "Procurar..."
$btnBrowse.Location = New-Object System.Drawing.Point(765, 78)
$btnBrowse.Size = New-Object System.Drawing.Size(80, 28)
$btnBrowse.Anchor = "Top, Right"

$grpOutput.Controls.AddRange(@($lblMode, $cmbMode, $chkJoin, $lblOutput, $txtOutput, $btnBrowse))

$btnDownload = New-Object System.Windows.Forms.Button
$btnDownload.Text = "Baixar trechos"
$btnDownload.Location = New-Object System.Drawing.Point(12, 398)
$btnDownload.Size = New-Object System.Drawing.Size(150, 34)

$btnCancel = New-Object System.Windows.Forms.Button
$btnCancel.Text = "Cancelar"
$btnCancel.Location = New-Object System.Drawing.Point(172, 398)
$btnCancel.Size = New-Object System.Drawing.Size(110, 34)
$btnCancel.Enabled = $false

$lblStatus = New-Object System.Windows.Forms.Label
$lblStatus.Text = "Pronto."
$lblStatus.Location = New-Object System.Drawing.Point(295, 406)
$lblStatus.Size = New-Object System.Drawing.Size(577, 22)
$lblStatus.Anchor = "Top, Left, Right"

$progressDownload = New-Object System.Windows.Forms.ProgressBar
$progressDownload.Location = New-Object System.Drawing.Point(12, 440)
$progressDownload.Size = New-Object System.Drawing.Size(860, 18)
$progressDownload.Anchor = "Top, Left, Right"
$progressDownload.Minimum = 0
$progressDownload.Maximum = 100
$progressDownload.Value = 0
$progressDownload.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous

$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Location = New-Object System.Drawing.Point(12, 466)
$txtLog.Size = New-Object System.Drawing.Size(860, 240)
$txtLog.Multiline = $true
$txtLog.ScrollBars = "Vertical"
$txtLog.ReadOnly = $true
$txtLog.Anchor = "Top, Bottom, Left, Right"

$form.Controls.AddRange(@(
        $grpSource,
        $grpSegments,
        $grpOutput,
        $btnDownload,
        $btnCancel,
        $lblStatus,
        $progressDownload,
        $txtLog
    ))

Append-Log -Target $txtLog -Line "GUI iniciada (multi-trechos)."
Append-Log -Target $txtLog -Line ("Log da sessao: " + $script:sessionLogFile)

$updateSegmentsInfo = {
    try {
        $parsed = Parse-SegmentsText -Text $txtSegments.Text
        $count = @($parsed.Segments).Count
        $errorCount = @($parsed.Errors).Count
        if ($count -eq 0 -and $errorCount -eq 0) {
            $lblSegmentsInfo.Text = "Nenhum trecho informado."
            $lblSegmentsInfo.ForeColor = [System.Drawing.Color]::Black
        }
        elseif ($errorCount -gt 0) {
            $lblSegmentsInfo.Text = ("{0} trecho(s) valido(s); {1} linha(s) com erro: {2}" -f $count, $errorCount, $parsed.Errors[0])
            $lblSegmentsInfo.ForeColor = [System.Drawing.Color]::Firebrick
        }
        else {
            $totalSeconds = 0.0
            foreach ($seg in $parsed.Segments) {
                $totalSeconds += ($seg.EndSeconds - $seg.StartSeconds)
            }
            $lblSegmentsInfo.Text = ("{0} trecho(s) valido(s); duracao total {1}." -f $count, (Format-SecondsForYtDlp -TotalSeconds $totalSeconds))
            $lblSegmentsInfo.ForeColor = [System.Drawing.Color]::DarkGreen
        }
    }
    catch {
        $lblSegmentsInfo.Text = "Nao foi possivel analisar os trechos."
        $lblSegmentsInfo.ForeColor = [System.Drawing.Color]::Firebrick
    }
}

$txtSegments.Add_TextChanged($updateSegmentsInfo)

$cmbMode.Add_SelectedIndexChanged({
        $isVideo = ($cmbMode.SelectedIndex -eq 0)
        $chkJoin.Enabled = $isVideo
        if (-not $isVideo) {
            $chkJoin.Checked = $false
        }
    })

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
                Stop-DownloadProcess -Process $script:activeProcess
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
                Append-Log -Target $txtLog -Line "Processo cancelado pelo usuario."
                Set-StatusText -Target $lblStatus -Text "Cancelado."
            }
            catch {
                [System.Windows.Forms.MessageBox]::Show(
                    "Nao foi possivel cancelar o processo: $($_.Exception.Message)",
                    "Recortar YouTube",
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
                    -ErrorHint "yt-dlp nao encontrado. Instale com: winget install yt-dlp.yt-dlp"
            }

            $ffmpegPath = Resolve-CommandPathOrThrow `
                    -CommandName "ffmpeg" `
                    -CandidatePaths $ffmpegCandidates `
                    -ErrorHint "ffmpeg nao encontrado. Instale com: winget install Gyan.FFmpeg"

            $url = Normalize-YouTubeUrl -InputText $txtUrl.Text

            $parsed = Parse-SegmentsText -Text $txtSegments.Text
            if (@($parsed.Errors).Count -gt 0) {
                throw ("Corrija os trechos antes de baixar:`n- " + ($parsed.Errors -join "`n- "))
            }
            $segments = @($parsed.Segments)
            if ($segments.Count -eq 0) {
                throw "Informe ao menos um trecho (um por linha), por exemplo: 1:07:12 - 1:07:31"
            }

            $outputDir = $txtOutput.Text.Trim()
            if ([string]::IsNullOrWhiteSpace($outputDir)) {
                throw "A pasta de saida nao pode ser vazia."
            }

            if (-not (Test-Path -LiteralPath $outputDir)) {
                New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
            }

            if ($cmbMode.SelectedIndex -eq 0) {
                # --- Video MP4 via worker multi-trechos ---
                $workerPath = Join-Path -Path $PSScriptRoot -ChildPath "yt_clip_video_worker.ps1"
                if (-not (Test-Path -LiteralPath $workerPath)) {
                    throw "Worker de video nao encontrado: $workerPath"
                }
                $workerLogFile = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("yt_clip_worker_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

                $powershellPath = Join-Path -Path $PSHOME -ChildPath "powershell.exe"
                if (-not (Test-Path -LiteralPath $powershellPath)) {
                    $powershellPath = "powershell.exe"
                }

                $segmentSpecs = @(
                    $segments | ForEach-Object {
                        "{0}-{1}" -f (Format-SecondsForYtDlp -TotalSeconds $_.StartSeconds), (Format-SecondsForYtDlp -TotalSeconds $_.EndSeconds)
                    }
                ) -join ";"

                $workerArgs = @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", $workerPath,
                    "-YtDlpPath", $ytDlpPath,
                    "-FfmpegPath", $ffmpegPath,
                    "-Url", $url,
                    "-Segments", $segmentSpecs,
                    "-OutputDir", $outputDir,
                    "-MaxParallel", "3",
                    "-ProgressLogFile", $workerLogFile
                )

                if ($chkJoin.Checked -and $segments.Count -ge 2) {
                    $workerArgs += "-JoinSegments"
                }

                if ($ytDlpPrefixArgs.Count -gt 0) {
                    $workerArgs += "-UsePythonModule"
                }

                Append-Log -Target $txtLog -Line ("Processando {0} trecho(s) em MP4{1}." -f $segments.Count, $(if ($chkJoin.Checked -and $segments.Count -ge 2) { " + montagem unica com transicao" } else { "" }))

                Start-YtDlpDownload `
                    -ExecutablePath $powershellPath `
                    -Arguments $workerArgs `
                    -LogBox $txtLog `
                    -StatusLabel $lblStatus `
                    -DownloadButton $btnDownload `
                    -CancelButton $btnCancel `
                    -ProgressBar $progressDownload `
                    -ExpectedOutputPath "" `
                    -PollLogFile $workerLogFile
            }
            else {
                # --- MP3: uma execucao do yt-dlp com N secoes ---
                $ytDlpArgs = @(
                    "--newline",
                    "--no-playlist",
                    "--force-overwrites",
                    "--replace-in-metadata", "title", "\s*[|｜]\s*.+$", "",
                    "--replace-in-metadata", "title", "\s*[：:]\s*", " - ",
                    "--replace-in-metadata", "title", "[/\\]+", "-"
                )

                foreach ($seg in $segments) {
                    $section = "*{0}-{1}" -f (Format-SecondsForYtDlp -TotalSeconds $seg.StartSeconds), (Format-SecondsForYtDlp -TotalSeconds $seg.EndSeconds)
                    $ytDlpArgs += @("--download-sections", $section)
                }

                $outputTemplate = if ($segments.Count -gt 1) {
                    "%(title)s [trecho %(section_start)s-%(section_end)s].%(ext)s"
                }
                else {
                    $suffix = "{0}-{1}" -f (Format-SecondsForFileName -TotalSeconds $segments[0].StartSeconds), (Format-SecondsForFileName -TotalSeconds $segments[0].EndSeconds)
                    "%(title)s [{0}].%(ext)s" -f $suffix
                }

                $ytDlpArgs += @(
                    "--paths", $outputDir,
                    "--windows-filenames",
                    "--ffmpeg-location", $ffmpegPath,
                    "--print", "after_move:filepath",
                    "-o", $outputTemplate,
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "0",
                    $url
                )

                Append-Log -Target $txtLog -Line ("Processando {0} trecho(s) em MP3." -f $segments.Count)

                Start-YtDlpDownload `
                    -ExecutablePath $ytDlpPath `
                    -Arguments ($ytDlpPrefixArgs + $ytDlpArgs) `
                    -LogBox $txtLog `
                    -StatusLabel $lblStatus `
                    -DownloadButton $btnDownload `
                    -CancelButton $btnCancel `
                    -ProgressBar $progressDownload `
                    -ExpectedOutputPath ""
            }
        }
        catch {
            Append-Log -Target $txtLog -Line ("ERRO: {0}" -f $_.Exception.Message)
            Set-StatusText -Target $lblStatus -Text "Erro. Veja o log para detalhes."
            Set-ButtonState -Target $btnDownload -Enabled $true
            Set-ButtonState -Target $btnCancel -Enabled $false
            Set-ProgressMarquee -Target $progressDownload -Enabled $false
            [System.Windows.Forms.MessageBox]::Show(
                $_.Exception.Message,
                "Validacao",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Error
            ) | Out-Null
        }
    })

$form.Add_FormClosing({
        if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
            [System.Windows.Forms.MessageBox]::Show(
                "Ha um processamento em andamento. Use Cancelar antes de fechar a janela.",
                "Recortar YouTube",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            ) | Out-Null
            $_.Cancel = $true
            return
        }
    })

[void]$form.ShowDialog()
