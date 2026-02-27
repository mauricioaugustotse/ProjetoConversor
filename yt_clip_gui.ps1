Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:activeProcess = $null
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

function Escape-Argument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }

    return $Value
}

function Join-Arguments {
    param([Parameter(Mandatory = $true)][string[]]$Args)
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

function Start-YtDlpDownload {
    param(
        [Parameter(Mandatory = $true)][string]$ExecutablePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][System.Windows.Forms.TextBox]$LogBox,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Label]$StatusLabel,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$DownloadButton,
        [Parameter(Mandatory = $true)][System.Windows.Forms.Button]$CancelButton
    )

    if ($script:activeProcess -and -not $script:activeProcess.HasExited) {
        throw "A download is already running."
    }

    $argumentText = Join-Arguments -Args $Arguments
    Append-Log -Target $LogBox -Line ""
    Append-Log -Target $LogBox -Line ("> yt-dlp $argumentText")
    Append-Log -Target $LogBox -Line ""

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ExecutablePath
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
                    Append-Log -Target $LogBox -Line $eventArgs.Data
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
                    Append-Log -Target $LogBox -Line $eventArgs.Data
                }
            }
            catch {
                # Ignore callback errors to avoid crashing the UI host.
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
                    # Keep default non-zero code.
                }

                $script:activeProcess = $null

                Set-ButtonState -Target $DownloadButton -Enabled $true
                Set-ButtonState -Target $CancelButton -Enabled $false

                if ($exitCode -eq 0) {
                    Set-StatusText -Target $StatusLabel -Text "Done. Clip downloaded successfully."
                    Append-Log -Target $LogBox -Line ""
                    Append-Log -Target $LogBox -Line "Download finished successfully."
                }
                else {
                    Set-StatusText -Target $StatusLabel -Text "Failed. Check log for details."
                    Append-Log -Target $LogBox -Line ""
                    Append-Log -Target $LogBox -Line "Download failed."
                }
            }
            catch {
                # Ignore callback errors to avoid closing the GUI unexpectedly.
            }
        })

    $started = $process.Start()
    if (-not $started) {
        throw "Could not start yt-dlp process."
    }

    $script:activeProcess = $process
    Set-ButtonState -Target $DownloadButton -Enabled $false
    Set-ButtonState -Target $CancelButton -Enabled $true
    Set-StatusText -Target $StatusLabel -Text "Downloading clip..."

    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
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

$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Location = New-Object System.Drawing.Point(15, 300)
$txtLog.Size = New-Object System.Drawing.Size(810, 270)
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
            )

            $ffmpegCandidates = @(
                (Join-Path -Path $localAppData -ChildPath "Microsoft\WinGet\Links\ffmpeg.exe"),
                (Join-Path -Path $programData -ChildPath "chocolatey\bin\ffmpeg.exe"),
                (Join-Path -Path $userProfile -ChildPath "scoop\shims\ffmpeg.exe")
            )

            $ytDlpPath = Resolve-CommandPathOrThrow `
                -CommandName "yt-dlp" `
                -CandidatePaths $ytDlpCandidates `
                -ErrorHint "yt-dlp not found. Install with: winget install yt-dlp.yt-dlp"

            [void](Resolve-CommandPathOrThrow `
                    -CommandName "ffmpeg" `
                    -CandidatePaths $ffmpegCandidates `
                    -ErrorHint "ffmpeg not found. Install with: winget install Gyan.FFmpeg")

            $url = $txtUrl.Text.Trim()
            if ([string]::IsNullOrWhiteSpace($url)) {
                throw "URL cannot be empty."
            }

            if ($url -notmatch '^https?://') {
                throw "Please provide a valid URL (starting with http:// or https://)."
            }

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

            $args = @(
                "--newline",
                "--download-sections", $section,
                "--paths", $outputDir,
                "--windows-filenames",
                "-o", "%(title)s [%(section_start)s-%(section_end)s].%(ext)s"
            )

            if ($cmbMode.SelectedIndex -eq 0) {
                $args += @(
                    "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                    "--merge-output-format", "mp4"
                )
            }
            else {
                $args += @(
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "0"
                )
            }

            $args += $url

            Start-YtDlpDownload `
                -ExecutablePath $ytDlpPath `
                -Arguments $args `
                -LogBox $txtLog `
                -StatusLabel $lblStatus `
                -DownloadButton $btnDownload `
                -CancelButton $btnCancel
        }
        catch {
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
            $confirm = [System.Windows.Forms.MessageBox]::Show(
                "A download is running. Close and cancel it?",
                "yt clip",
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
