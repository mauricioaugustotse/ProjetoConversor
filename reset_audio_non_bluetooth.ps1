param(
    [switch]$DeepReinstall
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $ScriptRoot "audio_reset_non_bluetooth_$Stamp.log"
$SnapshotPath = Join-Path $ScriptRoot "audio_reset_non_bluetooth_snapshot_$Stamp.json"

function Write-Log {
    param([string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Test-BluetoothLike {
    param([string]$Text)

    if ([string]::IsNullOrWhiteSpace($Text)) {
        return $false
    }

    return $Text -match "BTH|Bluetooth|BTHENUM|BTHLE|APXENUM|Galaxy Buds|Samsung Soundbar|Bose|JBL|S23 Ultra|Hands-Free|A2DP|AVRCP"
}

function Test-AudioResetTarget {
    param($Device)

    $text = @(
        $Device.FriendlyName
        $Device.InstanceId
        $Device.DeviceID
        $Device.Name
    ) -join " "

    if (Test-BluetoothLike $text) {
        return $false
    }

    return $text -match "Realtek|INTELAUDIO|Smart Sound|microfones digitais|USB Audio|HDAUDIO|HD Audio Driver for Display Audio"
}

function Invoke-PnpUtil {
    param([string[]]$Arguments)

    Write-Log ("pnputil " + ($Arguments -join " "))

    $exe = Join-Path $env:WINDIR "System32\pnputil.exe"
    $stdoutPath = Join-Path $env:TEMP ("pnputil_stdout_{0}.txt" -f ([guid]::NewGuid().ToString("N")))
    $stderrPath = Join-Path $env:TEMP ("pnputil_stderr_{0}.txt" -f ([guid]::NewGuid().ToString("N")))

    $process = Start-Process -FilePath $exe `
        -ArgumentList $Arguments `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $output = @()
    if (Test-Path $stdoutPath) {
        $output += Get-Content -Path $stdoutPath -ErrorAction SilentlyContinue
    }
    if (Test-Path $stderrPath) {
        $output += Get-Content -Path $stderrPath -ErrorAction SilentlyContinue | ForEach-Object { "STDERR: $_" }
    }

    if ($output) {
        $output | ForEach-Object { Write-Log "  $_" }
    }

    $exitCode = $process.ExitCode
    Write-Log "  exit=$exitCode"
    Remove-Item -Path $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    return $exitCode
}

function Select-DeviceSummary {
    param($Device)

    [pscustomobject]@{
        Status = $Device.Status
        Class = $Device.Class
        FriendlyName = $Device.FriendlyName
        InstanceId = $Device.InstanceId
        Problem = $Device.Problem
        ConfigManagerErrorCode = $Device.ConfigManagerErrorCode
    }
}

Write-Log "Starting non-Bluetooth audio reset."
Write-Log "DeepReinstall=$DeepReinstall"

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
Write-Log "IsAdministrator=$isAdmin"
if (-not $isAdmin) {
    Write-Log "WARNING: Not running as Administrator. Device reset and service restart may fail."
}

$mediaDevices = @(Get-PnpDevice -Class MEDIA -ErrorAction SilentlyContinue | Where-Object { Test-AudioResetTarget $_ })

# Include the parent USB node for the stale USB Audio device, if present.
$usbParents = @(Get-PnpDevice -ErrorAction SilentlyContinue | Where-Object {
    $text = @($_.FriendlyName, $_.InstanceId) -join " "
    -not (Test-BluetoothLike $text) -and
    ($_.InstanceId -match "^USB\\VID_3654&PID_5678" -or $_.FriendlyName -eq "USB Audio")
})

$targets = @($mediaDevices + $usbParents) |
    Where-Object { $_ -and $_.InstanceId } |
    Sort-Object InstanceId -Unique

$audioEndpointsBefore = @(Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue |
    Where-Object {
        $text = @($_.FriendlyName, $_.InstanceId) -join " "
        -not (Test-BluetoothLike $text)
    } |
    ForEach-Object { Select-DeviceSummary $_ })

$driverSnapshot = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
    Where-Object {
        $text = @($_.DeviceName, $_.DeviceID) -join " "
        -not (Test-BluetoothLike $text) -and
        $text -match "Realtek|INTELAUDIO|Smart Sound|USB Audio|HDAUDIO|HD Audio Driver for Display Audio"
    } |
    Select-Object DeviceName, Manufacturer, DriverProviderName, DriverVersion, DriverDate, InfName, DeviceID)

$snapshot = [ordered]@{
    CreatedAt = (Get-Date).ToString("o")
    Exclusion = "Bluetooth/BTH/BTHENUM/BTHLE/APXENUM/Galaxy Buds/Samsung Soundbar/Bose/JBL/S23/Hands-Free/A2DP/AVRCP"
    Targets = @($targets | ForEach-Object { Select-DeviceSummary $_ })
    AudioEndpointsBefore = $audioEndpointsBefore
    DriversBefore = $driverSnapshot
}

$snapshot | ConvertTo-Json -Depth 6 | Set-Content -Path $SnapshotPath -Encoding UTF8
Write-Log "Snapshot saved: $SnapshotPath"

if (-not $targets -or $targets.Count -eq 0) {
    Write-Log "No non-Bluetooth audio hardware targets found."
    exit 2
}

Write-Log "Targets:"
$targets | ForEach-Object {
    Write-Log ("  {0} | {1} | {2} | {3}" -f $_.Status, $_.Class, $_.FriendlyName, $_.InstanceId)
}

$phantomTargets = @($targets | Where-Object { $_.Status -ne "OK" })
$presentTargets = @($targets | Where-Object { $_.Status -eq "OK" })

foreach ($device in $phantomTargets) {
    Write-Log "Removing phantom/non-present device: $($device.FriendlyName)"
    [void](Invoke-PnpUtil @("/remove-device", $device.InstanceId))
}

foreach ($device in $presentTargets) {
    if ($DeepReinstall) {
        Write-Log "Removing present device for reinstall: $($device.FriendlyName)"
        [void](Invoke-PnpUtil @("/remove-device", $device.InstanceId))
    } else {
        Write-Log "Restarting present device: $($device.FriendlyName)"
        [void](Invoke-PnpUtil @("/restart-device", $device.InstanceId))
    }
}

Write-Log "Scanning for hardware changes."
[void](Invoke-PnpUtil @("/scan-devices"))

try {
    Write-Log "Restarting Windows Audio service."
    Restart-Service -Name Audiosrv -Force -ErrorAction Stop
    Write-Log "Windows Audio service restarted."
} catch {
    Write-Log "Windows Audio service restart failed: $($_.Exception.Message)"
}

$postMedia = @(Get-PnpDevice -Class MEDIA -ErrorAction SilentlyContinue |
    Where-Object { Test-AudioResetTarget $_ } |
    ForEach-Object { Select-DeviceSummary $_ })

$postEndpoints = @(Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue |
    Where-Object {
        $text = @($_.FriendlyName, $_.InstanceId) -join " "
        -not (Test-BluetoothLike $text)
    } |
    ForEach-Object { Select-DeviceSummary $_ })

Write-Log "Post-reset MEDIA devices:"
$postMedia | ForEach-Object {
    Write-Log ("  {0} | {1} | {2} | {3}" -f $_.Status, $_.Class, $_.FriendlyName, $_.InstanceId)
}

Write-Log "Post-reset AudioEndpoint devices:"
$postEndpoints | ForEach-Object {
    Write-Log ("  {0} | {1} | {2} | {3}" -f $_.Status, $_.Class, $_.FriendlyName, $_.InstanceId)
}

Write-Log "Completed. Log: $LogPath"
Write-Log "Snapshot: $SnapshotPath"
