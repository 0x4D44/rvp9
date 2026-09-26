# Run from Crucible with Windows PowerShell 5.1:
#   powershell -NoProfile -File .\capture-repeats.ps1 -Count 3 -Preflight
#   powershell -NoProfile -File .\capture-repeats.ps1 -Count 3
# After an interrupted batch, choose unused names with -StartAt (for example,
# -StartAt 4 -Count 2). Preflight checks everything without creating artifacts.
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 99)]
    [int]$Count,
    [ValidateRange(1, 99)]
    [int]$StartAt = 1,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob'
$expectedHead = '48cab42d2dba1a46c502a5da2faf907d79d78ac1'
$expectedEncoderSha256 = 'c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1'
$expectedPdbSha256 = '085c7c7cc7f724ed7674797dc88bf27011e6662d27d7661e9758677004bd8f9c'
$inputPath = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$expectedInputSha256 = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutputSha256 = '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'
$sourcePath = Join-Path $worktree 'wrk_docs\evidence\2026-09-25-unit1434-three-clip-ip-attribution\capture-clip.ps1'
$scratchPath = Join-Path $worktree '.scratch'
$artifactRoot = Join-Path $scratchPath 'repeated-ducks-ip'
$encoderPath = Join-Path $scratchPath 'target\release\rvp9enc.exe'
$pdbPath = Join-Path $scratchPath 'target\release\rvp9enc.pdb'

function Assert-FileHash([string]$Name, [string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name is missing: $Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw "$Name SHA-256 mismatch: expected=$Expected actual=$actual path=$Path"
    }
}

function Replace-Once([string]$Text, [string]$Old, [string]$New, [string]$Name) {
    $first = $Text.IndexOf($Old, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $Text.IndexOf($Old, $first + $Old.Length, [StringComparison]::Ordinal) -ge 0) {
        throw "Expected exactly one $Name anchor in $sourcePath"
    }
    return $Text.Replace($Old, $New)
}

function Get-AdaptedCapture([string]$Source, [string]$RunName) {
    $text = Replace-Once $Source `
        'D:\worktrees\rvp9-2\20260925-TSK-HUM-unit1434-integrated-ip-profile' `
        $worktree 'worktree'
    $text = Replace-Once $text '88644d86641f99a4291c21842bdc17aec9baee8e' $expectedHead 'HEAD'
    $text = Replace-Once $text '1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a' `
        $expectedEncoderSha256 'encoder hash'
    $text = Replace-Once $text '27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4' `
        $expectedPdbSha256 'PDB hash'
    $text = Replace-Once $text `
        '$artifactRoot = Join-Path $worktree ''.scratch\unit1434-ip-profile''' `
        '$artifactRoot = Join-Path $worktree ''.scratch\repeated-ducks-ip''' 'artifact root'
    $runDirLine = '$outputDir = Join-Path $artifactRoot ''{0}''' -f $RunName
    $text = Replace-Once $text '$outputDir = Join-Path $artifactRoot $ClipName' $runDirLine 'run directory'

    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -ne 0) {
        throw "Adapted capture has PowerShell parse errors for ${RunName}: $($parseErrors -join '; ')"
    }
    return $text
}

function Assert-WprIdle {
    $wpr = (Get-Command -Name 'wpr.exe' -CommandType Application -ErrorAction Stop |
        Select-Object -First 1).Source
    $statusOutput = @(& $wpr -status 2>&1)
    $statusExit = $LASTEXITCODE
    $statusText = ($statusOutput | ForEach-Object { [string]$_ }) -join ' '
    if ($statusExit -ne 0 -or $statusText -notmatch '(?i)\bWPR is not recording\b') {
        throw "WPR is recording or its idle state is unknown: exit=$statusExit status=$statusText"
    }
}

function Get-UniqueLogField([string]$Log, [string]$Key) {
    $pattern = '(?m)^' + [regex]::Escape($Key) + '=([^\r\n]*)\r?$'
    $matches = [regex]::Matches($Log, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Key field in capture.log; found $($matches.Count)"
    }
    return $matches[0].Groups[1].Value
}

function Write-NewUtf8File([string]$Path, [string]$Content) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Content)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $stream.Write($bytes, 0, $bytes.Length)
    } finally {
        $stream.Dispose()
    }
}

function Invoke-TraceStats([string]$Xperf, [string]$TracePath) {
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $Xperf
    $psi.Arguments = '-i "{0}" -a tracestats' -f $TracePath
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $psi
    try {
        if (-not $process.Start()) {
            throw 'xperf did not start.'
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(120000)) {
            $process.Kill()
            [void]$process.WaitForExit(10000)
            throw 'xperf tracestats exceeded 120 seconds.'
        }
        if (-not $stdoutTask.Wait(5000) -or -not $stderrTask.Wait(5000)) {
            throw 'xperf output was incomplete.'
        }
        $report = [string]$stdoutTask.Result + [string]$stderrTask.Result
        if ($process.ExitCode -ne 0) {
            throw "xperf tracestats exited $($process.ExitCode): $report"
        }
        return $report
    } finally {
        $process.Dispose()
    }
}

function Assert-ZeroTraceLoss([string]$Report) {
    $pattern = '(?m)^\s*(?<Total>Total\s+)?#\s*Lost\s+(?<Kind>Buffers|Events)\s*:\s*(?<Count>\d+)\s*$'
    $lossRows = [regex]::Matches($Report, $pattern)
    $totalBuffers = 0
    $totalEvents = 0
    foreach ($row in $lossRows) {
        if ($row.Groups['Count'].Value -ne '0') {
            throw "Trace lost $($row.Groups['Kind'].Value): $($row.Value.Trim())"
        }
        if ($row.Groups['Total'].Success -and $row.Groups['Kind'].Value -eq 'Buffers') {
            $totalBuffers++
        }
        if ($row.Groups['Total'].Success -and $row.Groups['Kind'].Value -eq 'Events') {
            $totalEvents++
        }
    }
    if ($totalBuffers -ne 1 -or $totalEvents -ne 1) {
        throw "Missing or duplicate numeric total-loss rows: buffers=$totalBuffers events=$totalEvents"
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'Run this capture wrapper on Crucible Windows.'
}
if ($StartAt + $Count - 1 -gt 99) {
    throw 'Run names are limited to run-01 through run-99.'
}
if (-not (Test-Path -LiteralPath $worktree -PathType Container) -or
    -not (Test-Path -LiteralPath $scratchPath -PathType Container)) {
    throw "Profile worktree or its .scratch directory is missing: $worktree"
}
if (((Get-Item -LiteralPath $scratchPath).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Scratch path is a reparse point: $scratchPath"
}
if (Test-Path -LiteralPath $artifactRoot) {
    if (-not (Test-Path -LiteralPath $artifactRoot -PathType Container) -or
        ((Get-Item -LiteralPath $artifactRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Artifact root is not a plain directory: $artifactRoot"
    }
}

$headOutput = @(& git -C $worktree rev-parse HEAD 2>&1)
$gitExit = $LASTEXITCODE
$actualHead = (($headOutput | ForEach-Object { [string]$_ }) -join '').Trim().ToLowerInvariant()
if ($gitExit -ne 0 -or $actualHead -ne $expectedHead) {
    throw "Profile HEAD mismatch: expected=$expectedHead actual=$actualHead exit=$gitExit"
}
Assert-FileHash 'encoder' $encoderPath $expectedEncoderSha256
Assert-FileHash 'PDB' $pdbPath $expectedPdbSha256
Assert-FileHash 'Ducks input' $inputPath $expectedInputSha256

if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Committed capture script is missing: $sourcePath"
}
$sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
# The two values are the same committed script with LF and CRLF checkouts.
if ($sourceHash -notin @(
    '17edbb348c2fe7bf0ebd8f3d8a544001b62a482a034566a7fb7f67fa7deeedee',
    'ad5590df98117fff9c3c48f5cfdd47b9b9826e3a87017cb635642f36dd1d60fe'
)) {
    throw "Capture script SHA-256 mismatch: $sourceHash"
}
$source = [IO.File]::ReadAllText($sourcePath, [Text.Encoding]::UTF8)

$planned = @()
for ($number = $StartAt; $number -lt $StartAt + $Count; $number++) {
    $runName = 'run-{0:D2}' -f $number
    $runDir = Join-Path $artifactRoot $runName
    if (Test-Path -LiteralPath $runDir) {
        throw "Refusing to reuse an existing run directory: $runDir"
    }
    $adapted = Get-AdaptedCapture $source $runName
    $planned += [pscustomobject]@{ Name = $runName; Directory = $runDir; Script = $adapted }
}

Assert-WprIdle
$xperf = (Get-Command -Name 'xperf.exe' -CommandType Application -ErrorAction Stop |
    Select-Object -First 1).Source

foreach ($run in $planned) {
    Write-Output "PREFLIGHT run=$($run.Name) head=$actualHead exe_sha256=$expectedEncoderSha256 pdb_sha256=$expectedPdbSha256 input_sha256=$expectedInputSha256 output_root=$($run.Directory)"
}
if ($Preflight) {
    Write-Output 'PREFLIGHT_COMPLETE capture_started=false artifacts_created=false'
    return
}

if (-not (Test-Path -LiteralPath $artifactRoot)) {
    New-Item -ItemType Directory -Path $artifactRoot -ErrorAction Stop | Out-Null
}
foreach ($run in $planned) {
    if (Test-Path -LiteralPath $run.Directory) {
        throw "Refusing to reuse an existing run directory: $($run.Directory)"
    }
    New-Item -ItemType Directory -Path $run.Directory -ErrorAction Stop | Out-Null
    # Dot-source to give the committed runner its normal script scope. It
    # checks HEAD, assets, WPR idle state, encode status, and exact IVF hash
    # again before and during this individual capture.
    . ([scriptblock]::Create($run.Script)) -ClipName ducks

    $logPath = Join-Path $run.Directory 'capture.log'
    $tracePath = Join-Path $run.Directory 'capture.etl'
    $reportPath = Join-Path $run.Directory 'xperf-tracestats.txt'
    if (-not (Test-Path -LiteralPath $logPath -PathType Leaf) -or
        -not (Test-Path -LiteralPath $tracePath -PathType Leaf)) {
        throw "Capture artifacts are missing for $($run.Name)"
    }
    $log = [IO.File]::ReadAllText($logPath)
    if ((Get-UniqueLogField $log 'CAPTURE_SUCCESS') -ne 'True' -or
        (Get-UniqueLogField $log 'OUTPUT_SHA256') -ne $expectedOutputSha256) {
        throw "Capture status or exact IVF hash failed for $($run.Name)"
    }
    $cpuText = Get-UniqueLogField $log 'ENCODER_CPU_SECONDS'
    $cpuSeconds = 0.0
    if (-not [double]::TryParse($cpuText, [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::CurrentCulture, [ref]$cpuSeconds) -or
        $cpuSeconds -le 0 -or [double]::IsNaN($cpuSeconds) -or [double]::IsInfinity($cpuSeconds)) {
        throw "Missing or invalid encoder CPU seconds for $($run.Name): $cpuText"
    }
    $etlHash = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ((Get-UniqueLogField $log 'ETL_SHA256') -ne $etlHash -or
        (Get-Item -LiteralPath $tracePath).Length -le 0) {
        throw "ETL hash or size mismatch for $($run.Name)"
    }
    $report = Invoke-TraceStats $xperf $tracePath
    Write-NewUtf8File $reportPath $report
    Assert-ZeroTraceLoss $report
    $verificationPath = Join-Path $run.Directory 'verification.txt'
    $verification = "RUN=$($run.Name)`nENCODER_CPU_SECONDS=$cpuText`nOUTPUT_SHA256=$expectedOutputSha256`nETL_SHA256=$etlHash`nLOST_BUFFERS=0`nLOST_EVENTS=0`n"
    Write-NewUtf8File $verificationPath $verification
    Write-Output "CAPTURE_OK run=$($run.Name) cpu_seconds=$cpuText ivf_sha256=$expectedOutputSha256 etl_sha256=$etlHash lost_buffers=0 lost_events=0"
}
