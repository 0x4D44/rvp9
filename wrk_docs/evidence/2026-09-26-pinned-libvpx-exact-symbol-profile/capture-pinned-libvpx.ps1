param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('ducks', 'tree', 'town')]
    [string]$ClipName,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$exe = 'D:\worktrees\rvp9-2\acceptance-d639c265\build\libvpx\vpxenc.exe'
$expectedExeSha = '88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clips = @{
    ducks = [pscustomobject]@{
        Input = (Join-Path $inputRoot 'ducks_take_off_420_1080p30_60f.y4m')
        InputSha = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
        OutputSha = 'b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8'
    }
    tree = [pscustomobject]@{
        Input = (Join-Path $inputRoot 'in_to_tree_420_1080p30_60f.y4m')
        InputSha = 'ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'
        OutputSha = 'b930800c5bac04a67aa0a25b3344031c5aebe6f3efe4ba569d354301bf238964'
    }
    town = [pscustomobject]@{
        Input = (Join-Path $inputRoot 'old_town_cross_420_1080p30_60f.y4m')
        InputSha = 'b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'
        OutputSha = 'e05daf970ff4f9888a64b6edc2f4a2cd7151cf0ee2bc0b1de4170cce5f1b24bb'
    }
}
$clip = $clips[$ClipName]
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
if ($OutputRoot -notmatch '(?i)[\\/]\.scratch[\\/]') {
    throw "OutputRoot must be inside a task worktree .scratch: $OutputRoot"
}
$outputParent = Split-Path -Parent $OutputRoot
if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) {
    throw "OutputRoot parent must already exist: $outputParent"
}
if ($OutputRoot.Contains('"') -or $clip.Input.Contains('"')) {
    throw 'A quote character in a path cannot be passed to vpxenc safely.'
}

function Assert-Hash([string]$Path, [string]$ExpectedSha) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing asset: $Path" }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha) { throw "SHA-256 mismatch: $Path expected=$ExpectedSha actual=$actual" }
    return $actual
}

function Invoke-BoundedTool([string]$File, [string]$Arguments, [int]$Seconds) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $File
    $psi.Arguments = $Arguments
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    try {
        if (-not $process.Start()) { throw "Could not start $File $Arguments" }
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit($Seconds * 1000)) {
            $process.Kill()
            [void]$process.WaitForExit(10000)
            throw "$File $Arguments exceeded ${Seconds}s"
        }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output = ($stdout.Result + [Environment]::NewLine + $stderr.Result).Trim()
        }
    } finally {
        $process.Dispose()
    }
}

$exeSha = Assert-Hash -Path $exe -ExpectedSha $expectedExeSha
$inputSha = Assert-Hash -Path $clip.Input -ExpectedSha $clip.InputSha
$wprCommand = Get-Command -Name 'wpr.exe' -CommandType Application -ErrorAction Stop | Select-Object -First 1
$wpr = $wprCommand.Source
if (-not $wpr) { throw 'wpr.exe was not found' }
$status = Invoke-BoundedTool -File $wpr -Arguments '-status' -Seconds 20
if ($status.ExitCode -ne 0 -or $status.Output -notmatch '(?i)\bWPR is not recording\b') {
    throw "WPR is not confirmed idle: exit=$($status.ExitCode) output=$($status.Output)"
}
if ($PreflightOnly) {
    Write-Output "PREFLIGHT_OK clip=$ClipName exe_sha256=$exeSha input_sha256=$inputSha wpr=$wpr output_root=$OutputRoot"
    exit 0
}

if (-not (Test-Path -LiteralPath $OutputRoot)) {
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null
}
$clipDir = Join-Path $OutputRoot $ClipName
if (Test-Path -LiteralPath $clipDir) { throw "Refusing to overwrite clip artifacts: $clipDir" }
New-Item -ItemType Directory -Path $clipDir | Out-Null
$ivf = Join-Path $clipDir 'output.ivf'
$etl = Join-Path $clipDir 'capture.etl'
$stdoutPath = Join-Path $clipDir 'encoder.stdout.txt'
$stderrPath = Join-Path $clipDir 'encoder.stderr.txt'
$manifestPath = Join-Path $clipDir 'capture-manifest.json'
$logPath = Join-Path $clipDir 'capture.log'

function Write-RunLog([string]$Text) {
    [Console]::Out.WriteLine($Text)
    [System.IO.File]::AppendAllText($script:logPath, $Text + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false))
}

$argv = @(
    '--codec=vp9', '--passes=1', '--rt', '--cpu-used=6', '--end-usage=cbr',
    '--target-bitrate=1260', '--fps=30/1', '--threads=1', '--tile-columns=0',
    '--lag-in-frames=0', '--tune-content=default', '--aq-mode=0',
    '--noise-sensitivity=0', '--limit=60', '--kf-max-dist=60', '--quiet',
    '--ivf', '-o', $ivf, $clip.Input
)
$arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $ivf, $clip.Input
$manifest = [ordered]@{
    clip = $ClipName
    capture_method = 'wpr -start CPU (memory mode); one 60-frame encode; wpr -stop ETL'
    command_argv = $argv
    executable = $exe
    executable_sha256 = $exeSha
    input = $clip.Input
    input_sha256 = $inputSha
    expected_output_sha256 = $clip.OutputSha
    output = $ivf
    etl = $etl
    wpr = $wpr
    runner = $PSCommandPath
    runner_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    start_utc = [DateTime]::UtcNow.ToString('o')
    pid = $null
    process_cpu_seconds = $null
    wall_seconds = $null
    exit_code = $null
    timed_out = $false
    output_sha256 = $null
    output_bytes = $null
    etl_sha256 = $null
    etl_bytes = $null
    wpr_stop_exit = $null
    failure = $null
}
$wprStarted = $false
$traceFailure = $null
try {
    Write-RunLog "START clip=$ClipName exe_sha256=$exeSha input_sha256=$inputSha"
    $started = Invoke-BoundedTool -File $wpr -Arguments '-start CPU' -Seconds 30
    Write-RunLog "WPR_START_EXIT=$($started.ExitCode) $($started.Output)"
    if ($started.ExitCode -ne 0) { throw "WPR start failed: $($started.Output)" }
    $wprStarted = $true

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = $arguments
    $psi.WorkingDirectory = $clipDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) { throw 'vpxenc did not start' }
        $manifest.pid = $process.Id
        Write-RunLog "ENCODER_PID=$($process.Id)"
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(90000)) {
            $manifest.timed_out = $true
            $process.Kill()
            if (-not $process.WaitForExit(10000)) { throw 'vpxenc did not exit after timeout kill' }
        }
        $watch.Stop()
        $manifest.wall_seconds = [Math]::Round($watch.Elapsed.TotalSeconds, 6)
        $manifest.process_cpu_seconds = [Math]::Round($process.TotalProcessorTime.TotalSeconds, 6)
        $manifest.exit_code = $process.ExitCode
        [System.IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result, [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::WriteAllText($stderrPath, $stderrTask.Result, [System.Text.UTF8Encoding]::new($false))
        Write-RunLog "ENCODER_EXIT=$($manifest.exit_code) CPU_SECONDS=$($manifest.process_cpu_seconds) WALL_SECONDS=$($manifest.wall_seconds) TIMED_OUT=$($manifest.timed_out)"
    } finally {
        $watch.Stop()
        $process.Dispose()
    }
    if ($manifest.timed_out -or $manifest.exit_code -ne 0) { throw 'vpxenc failed or timed out' }
    if (-not (Test-Path -LiteralPath $ivf -PathType Leaf)) { throw 'vpxenc did not write output IVF' }
    $manifest.output_bytes = (Get-Item -LiteralPath $ivf).Length
    $manifest.output_sha256 = (Get-FileHash -LiteralPath $ivf -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-RunLog "OUTPUT_BYTES=$($manifest.output_bytes) OUTPUT_SHA256=$($manifest.output_sha256)"
    if ($manifest.output_bytes -le 0 -or $manifest.output_sha256 -ne $clip.OutputSha) {
        throw "libvpx IVF mismatch: expected=$($clip.OutputSha) actual=$($manifest.output_sha256)"
    }
} catch {
    $traceFailure = $_.Exception.Message
    Write-RunLog "CAPTURE_FAILURE=$traceFailure"
} finally {
    if ($wprStarted) {
        try {
            $stopped = Invoke-BoundedTool -File $wpr -Arguments ('-stop "{0}"' -f $etl) -Seconds 120
            $manifest.wpr_stop_exit = $stopped.ExitCode
            Write-RunLog "WPR_STOP_EXIT=$($stopped.ExitCode) $($stopped.Output)"
            if ($stopped.ExitCode -ne 0) { throw "WPR stop failed: $($stopped.Output)" }
            if (-not (Test-Path -LiteralPath $etl -PathType Leaf)) { throw 'WPR did not write ETL' }
            $manifest.etl_bytes = (Get-Item -LiteralPath $etl).Length
            $manifest.etl_sha256 = (Get-FileHash -LiteralPath $etl -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($manifest.etl_bytes -le 0) { throw 'WPR wrote an empty ETL' }
            Write-RunLog "ETL_BYTES=$($manifest.etl_bytes) ETL_SHA256=$($manifest.etl_sha256)"
        } catch {
            if (-not $traceFailure) { $traceFailure = $_.Exception.Message }
            Write-RunLog "WPR_STOP_FAILURE=$($_.Exception.Message)"
        }
    }
    $manifest.end_utc = [DateTime]::UtcNow.ToString('o')
    $manifest.failure = $traceFailure
    [System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 6) + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false))
}
if ($traceFailure) { throw "Capture failed: $traceFailure (see $manifestPath)" }
Write-RunLog "CAPTURE_OK clip=$ClipName manifest=$manifestPath etl=$etl"
