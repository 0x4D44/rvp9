param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('ducks', 'tree', 'town')]
    [string]$ClipName,
    [switch]$SmokeOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$ClipName = $ClipName.ToLowerInvariant()

$worktree = 'D:\worktrees\rvp9-2\20260925-TSK-HUM-unit1434-integrated-ip-profile'
$expectedHead = '88644d86641f99a4291c21842bdc17aec9baee8e'
$encoderPath = Join-Path $worktree '.scratch\target\release\rvp9enc.exe'
$pdbPath = Join-Path (Split-Path -Parent $encoderPath) 'rvp9enc.pdb'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clipSpecs = @{
    ducks = [pscustomobject]@{
        InputPath = Join-Path $inputRoot 'ducks_take_off_420_1080p30_60f.y4m'
        InputSha256 = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
        OutputSha256 = '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'
    }
    tree = [pscustomobject]@{
        InputPath = Join-Path $inputRoot 'in_to_tree_420_1080p30_60f.y4m'
        InputSha256 = 'ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'
        OutputSha256 = '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'
    }
    town = [pscustomobject]@{
        InputPath = Join-Path $inputRoot 'old_town_cross_420_1080p30_60f.y4m'
        InputSha256 = 'b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'
        OutputSha256 = '4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'
    }
}

$clip = $clipSpecs[$ClipName]
$artifactRoot = Join-Path $worktree '.scratch\unit1434-ip-profile'
$outputDir = Join-Path $artifactRoot $ClipName
if ($SmokeOnly) {
    $outputPath = Join-Path $outputDir 'smoke-output.ivf'
    $stdoutPath = Join-Path $outputDir 'smoke.stdout.txt'
    $stderrPath = Join-Path $outputDir 'smoke.stderr.txt'
    $logPath = Join-Path $outputDir 'smoke.log'
    $tracePath = $null
} else {
    $outputPath = Join-Path $outputDir 'output.ivf'
    $stdoutPath = Join-Path $outputDir 'encoder.stdout.txt'
    $stderrPath = Join-Path $outputDir 'encoder.stderr.txt'
    $logPath = Join-Path $outputDir 'capture.log'
    $tracePath = Join-Path $outputDir 'capture.etl'
}

function Write-RunLog([string]$Message) {
    [Console]::Out.WriteLine($Message)
    [System.IO.File]::AppendAllText(
        $script:logPath,
        $Message + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}

function Write-CommandOutput([string]$Prefix, [object[]]$Lines) {
    foreach ($line in $Lines) {
        Write-RunLog "$Prefix=$line"
    }
}

function Assert-FileHash([string]$Name, [string]$Path, [string]$ExpectedSha256) {
    Write-RunLog "ASSET_NAME=$Name"
    Write-RunLog "ASSET_PATH=$Path"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        Write-RunLog "ASSET_EXISTS=false"
        throw "Required asset is missing: $Path"
    }

    $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-RunLog "ASSET_EXISTS=true"
    Write-RunLog "ASSET_EXPECTED_SHA256=$ExpectedSha256"
    Write-RunLog "ASSET_ACTUAL_SHA256=$actualSha256"
    if ($actualSha256 -ne $ExpectedSha256) {
        throw "Asset hash mismatch for $Name`: expected=$ExpectedSha256 actual=$actualSha256"
    }
}

function New-EmptyArtifact([string]$Path) {
    $stream = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    $stream.Dispose()
}

function Invoke-Encoder {
    $argv = @(
        '--codec=vp9', '--passes=1', '--rt', '--cpu-used=6', '--end-usage=cbr',
        '--target-bitrate=1260', '--fps=30/1', '--threads=1', '--tile-columns=0',
        '--lag-in-frames=0', '--tune-content=default', '--aq-mode=0',
        '--noise-sensitivity=0', '--limit=60', '--kf-max-dist=60', '--quiet',
        '--ivf', '-o', $outputPath, $clip.InputPath
    )
    $argumentString = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath, $clip.InputPath
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $encoderPath
    $psi.Arguments = $argumentString
    $psi.WorkingDirectory = $worktree
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    Write-RunLog "ENCODER_PATH=$encoderPath"
    Write-RunLog ('ENCODER_ARGV_JSON=' + (ConvertTo-Json -InputObject $argv -Compress))
    Write-RunLog "ENCODER_OUTPUT_PATH=$outputPath"
    Write-RunLog "ENCODER_STDOUT_PATH=$stdoutPath"
    Write-RunLog "ENCODER_STDERR_PATH=$stderrPath"
    Write-RunLog 'ENCODER_TIMEOUT_SECONDS=90'
    Write-RunLog "ENCODER_START_UTC=$([DateTime]::UtcNow.ToString('o'))"

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $stdoutTask = $null
    $stderrTask = $null
    $timedOut = $false
    $stillRunning = $false
    $exitCode = $null
    $cpuSeconds = $null
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) {
            throw 'Encoder process did not start.'
        }
        $processId = $process.Id
        Write-RunLog "ENCODER_PROCESS_ID=$processId"
        Write-RunLog 'ENCODER_PROCESS_STARTED=true'
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()

        while (-not $process.HasExited) {
            $remainingMs = [int][Math]::Ceiling((90.0 - $watch.Elapsed.TotalSeconds) * 1000.0)
            if ($remainingMs -le 0) {
                $timedOut = $true
                Write-RunLog 'ENCODER_TIMEOUT_REACHED=true'
                try {
                    if (-not $process.HasExited) {
                        $process.Kill()
                        Write-RunLog 'ENCODER_KILL_REQUESTED=true'
                    }
                } catch {
                    Write-RunLog "ENCODER_KILL_ERROR=$($_.Exception.Message)"
                }
                if (-not $process.WaitForExit(10000)) {
                    $stillRunning = $true
                    Write-RunLog 'ENCODER_EXIT_AFTER_KILL=still-running-after-10-seconds'
                }
                break
            }
            [void]$process.WaitForExit([Math]::Min(100, $remainingMs))
        }

        if (-not $stillRunning -and $process.HasExited) {
            $exitCode = $process.ExitCode
            try {
                $cpuSeconds = [Math]::Round($process.TotalProcessorTime.TotalSeconds, 4)
            } catch {
                Write-RunLog "ENCODER_CPU_TIME_ERROR=$($_.Exception.Message)"
            }
        }
    } catch {
        Write-RunLog "ENCODER_PROCESS_ERROR=$($_.Exception.Message)"
        throw
    } finally {
        $watch.Stop()
        Write-RunLog "ENCODER_TIMED_OUT=$timedOut"
        if ($null -eq $exitCode) {
            Write-RunLog 'ENCODER_EXIT_CODE=unavailable'
        } else {
            Write-RunLog "ENCODER_EXIT_CODE=$exitCode"
        }
        if ($null -eq $cpuSeconds) {
            Write-RunLog 'ENCODER_CPU_SECONDS=unavailable'
        } else {
            Write-RunLog "ENCODER_CPU_SECONDS=$cpuSeconds"
        }
        Write-RunLog "ENCODER_ELAPSED_SECONDS=$([Math]::Round($watch.Elapsed.TotalSeconds, 4))"
        Write-RunLog "ENCODER_END_UTC=$([DateTime]::UtcNow.ToString('o'))"

        $utf8 = [System.Text.UTF8Encoding]::new($false)
        foreach ($streamResult in @(
            [pscustomobject]@{ Name = 'stdout'; Task = $stdoutTask; Path = $stdoutPath },
            [pscustomobject]@{ Name = 'stderr'; Task = $stderrTask; Path = $stderrPath }
        )) {
            if ($null -eq $streamResult.Task) {
                Write-RunLog "ENCODER_$($streamResult.Name.ToUpperInvariant())_CAPTURE=process-not-started"
                continue
            }
            $streamReady = $false
            if (-not $stillRunning) {
                try {
                    $streamReady = $streamResult.Task.Wait(5000)
                } catch {
                    Write-RunLog "ENCODER_$($streamResult.Name.ToUpperInvariant())_WAIT_ERROR=$($_.Exception.Message)"
                }
            }
            if ($streamReady) {
                try {
                    [System.IO.File]::WriteAllText($streamResult.Path, [string]$streamResult.Task.Result, $utf8)
                    Write-RunLog "ENCODER_$($streamResult.Name.ToUpperInvariant())_CAPTURE=complete"
                } catch {
                    Write-RunLog "ENCODER_$($streamResult.Name.ToUpperInvariant())_CAPTURE_ERROR=$($_.Exception.Message)"
                }
            } else {
                Write-RunLog "ENCODER_$($streamResult.Name.ToUpperInvariant())_CAPTURE=not-complete"
            }
        }
        $process.Dispose()
    }

    $outputBytes = 0
    $outputSha256 = ''
    if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
        $outputBytes = (Get-Item -LiteralPath $outputPath).Length
        $outputSha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    Write-RunLog "OUTPUT_BYTES=$outputBytes"
    Write-RunLog "OUTPUT_SHA256=$outputSha256"
    Write-RunLog "OUTPUT_EXPECTED_SHA256=$($clip.OutputSha256)"
    Write-RunLog "OUTPUT_HASH_MATCH=$($outputSha256 -eq $clip.OutputSha256)"

    [pscustomobject]@{
        TimedOut = $timedOut
        StillRunningAfterKill = $stillRunning
        ExitCode = $exitCode
        OutputBytes = $outputBytes
        OutputSha256 = $outputSha256
    }
}

if (-not (Test-Path -LiteralPath $worktree -PathType Container)) {
    throw "Integrated Crucible worktree is missing: $worktree"
}
$scratchPath = Join-Path $worktree '.scratch'
if (-not (Test-Path -LiteralPath $scratchPath -PathType Container)) {
    throw "Expected scratch directory is missing: $scratchPath"
}
if (Test-Path -LiteralPath $artifactRoot -PathType Leaf) {
    throw "Artifact root exists as a file: $artifactRoot"
}
if (-not (Test-Path -LiteralPath $artifactRoot -PathType Container)) {
    New-Item -ItemType Directory -Path $artifactRoot | Out-Null
}
if (Test-Path -LiteralPath $outputDir -PathType Leaf) {
    throw "Clip artifact path exists as a file: $outputDir"
}
if (-not (Test-Path -LiteralPath $outputDir -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

$plannedArtifacts = @($outputPath, $stdoutPath, $stderrPath, $logPath)
if (-not $SmokeOnly) {
    $plannedArtifacts += $tracePath
}
$existingArtifacts = @($plannedArtifacts | Where-Object { Test-Path -LiteralPath $_ })
if ($existingArtifacts.Count -gt 0) {
    throw "Refusing to overwrite existing capture artifacts: $($existingArtifacts -join ', ')"
}
New-EmptyArtifact -Path $logPath
New-EmptyArtifact -Path $stdoutPath
New-EmptyArtifact -Path $stderrPath

$scriptFailure = $null
$wprStartAttempted = $false
$wprStopExit = $null
$wprStopFailed = $false
$etlSha256 = ''
$etlBytes = 0
try {
    Write-RunLog "CAPTURE_START_UTC=$([DateTime]::UtcNow.ToString('o'))"
    Write-RunLog "CAPTURE_CLIP=$ClipName"
    Write-RunLog "CAPTURE_MODE=$(if ($SmokeOnly) { 'smoke-only' } else { 'wpr-memory' })"
    Write-RunLog "WORKTREE=$worktree"
    Write-RunLog "EXPECTED_HEAD=$expectedHead"

    $headOutput = @(& git -C $worktree rev-parse HEAD 2>&1)
    $gitExit = $LASTEXITCODE
    Write-CommandOutput -Prefix 'GIT_HEAD_OUTPUT' -Lines $headOutput
    Write-RunLog "GIT_HEAD_EXIT=$gitExit"
    $actualHead = (($headOutput | ForEach-Object { [string]$_ }) -join '').Trim().ToLowerInvariant()
    Write-RunLog "ACTUAL_HEAD=$actualHead"
    if ($gitExit -ne 0 -or $actualHead -ne $expectedHead) {
        throw "Integrated worktree HEAD mismatch: expected=$expectedHead actual=$actualHead git_exit=$gitExit"
    }

    Assert-FileHash -Name 'encoder' -Path $encoderPath -ExpectedSha256 '1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a'
    Assert-FileHash -Name 'pdb' -Path $pdbPath -ExpectedSha256 '27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4'
    Assert-FileHash -Name "input-$ClipName" -Path $clip.InputPath -ExpectedSha256 $clip.InputSha256

    if ($SmokeOnly) {
        Write-RunLog 'WPR_STATUS=skipped-smoke-only'
        Write-RunLog 'WPR_START=skipped-smoke-only'
    } else {
        $wprCommand = Get-Command -Name 'wpr.exe' -CommandType Application -ErrorAction Stop | Select-Object -First 1
        $wprExecutable = $wprCommand.Source
        Write-RunLog "WPR_PATH=$wprExecutable"
        Write-RunLog 'WPR_MODE=memory; -filemode omitted'
        Write-RunLog 'WPR_STATUS_COMMAND=wpr -status'
        $statusOutput = @(& $wprExecutable -status 2>&1)
        $statusExit = $LASTEXITCODE
        Write-CommandOutput -Prefix 'WPR_STATUS_OUTPUT' -Lines $statusOutput
        Write-RunLog "WPR_STATUS_EXIT=$statusExit"
        $statusText = ($statusOutput | ForEach-Object { [string]$_ }) -join ' '
        if ($statusExit -eq 0 -and $statusText -match '(?i)\bWPR is not recording\b') {
            Write-RunLog 'WPR_STATUS_CLASSIFICATION=idle'
        } elseif ($statusText -match '(?i)\bWPR is recording\b') {
            Write-RunLog 'WPR_STATUS_CLASSIFICATION=recording'
            throw "WPR is already recording: $statusText"
        } else {
            Write-RunLog 'WPR_STATUS_CLASSIFICATION=unknown'
            throw "Could not verify that WPR is idle: exit=$statusExit status=$statusText"
        }

        Write-RunLog 'WPR_START_COMMAND=wpr -start CPU'
        $wprStartAttempted = $true
        $startOutput = @(& $wprExecutable -start CPU 2>&1)
        $startExit = $LASTEXITCODE
        Write-CommandOutput -Prefix 'WPR_START_OUTPUT' -Lines $startOutput
        Write-RunLog "WPR_START_EXIT=$startExit"
        Write-RunLog "WPR_START_STATUS=$(if ($startExit -eq 0) { 'started' } else { 'failed' })"
        if ($startExit -ne 0) {
            throw "WPR start failed with exit code $startExit"
        }
    }

    $encodeResult = Invoke-Encoder
    if ($encodeResult.TimedOut) {
        throw 'Encoder exceeded its 90-second limit.'
    }
    if ($encodeResult.StillRunningAfterKill) {
        throw 'Encoder was still running 10 seconds after the timeout kill request.'
    }
    if ($null -eq $encodeResult.ExitCode -or $encodeResult.ExitCode -ne 0) {
        throw "Encoder failed with exit code $($encodeResult.ExitCode)."
    }
    if ($encodeResult.OutputBytes -le 0 -or $encodeResult.OutputSha256 -ne $clip.OutputSha256) {
        throw "Encoded IVF mismatch: expected=$($clip.OutputSha256) actual=$($encodeResult.OutputSha256) bytes=$($encodeResult.OutputBytes)"
    }
} catch {
    $scriptFailure = $_.Exception.Message
    Write-RunLog "CAPTURE_FAILURE=$scriptFailure"
} finally {
    if ($wprStartAttempted) {
        try {
            Write-RunLog "WPR_STOP_COMMAND=wpr -stop $tracePath"
            $stopOutput = @(& $wprExecutable -stop $tracePath 2>&1)
            $wprStopExit = $LASTEXITCODE
            Write-CommandOutput -Prefix 'WPR_STOP_OUTPUT' -Lines $stopOutput
            Write-RunLog "WPR_STOP_EXIT=$wprStopExit"
            if ($wprStopExit -ne 0) {
                $wprStopFailed = $true
                Write-RunLog 'WPR_STOP_STATUS=failed'
            } else {
                Write-RunLog 'WPR_STOP_STATUS=stopped'
            }

            $stopText = ($stopOutput | ForEach-Object { [string]$_ }) -join ' '
            if ($stopText -match '(?i)\bno\s+(?:events?\s+)?(?:were\s+)?lost\b|\b0\s+(?:events?\s+)?lost\b|\blost\s+0\s+events?\b') {
                Write-RunLog 'WPR_LOSS_WORDING=explicit-no-loss-wording-present'
            } elseif ($stopText -match '(?i)\b(lost|loss|drop|overrun|overflow|discard)\b') {
                Write-RunLog 'WPR_LOSS_WORDING=loss-related-wording-present; inspect WPR_STOP_OUTPUT'
            } else {
                Write-RunLog 'WPR_LOSS_WORDING=no-loss-wording-reported'
            }

            if (Test-Path -LiteralPath $tracePath -PathType Leaf) {
                $etlBytes = (Get-Item -LiteralPath $tracePath).Length
                $etlSha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
                Write-RunLog "ETL_BYTES=$etlBytes"
                Write-RunLog "ETL_SHA256=$etlSha256"
                if ($etlBytes -le 0) {
                    $wprStopFailed = $true
                    Write-RunLog 'ETL_STATUS=empty'
                } else {
                    Write-RunLog 'ETL_STATUS=present'
                }
            } else {
                $wprStopFailed = $true
                Write-RunLog 'ETL_STATUS=missing'
            }
        } catch {
            $wprStopFailed = $true
            Write-RunLog "WPR_STOP_ERROR=$($_.Exception.Message)"
        }
    } elseif (-not $SmokeOnly) {
        Write-RunLog 'WPR_STOP_STATUS=not-started-after-idle-check'
    }

    if ($null -ne $wprStopExit -and $wprStopExit -ne 0 -and -not $scriptFailure) {
        $scriptFailure = "WPR stop failed with exit code $wprStopExit"
    }
    if ($wprStopFailed -and -not $scriptFailure) {
        $scriptFailure = 'WPR stop or ETL validation failed.'
    }
    Write-RunLog "CAPTURE_SUCCESS=$($null -eq $scriptFailure)"
    Write-RunLog "CAPTURE_END_UTC=$([DateTime]::UtcNow.ToString('o'))"
}

if ($scriptFailure) {
    throw "Capture failed: $scriptFailure"
}
Write-RunLog "CAPTURE_COMPLETE clip=$ClipName smoke_only=$SmokeOnly output=$outputPath output_sha256=$((Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()) etl_sha256=$etlSha256"
