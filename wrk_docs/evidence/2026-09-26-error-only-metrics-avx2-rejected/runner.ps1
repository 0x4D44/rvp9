param(
    [switch]$SyntaxOnly,
    [string]$CandidateWorktreePath,
    [string]$CandidateExePath,
    [string]$CandidateExeSha256,
    [string[]]$CandidateSourceRelativePath = @(),
    [string[]]$CandidateSourceSha256 = @(),
    [string]$CandidateRevision = 'candidate',
    [string]$BaselineExePath = 'D:\worktrees\rvp9-2\20260925-TSK-HUM-unit1434-integrated-ip-profile\.scratch\target\release\rvp9enc.exe',
    [string]$BaselinePdbPath = 'D:\worktrees\rvp9-2\20260925-TSK-HUM-unit1434-integrated-ip-profile\.scratch\target\release\rvp9enc.pdb',
    [string]$InputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs',
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

if ($SyntaxOnly) {
    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile($PSCommandPath, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -gt 0) {
        $parseErrors | ForEach-Object { [Console]::Error.WriteLine($_.ToString()) }
        exit 1
    }
    Write-Output 'SYNTAX_OK'
    exit 0
}

if ([string]::IsNullOrWhiteSpace($CandidateWorktreePath)) { throw 'CandidateWorktreePath is required.' }
if ([string]::IsNullOrWhiteSpace($CandidateExeSha256) -or $CandidateExeSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'CandidateExeSha256 must be a 64-character SHA-256 hex string.'
}
if ($CandidateSourceRelativePath.Count -eq 0 -or $CandidateSourceRelativePath.Count -ne $CandidateSourceSha256.Count) {
    throw 'Pass matching, non-empty CandidateSourceRelativePath and CandidateSourceSha256 arrays.'
}
foreach ($sha in $CandidateSourceSha256) {
    if ($sha -notmatch '^[0-9a-fA-F]{64}$') { throw "Invalid candidate source SHA-256: $sha" }
}

$CandidateExeSha256 = $CandidateExeSha256.ToLowerInvariant()
$CandidateSourceSha256 = @($CandidateSourceSha256 | ForEach-Object { $_.ToLowerInvariant() })
$BaselineExeSha256 = '1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a'
$BaselinePdbSha256 = '27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4'
$CandidateWorktreePath = [System.IO.Path]::GetFullPath($CandidateWorktreePath)
if ([string]::IsNullOrWhiteSpace($CandidateExePath)) {
    $CandidateExePath = Join-Path $CandidateWorktreePath '.scratch\target\release\rvp9enc.exe'
}

$clips = @(
    [pscustomobject]@{ Name='ducks'; Path=(Join-Path $InputRoot 'ducks_take_off_420_1080p30_60f.y4m'); InputSha='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; OutputSha='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' },
    [pscustomobject]@{ Name='tree'; Path=(Join-Path $InputRoot 'in_to_tree_420_1080p30_60f.y4m'); InputSha='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; OutputSha='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' },
    [pscustomobject]@{ Name='town'; Path=(Join-Path $InputRoot 'old_town_cross_420_1080p30_60f.y4m'); InputSha='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; OutputSha='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f' }
)
$encoders = @(
    [pscustomobject]@{ Arm='baseline'; Revision='88644d86'; Exe=$BaselineExePath; ExeSha=$BaselineExeSha256; Pdb=$BaselinePdbPath; PdbSha=$BaselinePdbSha256 },
    [pscustomobject]@{ Arm='candidate'; Revision=$CandidateRevision; Exe=$CandidateExePath; ExeSha=$CandidateExeSha256; Pdb=$null; PdbSha=$null }
)

$scriptRoot = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($OutputRoot)) { $OutputRoot = Join-Path $scriptRoot 'error-only-metrics-run-ab' }
$runStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss.fffZ')
$root = Join-Path $OutputRoot "run-$runStamp"
[void](New-Item -ItemType Directory -Path $root -Force)
$runnerHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()

$assets = @(
    [pscustomobject]@{ Name='baseline_exe'; Path=$BaselineExePath; ExpectedSha=$BaselineExeSha256 },
    [pscustomobject]@{ Name='baseline_pdb'; Path=$BaselinePdbPath; ExpectedSha=$BaselinePdbSha256 },
    [pscustomobject]@{ Name='candidate_exe'; Path=$CandidateExePath; ExpectedSha=$CandidateExeSha256 }
)
for ($i = 0; $i -lt $CandidateSourceRelativePath.Count; $i++) {
    $relative = $CandidateSourceRelativePath[$i]
    if ([System.IO.Path]::IsPathRooted($relative)) { throw "Candidate source path must be relative to the candidate worktree: $relative" }
    $sourcePath = [System.IO.Path]::GetFullPath((Join-Path $CandidateWorktreePath $relative))
    $rootPrefix = $CandidateWorktreePath.TrimEnd('\') + '\'
    if (-not $sourcePath.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Candidate source path escapes the candidate worktree: $relative"
    }
    $assets += [pscustomobject]@{ Name="candidate_source_$($i + 1)"; Path=$sourcePath; ExpectedSha=$CandidateSourceSha256[$i] }
}
foreach ($clip in $clips) {
    $assets += [pscustomobject]@{ Name="input_$($clip.Name)"; Path=$clip.Path; ExpectedSha=$clip.InputSha }
}

$identity = @()
foreach ($asset in $assets) {
    $bytes = ''
    $actualSha = ''
    $errorText = ''
    if (Test-Path -LiteralPath $asset.Path -PathType Leaf) {
        try {
            $item = Get-Item -LiteralPath $asset.Path
            $bytes = $item.Length
            $actualSha = (Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash.ToLowerInvariant()
        } catch {
            $errorText = $_.Exception.Message
        }
    } else {
        $errorText = 'missing file'
    }
    $match = ($actualSha -eq $asset.ExpectedSha)
    $identity += [pscustomobject]@{ Name=$asset.Name; Path=$asset.Path; Bytes=$bytes; ExpectedSha256=$asset.ExpectedSha; ActualSha256=$actualSha; Match=$match; Error=$errorText }
}
$identityPath = Join-Path $root 'identity.csv'
$identity | Export-Csv -LiteralPath $identityPath -NoTypeInformation -Encoding UTF8

$rawPath = Join-Path $root 'raw.csv'
$loadPath = Join-Path $root 'load.csv'
$summaryPath = Join-Path $root 'summary.csv'
$stabilityPath = Join-Path $root 'output-stability.csv'
$manifestPath = Join-Path $root 'manifest.txt'
Set-Content -LiteralPath $rawPath -Value '"RunId","Phase","PairIndex","Clip","Arm","Revision","Order","StartUtc","EndUtc","WallSeconds","ProcessCpuSeconds","ExitCode","TimedOut","TimeoutReason","ProcessStillRunning","ProcessError","OutputPath","OutputBytes","OutputSha256","ExpectedOutputSha256","OutputMatchesExpected","StdoutPath","StderrPath","LoadSamplesDuringRun","LoadLateSamples","LoadErrors","LoadMaxGapSeconds","LoadMeanPct","LoadMaxPct","LoadStatus"' -Encoding UTF8
Set-Content -LiteralPath $loadPath -Value '"RunId","SampleUtc","ProcessElapsedSeconds","CounterCallMs","CpuBusyPct","DuringChild","GapSeconds","Status","Error"' -Encoding UTF8

$counterPath = '\Processor(_Total)\% Processor Time'
$counterEnabled = $true
$counterPreflight = ''
try { $counterPreflight = [math]::Round([double](Get-Counter -Counter $counterPath -MaxSamples 1 -ErrorAction Stop).CounterSamples[0].CookedValue, 2) }
catch { $counterEnabled = $false; $counterPreflight = $_.Exception.Message }

$manifest = @(
    'experiment=error-only metrics candidate vs integrated 88644d86 process-CPU A/B',
    "started_utc=$([DateTime]::UtcNow.ToString('o'))", "runner_path=$PSCommandPath", "runner_sha256=$runnerHash", "host=$env:COMPUTERNAME",
    "baseline_encoder=$BaselineExePath", "baseline_exe_sha256=$BaselineExeSha256", "baseline_pdb=$BaselinePdbPath", "baseline_pdb_sha256=$BaselinePdbSha256",
    "candidate_revision=$CandidateRevision", "candidate_worktree=$CandidateWorktreePath", "candidate_encoder=$CandidateExePath", "candidate_exe_sha256=$CandidateExeSha256",
    "candidate_sources=$((0..($CandidateSourceRelativePath.Count - 1) | ForEach-Object { "$($CandidateSourceRelativePath[$_])=$($CandidateSourceSha256[$_])" }) -join ';')",
    "inputs=$($clips.Path -join ';')",
    'settings=CPU6 CBR1260 fps30/1 threads1 tile0 AQ0 noise0 lag0 limit60 KF60',
    'warmups=one per arm per clip; measured_pairs=15 per clip; odd baseline/candidate, even candidate/baseline',
    'clip_order=odd pairs ducks/tree/town; even pairs town/tree/ducks; ratios=candidate divided by baseline; all rows retained; host load is diagnostic only',
    'process_timeout_seconds=90', 'total_timeout_seconds=1200', "host_counter=$counterPath", "counter_preflight=$counterPreflight", "counter_enabled=$counterEnabled",
    "identity_csv=$identityPath", "raw_csv=$rawPath", "load_csv=$loadPath", "summary_csv=$summaryPath", "output_stability_csv=$stabilityPath"
)
Set-Content -LiteralPath $manifestPath -Value $manifest -Encoding UTF8
if (@($identity | Where-Object { -not $_.Match }).Count -gt 0) {
    $badAssets = @($identity | Where-Object { -not $_.Match } | ForEach-Object { "$($_.Name): expected=$($_.ExpectedSha256) actual=$($_.ActualSha256) error=$($_.Error)" }) -join '; '
    throw "Pinned asset identity check failed. See $identityPath. $badAssets"
}

$overall = [System.Diagnostics.Stopwatch]::StartNew()
$perRunLimit = 90
$totalLimit = 1200
function Add-CsvRow([string]$Path, $Row) {
    $Row | Export-Csv -LiteralPath $Path -Append -NoTypeInformation -Encoding UTF8
}

function Invoke-Encode($Clip, $Encoder, [string]$Phase, [int]$PairIndex, [string]$Order) {
    if ($script:overall.Elapsed.TotalSeconds -ge $script:totalLimit) { throw 'Total run limit reached before next encode.' }
    $runId = if ($Phase -eq 'warmup') { "warm-$($Clip.Name)-$($Encoder.Arm)" } else { "p$('{0:D2}' -f $PairIndex)-$($Clip.Name)-$($Encoder.Arm)" }
    $outputPath = Join-Path $script:root "$runId.ivf"
    $stdoutPath = Join-Path $script:root "$runId.stdout.txt"
    $stderrPath = Join-Path $script:root "$runId.stderr.txt"
    $arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$Clip.Path
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $Encoder.Exe
    $psi.Arguments = $arguments
    $psi.WorkingDirectory = $script:root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $loadValues = New-Object 'System.Collections.Generic.List[double]'
    $loadErrors = 0
    $lateSamples = 0
    $maxGap = 0.0
    $lastSampleElapsed = 0.0
    $timedOut = $false
    $timeoutReason = ''
    $processStillRunning = $false
    $processError = ''
    $exitCode = ''
    $cpu = 0.0
    $wall = 0.0
    $start = [DateTime]::UtcNow
    $end = $start
    $stdoutTask = $null
    $stderrTask = $null
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $nextCounter = 0.25
    [Console]::Out.WriteLine("START $runId")
    try {
        if (-not $process.Start()) { throw 'Process did not start.' }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $start = $process.StartTime.ToUniversalTime()
        while (-not $process.HasExited) {
            $remainingTotal = $script:totalLimit - $script:overall.Elapsed.TotalSeconds
            if ($watch.Elapsed.TotalSeconds -ge $script:perRunLimit -or $remainingTotal -le 0) {
                $timedOut = $true
                $timeoutReason = if ($remainingTotal -le 0) { 'total-1200s' } else { 'child-90s' }
                try { $process.Kill() } catch { $processError = "kill failed: $($_.Exception.Message)" }
                if ($remainingTotal -gt 0) { [void]$process.WaitForExit([int][math]::Min(1000, $remainingTotal * 1000)) }
                break
            }
            if ($script:counterEnabled -and $watch.Elapsed.TotalSeconds -ge $nextCounter) {
                $counterWatch = [System.Diagnostics.Stopwatch]::StartNew()
                try {
                    $counter = Get-Counter -Counter $script:counterPath -MaxSamples 1 -ErrorAction Stop
                    $busy = [math]::Round([double]$counter.CounterSamples[0].CookedValue, 3)
                    $sampleUtc = [DateTime]::UtcNow
                    $elapsed = $watch.Elapsed.TotalSeconds
                    $during = -not $process.HasExited
                    $gap = $elapsed - $lastSampleElapsed
                    if ($gap -gt $maxGap) { $maxGap = $gap }
                    if ($during) { $loadValues.Add($busy) } else { $lateSamples++ }
                    Add-CsvRow $script:loadPath ([pscustomobject]@{ RunId=$runId; SampleUtc=$sampleUtc.ToString('o'); ProcessElapsedSeconds=[math]::Round($elapsed,4); CounterCallMs=[math]::Round($counterWatch.Elapsed.TotalMilliseconds,1); CpuBusyPct=$busy; DuringChild=$during; GapSeconds=[math]::Round($gap,4); Status=$(if($during){'ok'}else{'child-exited-during-sample'}); Error='' })
                } catch {
                    $loadErrors++
                    $sampleUtc = [DateTime]::UtcNow
                    $elapsed = $watch.Elapsed.TotalSeconds
                    $gap = $elapsed - $lastSampleElapsed
                    if ($gap -gt $maxGap) { $maxGap = $gap }
                    Add-CsvRow $script:loadPath ([pscustomobject]@{ RunId=$runId; SampleUtc=$sampleUtc.ToString('o'); ProcessElapsedSeconds=[math]::Round($elapsed,4); CounterCallMs=[math]::Round($counterWatch.Elapsed.TotalMilliseconds,1); CpuBusyPct=''; DuringChild=(-not $process.HasExited); GapSeconds=[math]::Round($gap,4); Status='error'; Error=$_.Exception.Message })
                }
                $lastSampleElapsed = $watch.Elapsed.TotalSeconds
                while ($nextCounter -le $lastSampleElapsed) { $nextCounter += 1.0 }
            }
            [void]$process.WaitForExit(100)
        }
    } catch {
        $processError = $_.Exception.Message
    }
    $watch.Stop()
    try {
        $process.Refresh()
        if ($process.HasExited) {
            $exitCode = $process.ExitCode
            $cpu = [math]::Round($process.TotalProcessorTime.TotalSeconds, 6)
            $end = $process.ExitTime.ToUniversalTime()
        } else {
            $processStillRunning = $true
            $end = [DateTime]::UtcNow
            try { $cpu = [math]::Round($process.TotalProcessorTime.TotalSeconds, 6) } catch { }
        }
    } catch {
        if ([string]::IsNullOrWhiteSpace($processError)) { $processError = $_.Exception.Message }
        $end = [DateTime]::UtcNow
    }
    $wall = [math]::Round(($end - $start).TotalSeconds, 6)
    if ($stdoutTask -and $stdoutTask.IsCompleted) {
        [System.IO.File]::WriteAllText($stdoutPath, $stdoutTask.Result, [System.Text.UTF8Encoding]::new($false))
    }
    if ($stderrTask -and $stderrTask.IsCompleted) {
        [System.IO.File]::WriteAllText($stderrPath, $stderrTask.Result, [System.Text.UTF8Encoding]::new($false))
    }
    $bytes = 0
    $hash = ''
    if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
        $bytes = (Get-Item -LiteralPath $outputPath).Length
        $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $expectedSha = $Clip.OutputSha
    $mean = ''
    $peak = ''
    if ($loadValues.Count -gt 0) {
        $mean = [math]::Round(($loadValues | Measure-Object -Average).Average, 3)
        $peak = [math]::Round(($loadValues | Measure-Object -Maximum).Maximum, 3)
    }
    $loadStatus = if (-not $script:counterEnabled) { 'unavailable' } elseif ($loadErrors -gt 0 -or $maxGap -gt 2.5) { 'gaps-or-errors' } elseif ($loadValues.Count -eq 0) { 'no-during-run-sample' } else { 'sampled' }
    $row = [pscustomobject]@{
        RunId=$runId; Phase=$Phase; PairIndex=$PairIndex; Clip=$Clip.Name; Arm=$Encoder.Arm; Revision=$Encoder.Revision; Order=$Order
        StartUtc=$start.ToString('o'); EndUtc=$end.ToString('o'); WallSeconds=$wall; ProcessCpuSeconds=$cpu; ExitCode=$exitCode
        TimedOut=$timedOut; TimeoutReason=$timeoutReason; ProcessStillRunning=$processStillRunning; ProcessError=$processError
        OutputPath=$outputPath; OutputBytes=$bytes; OutputSha256=$hash; ExpectedOutputSha256=$expectedSha; OutputMatchesExpected=($hash -eq $expectedSha)
        StdoutPath=$(if(Test-Path -LiteralPath $stdoutPath -PathType Leaf){$stdoutPath}else{''}); StderrPath=$(if(Test-Path -LiteralPath $stderrPath -PathType Leaf){$stderrPath}else{''})
        LoadSamplesDuringRun=$loadValues.Count; LoadLateSamples=$lateSamples; LoadErrors=$loadErrors; LoadMaxGapSeconds=[math]::Round($maxGap,4); LoadMeanPct=$mean; LoadMaxPct=$peak; LoadStatus=$loadStatus
    }
    Add-CsvRow $script:rawPath $row
    [Console]::Out.WriteLine(("DONE {0} exit={1} wall={2:F3}s cpu={3:F3}s bytes={4} sha={5} load={6}" -f $runId,$exitCode,$wall,$cpu,$bytes,$hash,$loadStatus))
    $process.Dispose()
    if ($timedOut -or $processStillRunning -or $exitCode -ne 0 -or $hash -ne $expectedSha -or -not [string]::IsNullOrWhiteSpace($processError)) {
        throw "Encode failed, timed out, or output hash mismatched: $runId expected=$expectedSha got=$hash error=$processError"
    }
}

function Get-Median([double[]]$Values) {
    $sorted = @($Values | Sort-Object)
    if ($sorted.Count -eq 0) { return [double]::NaN }
    if ($sorted.Count % 2 -eq 1) { return [double]$sorted[[int][math]::Floor($sorted.Count / 2)] }
    return ([double]$sorted[[int]($sorted.Count / 2 - 1)] + [double]$sorted[[int]($sorted.Count / 2)]) / 2.0
}

$runError = ''
try {
    $warmups = @(
        [pscustomobject]@{ Clip='ducks'; Arms=@('baseline','candidate') },
        [pscustomobject]@{ Clip='tree'; Arms=@('candidate','baseline') },
        [pscustomobject]@{ Clip='town'; Arms=@('baseline','candidate') }
    )
    foreach ($warmup in $warmups) {
        $clip = $clips | Where-Object { $_.Name -eq $warmup.Clip }
        foreach ($arm in $warmup.Arms) {
            $encoder = $encoders | Where-Object { $_.Arm -eq $arm }
            Invoke-Encode $clip $encoder 'warmup' 0 ($warmup.Arms -join '/')
        }
    }
    for ($pair = 1; $pair -le 15; $pair++) {
        if ($pair % 2 -eq 1) { $clipOrder = @('ducks','tree','town'); $armOrder = @('baseline','candidate') }
        else { $clipOrder = @('town','tree','ducks'); $armOrder = @('candidate','baseline') }
        foreach ($clipName in $clipOrder) {
            $clip = $clips | Where-Object { $_.Name -eq $clipName }
            foreach ($arm in $armOrder) {
                $encoder = $encoders | Where-Object { $_.Arm -eq $arm }
                Invoke-Encode $clip $encoder 'measured' $pair ($armOrder -join '/')
            }
        }
    }
} catch {
    $runError = $_.Exception.Message
}
$overall.Stop()

$allRows = @(Import-Csv -LiteralPath $rawPath)
$measured = @($allRows | Where-Object { $_.Phase -eq 'measured' })
$allCpuRatios = @()
$allWallRatios = @()
$summaries = @()
foreach ($clip in $clips) {
    $cpuRatios = @()
    $wallRatios = @()
    for ($pair = 1; $pair -le 15; $pair++) {
        $baseline = @($measured | Where-Object { $_.Clip -eq $clip.Name -and [int]$_.PairIndex -eq $pair -and $_.Arm -eq 'baseline' }) | Select-Object -First 1
        $candidate = @($measured | Where-Object { $_.Clip -eq $clip.Name -and [int]$_.PairIndex -eq $pair -and $_.Arm -eq 'candidate' }) | Select-Object -First 1
        if ($baseline -and $candidate -and [double]$baseline.ProcessCpuSeconds -gt 0 -and [double]$candidate.ProcessCpuSeconds -gt 0) {
            $cpuRatios += [double]$candidate.ProcessCpuSeconds / [double]$baseline.ProcessCpuSeconds
            if ([double]$baseline.WallSeconds -gt 0 -and [double]$candidate.WallSeconds -gt 0) { $wallRatios += [double]$candidate.WallSeconds / [double]$baseline.WallSeconds }
        }
    }
    $allCpuRatios += $cpuRatios
    $allWallRatios += $wallRatios
    if ($cpuRatios.Count -gt 0) {
        $cpuLog = 0.0; foreach ($ratio in $cpuRatios) { $cpuLog += [math]::Log($ratio) }
        $wallLog = 0.0; foreach ($ratio in $wallRatios) { $wallLog += [math]::Log($ratio) }
        $median = Get-Median ([double[]]$cpuRatios)
        $deviations = @(); foreach ($ratio in $cpuRatios) { $deviations += [math]::Abs($ratio - $median) }
        $mad = Get-Median ([double[]]$deviations)
        $sorted = @($cpuRatios | Sort-Object)
        $outlierCount = 0
        if ($sorted.Count -ge 4) {
            $q1 = [double]$sorted[[int][math]::Floor(($sorted.Count - 1) * 0.25)]
            $q3 = [double]$sorted[[int][math]::Floor(($sorted.Count - 1) * 0.75)]
            $iqr = $q3 - $q1
            $outlierCount = @($cpuRatios | Where-Object { $_ -lt ($q1 - 1.5 * $iqr) -or $_ -gt ($q3 + 1.5 * $iqr) }).Count
        }
        $summaries += [pscustomobject]@{ Clip=$clip.Name; PlannedPairs=15; ValidPairs=$cpuRatios.Count; CpuGeoMeanCandidateOverBaseline=[math]::Round([math]::Exp($cpuLog / $cpuRatios.Count),6); CpuMedian=[math]::Round($median,6); CpuMin=[math]::Round(($cpuRatios | Measure-Object -Minimum).Minimum,6); CpuMax=[math]::Round(($cpuRatios | Measure-Object -Maximum).Maximum,6); CpuMad=[math]::Round($mad,6); TukeyOutlierCountDescriptive=$outlierCount; WallGeoMeanCandidateOverBaseline=$(if($wallRatios.Count -gt 0){[math]::Round([math]::Exp($wallLog / $wallRatios.Count),6)}else{''}); AllValidPairsIncluded=$true; AllPairsComplete=($cpuRatios.Count -eq 15) }
    }
}
if ($allCpuRatios.Count -gt 0) {
    $cpuLog = 0.0; foreach ($ratio in $allCpuRatios) { $cpuLog += [math]::Log($ratio) }
    $wallLog = 0.0; foreach ($ratio in $allWallRatios) { $wallLog += [math]::Log($ratio) }
    $summaries += [pscustomobject]@{ Clip='all_clips_equal_pair'; PlannedPairs=45; ValidPairs=$allCpuRatios.Count; CpuGeoMeanCandidateOverBaseline=[math]::Round([math]::Exp($cpuLog / $allCpuRatios.Count),6); CpuMedian=[math]::Round((Get-Median ([double[]]$allCpuRatios)),6); CpuMin=[math]::Round(($allCpuRatios | Measure-Object -Minimum).Minimum,6); CpuMax=[math]::Round(($allCpuRatios | Measure-Object -Maximum).Maximum,6); CpuMad=''; TukeyOutlierCountDescriptive=''; WallGeoMeanCandidateOverBaseline=$(if($allWallRatios.Count -gt 0){[math]::Round([math]::Exp($wallLog / $allWallRatios.Count),6)}else{''}); AllValidPairsIncluded=$true; AllPairsComplete=($allCpuRatios.Count -eq 45) }
}
if ($summaries.Count -gt 0) { $summaries | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8 }

$stability = @()
foreach ($clip in $clips) {
    foreach ($encoder in $encoders) {
        $rows = @($allRows | Where-Object { $_.Clip -eq $clip.Name -and $_.Arm -eq $encoder.Arm })
        $hashes = @($rows | ForEach-Object { $_.OutputSha256 } | Where-Object { $_ } | Sort-Object -Unique)
        $bad = @($rows | Where-Object { $_.OutputMatchesExpected -ne 'True' }).Count
        $stability += [pscustomobject]@{ Clip=$clip.Name; Arm=$encoder.Arm; RunCount=$rows.Count; DistinctOutputHashCount=$hashes.Count; Hashes=($hashes -join ';'); ExpectedSha256=$clip.OutputSha; AllMatchExpected=($rows.Count -gt 0 -and $bad -eq 0); AllRowsRetained=$true }
    }
}
$stability | Export-Csv -LiteralPath $stabilityPath -NoTypeInformation -Encoding UTF8
$safeRunError = $runError -replace '[\r\n]+', ' '
Add-Content -LiteralPath $manifestPath -Value "completed_utc=$([DateTime]::UtcNow.ToString('o'))"
Add-Content -LiteralPath $manifestPath -Value "total_elapsed_seconds=$([math]::Round($overall.Elapsed.TotalSeconds,3))"
Add-Content -LiteralPath $manifestPath -Value "raw_rows=$($allRows.Count)"
Add-Content -LiteralPath $manifestPath -Value "measured_rows=$($measured.Count)"
Add-Content -LiteralPath $manifestPath -Value "warmup_rows=$(@($allRows | Where-Object { $_.Phase -eq 'warmup' }).Count)"
Add-Content -LiteralPath $manifestPath -Value "run_error=$safeRunError"
Write-Output "RUNNER_SHA256=$runnerHash"
Write-Output "RAW_PATH=$rawPath"
Write-Output "SUMMARY_PATH=$summaryPath"
Write-Output "LOAD_PATH=$loadPath"
Write-Output "IDENTITY_PATH=$identityPath"
Write-Output "STABILITY_PATH=$stabilityPath"
if (-not [string]::IsNullOrWhiteSpace($runError)) { throw "A/B run did not complete: $runError. Partial rows and diagnostics are retained in $root" }
Write-Output "COMPLETE root=$root total_seconds=$([math]::Round($overall.Elapsed.TotalSeconds,1)) measured_rows=$($measured.Count)"
