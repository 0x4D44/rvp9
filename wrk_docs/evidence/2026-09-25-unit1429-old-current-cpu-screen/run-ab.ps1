param([switch]$SyntaxOnly)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSCommandPath
$runnerHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$oldExe = 'D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit\.scratch\profile\rvp9enc.exe'
$oldPdb = 'D:\worktrees\rvp9-2\20260924-TSK-HUM-fresh-x86-motion-search-attribution-unit\.scratch\profile\rvp9enc.pdb'
$newExe = 'D:\worktrees\rvp9-2\20260925-TSK-CDX@CRUCIBLE-unit1421-cpu-profile\.scratch\target\release\rvp9enc.exe'
$newPdb = 'D:\worktrees\rvp9-2\20260925-TSK-CDX@CRUCIBLE-unit1421-cpu-profile\.scratch\target\release\rvp9enc.pdb'
$clips = @(
    [pscustomobject]@{ Name='ducks'; Path='D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'; Sha='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; OutSha='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' },
    [pscustomobject]@{ Name='tree'; Path='D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\in_to_tree_420_1080p30_60f.y4m'; Sha='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; OutSha='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' },
    [pscustomobject]@{ Name='town'; Path='D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\old_town_cross_420_1080p30_60f.y4m'; Sha='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; OutSha='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f' }
)
$encoders = @(
    [pscustomobject]@{ Arm='old'; Revision='5b6822c0'; Exe=$oldExe; ExeSha='8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e'; Pdb=$oldPdb; PdbSha='5dca6f9dca6cbc1ecbf924b090231b5fd86a149f590dec3985c3234d41014f2e' },
    [pscustomobject]@{ Arm='current'; Revision='e588d13f'; Exe=$newExe; ExeSha='c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98'; Pdb=$newPdb; PdbSha='a3995992a6eb0ce343fed14f01830737ce577b708dde8379d6913902c7ae67fc' }
)
$libvpx = 'D:\worktrees\rvp9-2\acceptance-d639c265\build\libvpx\vpxenc.exe'
$assets = @()
foreach ($e in $encoders) {
    $assets += [pscustomobject]@{ Name="$($e.Arm)_exe"; Path=$e.Exe; ExpectedSha=$e.ExeSha }
    $assets += [pscustomobject]@{ Name="$($e.Arm)_pdb"; Path=$e.Pdb; ExpectedSha=$e.PdbSha }
}
foreach ($c in $clips) { $assets += [pscustomobject]@{ Name="input_$($c.Name)"; Path=$c.Path; ExpectedSha=$c.Sha } }
$assets += [pscustomobject]@{ Name='pinned_libvpx_not_run'; Path=$libvpx; ExpectedSha='88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d' }
$identity = @()
foreach ($a in $assets) {
    if (-not (Test-Path -LiteralPath $a.Path -PathType Leaf)) { throw "Missing asset $($a.Path)" }
    $item = Get-Item -LiteralPath $a.Path
    $hash = (Get-FileHash -LiteralPath $a.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $ok = $hash -eq $a.ExpectedSha
    $identity += [pscustomobject]@{ Name=$a.Name; Path=$a.Path; Bytes=$item.Length; ExpectedSha256=$a.ExpectedSha; ActualSha256=$hash; Match=$ok }
    if (-not $ok) { throw "Asset hash mismatch: $($a.Path)" }
}
$identityPath = Join-Path $root 'identity.csv'
$identity | Export-Csv -LiteralPath $identityPath -NoTypeInformation -Encoding UTF8
$rawPath = Join-Path $root 'raw.csv'
$loadPath = Join-Path $root 'load.csv'
$summaryPath = Join-Path $root 'summary.csv'
$stabilityPath = Join-Path $root 'output-stability.csv'
$manifestPath = Join-Path $root 'manifest.txt'
Set-Content -LiteralPath $rawPath -Value '"RunId","Phase","PairIndex","Clip","Arm","Revision","Order","StartUtc","EndUtc","WallSeconds","ProcessCpuSeconds","ExitCode","TimedOut","OutputPath","OutputBytes","OutputSha256","ExpectedOutputSha256","OutputMatchesExpected","StdoutPath","StderrPath","LoadSamplesDuringRun","LoadLateSamples","LoadErrors","LoadMaxGapSeconds","LoadMeanPct","LoadMaxPct","LoadStatus"' -Encoding UTF8
Set-Content -LiteralPath $loadPath -Value '"RunId","SampleUtc","ProcessElapsedSeconds","CounterCallMs","CpuBusyPct","DuringChild","GapSeconds","Status","Error"' -Encoding UTF8
$counterPath = '\Processor(_Total)\% Processor Time'
$counterEnabled = $true
$counterPreflight = ''
try { $counterPreflight = [math]::Round([double](Get-Counter -Counter $counterPath -MaxSamples 1 -ErrorAction Stop).CounterSamples[0].CookedValue, 2) }
catch { $counterEnabled = $false; $counterPreflight = $_.Exception.Message }
$manifest = @(
    'experiment=Unit1429 old 5b6822c0 vs current e588d13f process-CPU A/B',
    "started_utc=$([DateTime]::UtcNow.ToString('o'))", "runner_path=$PSCommandPath", "runner_sha256=$runnerHash", "host=$env:COMPUTERNAME",
    "old_encoder=$oldExe", "current_encoder=$newExe", "inputs=$($clips.Path -join ';')",
    'settings=CPU6 CBR1260 fps30/1 threads1 tile0 AQ0 noise0 lag0 limit60 KF60',
    'warmups=one per arm per clip; measured_pairs=15 per clip; odd old/current, even current/old',
    'clip_order=odd pairs ducks/tree/town; even pairs town/tree/ducks; ratios=current divided by old; all rows retained',
    'process_timeout_seconds=90', 'total_timeout_seconds=1200', "host_counter=$counterPath", "counter_preflight=$counterPreflight", "counter_enabled=$counterEnabled",
    'libvpx hashed for identity only; not executed', "identity_csv=$identityPath", "raw_csv=$rawPath", "load_csv=$loadPath", "summary_csv=$summaryPath", "output_stability_csv=$stabilityPath"
)
Set-Content -LiteralPath $manifestPath -Value $manifest -Encoding UTF8
if ($SyntaxOnly) { Write-Output "SYNTAX_AND_PREFLIGHT_OK runner_sha256=$runnerHash counter=$counterPreflight"; exit 0 }
$overall = [System.Diagnostics.Stopwatch]::StartNew()
$perRunLimit = 90
$totalLimit = 1200
function Add-CsvRow([string]$path, $row) { $row | Export-Csv -LiteralPath $path -Append -NoTypeInformation -Encoding UTF8 }
function Invoke-Encode($clip, $encoder, [string]$phase, [int]$pairIndex, [string]$order) {
    if ($script:overall.Elapsed.TotalSeconds -ge $script:totalLimit) { throw 'Total run limit reached before next encode' }
    $runId = if ($phase -eq 'warmup') { "warm-$($clip.Name)-$($encoder.Arm)" } else { "p$('{0:D2}' -f $pairIndex)-$($clip.Name)-$($encoder.Arm)" }
    $out = Join-Path $root "$runId.ivf"
    $stdout = Join-Path $root "$runId.stdout.txt"
    $stderr = Join-Path $root "$runId.stderr.txt"
    $arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $out,$clip.Path
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $encoder.Exe; $psi.Arguments = $arguments; $psi.WorkingDirectory = $root
    $psi.UseShellExecute = $false; $psi.CreateNoWindow = $true; $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true
    $proc = New-Object System.Diagnostics.Process; $proc.StartInfo = $psi
    $loadValues = New-Object 'System.Collections.Generic.List[double]'
    $loadErrors = 0; $lateSamples = 0; $maxGap = 0.0; $lastSampleElapsed = 0.0; $timedOut = $false; $nextCounter = 0.25
    Write-Output "START $runId"
    if (-not $proc.Start()) { throw "Process did not start: $runId" }
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync(); $stderrTask = $proc.StandardError.ReadToEndAsync()
    $wallWatch = [System.Diagnostics.Stopwatch]::StartNew()
    while (-not $proc.HasExited) {
        if ($wallWatch.Elapsed.TotalSeconds -ge $perRunLimit) { $timedOut = $true; $proc.Kill(); [void]$proc.WaitForExit(10000); break }
        if ($counterEnabled -and $wallWatch.Elapsed.TotalSeconds -ge $nextCounter) {
            $cw = [System.Diagnostics.Stopwatch]::StartNew()
            try {
                $s = Get-Counter -Counter $counterPath -MaxSamples 1 -ErrorAction Stop
                $busy = [math]::Round([double]$s.CounterSamples[0].CookedValue,3); $at = [DateTime]::UtcNow; $elapsed = $wallWatch.Elapsed.TotalSeconds
                $during = -not $proc.HasExited; $gap = $elapsed - $lastSampleElapsed
                if (($lastSampleElapsed -gt 0 -and $gap -gt $maxGap) -or ($lastSampleElapsed -eq 0 -and $elapsed -gt $maxGap)) { $maxGap = $gap }
                if (-not $during) { $lateSamples++ } else { $loadValues.Add($busy) }
                Add-CsvRow $loadPath ([pscustomobject]@{ RunId=$runId; SampleUtc=$at.ToString('o'); ProcessElapsedSeconds=[math]::Round($elapsed,4); CounterCallMs=[math]::Round($cw.Elapsed.TotalMilliseconds,1); CpuBusyPct=$busy; DuringChild=$during; GapSeconds=[math]::Round($gap,4); Status=$(if($during){'ok'}else{'child-exited-during-sample'}); Error='' })
            } catch {
                $loadErrors++; $at=[DateTime]::UtcNow; $elapsed=$wallWatch.Elapsed.TotalSeconds; $gap=$elapsed-$lastSampleElapsed
                if ($lastSampleElapsed -gt 0 -and $gap -gt $maxGap) { $maxGap = $gap }
                Add-CsvRow $loadPath ([pscustomobject]@{ RunId=$runId; SampleUtc=$at.ToString('o'); ProcessElapsedSeconds=[math]::Round($elapsed,4); CounterCallMs=[math]::Round($cw.Elapsed.TotalMilliseconds,1); CpuBusyPct=''; DuringChild=(-not $proc.HasExited); GapSeconds=[math]::Round($gap,4); Status='error'; Error=$_.Exception.Message })
            }
            $lastSampleElapsed = $wallWatch.Elapsed.TotalSeconds; while ($nextCounter -le $lastSampleElapsed) { $nextCounter += 1.0 }
        }
        [void]$proc.WaitForExit(100)
    }
    $proc.WaitForExit(); $proc.Refresh(); $exitCode=$proc.ExitCode
    $cpu=[math]::Round($proc.TotalProcessorTime.TotalSeconds,6)
    $wall=[math]::Round(($proc.ExitTime.ToUniversalTime()-$proc.StartTime.ToUniversalTime()).TotalSeconds,6)
    $start=$proc.StartTime.ToUniversalTime().ToString('o'); $end=$proc.ExitTime.ToUniversalTime().ToString('o')
    [System.IO.File]::WriteAllText($stdout,$stdoutTask.Result,[System.Text.UTF8Encoding]::new($false))
    [System.IO.File]::WriteAllText($stderr,$stderrTask.Result,[System.Text.UTF8Encoding]::new($false))
    $bytes=0; $hash=''; if (Test-Path -LiteralPath $out -PathType Leaf) { $bytes=(Get-Item -LiteralPath $out).Length; $hash=(Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant() }
    $mean=''; $peak=''; if ($loadValues.Count -gt 0) { $mean=[math]::Round(($loadValues|Measure-Object -Average).Average,3); $peak=[math]::Round(($loadValues|Measure-Object -Maximum).Maximum,3) }
    $loadStatus=if(-not $counterEnabled){'unavailable'}elseif($loadErrors -gt 0 -or $maxGap -gt 2.5){'gaps-or-errors'}elseif($loadValues.Count -eq 0){'no-during-run-sample'}else{'sampled'}
    $row=[pscustomobject]@{ RunId=$runId; Phase=$phase; PairIndex=$pairIndex; Clip=$clip.Name; Arm=$encoder.Arm; Revision=$encoder.Revision; Order=$order; StartUtc=$start; EndUtc=$end; WallSeconds=$wall; ProcessCpuSeconds=$cpu; ExitCode=$exitCode; TimedOut=$timedOut; OutputPath=$out; OutputBytes=$bytes; OutputSha256=$hash; ExpectedOutputSha256=$clip.OutSha; OutputMatchesExpected=($hash -eq $clip.OutSha); StdoutPath=$stdout; StderrPath=$stderr; LoadSamplesDuringRun=$loadValues.Count; LoadLateSamples=$lateSamples; LoadErrors=$loadErrors; LoadMaxGapSeconds=[math]::Round($maxGap,4); LoadMeanPct=$mean; LoadMaxPct=$peak; LoadStatus=$loadStatus }
    Add-CsvRow $rawPath $row
    Write-Output ("DONE {0} exit={1} wall={2:F3}s cpu={3:F3}s bytes={4} sha={5} load={6}" -f $runId,$exitCode,$wall,$cpu,$bytes,$hash,$loadStatus)
    $proc.Dispose()
    if ($timedOut -or $exitCode -ne 0) { throw "Encode failed or timed out: $runId" }
}
$warmups = @(
    [pscustomobject]@{ Clip='ducks'; Arms=@('old','current') },
    [pscustomobject]@{ Clip='tree'; Arms=@('current','old') },
    [pscustomobject]@{ Clip='town'; Arms=@('old','current') }
)
foreach ($w in $warmups) { $c=$clips|Where-Object Name -eq $w.Clip; foreach($a in $w.Arms){$e=$encoders|Where-Object Arm -eq $a; Invoke-Encode $c $e 'warmup' 0 ($w.Arms -join '/')} }
for ($pair=1; $pair -le 15; $pair++) {
    if ($pair % 2 -eq 1) { $order=@('ducks','tree','town'); $arms=@('old','current') } else { $order=@('town','tree','ducks'); $arms=@('current','old') }
    foreach ($name in $order) { $c=$clips|Where-Object Name -eq $name; foreach($a in $arms){$e=$encoders|Where-Object Arm -eq $a; Invoke-Encode $c $e 'measured' $pair ($arms -join '/')} }
}
$measured = Import-Csv -LiteralPath $rawPath | Where-Object Phase -eq 'measured'
function Get-Median([double[]]$v) { $s=@($v|Sort-Object); if($s.Count%2 -eq 1){return [double]$s[[int][math]::Floor($s.Count/2)]}; return ([double]$s[[int]($s.Count/2-1)]+[double]$s[[int]($s.Count/2)])/2.0 }
$allCpu=@(); $allWall=@(); $summaries=@()
foreach ($c in $clips) {
    $cr=@(); $wr=@()
    for($pair=1;$pair -le 15;$pair++){
        $a=$measured|Where-Object{$_.Clip -eq $c.Name -and [int]$_.PairIndex -eq $pair -and $_.Arm -eq 'old'}
        $b=$measured|Where-Object{$_.Clip -eq $c.Name -and [int]$_.PairIndex -eq $pair -and $_.Arm -eq 'current'}
        if(-not $a -or -not $b -or [double]$a.ProcessCpuSeconds -le 0 -or [double]$b.ProcessCpuSeconds -le 0){throw "Invalid pair $($c.Name) $pair"}
        $cr += [double]$b.ProcessCpuSeconds/[double]$a.ProcessCpuSeconds; $wr += [double]$b.WallSeconds/[double]$a.WallSeconds
    }
    $allCpu += $cr; $allWall += $wr; $log=0.0; foreach($x in $cr){$log += [math]::Log($x)}; $geo=[math]::Exp($log/$cr.Count)
    $wlog=0.0; foreach($x in $wr){$wlog += [math]::Log($x)}; $wgeo=[math]::Exp($wlog/$wr.Count)
    $med=Get-Median ([double[]]$cr); $dev=@(); foreach($x in $cr){$dev += [math]::Abs($x-$med)}; $mad=Get-Median ([double[]]$dev)
    $s=@($cr|Sort-Object); $iqr=[double]$s[11]-[double]$s[3]; $n=@($cr|Where-Object{$_ -lt ([double]$s[3]-1.5*$iqr) -or $_ -gt ([double]$s[11]+1.5*$iqr)}).Count
    $summaries += [pscustomobject]@{ Clip=$c.Name; PairCount=15; CpuGeoMeanCurrentOverOld=[math]::Round($geo,6); CpuMedian=[math]::Round($med,6); CpuMin=[math]::Round(($cr|Measure-Object -Minimum).Minimum,6); CpuMax=[math]::Round(($cr|Measure-Object -Maximum).Maximum,6); CpuMad=[math]::Round($mad,6); TukeyOutlierCountDescriptive=$n; WallGeoMeanCurrentOverOld=[math]::Round($wgeo,6); AllRowsIncluded=$true }
}
$log=0.0; foreach($x in $allCpu){$log += [math]::Log($x)}; $geo=[math]::Exp($log/$allCpu.Count)
$wlog=0.0; foreach($x in $allWall){$wlog += [math]::Log($x)}; $wgeo=[math]::Exp($wlog/$allWall.Count)
$summaries += [pscustomobject]@{ Clip='corpus_equal_clip'; PairCount=45; CpuGeoMeanCurrentOverOld=[math]::Round($geo,6); CpuMedian=[math]::Round((Get-Median ([double[]]$allCpu)),6); CpuMin=[math]::Round(($allCpu|Measure-Object -Minimum).Minimum,6); CpuMax=[math]::Round(($allCpu|Measure-Object -Maximum).Maximum,6); CpuMad=''; TukeyOutlierCountDescriptive=''; WallGeoMeanCurrentOverOld=[math]::Round($wgeo,6); AllRowsIncluded=$true }
$summaries|Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding UTF8
$all=Import-Csv -LiteralPath $rawPath; $stability=@()
foreach($c in $clips){foreach($e in $encoders){$r=$all|Where-Object{$_.Clip -eq $c.Name -and $_.Arm -eq $e.Arm};$h=@($r|ForEach-Object OutputSha256|Sort-Object -Unique);$bad=@($r|Where-Object{$_.OutputMatchesExpected -ne 'True'}).Count;$stability += [pscustomobject]@{Clip=$c.Name;Arm=$e.Arm;RunCount=$r.Count;DistinctOutputHashCount=$h.Count;Hashes=$h -join ';';ExpectedSha256=$c.OutSha;AllMatchExpected=($bad -eq 0)}}}
$stability|Export-Csv -LiteralPath $stabilityPath -NoTypeInformation -Encoding UTF8
Add-Content -LiteralPath $manifestPath -Value "completed_utc=$([DateTime]::UtcNow.ToString('o'))"
Add-Content -LiteralPath $manifestPath -Value "total_elapsed_seconds=$([math]::Round($overall.Elapsed.TotalSeconds,3))"
Add-Content -LiteralPath $manifestPath -Value "measured_rows=$($measured.Count)"
Add-Content -LiteralPath $manifestPath -Value "warmup_rows=$(@((Import-Csv $rawPath)|Where-Object Phase -eq 'warmup').Count)"
Write-Output "COMPLETE runner_sha256=$runnerHash root=$root total_seconds=$([math]::Round($overall.Elapsed.TotalSeconds,1))"
Write-Output "SUMMARY_PATH=$summaryPath"
Write-Output "RAW_PATH=$rawPath"
Write-Output "LOAD_PATH=$loadPath"
Write-Output "STABILITY_PATH=$stabilityPath"
