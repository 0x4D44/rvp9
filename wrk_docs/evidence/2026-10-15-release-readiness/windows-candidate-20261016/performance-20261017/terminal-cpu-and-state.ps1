$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-Date -Format o
foreach($idValue in @(29232,34176)) {
 $owned=Get-Process -Id $idValue -ErrorAction SilentlyContinue
 if($owned) { "owned_pid=$idValue still_present=True name=$($owned.ProcessName)" } else { "owned_pid=$idValue present=False" }
}
$baseline=@{}
$skipped=0
foreach($proc in @(Get-Process)) {
 try { $baseline[$proc.Id]=@{name=$proc.ProcessName;start=$proc.StartTime.ToUniversalTime().Ticks;cpu=$proc.TotalProcessorTime.TotalSeconds} } catch { $skipped++ }
}
$watch=[Diagnostics.Stopwatch]::StartNew()
Start-Sleep -Seconds 5
$rows=@();$sum=0.0
foreach($proc in @(Get-Process)) {
 try {
  if(!$baseline.ContainsKey($proc.Id)) { continue }
  $before=$baseline[$proc.Id]
  if($before.start -ne $proc.StartTime.ToUniversalTime().Ticks) { continue }
  $delta=$proc.TotalProcessorTime.TotalSeconds-$before.cpu
  if($delta -gt 0) { $sum+=$delta; $rows+=[pscustomobject]@{pid=$proc.Id;name=$proc.ProcessName;cpu_seconds_delta=$delta;busy_core_equivalent=$delta/$watch.Elapsed.TotalSeconds} }
 } catch { $skipped++ }
}
"current_sample_seconds=$($watch.Elapsed.TotalSeconds) attributable_cpu_seconds=$sum skipped_or_protected=$skipped"
$rows | Sort-Object cpu_seconds_delta -Descending | Select-Object -First 15 | Format-Table
'This current process-delta sample is not attribution for the earlier 60second canonical load window.'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
$d='C:/Users/marti/AppData/Local/Programs/Deltic/deltic.exe'
$q=Start-Process $d -ArgumentList @('quiet','status','--json') -NoNewWindow -PassThru -RedirectStandardOutput "$r/performance-logs/terminal-quiet.json" -RedirectStandardError "$r/performance-logs/terminal-quiet.stderr.txt"
if(!$q.WaitForExit(10000)) { Stop-Process -Id $q.Id; throw 'read-only quiet query timed out' }
Get-Content "$r/performance-logs/terminal-quiet.json"
Get-Process | Where-Object { $_.ProcessName -match 'cargo|rustc|rvp9|conformance|bash|headtohead|powershell|deltic' } | Select-Object Id,ProcessName,StartTime | Format-Table
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
"source_head=$(git -C $w rev-parse HEAD)"
"source_status=$(git -C $w status --porcelain=v1)"
$manifest="$r/production-speed/manifest.txt"
"manifest_sha256=$((Get-FileHash $manifest -Algorithm SHA256).Hash.ToLower())"
"group_summaries=$(@(Get-ChildItem "$r/production-speed" -Recurse -Filter 'summary.csv' -File -Force).Count)"
"raw_csvs=$(@(Get-ChildItem "$r/production-speed" -Recurse -Filter 'raw.csv' -File -Force).Count)"
foreach($name in @('campaign.log','campaign.exit.txt','campaign.command.txt','quiet-start.log','quiet-start.exit.txt','quiet-grant.json','quiet-end.log','started.txt','ended.txt','final-source.txt')) {
 $f=Get-Item "$r/performance-logs/$name"
 "$name|bytes=$($f.Length)|sha256=$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())"
}
