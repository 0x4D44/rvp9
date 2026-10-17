$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
$l="$r/performance-logs"
$phase="$r/production-speed"
$pidValue=$null
if(Test-Path "$l/launcher.pid") { $pidValue=[int](Get-Content "$l/launcher.pid") }
$terminal=$null
if(Test-Path "$l/campaign.exit.txt") { $terminal=[int](Get-Content "$l/campaign.exit.txt") }
$groups=@()
$ledgers=@()
if(Test-Path "$phase/speed") { $ledgers=@(Get-ChildItem "$phase/speed" -Recurse -File -Filter 'speed-summary.csv' -Force) }
foreach($ledger in $ledgers) {
 foreach($row in @(Import-Csv $ledger.FullName)) {
  $g=Join-Path $ledger.Directory.FullName "groups/$($row.group_id)"
  $names=@('headtohead.log','run/raw.csv','run/summary.csv','run/manifest.txt')
  $missing=@($names | Where-Object { !(Test-Path (Join-Path $g $_)) })
  if($missing.Count) { continue }
  $files=@()
  foreach($name in $names) {
   $i=Get-Item (Join-Path $g $name)
   $files+=@{name=$name;path=$i.FullName;bytes=$i.Length;sha256=(Get-FileHash $i.FullName -Algorithm SHA256).Hash.ToLower()}
  }
  $groups+=@{id=$row.group_id;status=$row.status;separate_ratio=$row.corpus_separate_ratio;paired_ratio=$row.corpus_paired_ratio;files=$files}
 }
}
$quiet=deltic quiet status --json | ConvertFrom-Json
$tail=@()
if(Test-Path "$l/campaign.log") { $tail=@(Get-Content "$l/campaign.log" -Tail 5) }
$own=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name)
@{observed_utc=[DateTime]::UtcNow.ToString('o');native_root_pid=$pidValue;terminal_exit=$terminal;quiet=$quiet;groups=$groups;tail=$tail;processes=$own} | ConvertTo-Json -Depth 9 -Compress
