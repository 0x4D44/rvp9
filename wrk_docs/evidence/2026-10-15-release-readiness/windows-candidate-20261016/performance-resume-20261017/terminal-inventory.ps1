$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
"utc=$([DateTime]::UtcNow.ToString('o'))"
foreach($x in @(37136,43936)) { "pid=$x present=$($null -ne (Get-Process -Id $x -ErrorAction SilentlyContinue))" }
"speed_path_exists=$(Test-Path "$r/production-speed/speed")"
"encoder_count=$([Diagnostics.Process]::GetProcessesByName('rvp9enc').Count)"
"headtohead_count=$([Diagnostics.Process]::GetProcessesByName('encode-headtohead').Count)"
deltic quiet status --json
foreach($name in @('campaign.log','campaign.exit.txt','campaign.command.txt','campaign.sh','quiet-start.log','quiet-start.exit.txt','quiet-grant.json','quiet-end.log','started.txt','ended.txt','final-source.txt','launcher.pid')) {
 $f=Get-Item "$r/performance-resume-logs/$name"; $h=Get-FileHash $f.FullName -Algorithm SHA256
 "performance-resume-logs/$name|bytes=$($f.Length)|sha256=$($h.Hash.ToLower())"
}
$f=Get-Item "$r/production-speed/manifest.txt"; $h=Get-FileHash $f.FullName -Algorithm SHA256
"production-speed/manifest.txt|bytes=$($f.Length)|sha256=$($h.Hash.ToLower())"
