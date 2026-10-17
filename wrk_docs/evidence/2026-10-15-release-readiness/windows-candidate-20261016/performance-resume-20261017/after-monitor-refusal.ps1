$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
"utc=$([DateTime]::UtcNow.ToString('o'))"
"root37136_present=$($null -ne (Get-Process -Id 37136 -ErrorAction SilentlyContinue))"
Get-Content "$r/performance-resume-logs/campaign.log" -Tail 8
if(Test-Path "$r/performance-resume-logs/campaign.exit.txt") { Get-Content "$r/performance-resume-logs/campaign.exit.txt" }
Get-Content "$r/production-speed/manifest.txt" -Tail 7
[Diagnostics.Process]::GetProcesses() | Where-Object { $_.ProcessName -match '^(cargo|rustc|rvp9enc|rvp9dec|encode-headtohead|soak)$' } | Select-Object Id,ProcessName
