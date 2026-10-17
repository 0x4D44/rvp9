$ProgressPreference='SilentlyContinue'
Get-Date -Format o
Get-Process -Id 34176 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime | Format-List
Get-Process | Where-Object { $_.ProcessName -match 'cargo|rustc|rvp9|conformance|bash|headtohead|powershell|deltic' } | Select-Object Id,ProcessName,StartTime | Format-Table
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Content "$r/performance-logs/campaign.exit.txt" -ErrorAction SilentlyContinue
Get-Content "$r/performance-logs/campaign.log" -Tail 4
Get-Content "$r/performance-logs/quiet-end.log" -ErrorAction SilentlyContinue
