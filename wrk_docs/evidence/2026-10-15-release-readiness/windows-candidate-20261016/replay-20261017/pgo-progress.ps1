Get-Date -Format o
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|bash' } | Select-Object ProcessId,ParentProcessId,Name | Format-Table
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Content "$r/logs/pgo-build.log" -Tail 7
