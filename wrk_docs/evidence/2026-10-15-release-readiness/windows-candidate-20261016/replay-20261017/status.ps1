Get-Date -Format o
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-ChildItem "$r/logs" -Filter '*.exit.txt' | ForEach-Object { "$($_.Name)=$(Get-Content $_.FullName)" }
Get-ChildItem "$r/logs" -Filter '*.log' | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.Name; Get-Content $_.FullName -Tail 5 }
