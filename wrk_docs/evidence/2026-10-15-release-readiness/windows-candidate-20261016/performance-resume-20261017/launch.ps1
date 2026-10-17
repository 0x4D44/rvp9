$ErrorActionPreference='Stop'
$env:PATH='C:/Program Files/Git/bin;C:/Users/marti/.cargo/bin;'+$env:PATH
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/performance-resume-logs/campaign.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/performance-resume-logs/launcher.pid"
"native_root_pid=$($p.Id)"
if(!$p.WaitForExit(88000000)) { taskkill /PID $p.Id /T /F; deltic quiet end --owner rvp9-resume-speed474-20261017; throw 'bounded campaign launcher timeout' }
$p.Refresh(); "launcher_completed=$($p.HasExited)"
