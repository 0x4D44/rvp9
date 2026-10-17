$ErrorActionPreference='Stop'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-asanapr12-474-b98479dc'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/preflight.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/runner.pid"
"native_asan_root_pid=$($p.Id)"
if(!$p.WaitForExit(180000)) { taskkill /PID $p.Id /T /F; throw 'native ASan preflight bound reached' }
$p.Refresh(); "native_asan_launcher_completed=$($p.HasExited)"
