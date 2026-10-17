$ErrorActionPreference='Stop'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/build.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/build-runner.pid"
"native_soak_build_pid=$($p.Id)"
if(!$p.WaitForExit(960000)) { taskkill /PID $p.Id /T /F; throw 'soak build outer bound reached' }
$p.Refresh(); "native_soak_build_launcher_completed=$($p.HasExited)"
