$ErrorActionPreference='Stop'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b'
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/package.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/runner.pid"
"native_package_root_pid=$($p.Id)"
if(!$p.WaitForExit(1860000)) { taskkill /PID $p.Id /T /F; throw 'package native outer bound reached' }
$p.Refresh(); "native_package_launcher_completed=$($p.HasExited)"
