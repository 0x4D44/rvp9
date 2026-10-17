$ErrorActionPreference='Stop'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661'
$exe='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6/cargo-target/release/soak.exe'
if([System.Diagnostics.Process]::GetProcessesByName('soak').Count -ne 0) { throw 'pre-existing soak process; refusing ambiguous ownership' }
$p=Start-Process 'C:/Program Files/Git/bin/bash.exe' -ArgumentList @('--noprofile','--norc',"$r/run.sh") -PassThru -NoNewWindow
$p.Id | Set-Content "$r/run-runner.pid"
"native_soak_root_pid=$($p.Id)"
'utc,pid,working_set_bytes,private_bytes,handle_count,cpu_seconds' | Set-Content "$r/native-memory-handles.csv" -Encoding utf8
$clock=[Diagnostics.Stopwatch]::StartNew()
while(!$p.WaitForExit(1000)) {
 foreach($s in [System.Diagnostics.Process]::GetProcessesByName('soak')) {
  if($s.MainModule.FileName.Replace('\','/') -eq $exe) {
   $s.Id | Set-Content "$r/soak.pid"
   "$([DateTime]::UtcNow.ToString('o')),$($s.Id),$($s.WorkingSet64),$($s.PrivateMemorySize64),$($s.HandleCount),$($s.TotalProcessorTime.TotalSeconds)" | Add-Content "$r/native-memory-handles.csv" -Encoding utf8
  }
 }
 if($clock.Elapsed.TotalSeconds -gt 210) { taskkill /PID $p.Id /T /F; throw 'soak outer bound reached' }
}
$p.Refresh(); "native_soak_launcher_completed=$($p.HasExited)"
"native_soak_executable_count_after=$([System.Diagnostics.Process]::GetProcessesByName('soak').Count)"
