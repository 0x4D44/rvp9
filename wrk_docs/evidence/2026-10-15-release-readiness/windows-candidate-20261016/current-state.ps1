$ErrorActionPreference='Continue'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$e='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-speed474-cb58beec'
Get-Date -Format o
hostname
"actual_head=$(git -C $w rev-parse HEAD)"
"actual_status=$(git -C $w status --porcelain=v1)"
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead|deltic' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
deltic quiet status --json
foreach($p in @("$w/.scratch","$w/target","$w/.scratch/windows-refresh","$w/.scratch/windows-refresh-pgo","$w/.scratch/windows-performance",$e)) {
 "path=$p exists=$(Test-Path $p)"
 if(Test-Path $p) { Get-Item $p | Select-Object FullName,LastWriteTime | Format-List; Get-ChildItem $p -Force | Select-Object Name,LastWriteTime,Length | Format-Table }
}
'build reference files:'
Get-ChildItem "$w/build/libvpx" -ErrorAction Continue | Select-Object Name,Length | Format-Table
'External retained evidence inventory:'
Get-ChildItem $e -Recurse -File | Select-Object FullName,Length | Format-List
