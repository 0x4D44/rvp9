$ProgressPreference='SilentlyContinue'
Get-Date -Format o
$dll=Get-Item 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll'
$dll.VersionInfo | Select-Object FileVersion,ProductVersion,FileDescription,OriginalFilename | Format-List
Get-Process | Where-Object { $_.ProcessName -match 'cargo|rustc|rvp9|preflight|conformance|bash|headtohead' } | Select-Object Id,ProcessName,StartTime | Format-Table
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
"source_head=$(git -C $w rev-parse HEAD)"
"source_status=$(git -C $w status --porcelain=v1)"
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-asan474-a78b6878'
foreach($name in @('preflight.rs','toolchain.log','compile.command.txt','compile.log','compile.exit.txt','run.log','run.exit.txt','artifacts.txt')) {
 $f=Get-Item "$r/$name"
 "$name|bytes=$($f.Length)|sha256=$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())"
}
