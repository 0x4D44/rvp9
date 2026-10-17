$ErrorActionPreference='Stop'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-soak474-d407b661'
[DateTime]::UtcNow.ToString('o')
foreach($id in @(11832,40100)) { $q=Get-Process -Id $id -ErrorAction SilentlyContinue; "pid=$id present=$($null -ne $q)" }
"soak_executable_count=$([System.Diagnostics.Process]::GetProcessesByName('soak').Count)"
$task='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
"head=$(git -C $task rev-parse HEAD)"
"status=$(git -C $task status --porcelain)"
foreach($name in @('build.log','build-identity.log','build.exit.txt','soak-binary.sha256','run-identity.log','run.command.txt','run.log','run.exit.txt','terminal.log','native-memory-handles.csv','run-runner.pid','soak.pid')) {
 $f=Get-Item "$r/$name"; $h=Get-FileHash $f.FullName -Algorithm SHA256
 "$name|bytes=$($f.Length)|sha256=$($h.Hash.ToLower())"
}
"transient_files=$(@(Get-ChildItem "$r/transient" -Recurse -File).Count)"
