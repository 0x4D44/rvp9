$ErrorActionPreference='Stop'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Date -Format o
if((git -C $w rev-parse HEAD) -ne '474dfd99a673df85f750136f7552ac156b7a24e9') { throw 'source changed' }
if((git -C $w status --porcelain=v1)) { throw 'source dirty' }
$dirs=@(Get-ChildItem "$r/pgo/runs" -Directory)
if($dirs.Count -ne 1) { throw 'ambiguous PGO run' }
$run=$dirs[0].FullName
"head=$(git -C $w rev-parse HEAD)"
'git_status=clean'
"run_id=$($dirs[0].Name)"
$manifest=Get-Content "$run/manifest.txt"
if($manifest -notcontains 'frames_per_row=60' -or $manifest -notcontains 'training_rows=28') { throw 'wrong training recipe' }
$raw=@(Get-ChildItem "$run/profiles" -Filter '*.profraw')
"observed_raw_profile_count=$($raw.Count)"
if($raw.Count -lt 28) { throw 'too few raw profiles' }
$bin="$r/pgo/optimized/release/rvp9enc.exe"
& $bin --help
"optimized_binary_help_exit=$LASTEXITCODE"
if($LASTEXITCODE) { throw 'PGO binary help failed' }
$files=@("$run/manifest.txt",$bin,"$run/encoder.profdata","$w/scripts/build-encoder-pgo.sh","$r/logs/pgo-build.log","$r/logs/pgo-input-identity.log","$r/logs/pgo-plan.log")
$files+=@(Get-ChildItem "$r/pgo/inputs" -File | ForEach-Object {$_.FullName})
$files+=@($raw | ForEach-Object {$_.FullName})
foreach($f in $files) {
 $i=Get-Item $f
 "$($i.FullName)|bytes=$($i.Length)|sha256=$((Get-FileHash $f -Algorithm SHA256).Hash.ToLower())"
}
foreach($i in @(Get-ChildItem "$r/logs" -File)) {
 "$($i.Name)|bytes=$($i.Length)|sha256=$((Get-FileHash $i.FullName -Algorithm SHA256).Hash.ToLower())"
}
& 'C:/Users/marti/.rustup/toolchains/1.95.0-x86_64-pc-windows-msvc/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe' show "$run/encoder.profdata"
Get-Item "$w/target" | Select-Object FullName,LinkType,Target | Format-List
deltic quiet status --json
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
