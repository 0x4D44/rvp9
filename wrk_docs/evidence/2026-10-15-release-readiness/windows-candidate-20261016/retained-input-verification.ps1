$ErrorActionPreference='Stop'
$e='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-speed474-cb58beec/speed'
$lines=Get-Content "$e/manifest.txt"
Get-Date -Format o
$count=0
foreach($line in $lines) {
 if($line -match '^(input_(?:clean|noisy)_.+)_path=(.+)$') {
  $key=$matches[1]
  $posix=$matches[2]
  $filename=($posix -split '/')[-1]
  $path="$e/inputs/$filename"
  $expected=@($lines | Where-Object { $_.StartsWith("${key}_sha256=") })
  if($expected.Count -ne 1) { throw "ambiguous hash for:$key" }
  $expectedHash=$expected[0].Split('=')[1]
  $i=Get-Item $path
  $actual=(Get-FileHash $path -Algorithm SHA256).Hash.ToLower()
  "$filename|bytes=$($i.Length)|sha256=$actual|manifest_match=$($actual -eq $expectedHash)"
  if($actual -ne $expectedHash) { throw 'retained input hash mismatch' }
  $count++
 }
}
"verified_input_count=$count"
if($count -ne 16) { throw 'expected16 inputs' }
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
foreach($p in @("$w/.scratch/windows-refresh-pgo/build/runs/20261016T142333Z-1801/manifest.txt","$w/.scratch/windows-refresh-pgo/build/runs/20261016T142333Z-1801/encoder.profdata","$w/.scratch/windows-refresh-pgo/build/optimized/release/rvp9enc.exe","$w/target/release/conformance.exe","$w/target/release/encode-headtohead.exe")) { "actual_artifact=$p exists=$(Test-Path $p)" }
"actual_head=$(git -C $w rev-parse HEAD)"
"actual_status=$(git -C $w status --porcelain=v1)"
deltic quiet status --json
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
