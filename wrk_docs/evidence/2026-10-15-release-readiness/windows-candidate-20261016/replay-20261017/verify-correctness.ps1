$ErrorActionPreference='Stop'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Date -Format o
if((git -C $w rev-parse HEAD) -ne '474dfd99a673df85f750136f7552ac156b7a24e9') { throw 'source changed' }
if((git -C $w status --porcelain=v1)) { throw 'source dirty' }
"head=$(git -C $w rev-parse HEAD)"
'git_status=clean'
$bad=@(Get-ChildItem "$r/logs" -Filter '*.exit.txt' | Where-Object { (Get-Content $_.FullName) -ne '0' })
if($bad.Count) { throw 'nonzero phase exit' }
$alpha=@(Get-Content "$r/alpha32/qualification.txt" | Where-Object { $_ -match '\|PASS\|' })
"alpha_pass_rows=$($alpha.Count)"
if($alpha.Count -ne 32) { throw 'expected alpha32' }
$reverse=@(Get-Content "$r/reverse24/qualification.txt" | Where-Object { $_ -match '\|PASS\|' })
"reverse_pass_rows=$($reverse.Count)"
if($reverse.Count -ne 24) { throw 'expected reverse24' }
$files=@(Get-ChildItem "$r/reverse24" -Filter '*.i420')
"reverse_output_count=$($files.Count)"
if($files.Count -ne 48) { throw 'expected48 reverse outputs' }
foreach($row in $reverse) {
 $label=$row.Split('|')[0]
 $a=Get-Item "$r/reverse24/$label.rvp9.i420"
 $b=Get-Item "$r/reverse24/$label.vpx.i420"
 $ah=(Get-FileHash $a.FullName -Algorithm SHA256).Hash.ToLower()
 $bh=(Get-FileHash $b.FullName -Algorithm SHA256).Hash.ToLower()
 $ok=($a.Length -eq 147456 -and $b.Length -eq 147456 -and $ah -eq $bh)
 "$label|rvp9_bytes=$($a.Length)|vpx_bytes=$($b.Length)|rvp9_sha256=$ah|vpx_sha256=$bh|pass=$ok"
 if(!$ok) { throw 'independent reverse verification failed' }
}
'independent_reverse_overall=PASS'
foreach($f in @('conformance','rvp9enc','rvp9dec','bdrate','bdrate-svc','encode-headtohead','realtime')) {
 $file=Get-Item "$r/cargo-target/release/$f.exe"
 "$f.exe|bytes=$($file.Length)|sha256=$((Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLower())"
}
foreach($file in @(Get-ChildItem "$r/logs" -File)) {
 "$($file.Name)|bytes=$($file.Length)|sha256=$((Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLower())"
}
foreach($file in @("$r/alpha32/qualification.txt","$r/reverse24/qualification.txt")) {
 "$file|sha256=$((Get-FileHash $file -Algorithm SHA256).Hash.ToLower())"
}
Get-Item "$w/target" | Select-Object FullName,LinkType,Target | Format-List
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
