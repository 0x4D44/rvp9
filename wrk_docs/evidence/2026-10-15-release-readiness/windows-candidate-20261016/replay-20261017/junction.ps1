$ErrorActionPreference='Stop'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
if(!(Test-Path $r)) { throw 'registered scratch absent' }
foreach($name in @('replay-scratch-registration.json','replay-scratch-registration.stderr.txt')) { Move-Item "$w/$name" "$r/$name" }
if(Test-Path "$w/target") { throw 'target exists; refusing replacement' }
if(Test-Path "$r/cargo-target") { throw 'external cargo-target exists unexpectedly' }
New-Item -ItemType Directory "$r/cargo-target" | Out-Null
New-Item -ItemType Directory "$r/logs" | Out-Null
New-Item -ItemType Junction -Path "$w/target" -Target "$r/cargo-target" | Select-Object FullName,LinkType,Target | Format-List
if((git -C $w status --porcelain=v1)) { throw 'task no longer clean' }
"actual_head=$(git -C $w rev-parse HEAD)"
"actual_status=$(git -C $w status --porcelain=v1)"
Get-Content "$r/replay-scratch-registration.json"
