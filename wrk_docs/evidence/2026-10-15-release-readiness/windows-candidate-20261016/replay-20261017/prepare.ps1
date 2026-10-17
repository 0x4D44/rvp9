$ErrorActionPreference='Stop'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
Get-Date -Format o
if((git -C $w rev-parse HEAD) -ne '474dfd99a673df85f750136f7552ac156b7a24e9') { throw 'wrong candidate' }
if((git -C $w status --porcelain=v1)) { throw 'dirty candidate' }
if(Test-Path "$w/target") { throw 'target exists; refusing replacement' }
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
deltic quiet status --json
$p=Start-Process 'C:/Users/marti/AppData/Local/Programs/Deltic/deltic.exe' -ArgumentList @('scratch','new','--label','rvp9-native-replay474','--json') -WorkingDirectory $w -NoNewWindow -PassThru -RedirectStandardOutput "$w/replay-scratch-registration.json" -RedirectStandardError "$w/replay-scratch-registration.stderr.txt"
if(!$p.WaitForExit(30000)) { throw 'scratch registration timed out' }
Get-Content "$w/replay-scratch-registration.json"
