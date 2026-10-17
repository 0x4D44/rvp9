$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
"utc=$([DateTime]::UtcNow.ToString('o'))"
$expected='ae4ff31cd057d49db9e0147fb3f8970f7086e23b5ec0ae06076de2eb060b67df'
$items=@(Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -OperationTimeoutSec 5)
$found=0
foreach($x in $items) {
 if($x.CommandLine -match '(?i)-EncodedCommand\s+"?([A-Za-z0-9+/=]+)') {
  $actual=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::ASCII.GetBytes($matches[1]))).Replace('-','').ToLower()
  if($actual -eq $expected) {
   "owned_inventory_pid=$($x.ProcessId) parent=$($x.ParentProcessId) creation=$($x.CreationDate.ToUniversalTime().ToString('o')) encoded_command_sha256=$actual"
   Stop-Process -Id $x.ProcessId
   "owned_inventory_stopped=True"
   $found++
  }
 }
}
"owned_inventory_matches=$found"
"root37136_present=$($null -ne (Get-Process -Id 37136 -ErrorAction SilentlyContinue))"
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Content "$r/performance-resume-logs/quiet-end.log"
