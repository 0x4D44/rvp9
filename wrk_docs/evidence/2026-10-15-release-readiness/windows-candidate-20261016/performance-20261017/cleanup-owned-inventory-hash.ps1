$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
Get-Date -Format o
$expected='4e2a3d7095348fce9c90a2fc8c56b04e071d59112d1e37936c2b17a9699496e1'
$owned=Get-CimInstance Win32_Process -Filter 'ProcessId=29232' -OperationTimeoutSec 5
if($owned) {
 if($owned.CommandLine -notmatch '(?i)-EncodedCommand\s+"?([A-Za-z0-9+/=]+)') { throw 'unknown command format' }
 $actual=[BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::ASCII.GetBytes($matches[1]))).Replace('-','').ToLower()
 $matched=$actual -eq $expected
 "inventory_pid=$($owned.ProcessId) parent=$($owned.ParentProcessId) creation=$($owned.CreationDate.ToUniversalTime().ToString('o')) owned_command_hash=$actual owned_command_match=$matched"
 if(!$matched) { throw 'Refusing cleanup: command does not match owned inventory' }
 Stop-Process -Id 29232
 'owned_timed_out_inventory_stopped=True'
}
Get-Process -Id 29232,34176 -ErrorAction SilentlyContinue | Select-Object Id,ProcessName,StartTime | Format-List
'no_output_above_means_both_owned_process_ids_absent'
