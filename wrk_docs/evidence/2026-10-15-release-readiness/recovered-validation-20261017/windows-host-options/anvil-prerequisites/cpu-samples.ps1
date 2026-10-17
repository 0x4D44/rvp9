$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc_begin=$([DateTime]::UtcNow.ToString('o'))"
'CPU sample scope:10 one-second Get-Counter samples only; NOT canonical60s guard or a timing qualification.'
Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 10 | ForEach-Object { $_.CounterSamples | Select-Object @{n='utc';e={$_.Timestamp.ToUniversalTime().ToString('o')}},CookedValue,Status } | ConvertTo-Json
"utc_end=$([DateTime]::UtcNow.ToString('o'))"
