$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r=[ordered]@{host=$env:COMPUTERNAME;utc=[DateTime]::UtcNow.ToString('o')}
$r.computer=Get-CimInstance Win32_ComputerSystem -OperationTimeoutSec 5 | Select-Object Manufacturer,Model,NumberOfProcessors,NumberOfLogicalProcessors,TotalPhysicalMemory
$r.cpu=@(Get-CimInstance Win32_Processor -OperationTimeoutSec 5 | Select-Object Name,Manufacturer,Architecture,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,AddressWidth)
$r.os=Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 5 | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory,LastBootUpTime
$r.disks=@(Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' -OperationTimeoutSec 5 | Select-Object DeviceID,FileSystem,Size,FreeSpace)
$r.power_plan=(& powercfg /getactivescheme | Out-String).Trim()
$r | ConvertTo-Json -Depth 6
