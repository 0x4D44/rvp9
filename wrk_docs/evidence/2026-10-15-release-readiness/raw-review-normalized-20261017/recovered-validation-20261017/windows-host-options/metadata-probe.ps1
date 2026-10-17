$ErrorActionPreference='Stop'
$paths=@('D:\language\rvp9-2','D:\language\rvp9','C:\language\rvp9')
$result=[ordered]@{host=$env:COMPUTERNAME;utc=[DateTime]::UtcNow.ToString('o');repos=@();toolchains=@();ffmpeg=@();sample=$null}
$result.toolchains=@(& rustup toolchain list 2>&1 | ForEach-Object {"$_"})
$result.ffmpeg=@(& where.exe ffmpeg 2>$null)
foreach($path in $paths){
 if(Test-Path -LiteralPath (Join-Path $path 'Cargo.toml')){
  $files=@('build\libvpx\vpxenc.exe','build\libvpx\vpxdec.exe','tests\corpus\in_to_tree_420_720p50.y4m','tests\corpus\ducks_take_off_420_720p50.y4m','tests\corpus\old_town_cross_420_720p50.y4m','tests\corpus\foreman_cif.y4m','target\release\rvp9enc.exe','target\release\encode-headtohead.exe')
  $checks=@();foreach($rel in $files){$full=Join-Path $path $rel;$item=Get-Item -LiteralPath $full -ErrorAction SilentlyContinue;$checks+=@{file=$rel;present=($null -ne $item);bytes=if($null -ne $item){$item.Length}else{$null}}}
  $head=(& git -C $path rev-parse HEAD 2>&1 | Out-String).Trim()
  $changes=@(& git -C $path status --porcelain 2>&1).Count
  $result.repos+=@{path=$path;head=$head;status_line_count=$changes;files=$checks}
 }
}
$cpu=Get-CimInstance Win32_ComputerSystem
$sample=Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'"
$result.sample=@{logical_processors=$cpu.NumberOfLogicalProcessors;percent_processor_time=$sample.PercentProcessorTime;note='One instantaneous metadata sample only; not canonical60s idle qualification.'}
$result | ConvertTo-Json -Depth 8
