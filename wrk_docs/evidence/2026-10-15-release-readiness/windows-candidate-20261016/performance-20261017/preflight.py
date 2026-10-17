import subprocess,base64,pathlib
p=pathlib.Path(__file__).parent
s=r'''$ErrorActionPreference='Stop'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6'
Get-Date -Format o
if((git -C $w rev-parse HEAD) -ne '474dfd99a673df85f750136f7552ac156b7a24e9') { throw 'wrong source' }
if((git -C $w status --porcelain=v1)) { throw 'source dirty' }
if(Test-Path "$r/production-speed") { throw 'speed root exists; no reuse allowed' }
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Format-List
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | Format-List
powercfg /GETACTIVESCHEME
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'cargo|rustc|rvp9|conformance|bash|headtohead' } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | Format-List
deltic quiet status --json
$files=@("$r/pgo/runs/20261017T052417Z-1120/manifest.txt","$r/pgo/optimized/release/rvp9enc.exe","$r/pgo/runs/20261017T052417Z-1120/encoder.profdata")
$files+=@('bdrate','bdrate-svc','encode-headtohead','realtime','rvp9dec' | ForEach-Object { "$r/cargo-target/release/$_.exe" })
$files+=@('vpxenc','vpxdec','vpx_temporal_svc_reference' | ForEach-Object { "$w/build/libvpx/$_.exe" })
foreach($f in $files) { $i=Get-Item $f; "$($i.FullName)|bytes=$($i.Length)|sha256=$((Get-FileHash $f -Algorithm SHA256).Hash.ToLower())" }
New-Item -ItemType Directory "$r/performance-logs" | Out-Null
'''
p.joinpath('preflight.ps1').write_text(s)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','marti@crucible','powershell -NoProfile -EncodedCommand '+base64.b64encode(s.encode('utf-16le')).decode()],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
p.joinpath('preflight.raw.txt').write_bytes(r.stdout);print(r.stdout.decode(errors='replace'));raise SystemExit(r.returncode)
