$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc=$([DateTime]::UtcNow.ToString('o'))"
'Adaptive targeted check of observed yasm/perl portable-Strawberry root; no filesystem search.'
$r='C:/Users/marti/strawberry-portable'
foreach($rel in @('c/bin/gcc.exe','c/bin/gmake.exe','c/bin/make.exe','c/bin/ar.exe','c/bin/ranlib.exe','c/bin/yasm.exe','c/bin/nasm.exe','perl/bin/perl.exe')) {
 $f=Get-Item "$r/$rel" -ErrorAction SilentlyContinue
 "portable_tool=$rel exists=$($null -ne $f) bytes=$($f.Length)"
}
if(Test-Path "$r/c/bin/gcc.exe") { & "$r/c/bin/gcc.exe" --version | Select-Object -First 1; & "$r/c/bin/gcc.exe" -dumpmachine }
if(Test-Path "$r/c/bin/gmake.exe") { & "$r/c/bin/gmake.exe" --version | Select-Object -First 1 }
if(Test-Path "$r/c/bin/make.exe") { & "$r/c/bin/make.exe" --version | Select-Object -First 1 }
& "$r/c/bin/yasm.exe" --version | Select-Object -First 1
& "$r/c/bin/nasm.exe" --version | Select-Object -First 1
& "$r/perl/bin/perl.exe" -e 'print "$^V\n"'
