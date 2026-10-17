$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc=$([DateTime]::UtcNow.ToString('o'))"
'filtered_rvp9_deltic_roster_begin'
deltic roster --repo rvp9 --json
'filtered_rvp9_deltic_roster_end'
foreach($name in @('gcc','make','yasm','nasm','perl','clang','cmake')) {
 $cmd=Get-Command $name -ErrorAction SilentlyContinue
 if($cmd) { "path_tool=$name path=$($cmd.Source)" } else { "path_tool=$name absent=True" }
}
& 'C:/FPC/3.2.2/bin/i386-Win32/gcc.exe' --version | Select-Object -First 1
& 'C:/FPC/3.2.2/bin/i386-Win32/gcc.exe' -dumpmachine
& 'C:/FPC/3.2.2/bin/i386-Win32/make.exe' --version | Select-Object -First 1
foreach($glob in @('C:/Program Files/Git/usr/bin/make.exe','C:/Program Files/Git/usr/bin/perl.exe','C:/msys64/usr/bin/make.exe','C:/msys64/usr/bin/perl.exe','C:/msys64/mingw64/bin/yasm.exe','C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/x64/bin/clang.exe','C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/bin/clang.exe')) {
 $files=@(Get-Item $glob -ErrorAction SilentlyContinue)
 "known_tool_glob=$glob matches=$($files.Count)"
 foreach($f in $files) { "installed_tool=$($f.FullName) bytes=$($f.Length) fileversion=$($f.VersionInfo.FileVersion)" }
}
