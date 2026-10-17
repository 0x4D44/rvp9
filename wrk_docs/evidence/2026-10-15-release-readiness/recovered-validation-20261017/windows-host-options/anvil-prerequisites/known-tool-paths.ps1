$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc=$([DateTime]::UtcNow.ToString('o'))"
$patterns=@('C:/Program Files/Git/bin/bash.exe','C:/msys64/usr/bin/bash.exe','C:/msys64/mingw64/bin/gcc.exe','C:/msys64/ucrt64/bin/gcc.exe','C:/Strawberry/c/bin/gcc.exe','C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe','C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/link.exe','C:/Users/marti/.rustup/toolchains/*/lib/rustlib/x86_64-pc-windows-msvc/bin/rust-lld.exe','C:/Users/marti/.rustup/toolchains/*/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe','C:/Program Files/PowerShell/7/pwsh.exe')
foreach($glob in $patterns) {
 $files=@(Get-Item -Path $glob -ErrorAction SilentlyContinue)
 "known_tool_glob=$glob matches=$($files.Count)"
 foreach($f in $files) { "installed_tool=$($f.FullName) bytes=$($f.Length) fileversion=$($f.VersionInfo.FileVersion)" }
}
if(Test-Path 'C:/Program Files/Git/bin/bash.exe') { & 'C:/Program Files/Git/bin/bash.exe' --version | Select-Object -First 1 }
if(Test-Path 'C:/Strawberry/c/bin/gcc.exe') { & 'C:/Strawberry/c/bin/gcc.exe' --version | Select-Object -First 1 }
if(Test-Path 'C:/Program Files/PowerShell/7/pwsh.exe') { & 'C:/Program Files/PowerShell/7/pwsh.exe' -NoProfile -Command '[ordered]@{runtime=[Runtime.InteropServices.RuntimeInformation]::FrameworkDescription;Sse2=[Runtime.Intrinsics.X86.Sse2]::IsSupported;Ssse3=[Runtime.Intrinsics.X86.Ssse3]::IsSupported;Sse41=[Runtime.Intrinsics.X86.Sse41]::IsSupported;Avx2=[Runtime.Intrinsics.X86.Avx2]::IsSupported;Bmi2=[Runtime.Intrinsics.X86.Bmi2]::IsSupported;Fma=[Runtime.Intrinsics.X86.Fma]::IsSupported;Avx512F=[Runtime.Intrinsics.X86.Avx512F]::IsSupported}|ConvertTo-Json' }
