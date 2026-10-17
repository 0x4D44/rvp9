$ErrorActionPreference='Continue'
$ProgressPreference='SilentlyContinue'
Get-Date -Format o
$patterns=@(
'C:/Users/marti/.rustup/toolchains/*/bin/clang_rt.asan_dynamic-x86_64.dll',
'C:/Users/marti/.rustup/toolchains/*/lib/rustlib/x86_64-pc-windows-msvc/bin/clang_rt.asan_dynamic-x86_64.dll',
'C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll',
'C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/x64/bin/clang_rt.asan_dynamic-x86_64.dll',
'C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/x64/lib/clang/*/lib/windows/clang_rt.asan_dynamic-x86_64.dll',
'C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/Llvm/lib/clang/*/lib/windows/clang_rt.asan_dynamic-x86_64.dll',
'C:/Strawberry/c/bin/clang_rt.asan_dynamic-x86_64.dll')
$found=@()
foreach($pattern in $patterns) { "searched_glob=$pattern"; $found+=@(Get-ChildItem $pattern -File -ErrorAction SilentlyContinue) }
$found=@($found | Sort-Object FullName -Unique)
"candidate_count=$($found.Count)"
if($found.Count -gt 32) { throw 'Refusing unexpectedly broad candidate enumeration' }
foreach($f in $found) {
 "RUNTIME_CANDIDATE_BEGIN=$($f.FullName)"
 "bytes=$($f.Length) sha256=$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())"
 & 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/dumpbin.exe' /EXPORTS $f.FullName
 "RUNTIME_CANDIDATE_END=$($f.FullName)"
}
