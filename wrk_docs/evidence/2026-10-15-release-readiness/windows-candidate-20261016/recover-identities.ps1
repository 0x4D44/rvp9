$ErrorActionPreference='Continue'
$w='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
$e='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-speed474-cb58beec/speed'
Get-Date -Format o
foreach($f in @("$e/manifest.txt","$w/build/libvpx/vpxenc.exe","$w/build/libvpx/vpxdec.exe","$w/build/libvpx/vpx_temporal_svc_reference.exe","$w/build/libvpx/libvpx.a","$w/build/libvpx/vpx_config.h","$w/build/libvpx/config.mk","$w/scripts/build-encoder-pgo.sh","$w/scripts/qualify-production-encoder.sh")) {
 $i=Get-Item $f
 "$($i.FullName)|bytes=$($i.Length)|sha256=$((Get-FileHash $f -Algorithm SHA256).Hash.ToLower())"
}
"prepared_input_count=$(@(Get-ChildItem "$e/inputs" -Filter '*.y4m').Count)"
"group_summary_count=$(@(Get-ChildItem $e -Recurse -Filter 'summary.csv').Count)"
"headtohead_runs_csv_count=$(@(Get-ChildItem $e -Recurse -Filter 'runs.csv').Count)"
& 'C:/Users/marti/.cargo/bin/rustc.exe' +1.95.0 -vV
& 'C:/Users/marti/.cargo/bin/cargo.exe' +1.95.0 -vV
& 'C:/Users/marti/.rustup/toolchains/1.95.0-x86_64-pc-windows-msvc/lib/rustlib/x86_64-pc-windows-msvc/bin/llvm-profdata.exe' --version
& 'C:/Users/marti/.cargo/bin/rustc.exe' +nightly -vV
& 'C:/Users/marti/.cargo/bin/rustup.exe' component list --installed --toolchain nightly
& 'C:/Users/marti/.cargo/bin/cargo-fuzz.exe' --version
Get-ChildItem 'C:/Users/marti/.rustup/toolchains/nightly-x86_64-pc-windows-msvc/lib' -Recurse -File -Filter '*asan*' | Select-Object FullName,Length | Format-List
Get-ChildItem 'C:/Program Files/Microsoft Visual Studio/*/*/VC/Tools/MSVC/*/bin/Hostx64/x64/clang_rt.asan_dynamic-x86_64.dll' -ErrorAction SilentlyContinue | Select-Object FullName,Length | Format-List
Get-Command clang.exe,cc.exe,gcc.exe,cargo-fuzz.exe -ErrorAction SilentlyContinue | Select-Object Name,Source | Format-List
Get-CimInstance Win32_Processor | Select-Object Name | Format-List
