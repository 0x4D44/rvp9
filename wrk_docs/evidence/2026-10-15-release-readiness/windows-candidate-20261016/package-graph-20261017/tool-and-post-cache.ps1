$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
foreach($name in @('cargo.exe','rustc.exe','rustdoc.exe')) {
 $f=Get-Item "C:/Users/marti/.rustup/toolchains/1.95.0-x86_64-pc-windows-msvc/bin/$name"
 "$name|bytes=$($f.Length)|sha256=$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())"
}
foreach($name in @('rvp9-core','rvp9-io','rvp9-decoder','rvp9-encoder')) {
 $p="C:/Users/marti/.cargo/registry/src/-e838b5265d494894/$name-0.2.0-alpha.4"
 "registry_source_after_verification=$p exists=$(Test-Path $p)"
}
