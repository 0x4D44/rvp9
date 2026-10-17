$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b'
"utc=$([DateTime]::UtcNow.ToString('o'))"
"root22720_present=$($null -ne (Get-Process -Id 22720 -ErrorAction SilentlyContinue))"
"cargo_count=$([Diagnostics.Process]::GetProcessesByName('cargo').Count)"
"rustc_count=$([Diagnostics.Process]::GetProcessesByName('rustc').Count)"
foreach($name in @('rvp9-core','rvp9-io','rvp9-decoder','rvp9-encoder','rvp9-cli')) {
 $base="$name-0.2.0-alpha.4"
 Copy-Item "$r/cargo-target/package/tmp-registry/index/rv/p9/$name" "$r/tmp-index-$name.json"
 foreach($path in @("$r/cargo-target/package/$base.crate","$r/cargo-target/package/tmp-registry/$base.crate","C:/Users/marti/.cargo/registry/cache/-e838b5265d494894/$base.crate")) {
  if(Test-Path $path) { $f=Get-Item $path; "archive_identity=$path|bytes=$($f.Length)|sha256=$((Get-FileHash $path -Algorithm SHA256).Hash.ToLower())" }
 }
 $source="C:/Users/marti/.cargo/registry/src/-e838b5265d494894/$base"
 if(Test-Path "$source/.cargo-checksum.json") { Copy-Item "$source/.cargo-checksum.json" "$r/tmp-source-$name.checksum.json"; "resolved_registry_source=$source" }
}
$task='D:/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi'
"head=$(git -C $task rev-parse HEAD)"
"status=$(git -C $task status --porcelain)"
foreach($f in @(Get-ChildItem $r -File)) {
 "download_file=$($f.Name)|bytes=$($f.Length)|sha256=$((Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower())"
}
