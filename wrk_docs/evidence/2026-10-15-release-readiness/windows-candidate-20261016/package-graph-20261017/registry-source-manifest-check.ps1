$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$r='C:/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-packagegraph474-3255285b'
foreach($name in @('rvp9-core','rvp9-io','rvp9-decoder','rvp9-encoder')) {
 $base="$name-0.2.0-alpha.4"
 $source="C:/Users/marti/.cargo/registry/src/-e838b5265d494894/$base"
 foreach($member in @('Cargo.toml','Cargo.toml.orig','.cargo_vcs_info.json')) {
  $actual=(Get-FileHash "$source/$member" -Algorithm SHA256).Hash.ToLower()
  $archive=(Get-FileHash "$r/$base.inspected-$member.txt" -Algorithm SHA256).Hash.ToLower()
  "$name member=$member registry_source_sha256=$actual archive_member_sha256=$archive equal=$($actual -eq $archive)"
  if($actual -ne $archive) { throw 'registry source manifest mismatch' }
 }
 "registry_source_checksum_record_exists=$(Test-Path "$source/.cargo-checksum.json")"
}
