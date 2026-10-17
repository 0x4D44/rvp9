$ProgressPreference='SilentlyContinue'
"host=$env:COMPUTERNAME utc=$([DateTime]::UtcNow.ToString('o'))"
deltic --version
deltic session repos list --help
foreach($root in @('D:/language/rvp9-2','D:/language/rvp9','C:/language/rvp9')) {
 "known_rvp9_root=$root exists=$(Test-Path $root) git_exists=$(Test-Path "$root/.git") cargo_exists=$(Test-Path "$root/Cargo.toml")"
}
