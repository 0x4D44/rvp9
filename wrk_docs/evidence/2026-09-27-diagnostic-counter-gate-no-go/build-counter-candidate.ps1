$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-test-release-diagnostic-counter-gating-o'
$profile = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo\runs\20260927T111441Z-3232706\encoder.profdata'
Set-Location $root
$env:CARGO_TARGET_DIR = Join-Path $root '.scratch\counter-pgo-target'
$env:RUSTFLAGS = "-Ctarget-cpu=native -Cprofile-use=$profile"
cargo build --locked --release -p rvp9-cli --bin rvp9enc
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Get-FileHash -Algorithm SHA256 (Join-Path $env:CARGO_TARGET_DIR 'release\rvp9enc.exe') | Select-Object -ExpandProperty Hash
