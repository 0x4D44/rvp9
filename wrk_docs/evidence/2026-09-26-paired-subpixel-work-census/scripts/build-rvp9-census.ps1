$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\rvp9-work-census'
$source = Join-Path $root 'instrumented-src'
$target = Join-Path $root 'target'
$head = (& git -C $source rev-parse HEAD).Trim()
if ($head -ne '48cab42d2dba1a46c502a5da2faf907d79d78ac1') { throw "Source HEAD mismatch: $head" }
$files = @(
    [pscustomobject]@{Path='crates\rvp9-core\src\mv_search\sub_pixel.rs'; Hash='b32a0bd226a5d24f9f7ce510440c274a0d2c1304242e32e9afd1a63c6d605a1d'},
    [pscustomobject]@{Path='crates\rvp9-cli\src\bin\rvp9enc.rs'; Hash='ab981ea27ea60ea687a64c6579c3a1733c3dcb7073673cafac560f49e8af852c'}
)
foreach ($file in $files) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $source $file.Path)).Hash.ToLowerInvariant()
    if ($actual -ne $file.Hash) { throw "Source mismatch: $($file.Path) actual=$actual" }
}
if (Test-Path -LiteralPath $target) { throw "Refusing existing target: $target" }
Set-Location -LiteralPath $source
$env:CARGO_TARGET_DIR = $target
& deltic timeout 1800 cargo build --release -p rvp9-cli --bin rvp9enc
if ($LASTEXITCODE -ne 0) { throw "Release build failed: $LASTEXITCODE" }
$exe = Join-Path $target 'release\rvp9enc.exe'
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
Write-Output "RVP9_CENSUS_EXE_SHA256=$hash"
