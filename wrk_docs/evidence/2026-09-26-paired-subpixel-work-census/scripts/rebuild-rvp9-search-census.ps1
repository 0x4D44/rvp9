$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\rvp9-work-census'
$source = Join-Path $root 'instrumented-src'
$target = Join-Path $root 'target'
$head = (& git -C $source rev-parse HEAD).Trim()
if ($head -ne '48cab42d2dba1a46c502a5da2faf907d79d78ac1') { throw "Source HEAD mismatch: $head" }
$files = @(
    [pscustomobject]@{Path='crates\rvp9-core\src\mv_search\sub_pixel.rs'; Hash='ace63292161e9fe1588f05c2b36759d5f066004b7a9e6baba0b4d4ee096bba8e'},
    [pscustomobject]@{Path='crates\rvp9-cli\src\bin\rvp9enc.rs'; Hash='a62823f8b5a3d3ae24aa1b9b8584aa8a4c41b1f99786a70032a620a0a23d8740'}
)
foreach ($file in $files) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $source $file.Path)).Hash.ToLowerInvariant()
    if ($actual -ne $file.Hash) { throw "Source mismatch: $($file.Path) actual=$actual" }
}
$exe = Join-Path $target 'release\rvp9enc.exe'
$prior = Join-Path $root 'rvp9enc-phase-only.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing previous binary: $exe" }
if (Test-Path -LiteralPath $prior) { throw "Refusing to overwrite $prior" }
$priorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
if ($priorHash -ne 'b8e3219e948241dc109408c86fab5f8cd1f1f4e671b3796d57381ce1dddc5580') { throw "Previous binary mismatch: $priorHash" }
Copy-Item -LiteralPath $exe -Destination $prior
Set-Location -LiteralPath $source
$env:CARGO_TARGET_DIR = $target
& deltic timeout 1800 cargo build --release -p rvp9-cli --bin rvp9enc
if ($LASTEXITCODE -ne 0) { throw "Release build failed: $LASTEXITCODE" }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
Write-Output "RVP9_SEARCH_CENSUS_EXE_SHA256=$hash"
