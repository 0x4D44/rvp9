# build-pgo.ps1 — three-stage profile-guided optimization build of
# `decode-headtohead` (and any libraries it links — i.e. the whole
# decoder hot path).
#
# Stage 1 — instrument: build with -Cprofile-generate, write into
#   target/x86_64-pc-windows-msvc/release/.
# Stage 2 — train: run the instrumented binary on its default
#   9-row matrix (HLD §5). Each row decodes 1000 timed frames, which
#   gives ~9000 frames of profile coverage across resolutions, tile
#   layouts, and thread counts.
# Stage 3 — optimize: rebuild with -Cprofile-use, merging the .profraw
#   files into a .profdata that the rustc backend feeds to LLVM.
# Stage 4 — install: copy the PGO binary to the conventional
#   `target/release/` slot so `scripts/bench-headtohead.ps1` and any
#   other tools that hardcode that path pick it up.
#
# Requirements (pre-installed in this workspace):
#   - cargo-pgo (`cargo install cargo-pgo`)
#   - rustup component `llvm-tools-preview` (provides llvm-profdata)
#
# Output measured 2026-05-25 on the dev host:
#   geomean -16% p50 / -18.5% mean across the 9-fixture matrix
#   (largest wins on the 1080p single-tile fixtures: ~30% on
#   tile_1x4 t=1 / testsrc_1080p30 t=2).
#
# Trade-off vs. plain release: build wall-time roughly 3x (instrument
# + train + optimize). Run only when shipping a benchmarked binary
# or before a `bench-headtohead` measurement run.

$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel).Trim()
Set-Location $repo

Write-Host "[PGO 1/3] Building instrumented decode-headtohead..." -ForegroundColor Cyan
cargo pgo instrument build -- -p rvp9-bench --bin decode-headtohead
if ($LASTEXITCODE -ne 0) { throw "PGO instrument build failed" }

$instrBin = Join-Path $repo 'target/x86_64-pc-windows-msvc/release/decode-headtohead.exe'
if (-not (Test-Path $instrBin)) { throw "Expected instrumented binary at $instrBin" }

Write-Host "[PGO 2/3] Running instrumented binary to collect profiles..." -ForegroundColor Cyan
$trainOut = Join-Path $repo 'target/pgo-train.tsv'
& $instrBin --out $trainOut
if ($LASTEXITCODE -ne 0) { throw "PGO training run failed" }

$profDir = Join-Path $repo 'target/pgo-profiles'
$profCount = (Get-ChildItem $profDir -Filter '*.profraw' -ErrorAction SilentlyContinue).Count
if ($profCount -lt 1) { throw "No .profraw files produced in $profDir" }
Write-Host "  collected $profCount .profraw file(s) in $profDir"

Write-Host "[PGO 3/3] Building PGO-optimized binary..." -ForegroundColor Cyan
cargo pgo optimize build -- -p rvp9-bench --bin decode-headtohead
if ($LASTEXITCODE -ne 0) { throw "PGO optimize build failed" }

$optimizedBin = Join-Path $repo 'target/x86_64-pc-windows-msvc/release/decode-headtohead.exe'
$installBin   = Join-Path $repo 'target/release/decode-headtohead.exe'
Copy-Item -Force $optimizedBin $installBin
Write-Host "[PGO done] PGO-optimized binary installed at $installBin" -ForegroundColor Green
Write-Host "           (original under target/x86_64-pc-windows-msvc/release/)"
Write-Host ""
Write-Host "To benchmark:  pwsh scripts/bench-headtohead.ps1 -SkipEncode -Repetitions 3"
Write-Host "To restore plain release:  cargo build --release -p rvp9-bench --bin decode-headtohead"
