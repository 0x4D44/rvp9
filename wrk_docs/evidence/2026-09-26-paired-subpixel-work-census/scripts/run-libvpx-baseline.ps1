$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\libvpx-work-census'
$exe = Join-Path $root 'baseline-build-rel\vpxenc.exe'
$inputPath = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$outputPath = Join-Path $root 'baseline-ducks.ivf'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = 'b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8'
if (Test-Path -LiteralPath $outputPath) { throw "Refusing to overwrite $outputPath" }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing $exe" }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant() -ne $expectedInput) {
    throw 'Input hash mismatch'
}
& deltic timeout 120 $exe --codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr `
    --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 `
    --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 `
    --kf-max-dist=60 --quiet --ivf -o $outputPath $inputPath
if ($LASTEXITCODE -ne 0) { throw "Encode failed: $LASTEXITCODE" }
$exeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
$outputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
Write-Output "EXE_SHA256=$exeHash"
Write-Output "OUTPUT_SHA256=$outputHash"
if ($outputHash -ne $expectedOutput) { throw 'Fresh reference build output differs from pinned reference output' }
Write-Output 'BASELINE_OUTPUT_EXACT=1'
