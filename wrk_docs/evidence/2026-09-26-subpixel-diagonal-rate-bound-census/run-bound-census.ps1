$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-census-cpu6-subpixel-rate-bound-opportun'
$exe = Join-Path $root 'target\release\rvp9enc.exe'
$input = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$output = Join-Path $root '.scratch\ducks-bound-census-02.ivf'
$stderr = Join-Path $root '.scratch\ducks-bound-census-02.stderr.txt'
$expectedExe = '3df3196912a4cbd9421e87312011ae9e4988375048bad8ef326ca5ff3ba85e99'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'
if ((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedExe) { throw 'EXE hash mismatch' }
if ((Get-FileHash -LiteralPath $input -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedInput) { throw 'Input hash mismatch' }
if (Test-Path -LiteralPath $output) { throw 'Output already exists' }
$env:RVP9_RATE_BOUND_CENSUS = '1'
$ErrorActionPreference = 'Continue'
& $exe --codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o $output $input 2> $stderr
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) { throw "Encode failed: $LASTEXITCODE" }
$actual = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expectedOutput) { throw "Output mismatch: $actual" }
Write-Output "output_sha256=$actual"
Get-Content -LiteralPath $stderr
