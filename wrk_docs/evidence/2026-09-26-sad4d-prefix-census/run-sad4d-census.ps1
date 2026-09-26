$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-measure-nstep-sad4d-early-exit-opportuni'
$encoder = Join-Path $root '.scratch\target\release\rvp9enc.exe'
$inputs = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$outputRoot = Join-Path $root '.scratch\sad4d-census'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$clips = @(
    @{ Name='ducks'; File='ducks_take_off_420_1080p30_60f.y4m'; Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' },
    @{ Name='tree'; File='in_to_tree_420_1080p30_60f.y4m'; Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' },
    @{ Name='town'; File='old_town_cross_420_1080p30_60f.y4m'; Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f' }
)
$exeHash = (Get-FileHash -LiteralPath $encoder -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Output "EXE_SHA256=$exeHash"
$env:RVP9_DIAG_SAD4D_EARLY = '1'
try {
    foreach ($clip in $clips) {
        $inputPath = Join-Path $inputs $clip.File
        $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($inputHash -ne $clip.Input) { throw "Input hash mismatch: $($clip.Name) $inputHash" }
        $outputPath = Join-Path $outputRoot ($clip.Name + '.ivf')
        $stderrPath = Join-Path $outputRoot ($clip.Name + '.stderr.txt')
        $encodeArgs = @('--codec=vp9','--passes=1','--rt','--cpu-used=6','--end-usage=cbr','--target-bitrate=1260','--fps=30/1','--threads=1','--tile-columns=0','--lag-in-frames=0','--tune-content=default','--aq-mode=0','--noise-sensitivity=0','--limit=60','--kf-max-dist=60','--quiet','--ivf','-o',$outputPath,$inputPath)
        $savedErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & deltic timeout 120 $encoder @encodeArgs 2> $stderrPath
        } finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        if ($LASTEXITCODE -ne 0) { throw "Encode failed: $($clip.Name) exit $LASTEXITCODE" }
        $outputHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($outputHash -ne $clip.Output) { throw "Output hash mismatch: $($clip.Name) $outputHash" }
        $counterLine = Get-Content -LiteralPath $stderrPath | Where-Object { $_ -match 'RVP9_DIAG_SAD4D_EARLY batched_16x16=' }
        if (@($counterLine).Count -ne 1) { throw "Missing or repeated counters: $($clip.Name)" }
        $counterLine = [regex]::Match([string]$counterLine, 'RVP9_DIAG_SAD4D_EARLY batched_16x16=\d+ all_four_prefix4_ge_best=\d+').Value
        Write-Output "$($clip.Name) input=$inputHash output=$outputHash $counterLine"
    }
} finally {
    Remove-Item Env:RVP9_DIAG_SAD4D_EARLY -ErrorAction SilentlyContinue
}
