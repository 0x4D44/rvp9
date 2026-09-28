$ErrorActionPreference = 'Stop'
$worktree = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $worktree 'target\release\rvp9enc.exe'
$outDir = Join-Path $PSScriptRoot 'mvref-full-census-20260928-c'
if (Test-Path $outDir) { throw "Output already exists: $outDir" }
if (-not (Test-Path $exe)) { throw "Missing diagnostic encoder: $exe" }

$inputs = @(
    @('ducks', 'ducks_take_off_420_1080p30_60f.y4m', '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7', '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'),
    @('tree', 'in_to_tree_420_1080p30_60f.y4m', 'ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791', '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'),
    @('town', 'old_town_cross_420_1080p30_60f.y4m', 'b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909', '4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f')
)
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$null = New-Item -ItemType Directory -Path $outDir
$exeHash = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant()
"exe_sha256=$exeHash" | Set-Content (Join-Path $outDir 'identity.txt')

foreach ($clip in $inputs) {
    $name, $filename, $expectedInput, $expectedOutput = $clip
    $inputPath = Join-Path $inputRoot $filename
    $inputHash = (Get-FileHash $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($inputHash -ne $expectedInput) { throw "$name input SHA mismatch: $inputHash" }
    "${name}_input_sha256=$inputHash" | Add-Content (Join-Path $outDir 'identity.txt')
    foreach ($gate in @('on', 'off')) {
        if ($gate -eq 'on') { $env:RVP9_DIAG_MVREF_FULL = '1' }
        else { Remove-Item Env:RVP9_DIAG_MVREF_FULL -ErrorAction SilentlyContinue }
        $prefix = "$name-$gate"
        $outputPath = Join-Path $outDir "$prefix.ivf"
        $stderrPath = Join-Path $outDir "$prefix.stderr.txt"
        $stdoutPath = Join-Path $outDir "$prefix.stdout.txt"
        $arguments = @(
            '--codec=vp9', '--passes=1', '--rt', '--cpu-used=6', '--end-usage=cbr',
            '--target-bitrate=1260', '--fps=30/1', '--threads=1', '--tile-columns=0',
            '--lag-in-frames=0', '--tune-content=default', '--aq-mode=0',
            '--noise-sensitivity=0', '--limit=60', '--kf-max-dist=60', '--quiet', '--ivf',
            '-o', $outputPath, $inputPath
        )
        $process = Start-Process -FilePath $exe -ArgumentList $arguments -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        if ($process.ExitCode -ne 0) { throw "$prefix encoder exit: $($process.ExitCode)" }
        $outputHash = (Get-FileHash $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($outputHash -ne $expectedOutput) { throw "$prefix output SHA mismatch: $outputHash" }
        $rows = @(Get-Content $stderrPath | Where-Object { $_ -like 'RVP9_DIAG_MVREF_FULL refs=*' })
        if ($gate -eq 'on' -and $rows.Count -ne 2) { throw "$prefix expected two census rows, got $($rows.Count)" }
        if ($gate -eq 'off' -and $rows.Count -ne 0) { throw "$prefix unexpectedly reported census" }
        "${prefix}_output_sha256=$outputHash" | Add-Content (Join-Path $outDir 'identity.txt')
        Write-Output "$prefix hash=$outputHash"
        foreach ($row in $rows) { Write-Output $row }
    }
}
Remove-Item Env:RVP9_DIAG_MVREF_FULL -ErrorAction SilentlyContinue
