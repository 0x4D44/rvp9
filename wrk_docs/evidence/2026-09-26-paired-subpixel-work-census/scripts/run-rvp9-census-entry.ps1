$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\rvp9-work-census'
$exe = Join-Path $root 'target\release\rvp9enc.exe'
$outDir = Join-Path $root 'diagnostic-runs-entry'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$expectedExe = 'd2637a286b0ee911309ae45b3ff7551166481b7457244bbc0a3a5bb90b9246b4'
$clips = @(
    [pscustomobject]@{Name='ducks'; File='ducks_take_off_420_1080p30_60f.y4m'; Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    [pscustomobject]@{Name='tree'; File='in_to_tree_420_1080p30_60f.y4m'; Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    [pscustomobject]@{Name='town'; File='old_town_cross_420_1080p30_60f.y4m'; Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant() -ne $expectedExe) {
    throw 'Instrumented binary hash mismatch'
}
if (Test-Path -LiteralPath $outDir) { throw "Refusing to overwrite $outDir" }
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant() -ne $clip.Input) {
        throw "Input hash mismatch: $($clip.Name)"
    }
}
[void](New-Item -ItemType Directory -Path $outDir)
$rows = @()
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    $outputPath = Join-Path $outDir "$($clip.Name).ivf"
    $psi = New-Object Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$inputPath
    $psi.WorkingDirectory = $outDir
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['RVP9_SUBPEL_CENSUS'] = '1'
    $process = New-Object Diagnostics.Process
    $process.StartInfo = $psi
    try {
        if (-not $process.Start()) { throw "Could not start $($clip.Name)" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(90000)) {
            $process.Kill()
            throw "Timed out $($clip.Name)"
        }
        [IO.File]::WriteAllText((Join-Path $outDir "$($clip.Name).stdout.txt"), $stdoutTask.Result, [Text.UTF8Encoding]::new($false))
        [IO.File]::WriteAllText((Join-Path $outDir "$($clip.Name).stderr.txt"), $stderrTask.Result, [Text.UTF8Encoding]::new($false))
        if ($process.ExitCode -ne 0) { throw "Encode failed: $($clip.Name) exit=$($process.ExitCode)" }
        $outputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
        if ($outputHash -ne $clip.Output) { throw "Output hash mismatch: $($clip.Name) actual=$outputHash" }
        $censusLines = @($stderrTask.Result -split "`r?`n" | Where-Object { $_ -match '^subpel_census,' })
        if ($censusLines.Count -lt 2) { throw "Missing census rows: $($clip.Name)" }
        $searchLines = @($stderrTask.Result -split "`r?`n" | Where-Object { $_ -match '^subpel_searches,' })
        if ($searchLines.Count -lt 2) { throw "Missing search rows: $($clip.Name)" }
        $rows += [pscustomobject]@{Clip=$clip.Name; InputSha256=$clip.Input; OutputSha256=$outputHash; BinarySha256=$expectedExe; CensusRows=$censusLines.Count-1}
        Write-Output "CENSUS_OK clip=$($clip.Name) rows=$($censusLines.Count-1) hash=$outputHash"
    } finally {
        $process.Dispose()
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $outDir 'manifest.csv') -NoTypeInformation -Encoding UTF8
