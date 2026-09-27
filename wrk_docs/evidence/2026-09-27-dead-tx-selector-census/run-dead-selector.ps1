$ErrorActionPreference = 'Stop'
$taskRoot = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-measure-dead-emit-side-luma-tx-selector'
$exe = Join-Path $taskRoot 'target\release\rvp9enc.exe'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$outRoot = Join-Path $taskRoot '.scratch\dead-selector-run'
if (Test-Path -LiteralPath $outRoot) { throw "Run root already exists: $outRoot" }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing $exe" }
New-Item -ItemType Directory -Path $outRoot | Out-Null
$clips = @(
    [pscustomobject]@{Name='ducks'; File='ducks_take_off_420_1080p30_60f.y4m'; Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    [pscustomobject]@{Name='tree'; File='in_to_tree_420_1080p30_60f.y4m'; Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    [pscustomobject]@{Name='town'; File='old_town_cross_420_1080p30_60f.y4m'; Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$identity = @([pscustomobject]@{Name='exe';Path=$exe;Sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()})
$rows = @()
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) { throw "Missing $inputPath" }
    $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($inputHash -ne $clip.Input) { throw "Input hash mismatch: $($clip.Name)" }
    $identity += [pscustomobject]@{Name=$clip.Name;Path=$inputPath;Sha256=$inputHash}
    $outputPath = Join-Path $outRoot "$($clip.Name).ivf"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$inputPath
    $psi.WorkingDirectory = $outRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['RVP9_DIAG_DEAD_TX_SELECTOR'] = '1'
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Could not start $($clip.Name)" }
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "Timed out $($clip.Name)" }
    $proc.Refresh()
    [System.IO.File]::WriteAllText((Join-Path $outRoot "$($clip.Name).stdout.txt"), $stdout.Result)
    [System.IO.File]::WriteAllText((Join-Path $outRoot "$($clip.Name).stderr.txt"), $stderr.Result)
    $hash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $rows += [pscustomobject]@{Clip=$clip.Name;CpuSeconds=$proc.TotalProcessorTime.TotalSeconds;ExitCode=$proc.ExitCode;OutputSha256=$hash;ExpectedSha256=$clip.Output;Matches=($hash -eq $clip.Output)}
    $proc.Dispose()
    if ($rows[-1].ExitCode -ne 0 -or -not $rows[-1].Matches) { throw "Output mismatch: $($clip.Name)" }
    Write-Output ("{0} CPU={1:F3}s IVF={2}" -f $clip.Name,$rows[-1].CpuSeconds,$hash)
}
$identity | Export-Csv -LiteralPath (Join-Path $outRoot 'identity.csv') -NoTypeInformation -Encoding UTF8
$rows | Export-Csv -LiteralPath (Join-Path $outRoot 'results.csv') -NoTypeInformation -Encoding UTF8
