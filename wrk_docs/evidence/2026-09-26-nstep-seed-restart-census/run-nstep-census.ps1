param([string]$RunLabel = 'census-01')
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-count-nstep-seed-restarts-on-pinned-clip'
$exe = Join-Path $worktree '.scratch\target-nstep-census\release\rvp9enc.exe'
$expectedExe = 'a3f2a7255522fc7ad3421c2c387fd2b144bc8ef0853230680a4fea9889e35d85'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clips = @(
    [pscustomobject]@{Name='ducks';File='ducks_take_off_420_1080p30_60f.y4m';Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7';Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    [pscustomobject]@{Name='tree';File='in_to_tree_420_1080p30_60f.y4m';Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791';Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    [pscustomobject]@{Name='town';File='old_town_cross_420_1080p30_60f.y4m';Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909';Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$actualExe = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualExe -ne $expectedExe) { throw "EXE hash mismatch: $actualExe" }
$runRoot = Join-Path (Join-Path $worktree '.scratch') $RunLabel
if (Test-Path -LiteralPath $runRoot) { throw "Refusing to overwrite $runRoot" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
$rows = @()
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($inputHash -ne $clip.Input) { throw "Input hash mismatch: $inputPath" }
    $outputPath = Join-Path $runRoot ($clip.Name + '.ivf')
    $args = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$inputPath
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = $args
    $psi.WorkingDirectory = $runRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Could not start $($clip.Name)" }
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "Timed out $($clip.Name)" }
    $proc.Refresh()
    $stderrText = $stderr.Result
    [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name + '.stderr.txt')), $stderrText)
    [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name + '.stdout.txt')), $stdout.Result)
    $match = [regex]::Match($stderrText, '(?m)^NSTEP_SEED_CENSUS calls=(\d+) starts=(\d+) repeated_starts=(\d+) max_extra_starts=(\d+)\r?$')
    if (-not $match.Success) { throw "Missing NSTEP census: $($clip.Name)" }
    $calls = [long]$match.Groups[1].Value
    $starts = [long]$match.Groups[2].Value
    $repeated = [long]$match.Groups[3].Value
    $maxExtra = [long]$match.Groups[4].Value
    if ($starts - $calls -ne $repeated -or $repeated -gt $maxExtra) { throw "Inconsistent NSTEP counts: $($clip.Name)" }
    $outputHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $rows += [pscustomobject]@{Clip=$clip.Name;Calls=$calls;DiamondStarts=$starts;RepeatedStarts=$repeated;MaxExtraStarts=$maxExtra;RepeatedPerCall=([math]::Round($repeated / $calls, 6));CpuSeconds=$proc.TotalProcessorTime.TotalSeconds;ExitCode=$proc.ExitCode;OutputSha256=$outputHash;ExpectedSha256=$clip.Output;OutputMatches=($outputHash -eq $clip.Output);InputSha256=$inputHash;ExeSha256=$actualExe}
    $rows | Export-Csv -LiteralPath (Join-Path $runRoot 'census.csv') -NoTypeInformation -Encoding UTF8
    Write-Output ("{0}: calls={1} starts={2} repeated={3} max_extra={4} hash_ok={5}" -f $clip.Name,$calls,$starts,$repeated,$maxExtra,($outputHash -eq $clip.Output))
    if ($proc.ExitCode -ne 0 -or $outputHash -ne $clip.Output) { throw "Encode failed or output mismatch: $($clip.Name)" }
    $proc.Dispose()
}
