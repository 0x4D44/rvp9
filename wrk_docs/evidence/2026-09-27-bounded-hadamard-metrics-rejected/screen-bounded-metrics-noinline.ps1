param([int]$Pairs = 5, [string]$RunLabel = 'bounded-noinline-screen')
$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-prototype-bounded-hadamard-metrics-cruci\.scratch'
$current = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root 'encoder-pgo-bounded-metrics-noinline\optimized\release\rvp9enc.exe'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clips = @(
    [pscustomobject]@{Name='ducks';File='ducks_take_off_420_1080p30_60f.y4m';Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7';Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    [pscustomobject]@{Name='tree';File='in_to_tree_420_1080p30_60f.y4m';Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791';Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    [pscustomobject]@{Name='town';File='old_town_cross_420_1080p30_60f.y4m';Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909';Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$runRoot = Join-Path $root $RunLabel
if (Test-Path -LiteralPath $runRoot) { throw "Run root already exists: $runRoot" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
$identity = @()
foreach ($asset in @(@{Name='current';Path=$current}, @{Name='candidate';Path=$candidate})) {
    if (-not (Test-Path -LiteralPath $asset.Path -PathType Leaf)) { throw "Missing $($asset.Path)" }
    $identity += [pscustomobject]@{Name=$asset.Name;Path=$asset.Path;Sha256=(Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash.ToLowerInvariant()}
}
if (($identity | Where-Object Name -eq current).Sha256 -ne '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee') { throw 'Current PGO binary hash mismatch' }
foreach ($clip in $clips) {
    $path = Join-Path $inputRoot $clip.File
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hash -ne $clip.Input) { throw "Input hash mismatch: $path" }
    $identity += [pscustomobject]@{Name=$clip.Name;Path=$path;Sha256=$hash}
}
$identity | Export-Csv -LiteralPath (Join-Path $runRoot 'identity.csv') -NoTypeInformation -Encoding UTF8
$rows = New-Object 'System.Collections.Generic.List[object]'
function Invoke-Encode($clip, [string]$arm, [int]$pair) {
    $exe = if ($arm -eq 'current') { $current } else { $candidate }
    $name = '{0}-{1:D2}-{2}' -f $clip.Name,$pair,$arm
    $out = Join-Path $runRoot "$name.ivf"
    $inputPath = Join-Path $inputRoot $clip.File
    $args = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $out,$inputPath
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName=$exe; $psi.Arguments=$args; $psi.WorkingDirectory=$runRoot; $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
    $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
    $proc = New-Object System.Diagnostics.Process; $proc.StartInfo=$psi
    if (-not $proc.Start()) { throw "Could not start $name" }
    $stdout=$proc.StandardOutput.ReadToEndAsync(); $stderr=$proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "Timed out $name" }
    $proc.Refresh()
    $cpu=$proc.TotalProcessorTime.TotalSeconds; $wall=($proc.ExitTime-$proc.StartTime).TotalSeconds; $exit=$proc.ExitCode
    [System.IO.File]::WriteAllText((Join-Path $runRoot "$name.stdout.txt"),$stdout.Result)
    [System.IO.File]::WriteAllText((Join-Path $runRoot "$name.stderr.txt"),$stderr.Result)
    $hash = if (Test-Path -LiteralPath $out -PathType Leaf) { (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant() } else { '' }
    $rows.Add([pscustomobject]@{Clip=$clip.Name;Pair=$pair;Arm=$arm;CpuSeconds=$cpu;WallSeconds=$wall;ExitCode=$exit;OutputSha256=$hash;ExpectedSha256=$clip.Output;OutputMatches=($hash -eq $clip.Output)})
    $rows | Export-Csv -LiteralPath (Join-Path $runRoot 'raw.csv') -NoTypeInformation -Encoding UTF8
    Write-Output ("{0} cpu={1:F3}s wall={2:F3}s hash={3}" -f $name,$cpu,$wall,$hash)
    $proc.Dispose()
    if ($exit -ne 0 -or $hash -ne $clip.Output) { throw "Encode failed or output mismatch: $name" }
}
foreach ($clip in $clips) { Invoke-Encode $clip 'current' 0; Invoke-Encode $clip 'candidate' 0 }
for ($pair=1; $pair -le $Pairs; $pair++) {
    $order=if($pair%2 -eq 1){@('current','candidate')}else{@('candidate','current')}
    $clipOrder=if($pair%2 -eq 1){$clips}else{@($clips[2],$clips[1],$clips[0])}
    foreach ($clip in $clipOrder) { foreach ($arm in $order) { Invoke-Encode $clip $arm $pair } }
}
$summary = @()
foreach ($clip in $clips) {
    $ratios = @()
    for ($pair=1; $pair -le $Pairs; $pair++) {
        $a = $rows | Where-Object {$_.Clip -eq $clip.Name -and $_.Pair -eq $pair -and $_.Arm -eq 'current'} | Select-Object -First 1
        $b = $rows | Where-Object {$_.Clip -eq $clip.Name -and $_.Pair -eq $pair -and $_.Arm -eq 'candidate'} | Select-Object -First 1
        $ratios += $b.CpuSeconds / $a.CpuSeconds
    }
    $geo = [math]::Exp((($ratios | ForEach-Object {[math]::Log($_)} | Measure-Object -Sum).Sum) / $ratios.Count)
    $summary += [pscustomobject]@{Clip=$clip.Name;Pairs=$Pairs;CandidateOverCurrentCpuGeoMean=[math]::Round($geo,6);Ratios=($ratios -join ';')}
}
$summary | Export-Csv -LiteralPath (Join-Path $runRoot 'summary.csv') -NoTypeInformation -Encoding UTF8
$summary | Format-Table | Out-String | Write-Output
