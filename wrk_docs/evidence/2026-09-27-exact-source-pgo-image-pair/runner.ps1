param([int]$Pairs = 15)
$ErrorActionPreference = 'Stop'
$runRoot = 'C:/Users/marti/AppData/Local/Temp/deltic-scratch/exact-source-pgo-pair-8d5903d4'
$control = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch/fresh-integrated-pgo/optimized/release/rvp9enc.exe'
$candidate = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-chroma-energy-certificate-pgo/.scratch/encoder-pgo-retention-candidate-cf928/optimized/release/rvp9enc.exe'
$inputRoot = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch/fresh-integrated-pgo/inputs'
$assets = @(
    @{ Name='control'; Path=$control; Hash='3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad' },
    @{ Name='candidate'; Path=$candidate; Hash='c0dc53bc3a3b8ede9d9c45e8606d0f4b09ab0f60c10e7e300d790e79d8b7bfbb' }
)
$clips = @(
    @{ Name='ducks'; File='ducks_take_off_420_1080p30_60f.y4m'; Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' },
    @{ Name='tree'; File='in_to_tree_420_1080p30_60f.y4m'; Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' },
    @{ Name='town'; File='old_town_cross_420_1080p30_60f.y4m'; Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f' }
)
$identity = New-Object 'System.Collections.Generic.List[object]'
foreach ($asset in $assets) {
    $actual = (Get-FileHash -LiteralPath $asset.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $asset.Hash) { throw "Binary hash mismatch: $($asset.Name)" }
    $identity.Add([pscustomobject]@{Name=$asset.Name;Path=$asset.Path;Sha256=$actual})
}
foreach ($clip in $clips) {
    $path = Join-Path $inputRoot $clip.File
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $clip.Input) { throw "Input hash mismatch: $($clip.Name)" }
    $identity.Add([pscustomobject]@{Name=$clip.Name;Path=$path;Sha256=$actual})
}
$identity | Export-Csv -LiteralPath (Join-Path $runRoot 'identity.csv') -NoTypeInformation -Encoding UTF8
$rows = New-Object 'System.Collections.Generic.List[object]'
function Invoke-Encode($clip, [string]$arm, [int]$pair) {
    $exe = if ($arm -eq 'control') { $control } else { $candidate }
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
    $rows.Add([pscustomobject]@{Clip=$clip.Name;Pair=$pair;Arm=$arm;CpuSeconds=$cpu;WallSeconds=$wall;ExitCode=$exit;OutputSha256=$hash;OutputMatches=($hash -eq $clip.Output)})
    $rows | Export-Csv -LiteralPath (Join-Path $runRoot 'raw.csv') -NoTypeInformation -Encoding UTF8
    Write-Output ("{0} cpu={1:F3}s wall={2:F3}s hash={3}" -f $name,$cpu,$wall,$hash)
    $proc.Dispose()
    if ($exit -ne 0 -or $hash -ne $clip.Output) { throw "Encode failed or output mismatch: $name" }
}
foreach ($clip in $clips) { Invoke-Encode $clip 'control' 0; Invoke-Encode $clip 'candidate' 0 }
for ($pair=1; $pair -le $Pairs; $pair++) {
    $order=if($pair%2 -eq 1){@('control','candidate')}else{@('candidate','control')}
    $clipOrder=if($pair%2 -eq 1){$clips}else{@($clips[2],$clips[1],$clips[0])}
    foreach ($clip in $clipOrder) { foreach ($arm in $order) { Invoke-Encode $clip $arm $pair } }
}
$summary = @()
foreach ($clip in $clips) {
    $ratios = @()
    for ($pair=1; $pair -le $Pairs; $pair++) {
        $a = $rows | Where-Object {$_.Clip -eq $clip.Name -and $_.Pair -eq $pair -and $_.Arm -eq 'control'} | Select-Object -First 1
        $b = $rows | Where-Object {$_.Clip -eq $clip.Name -and $_.Pair -eq $pair -and $_.Arm -eq 'candidate'} | Select-Object -First 1
        $ratios += $b.CpuSeconds / $a.CpuSeconds
    }
    $geo = [math]::Exp((($ratios | ForEach-Object {[math]::Log($_)} | Measure-Object -Sum).Sum) / $ratios.Count)
    $summary += [pscustomobject]@{Clip=$clip.Name;Pairs=$Pairs;CandidateOverControlCpuGeoMean=[math]::Round($geo,6);Ratios=($ratios -join ';')}
}
$summary | Export-Csv -LiteralPath (Join-Path $runRoot 'summary.csv') -NoTypeInformation -Encoding UTF8
$summary | Format-Table | Out-String | Write-Output
