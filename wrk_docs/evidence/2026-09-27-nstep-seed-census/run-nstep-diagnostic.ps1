param([string]$RunLabel = 'nstep-diagnostic')
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s'
$exe = Join-Path $worktree '.scratch\nstep-census-target\release\rvp9enc.exe'
$expectedExe = '36f82f6b507c89b7ccad0bebf87536bd4d43630a76ee76af716fc295c1487f43'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clips = @(
 [pscustomobject]@{Name='ducks';File='ducks_take_off_420_1080p30_60f.y4m';Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7';Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
 [pscustomobject]@{Name='tree';File='in_to_tree_420_1080p30_60f.y4m';Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791';Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
 [pscustomobject]@{Name='town';File='old_town_cross_420_1080p30_60f.y4m';Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909';Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$actualExe=(Get-FileHash $exe -Algorithm SHA256).Hash.ToLowerInvariant()
if($actualExe-ne$expectedExe){throw "EXE hash mismatch: $actualExe"}
$runRoot=Join-Path (Join-Path $worktree '.scratch') $RunLabel
if(Test-Path $runRoot){throw "Refusing to overwrite $runRoot"}
New-Item -ItemType Directory $runRoot|Out-Null
$rows=@()
foreach($clip in $clips){
 $inputPath=Join-Path $inputRoot $clip.File
 $inputHash=(Get-FileHash $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
 if($inputHash-ne$clip.Input){throw "Input mismatch: $inputPath"}
 $outputPath=Join-Path $runRoot ($clip.Name+'.ivf')
 $args='--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$inputPath
 $psi=New-Object Diagnostics.ProcessStartInfo
 $psi.FileName=$exe;$psi.Arguments=$args;$psi.WorkingDirectory=$runRoot;$psi.UseShellExecute=$false;$psi.CreateNoWindow=$true;$psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true
 $proc=New-Object Diagnostics.Process;$proc.StartInfo=$psi
 if(-not $proc.Start()){throw "Could not start $($clip.Name)"}
 $stdout=$proc.StandardOutput.ReadToEndAsync();$stderr=$proc.StandardError.ReadToEndAsync()
 if(-not $proc.WaitForExit(90000)){$proc.Kill();throw "Timed out $($clip.Name)"}
 $text=$stderr.Result
 [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name+'.stderr.txt')),$text)
 [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name+'.stdout.txt')),$stdout.Result)
 $m=[regex]::Match($text,'(?m)^\[NSTEP_CENSUS\] nstep_calls=(\d+) diamond_entries=(\d+) repeated_restart_entries=(\d+) initial_seed_sad_evaluations=(\d+) num00_skipped_restarts=(\d+)\r?$')
 if(-not $m.Success){throw "Missing census: $($clip.Name): $text"}
 $calls=[long]$m.Groups[1].Value;$diamond=[long]$m.Groups[2].Value;$repeated=[long]$m.Groups[3].Value;$initial=[long]$m.Groups[4].Value;$skipped=[long]$m.Groups[5].Value
 if($initial-ne$diamond-or$repeated-gt$initial){throw "Inconsistent census: $($clip.Name)"}
 $outputHash=(Get-FileHash $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
 $rows += [pscustomobject]@{Clip=$clip.Name;NstepCalls=$calls;DiamondEntries=$diamond;RepeatedRestartEntries=$repeated;InitialSeedSadEvaluations=$initial;Num00SkippedRestarts=$skipped;ReusableSeedEvaluations=$repeated;RepeatedPerCall=[math]::Round($repeated/$calls,6);SkippedPerCall=[math]::Round($skipped/$calls,6);ExitCode=$proc.ExitCode;OutputSha256=$outputHash;ExpectedSha256=$clip.Output;OutputMatches=($outputHash-eq$clip.Output);InputSha256=$inputHash;ExeSha256=$actualExe}
 $rows|Export-Csv (Join-Path $runRoot 'census.csv') -NoTypeInformation -Encoding UTF8
 Write-Output "$($clip.Name): calls=$calls diamond=$diamond repeated=$repeated initial=$initial skipped=$skipped hash_ok=$($outputHash-eq$clip.Output)"
 if($proc.ExitCode-ne0-or$outputHash-ne$clip.Output){throw "Encode failed: $($clip.Name)"}
 $proc.Dispose()
}
