$ErrorActionPreference = 'Stop'
$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-count-subpel-second-level-mv-cost-reject'
$exe = Join-Path $root 'target\release\rvp9enc.exe'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$outputRoot = Join-Path $root '.scratch\rate-census'
if (Test-Path -LiteralPath $outputRoot) { throw "Output root exists: $outputRoot" }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$clips = @(
    @{Name='ducks';File='ducks_take_off_420_1080p30_60f.y4m';Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7';Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    @{Name='tree';File='in_to_tree_420_1080p30_60f.y4m';Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791';Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    @{Name='town';File='old_town_cross_420_1080p30_60f.y4m';Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909';Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$rows = @()
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    if ((Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $clip.Input) { throw "Input mismatch: $($clip.Name)" }
    $out = Join-Path $outputRoot ($clip.Name + '.ivf')
    $args = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $out,$inputPath
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName=$exe; $psi.Arguments=$args; $psi.WorkingDirectory=$outputRoot; $psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
    $psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
    $psi.EnvironmentVariables['RVP9_DIAG_SUBPEL_RATE']='1'
    $proc = New-Object System.Diagnostics.Process; $proc.StartInfo=$psi
    if (-not $proc.Start()) { throw "Could not start $($clip.Name)" }
    $stdout=$proc.StandardOutput.ReadToEndAsync(); $stderr=$proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "Timed out $($clip.Name)" }
    [IO.File]::WriteAllText((Join-Path $outputRoot ($clip.Name + '.stderr.txt')), $stderr.Result)
    $hash = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($proc.ExitCode -ne 0 -or $hash -ne $clip.Output) { throw "Encode mismatch: $($clip.Name), exit=$($proc.ExitCode), hash=$hash" }
    foreach ($line in ($stderr.Result -split "`r?`n")) {
        if ($line -match '^subpel_rate block=(\d+) attempts=(\d+) valid=(\d+) rate_rejects=(\d+)$') {
            $rows += [pscustomobject]@{Clip=$clip.Name;Block=[int]$Matches[1];Attempts=[long]$Matches[2];Valid=[long]$Matches[3];RateRejects=[long]$Matches[4]}
        }
    }
    Write-Output "$($clip.Name) hash=$hash rows=$(($rows | Where-Object Clip -eq $clip.Name).Count)"
    $proc.Dispose()
}
$rows | Export-Csv -LiteralPath (Join-Path $outputRoot 'census.csv') -NoTypeInformation -Encoding UTF8
$rows | Group-Object Clip | ForEach-Object {
    $a=($_.Group | Measure-Object Attempts -Sum).Sum
    $v=($_.Group | Measure-Object Valid -Sum).Sum
    $r=($_.Group | Measure-Object RateRejects -Sum).Sum
    Write-Output "$($_.Name) attempts=$a valid=$v rate_rejects=$r fraction=$($r/$v)"
}
