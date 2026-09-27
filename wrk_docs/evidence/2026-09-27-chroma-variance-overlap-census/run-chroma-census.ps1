$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo 'target\release\rvp9enc.exe'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$clips = @(
    [pscustomobject]@{Name='ducks';File='ducks_take_off_420_1080p30_60f.y4m';Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7';Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'},
    [pscustomobject]@{Name='tree';File='in_to_tree_420_1080p30_60f.y4m';Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791';Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'},
    [pscustomobject]@{Name='town';File='old_town_cross_420_1080p30_60f.y4m';Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909';Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'}
)
$utf8 = New-Object System.Text.UTF8Encoding($false)
$rows = New-Object 'System.Collections.Generic.List[object]'
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    $inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($inputHash -ne $clip.Input) { throw "$($clip.Name) input hash mismatch: $inputHash" }
    $out = Join-Path $PSScriptRoot "$($clip.Name).ivf"
    $args = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $out,$inputPath
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $exe
    $psi.Arguments = $args
    $psi.WorkingDirectory = $repo
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['RVP9_DIAG_CHROMA_VARIANCE_OVERLAP'] = '1'
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Could not start $($clip.Name)" }
    $proc.StandardInput.Close()
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "$($clip.Name) timed out" }
    $exitCode = $proc.ExitCode
    $proc.Dispose()
    [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "$($clip.Name).stdout.txt"), $stdout.Result, $utf8)
    [System.IO.File]::WriteAllText((Join-Path $PSScriptRoot "$($clip.Name).stderr.txt"), $stderr.Result, $utf8)
    $outputHash = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($exitCode -ne 0 -or $outputHash -ne $clip.Output) { throw "$($clip.Name) exit=$exitCode hash=$outputHash" }
    foreach ($line in $stderr.Result.Split([Environment]::NewLine)) {
        if (-not $line.StartsWith('chroma_variance_overlap ')) { continue }
        $fields = @{}
        foreach ($part in $line.Split(' ')) {
            $pair = $part.Split('=')
            if ($pair.Length -eq 2) { $fields[$pair[0]] = $pair[1] }
        }
        $rows.Add([pscustomobject]@{Clip=$clip.Name;BlockSize=[int]$fields['block_size'];ProbeCalls=[long]$fields['probe_calls'];VetoU=[long]$fields['veto_u'];VetoV=[long]$fields['veto_v'];AllSkippable=[long]$fields['all_skippable'];DuplicatedU=[long]$fields['duplicated_u'];DuplicatedV=[long]$fields['duplicated_v'];DuplicatedPixels=[long]$fields['duplicated_pixels']})
    }
    Write-Output "$($clip.Name) exact output hash=$outputHash"
}
$rows | Export-Csv -LiteralPath (Join-Path $PSScriptRoot 'counts.csv') -NoTypeInformation -Encoding UTF8
@(
    "commit=$(git -C $repo rev-parse HEAD)",
    "exe_sha256=$((Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant())",
    'options=CPU6 CBR1260 1080p60f threads1 tiles0 aq0 noise0 lag0 kf60',
    'env=RVP9_DIAG_CHROMA_VARIANCE_OVERLAP=1'
) | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'identity.txt') -Encoding UTF8
$rows | Format-Table | Out-String | Write-Output
