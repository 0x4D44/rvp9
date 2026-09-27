$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-chroma-energy-census'
$exe = Join-Path $worktree 'target\release\rvp9enc.exe'
$source = Join-Path $worktree 'crates\rvp9-encoder\src\lib.rs'
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
$runRoot = Join-Path $worktree '.scratch\energy-census-60f'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash.ToLowerInvariant() -ne '90c805fd7ac27fba3cfbcfbc5fad8751265cdb198a8cd8afb28256112624a85c') { throw 'Diagnostic source hash mismatch' }
if (Test-Path -LiteralPath $runRoot) { throw "Run directory already exists: $runRoot" }
New-Item -ItemType Directory -Path $runRoot | Out-Null
$clips = @(
    [pscustomobject]@{ Name='ducks'; File='ducks_take_off_420_1080p30_60f.y4m'; Input='6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'; Output='8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' },
    [pscustomobject]@{ Name='tree'; File='in_to_tree_420_1080p30_60f.y4m'; Input='ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'; Output='5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' },
    [pscustomobject]@{ Name='town'; File='old_town_cross_420_1080p30_60f.y4m'; Input='b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'; Output='4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f' }
)
$exeHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
$rows = @()
foreach ($clip in $clips) {
    $inputPath = Join-Path $inputRoot $clip.File
    $inputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant()
    if ($inputHash -ne $clip.Input) { throw "Input hash mismatch: $($clip.Name)" }
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
    $psi.EnvironmentVariables['RVP9_DIAG_CHROMA_ENERGY'] = '1'
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Could not start $($clip.Name)" }
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit(90000)) { $proc.Kill(); throw "Timed out $($clip.Name)" }
    $exit = $proc.ExitCode
    [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name + '.stdout.txt')), $stdout.Result)
    [IO.File]::WriteAllText((Join-Path $runRoot ($clip.Name + '.stderr.txt')), $stderr.Result)
    $outputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
    $rows += [pscustomobject]@{ Clip=$clip.Name; ExitCode=$exit; InputSha256=$inputHash; OutputSha256=$outputHash; ExpectedSha256=$clip.Output; ExeSha256=$exeHash; DiagnosticLines=([regex]::Matches($stderr.Result, '\[CHROMA_ENERGY\]')).Count }
    $rows | Export-Csv -LiteralPath (Join-Path $runRoot 'identity.csv') -NoTypeInformation -Encoding UTF8
    Write-Output ("{0}: exit={1} output={2} diagnostic_lines={3}" -f $clip.Name,$exit,$outputHash,$rows[-1].DiagnosticLines)
    $proc.Dispose()
    if ($exit -ne 0 -or $outputHash -ne $clip.Output) { throw "Encode failed or output mismatch: $($clip.Name)" }
}
