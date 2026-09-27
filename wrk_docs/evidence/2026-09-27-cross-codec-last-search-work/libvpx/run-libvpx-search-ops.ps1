$ErrorActionPreference = 'Stop'
$root = 'C:/Users/marti/AppData/Local/Temp/deltic-scratch/libvpx-search-ops-census-f50573b0'
$exe = Join-Path $root 'build/vpxenc.exe'
$inputPath = 'D:/worktrees/rvp9-2/acceptance-d639c265/target/encoder-pgo/inputs/ducks_take_off_420_1080p30_60f.y4m'
$out = Join-Path $root 'ducks-work.ivf'
$stderrPath = Join-Path $root 'ducks-work.stderr.txt'
$expectedExe = '421f91fb690f05291d74f9dfed0bd4c81a837ec346a88f502290c60530f8868c'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = 'b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8'
if (Test-Path -LiteralPath $out) { throw "Output already exists: $out" }
$exeHash = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
$inputHash = (Get-FileHash -LiteralPath $inputPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($exeHash -ne $expectedExe -or $inputHash -ne $expectedInput) { throw 'Asset hash mismatch' }
$args = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $out,$inputPath
$psi = New-Object Diagnostics.ProcessStartInfo
$psi.FileName=$exe; $psi.Arguments=$args; $psi.WorkingDirectory=$root
$psi.UseShellExecute=$false; $psi.CreateNoWindow=$true
$psi.RedirectStandardOutput=$true; $psi.RedirectStandardError=$true
$psi.EnvironmentVariables['RVP9_DIAG_LIBVPX_FULLPEL_CENSUS']='1'
$proc = New-Object Diagnostics.Process; $proc.StartInfo=$psi
if (-not $proc.Start()) { throw 'Could not start vpxenc' }
$stdoutTask=$proc.StandardOutput.ReadToEndAsync(); $stderrTask=$proc.StandardError.ReadToEndAsync()
if (-not $proc.WaitForExit(180000)) { $proc.Kill(); throw 'Encode timed out' }
$exit=$proc.ExitCode
[IO.File]::WriteAllText((Join-Path $root 'ducks-work.stdout.txt'),$stdoutTask.Result)
[IO.File]::WriteAllText($stderrPath,$stderrTask.Result)
$proc.Dispose()
$outputHash=(Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
if ($exit -ne 0 -or $outputHash -ne $expectedOutput) { throw "Encode/output failed: exit=$exit hash=$outputHash" }
$lines=@($stderrTask.Result -split "`r?`n" | Where-Object { $_ -match '^LIBVPX_FULLPEL_WORK,' })
if ($lines.Count -lt 2) { throw 'Missing work rows' }
[IO.File]::WriteAllLines((Join-Path $root 'ducks-work.csv'),[string[]]$lines)
$records=@($lines | ConvertFrom-Csv)
$last=@($records | Where-Object { $_.context -eq 'rt_pick' -and $_.ref -eq 'LAST' -and $_.method -eq 'NSTEP' })
$entries=($last | ForEach-Object { [long]$_.entries } | Measure-Object -Sum).Sum
if ($entries -ne 378013) { throw "Unexpected LAST entries: $entries" }
$identity=@("exe=$exeHash","input=$inputHash","output=$outputHash","last_nstep_entries=$entries","command=$exe $args")
[IO.File]::WriteAllLines((Join-Path $root 'ducks-work-identity.txt'),[string[]]$identity)
$identity | ForEach-Object { Write-Output $_ }
$last | Format-Table context,shape,method,ref,entries,diamond_calls,diamond_steps,all_in_steps,diamond_sdx4df_calls,diamond_scalar_sdf_calls,variance_calls,refine_calls | Out-String | Write-Output
