$ErrorActionPreference = 'Stop'

$root = 'D:\worktrees\rvp9-2\20260926-TSK-CDX@CRUCIBLE-libvpx-fullpel-entry-census-x86'
$exe = Join-Path $root '.scratch\libvpx-fullpel-build-relative\vpxenc.exe'
$patch = Join-Path $root '.scratch\libvpx-fullpel-entry-census.patch'
$inputPath = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$outDir = Join-Path $root '.scratch\libvpx-fullpel-ducks'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = 'b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8'
$expectedPatch = 'dd5df8c9c7de79d3f3e8257032a135272551d786c4d1de79da0aecebd74343ff'

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $inputPath).Hash.ToLowerInvariant() -ne $expectedInput) {
    throw 'Ducks input hash mismatch'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $patch).Hash.ToLowerInvariant() -ne $expectedPatch) {
    throw 'Patch hash mismatch'
}
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing binary: $exe" }
if (Test-Path -LiteralPath $outDir) { throw "Refusing to overwrite: $outDir" }
[void](New-Item -ItemType Directory -Path $outDir)

$outputPath = Join-Path $outDir 'ducks.ivf'
$arguments = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf -o "{0}" "{1}"' -f $outputPath,$inputPath
$psi = New-Object Diagnostics.ProcessStartInfo
$psi.FileName = $exe
$psi.Arguments = $arguments
$psi.WorkingDirectory = $outDir
$psi.UseShellExecute = $false
$psi.CreateNoWindow = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.EnvironmentVariables['RVP9_DIAG_LIBVPX_FULLPEL_CENSUS'] = '1'
$process = New-Object Diagnostics.Process
$process.StartInfo = $psi
try {
    if (-not $process.Start()) { throw 'Could not start vpxenc' }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    if (-not $process.WaitForExit(180000)) {
        $process.Kill()
        throw 'Ducks encode timed out'
    }
    $utf8 = [Text.UTF8Encoding]::new($false)
    [IO.File]::WriteAllText((Join-Path $outDir 'ducks.stdout.txt'), $stdoutTask.Result, $utf8)
    [IO.File]::WriteAllText((Join-Path $outDir 'ducks.stderr.txt'), $stderrTask.Result, $utf8)
    if ($process.ExitCode -ne 0) { throw "vpxenc exit code $($process.ExitCode)" }
} finally {
    $process.Dispose()
}

$censusLines = @($stderrTask.Result -split "`r?`n" | Where-Object { $_ -match '^LIBVPX_FULLPEL_ENTRY,' })
$header = 'LIBVPX_FULLPEL_ENTRY,context,shape,method,ref,calls'
if ($censusLines.Count -lt 2 -or $censusLines[0] -ne $header) { throw 'Missing full-pel census rows' }
[IO.File]::WriteAllLines((Join-Path $outDir 'ducks.fullpel.csv'), [string[]]$censusLines, $utf8)
$counts = @{}
$total = [long]0
foreach ($line in ($censusLines | Select-Object -Skip 1)) {
    $fields = $line.Split(',')
    if ($fields.Count -ne 6) { throw "Malformed census row: $line" }
    $count = [long]::Parse($fields[5])
    $counts[$fields[1]] = [long]$counts[$fields[1]] + $count
    $total += $count
}

$outputHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
$binaryHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.ToLowerInvariant()
$manifest = @(
    "source=12f3a2ac603e8f10742105519e0cd03c3b8f71dd"
    "patch_sha256=$expectedPatch"
    "binary_sha256=$binaryHash"
    "input_sha256=$expectedInput"
    "output_sha256=$outputHash"
    "expected_output_sha256=$expectedOutput"
    "rows=$($censusLines.Count - 1)"
    "total=$total"
    "other=$([long]$counts['other'])"
    "rt_pick=$([long]$counts['rt_pick'])"
    "command=$exe $arguments"
)
[IO.File]::WriteAllLines((Join-Path $outDir 'manifest.txt'), [string[]]$manifest, $utf8)
if ($outputHash -ne $expectedOutput) { throw 'Ducks output hash mismatch' }
$manifest | ForEach-Object { Write-Output $_ }
