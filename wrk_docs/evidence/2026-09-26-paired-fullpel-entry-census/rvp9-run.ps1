$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-CDX@CRUCIBLE-verify-full-pel-census-crucible'
$artifact = Join-Path $worktree '.scratch\fullpel-census'
$target = Join-Path $worktree '.scratch\target'
$input = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'
$expectedTree = '3090f9649d6420c0ea914aea79e3a814258a1853'
$utf8 = New-Object System.Text.UTF8Encoding($false)
New-Item -ItemType Directory -Force -Path $artifact | Out-Null

function Sha256([string]$path) {
    return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Run-Bounded([string]$name, [int]$seconds, [string]$command, [hashtable]$extraEnv) {
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Get-Command deltic).Source
    $psi.Arguments = "timeout $seconds $command"
    $psi.WorkingDirectory = $worktree
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['CARGO_TARGET_DIR'] = $target
    foreach ($key in $extraEnv.Keys) { $psi.EnvironmentVariables[$key] = $extraEnv[$key] }
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    if (-not $proc.Start()) { throw "Could not start $name" }
    $proc.StandardInput.Close()
    $stdout = $proc.StandardOutput.ReadToEndAsync()
    $stderr = $proc.StandardError.ReadToEndAsync()
    $finished = $proc.WaitForExit(($seconds + 30) * 1000)
    if (-not $finished) { $proc.Kill(); $proc.WaitForExit() }
    [System.IO.File]::WriteAllText((Join-Path $artifact "$name.stdout.txt"), $stdout.Result, $utf8)
    [System.IO.File]::WriteAllText((Join-Path $artifact "$name.stderr.txt"), $stderr.Result, $utf8)
    $code = $proc.ExitCode
    $proc.Dispose()
    Write-Output "$name exit=$code command=deltic timeout $seconds $command"
    if (-not $finished -or $code -ne 0) { throw "$name failed; inspect $artifact" }
}

$head = (git -C $worktree rev-parse HEAD).Trim()
$tree = (git -C $worktree show --format=%T --no-patch HEAD).Trim()
if ($tree -ne $expectedTree) { throw "Source tree mismatch: $tree" }
$status = @(git -C $worktree status --porcelain)
if ($status.Count -ne 0) { throw "Task worktree is dirty: $($status -join '; ')" }
$inputHash = Sha256 $input
if ($inputHash -ne $expectedInput) { throw "Ducks input hash mismatch: $inputHash" }

$testCommand = 'cargo test --locked --release -p rvp9-core full_pel_census_reconciles_search_and_seed_calls -- --ignored'
Run-Bounded 'focused-test' 900 $testCommand @{RVP9_FULLPEL_CENSUS='1'}

$buildCommand = 'cargo build --locked --release -p rvp9-cli --bin rvp9enc'
Run-Bounded 'release-build' 1800 $buildCommand @{}
$exe = Join-Path $target 'release\rvp9enc.exe'
$exeHash = Sha256 $exe

$ivf = Join-Path $artifact 'ducks-census.ivf'
$encodeFlags = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf'
$encodeCommand = ('"{0}" {1} -o "{2}" "{3}"' -f $exe,$encodeFlags,$ivf,$input)
Run-Bounded 'ducks-encode' 900 $encodeCommand @{RVP9_FULLPEL_CENSUS='1'}
$outputHash = Sha256 $ivf
if ($outputHash -ne $expectedOutput) { throw "Ducks IVF hash mismatch: $outputHash" }

$lines = [System.IO.File]::ReadAllLines((Join-Path $artifact 'ducks-encode.stderr.txt'))
$frames = @($lines | Where-Object { $_ -eq '[FULLPEL_CENSUS] frame totals' }).Count
if ($frames -ne 59) { throw "Expected 59 inter-frame census summaries, got $frames" }
$fieldsToSum = @('entries','diamond_starts','local_seed_full_sads','sad4_batches','scalar_sads','refinements','variance_evaluations')
$totals = @{}
foreach ($line in $lines) {
    if (-not $line.StartsWith('origin=')) { continue }
    $fields = @{}
    foreach ($word in $line.Split(' ')) {
        $pair = $word.Split('=')
        if ($pair.Length -eq 2) { $fields[$pair[0]] = $pair[1] }
    }
    $key = @($fields.origin,$fields.shape,$fields.ref,$fields.method) -join '|'
    if (-not $totals.ContainsKey($key)) {
        $row = [ordered]@{origin=$fields.origin; shape=$fields.shape; ref=$fields.ref; method=$fields.method}
        foreach ($field in $fieldsToSum) { $row[$field] = [long]0 }
        $totals[$key] = $row
    }
    foreach ($field in $fieldsToSum) { $totals[$key][$field] += [long]$fields[$field] }
}
$rows = @($totals.Values | ForEach-Object { [pscustomobject]$_ } | Sort-Object origin,shape,ref,method)
$rows | Export-Csv -LiteralPath (Join-Path $artifact 'totals.csv') -NoTypeInformation -Encoding UTF8

$identity = @(
    "host=CRUCIBLE",
    "remote_worktree=$worktree",
    "commit=$head",
    "tree=$tree",
    "rustc=$(rustc --version)",
    "cargo_target=$target",
    "input=$input",
    "input_sha256=$inputHash",
    "exe=$exe",
    "exe_sha256=$exeHash",
    "output=$ivf",
    "output_sha256=$outputHash",
    "expected_output_sha256=$expectedOutput",
    "census_inter_frames=$frames",
    "test=deltic timeout 900 $testCommand",
    "build=deltic timeout 1800 $buildCommand",
    "encode=deltic timeout 900 $encodeCommand",
    'env=RVP9_FULLPEL_CENSUS=1'
)
[System.IO.File]::WriteAllLines((Join-Path $artifact 'identity.txt'), $identity, $utf8)
Write-Output "Ducks output SHA256=$outputHash census_frames=$frames totals_rows=$($rows.Count)"
$rows | Format-Table -AutoSize | Out-String | Write-Output
