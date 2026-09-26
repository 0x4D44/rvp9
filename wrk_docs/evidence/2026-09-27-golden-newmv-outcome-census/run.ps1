$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-CDX@CRUCIBLE-verify-full-pel-census-crucible'
$artifact = Join-Path $worktree '.scratch\golden-outcome'
$target = Join-Path $worktree '.scratch\target'
$input = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs\ducks_take_off_420_1080p30_60f.y4m'
$expectedInput = '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7'
$expectedOutput = '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089'
$expectedTree = 'f5e1e21565de192c378218322f517f51c77d66a5'
$expectedFresh = 236166
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

function Parse-Fields([string]$line) {
    $fields = @{}
    foreach ($word in $line.Split(' ')) {
        $pair = $word.Split('=')
        if ($pair.Length -eq 2) { $fields[$pair[0]] = $pair[1] }
    }
    return $fields
}

$head = (git -C $worktree rev-parse HEAD).Trim()
$tree = (git -C $worktree show --format=%T --no-patch HEAD).Trim()
if ($tree -ne $expectedTree) { throw "Source tree mismatch: $tree" }
$status = @(git -C $worktree status --porcelain)
if ($status.Count -ne 0) { throw "Task worktree is dirty: $($status -join '; ')" }
$inputHash = Sha256 $input
if ($inputHash -ne $expectedInput) { throw "Ducks input hash mismatch: $inputHash" }

$test1 = 'cargo test --locked --release -p rvp9-core full_pel_census_reconciles_search_and_seed_calls -- --ignored'
$test2 = 'cargo test --locked --release -p rvp9-core golden_outcome_report_reconciles_exclusive_buckets_and_reuse -- --ignored'
Run-Bounded 'census-test' 900 $test1 @{RVP9_FULLPEL_CENSUS='1'}
Run-Bounded 'outcome-test' 900 $test2 @{RVP9_FULLPEL_CENSUS='1'}

$buildCommand = 'cargo build --locked --release -p rvp9-cli --bin rvp9enc'
Run-Bounded 'release-build' 1800 $buildCommand @{}
$exe = Join-Path $target 'release\rvp9enc.exe'
$exeHash = Sha256 $exe

$ivf = Join-Path $artifact 'ducks-golden-outcome.ivf'
$encodeFlags = '--codec=vp9 --passes=1 --rt --cpu-used=6 --end-usage=cbr --target-bitrate=1260 --fps=30/1 --threads=1 --tile-columns=0 --lag-in-frames=0 --tune-content=default --aq-mode=0 --noise-sensitivity=0 --limit=60 --kf-max-dist=60 --quiet --ivf'
$encodeCommand = ('"{0}" {1} -o "{2}" "{3}"' -f $exe,$encodeFlags,$ivf,$input)
Run-Bounded 'ducks-encode' 900 $encodeCommand @{RVP9_FULLPEL_CENSUS='1'}
$outputHash = Sha256 $ivf

$lines = [System.IO.File]::ReadAllLines((Join-Path $artifact 'ducks-encode.stderr.txt'))
$frames = New-Object 'System.Collections.Generic.List[object]'
for ($i = 0; $i -lt $lines.Length; $i++) {
    if ($lines[$i] -ne '[FULLPEL_CENSUS] frame totals') { continue }
    if ($i + 1 -ge $lines.Length -or -not $lines[$i + 1].StartsWith('golden_newmv_fresh=')) {
        throw "Missing outcome line after frame header at stderr line $($i + 1)"
    }
    $fields = Parse-Fields $lines[$i + 1]
    $row = [ordered]@{ frame = $frames.Count + 1 }
    foreach ($key in @('golden_newmv_fresh','fullpel_entries','search_rejected','deduplicated','inter_lost','intra_replaced','chosen','terminal_total','cached_full_picks','golden_override_consumed')) {
        $row[$key] = [long]$fields[$key]
    }
    $row['reconciled'] = $fields['reconciled']
    $row['matches_entries'] = $fields['matches_entries']
    $frames.Add([pscustomobject]$row)
}
$frames | Export-Csv -LiteralPath (Join-Path $artifact 'golden-outcomes-per-frame.csv') -NoTypeInformation -Encoding UTF8
$keys = @('golden_newmv_fresh','fullpel_entries','search_rejected','deduplicated','inter_lost','intra_replaced','chosen','terminal_total','cached_full_picks','golden_override_consumed')
$aggregate = [ordered]@{}
foreach ($key in $keys) {
    $sum = [long]0
    foreach ($row in $frames) { $sum += $row.$key }
    $aggregate[$key] = $sum
}
[pscustomobject]$aggregate | Export-Csv -LiteralPath (Join-Path $artifact 'golden-outcomes-aggregate.csv') -NoTypeInformation -Encoding UTF8

$identity = @(
    "host=CRUCIBLE",
    "remote_worktree=$worktree",
    "commit=$head",
    "tree=$tree",
    "source_equivalent_to=fde00bf307e82608dd20c42ca197e2a713209c55",
    "rustc=$(rustc --version)",
    "cargo_target=$target",
    "input=$input",
    "input_sha256=$inputHash",
    "exe=$exe",
    "exe_sha256=$exeHash",
    "output=$ivf",
    "output_sha256=$outputHash",
    "expected_output_sha256=$expectedOutput",
    "census_inter_frames=$($frames.Count)",
    "census_test=deltic timeout 900 $test1",
    "outcome_test=deltic timeout 900 $test2",
    "build=deltic timeout 1800 $buildCommand",
    "encode=deltic timeout 900 $encodeCommand",
    'env=RVP9_FULLPEL_CENSUS=1'
)
[System.IO.File]::WriteAllLines((Join-Path $artifact 'identity.txt'), $identity, $utf8)
Write-Output "Ducks output SHA256=$outputHash census_frames=$($frames.Count)"
[pscustomobject]$aggregate | Format-List | Out-String | Write-Output
if ($outputHash -ne $expectedOutput) { throw "Ducks IVF hash mismatch: $outputHash" }
if ($frames.Count -ne 59) { throw "Expected 59 inter-frame census summaries, got $($frames.Count)" }
foreach ($row in $frames) {
    if ($row.reconciled -ne 'true' -or $row.matches_entries -ne 'true' -or
        $row.golden_newmv_fresh -ne $row.fullpel_entries -or $row.golden_newmv_fresh -ne $row.terminal_total) {
        throw "Outcome mismatch in inter frame $($row.frame)"
    }
}
if ($aggregate['golden_newmv_fresh'] -ne $expectedFresh) {
    throw "Expected $expectedFresh fresh GOLDEN searches, got $($aggregate['golden_newmv_fresh'])"
}
