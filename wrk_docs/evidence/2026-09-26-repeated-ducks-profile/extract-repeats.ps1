# Extract already-verified numbered Ducks ETLs with their exact PE/PDB pair.
# Run from Crucible: powershell -NoProfile -File .\extract-repeats.ps1 -StartAt 2 -Count 9
param(
    [ValidateRange(1, 99)][int]$StartAt = 1,
    [ValidateRange(1, 99)][int]$Count = 10
)
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob'
$root = Join-Path $worktree '.scratch\repeated-ducks-ip'
$exe = Join-Path $worktree '.scratch\target\release\rvp9enc.exe'
$pdb = Join-Path $worktree '.scratch\target\release\rvp9enc.pdb'
$extractor = Join-Path $worktree 'wrk_docs\evidence\2026-09-25-unit1434-three-clip-ip-attribution\extract-ip.py'
$exeHash = 'c4b24b2c5b16add984a2dd6d32e5a7080037290e58b254aff838a0dcf26d63c1'
$pdbHash = '085c7c7cc7f724ed7674797dc88bf27011e6662d27d7661e9758677004bd8f9c'
if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
foreach ($number in $StartAt..($StartAt + $Count - 1)) {
    $name = 'run-{0:D2}' -f $number
    $run = Join-Path $root $name
    $verification = Join-Path $run 'verification.txt'
    $captureLog = Join-Path $run 'capture.log'
    $etl = Join-Path $run 'capture.etl'
    $out = Join-Path $run 'extracted'
    if (-not (Test-Path -LiteralPath $verification -PathType Leaf) -or
        -not (Test-Path -LiteralPath $captureLog -PathType Leaf) -or
        -not (Test-Path -LiteralPath $etl -PathType Leaf)) {
        throw "Incomplete capture: $name"
    }
    if (Test-Path -LiteralPath $out) { throw "Refusing to overwrite extraction: $out" }
    $check = [IO.File]::ReadAllText($verification)
    if ($check -notmatch '(?m)^LOST_BUFFERS=0\r?$' -or
        $check -notmatch '(?m)^LOST_EVENTS=0\r?$') {
        throw "Capture has no verified zero-loss record: $name"
    }
    $log = [IO.File]::ReadAllText($captureLog)
    $matches = [regex]::Matches($log, '(?m)^ENCODER_PROCESS_ID=(\d+)\r?$')
    if ($matches.Count -ne 1) { throw "Missing or duplicate encoder PID: $name" }
    $pidNumber = [int]$matches[0].Groups[1].Value
    & deltic timeout 600 python $extractor --etl $etl --exe $exe --pdb $pdb `
        --expected-exe-sha256 $exeHash --expected-pdb-sha256 $pdbHash `
        --pid $pidNumber --outdir $out
    if ($LASTEXITCODE -ne 0) { throw "Exact ETL extraction failed: $name exit=$LASTEXITCODE" }
    Write-Output "EXTRACT_OK run=$name pid=$pidNumber out=$out"
}
