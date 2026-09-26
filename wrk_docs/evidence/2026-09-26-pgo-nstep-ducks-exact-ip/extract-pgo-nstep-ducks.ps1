# Extract already-verified numbered Ducks ETLs with their exact PE/PDB pair.
# Run from Crucible: powershell -NoProfile -File .\extract-repeats.ps1 -StartAt 2 -Count 9
param(
    [ValidateRange(1, 99)][int]$StartAt = 1,
    [ValidateRange(1, 99)][int]$Count = 10
)
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score'
$root = Join-Path $worktree '.scratch\pgo-nstep-ducks-ip'
$exe = Join-Path $worktree '.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$pdb = Join-Path $worktree '.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.pdb'
$extractor = Join-Path $worktree '.scratch\extract-ip-nstep.py'
$exeHash = '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee'
$pdbHash = '7672ad045a634a9278d7b9255e1a0b665e9e729f6a480438e9b22c50d330bc13'
if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
foreach ($number in $StartAt..($StartAt + $Count - 1)) {
    $name = 'run-{0:D2}' -f $number
    $run = Join-Path $root $name
    $verification = Join-Path $run 'verification.txt'
    $captureLog = Join-Path $run 'capture.log'
    $etl = Join-Path $run 'capture.etl'
    $out = Join-Path $run 'extracted-nstep-fixed'
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
