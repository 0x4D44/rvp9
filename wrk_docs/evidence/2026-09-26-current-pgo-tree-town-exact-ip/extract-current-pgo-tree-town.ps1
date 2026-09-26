# Crucible: powershell -NoProfile -File .\.scratch\extract-current-pgo-tree-town.ps1 -Preflight
# Then run the same command without -Preflight. Defaults: three runs per clip.
param(
    [ValidateSet('tree', 'town', 'both')][string]$Clip = 'both',
    [ValidateRange(1, 99)][int]$Count = 3,
    [ValidateRange(1, 99)][int]$StartAt = 1,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$diagnosticWorktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-capture-current-pgo-tree-and-town-exact'
$imageWorktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score'
$scratch = Join-Path $diagnosticWorktree '.scratch'
$exe = Join-Path $imageWorktree '.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$pdb = Join-Path $imageWorktree '.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.pdb'
$extractor = Join-Path $imageWorktree '.scratch\extract-ip-nstep.py'
$exeHash = '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee'
$pdbHash = '7672ad045a634a9278d7b9255e1a0b665e9e729f6a480438e9b22c50d330bc13'
$outputs = @{
    tree = '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'
    town = '4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'
}

function Assert-Hash([string]$Name, [string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw ("Missing {0}: {1}" -f $Name, $Path)
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected) {
        throw ("{0} SHA-256 mismatch: expected={1} actual={2} path={3}" -f $Name, $Expected, $actual, $Path)
    }
}

function Get-UniqueField([string]$Text, [string]$Key, [string]$Name) {
    $pattern = '(?m)^' + [regex]::Escape($Key) + '=([^\r\n]*)\r?$'
    $matches = [regex]::Matches($Text, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Key field in $Name; found $($matches.Count)"
    }
    return $matches[0].Groups[1].Value
}

if ($env:OS -ne 'Windows_NT') { throw 'Run extraction on Crucible Windows.' }
if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
Assert-Hash 'PGO EXE' $exe $exeHash
Assert-Hash 'PGO PDB' $pdb $pdbHash
if (-not (Test-Path -LiteralPath $extractor -PathType Leaf)) {
    throw "Exact PE/PDB extractor is missing: $extractor"
}
[void](Get-Command -Name 'deltic' -ErrorAction Stop)
[void](Get-Command -Name 'python' -ErrorAction Stop)
$selected = if ($Clip -eq 'both') { @('tree', 'town') } else { @($Clip) }
$planned = @()
foreach ($name in $selected) {
    $root = Join-Path $scratch "pgo-current-$name-ip"
    for ($number = $StartAt; $number -lt $StartAt + $Count; $number++) {
        $runName = 'run-{0:D2}' -f $number
        $run = Join-Path $root $runName
        $verification = Join-Path $run 'verification.txt'
        $captureLog = Join-Path $run 'capture.log'
        $etl = Join-Path $run 'capture.etl'
        $out = Join-Path $run 'extracted-nstep-fixed'
        if (-not (Test-Path -LiteralPath $verification -PathType Leaf) -or
            -not (Test-Path -LiteralPath $captureLog -PathType Leaf) -or
            -not (Test-Path -LiteralPath $etl -PathType Leaf)) {
            throw "Incomplete capture: $name/$runName"
        }
        if (Test-Path -LiteralPath $out) {
            throw "Refusing to overwrite extraction: $out"
        }
        $check = [IO.File]::ReadAllText($verification)
        if ((Get-UniqueField $check 'RUN' $verification) -ne $runName -or
            (Get-UniqueField $check 'OUTPUT_SHA256' $verification) -ne $outputs[$name] -or
            (Get-UniqueField $check 'LOST_BUFFERS' $verification) -ne '0' -or
            (Get-UniqueField $check 'LOST_EVENTS' $verification) -ne '0') {
            throw "Verification identity, exact IVF, or zero-loss proof failed: $name/$runName"
        }
        $etlHash = (Get-FileHash -LiteralPath $etl -Algorithm SHA256).Hash.ToLowerInvariant()
        if ((Get-UniqueField $check 'ETL_SHA256' $verification) -ne $etlHash -or
            (Get-Item -LiteralPath $etl).Length -le 0) {
            throw "ETL hash or size mismatch: $name/$runName"
        }
        $log = [IO.File]::ReadAllText($captureLog)
        if ((Get-UniqueField $log 'CAPTURE_SUCCESS' $captureLog) -ne 'True' -or
            (Get-UniqueField $log 'OUTPUT_SHA256' $captureLog) -ne $outputs[$name] -or
            (Get-UniqueField $log 'ETL_SHA256' $captureLog) -ne $etlHash -or
            (Get-UniqueField $log 'CAPTURE_CLIP' $captureLog) -ne $name) {
            throw "Capture log identity failed: $name/$runName"
        }
        $pidText = Get-UniqueField $log 'ENCODER_PROCESS_ID' $captureLog
        $pidNumber = 0
        if (-not [int]::TryParse($pidText, [ref]$pidNumber) -or $pidNumber -le 0) {
            throw "Missing or invalid encoder PID: $name/$runName"
        }
        $planned += [pscustomobject]@{
            Clip = $name
            Run = $runName
            Pid = $pidNumber
            Etl = $etl
            Out = $out
        }
    }
}
foreach ($item in $planned) {
    Write-Output "PREFLIGHT clip=$($item.Clip) run=$($item.Run) pid=$($item.Pid) etl=$($item.Etl) out=$($item.Out)"
}
if ($Preflight) {
    Write-Output 'PREFLIGHT_COMPLETE extraction_started=false artifacts_created=false'
    return
}

foreach ($item in $planned) {
    & deltic timeout 600 python $extractor --etl $item.Etl --exe $exe --pdb $pdb --expected-exe-sha256 $exeHash --expected-pdb-sha256 $pdbHash --pid $item.Pid --outdir $item.Out
    if ($LASTEXITCODE -ne 0) {
        throw "Exact PE/PDB extraction failed: $($item.Clip)/$($item.Run) exit=$LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $item.Out 'summary.json') -PathType Leaf)) {
        throw "Extractor returned success without summary.json: $($item.Clip)/$($item.Run)"
    }
    Write-Output "EXTRACT_OK clip=$($item.Clip) run=$($item.Run) pid=$($item.Pid) out=$($item.Out)"
}
