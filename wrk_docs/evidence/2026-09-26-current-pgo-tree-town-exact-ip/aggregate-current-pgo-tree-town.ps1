# Crucible: powershell -NoProfile -File .\.scratch\aggregate-current-pgo-tree-town.ps1 -Preflight
# Then run the same command without -Preflight. Aggregates each clip's three runs
# separately, using the exact PGO image and zero-loss checks from Ducks.
param(
    [ValidateSet('tree', 'town', 'both')][string]$Clip = 'both',
    [ValidateRange(1, 99)][int]$Count = 3,
    [ValidateRange(1, 99)][int]$StartAt = 1,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$diagnosticWorktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-capture-current-pgo-tree-and-town-exact'
$scratch = Join-Path $diagnosticWorktree '.scratch'
$aggregator = Join-Path $diagnosticWorktree 'wrk_docs\evidence\2026-09-26-pgo-nstep-ducks-exact-ip\aggregate-nstep-samples.py'
$outputs = @{
    tree = '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'
    town = '4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'
}

function Get-UniqueField([string]$Text, [string]$Key, [string]$Name) {
    $pattern = '(?m)^' + [regex]::Escape($Key) + '=([^\r\n]*)\r?$'
    $matches = [regex]::Matches($Text, $pattern)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Key field in $Name; found $($matches.Count)"
    }
    return $matches[0].Groups[1].Value
}

if ($env:OS -ne 'Windows_NT') { throw 'Run aggregation on Crucible Windows.' }
if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
if (-not (Test-Path -LiteralPath $aggregator -PathType Leaf)) {
    throw "Exact PGO aggregate script is missing: $aggregator"
}
[void](Get-Command -Name 'deltic' -ErrorAction Stop)
[void](Get-Command -Name 'python' -ErrorAction Stop)
$selected = if ($Clip -eq 'both') { @('tree', 'town') } else { @($Clip) }
$planned = @()
foreach ($name in $selected) {
    $root = Join-Path $scratch "pgo-current-$name-ip"
    $endAt = $StartAt + $Count - 1
    $out = Join-Path $root ('aggregate-run-{0:D2}-through-{1:D2}' -f $StartAt, $endAt)
    if (Test-Path -LiteralPath $out) { throw "Refusing existing aggregate: $out" }
    $argsList = @($aggregator, '--expected-runs', "$Count", '--start-at', "$StartAt")
    for ($number = $StartAt; $number -le $endAt; $number++) {
        $runName = 'run-{0:D2}' -f $number
        $run = Join-Path $root $runName
        $verification = Join-Path $run 'verification.txt'
        $extracted = Join-Path $run 'extracted-nstep-fixed'
        if (-not (Test-Path -LiteralPath $verification -PathType Leaf) -or
            -not (Test-Path -LiteralPath $extracted -PathType Container) -or
            -not (Test-Path -LiteralPath (Join-Path $extracted 'summary.json') -PathType Leaf)) {
            throw "Incomplete extracted run: $name/$runName"
        }
        $check = [IO.File]::ReadAllText($verification)
        if ((Get-UniqueField $check 'RUN' $verification) -ne $runName -or
            (Get-UniqueField $check 'OUTPUT_SHA256' $verification) -ne $outputs[$name] -or
            (Get-UniqueField $check 'LOST_BUFFERS' $verification) -ne '0' -or
            (Get-UniqueField $check 'LOST_EVENTS' $verification) -ne '0') {
            throw "Exact IVF or zero-loss proof failed: $name/$runName"
        }
        $argsList += @('--run', "$runName=$extracted")
    }
    $planned += [pscustomobject]@{ Clip = $name; Output = $out; Args = $argsList }
}
foreach ($item in $planned) {
    Write-Output "PREFLIGHT clip=$($item.Clip) expected_runs=$Count output=$($item.Output)"
}
if ($Preflight) {
    Write-Output 'PREFLIGHT_COMPLETE aggregation_started=false artifacts_created=false'
    return
}

foreach ($item in $planned) {
    New-Item -ItemType Directory -Path $item.Output -ErrorAction Stop | Out-Null
    $argsList = $item.Args + @('--out-dir', $item.Output)
    & deltic timeout 600 python @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "Exact PGO aggregate failed: $($item.Clip) exit=$LASTEXITCODE output=$($item.Output)"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $item.Output 'summary.json') -PathType Leaf)) {
        throw "Aggregator returned success without summary.json: $($item.Clip)"
    }
    Write-Output "AGGREGATE_OK clip=$($item.Clip) output=$($item.Output)"
}
