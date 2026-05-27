# bench-lf-overlap.ps1 — Phase 1 separate-LF-pool overlap A/B
#
# Per HLD `2026.05.27 - HLD - Separate LF thread pool.md` §11
# condition 4: interleaved drift-cancelling A/B on
# `testsrc_1080p30 t=4` + sanity rows. ≥8 iters/binary; report
# median, p50, stdev.
#
# Variant A (baseline): PGO-built without RVP9_LF_OVERLAP_POOL
#                       (target/release/decode-headtohead-baseline.exe)
# Variant B (candidate): PGO-built with RVP9_LF_OVERLAP_POOL
#                       (target/release/decode-headtohead-candidate.exe).
#                       Run with the env var set.
#
# Interleave A B A B ... so any host drift (thermal, background
# load) hits both variants equally.

param(
    [int] $Iters = 8,
    [switch] $SkipBuild
)

$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel).Trim()

$baselineExe = Join-Path $repo 'target/release/decode-headtohead-baseline.exe'
$candidateExe = Join-Path $repo 'target/release/decode-headtohead-candidate.exe'

if (-not (Test-Path $baselineExe)) {
    throw "Missing $baselineExe -- build PGO baseline first."
}
if (-not (Test-Path $candidateExe)) {
    throw "Missing $candidateExe -- build PGO candidate first."
}

$vectors = Join-Path $repo 'tests/conformance/vectors'
$fixtures = Join-Path $repo 'tests/corpus/fixtures'

# Rows the HLD §11 condition 4 requires.
$rows = @(
    @{ Label = 'testsrc_1080p30'; Path = (Join-Path $fixtures 'testsrc_1080p30.ivf'); Threads = 4 },
    @{ Label = 'tile_1x4';        Path = (Join-Path $vectors  'vp90-2-08-tile_1x4.webm'); Threads = 4 },
    @{ Label = 'testsrc_1080p60'; Path = (Join-Path $fixtures 'testsrc_1080p60.ivf'); Threads = 4 }
)

# Per-row, per-variant, per-iter samples of p50 µs.
$results = @{}
foreach ($row in $rows) {
    $key = "$($row.Label)_t$($row.Threads)"
    $results[$key] = @{ Baseline = @(); Candidate = @() }
}

$outDir = Join-Path $repo 'target/lf-overlap-bench'
New-Item -ItemType Directory -Force $outDir | Out-Null

function Invoke-OneRow {
    param([string]$Exe, [hashtable]$Row, [int]$Iter, [string]$Variant)
    $rowSpec = "$($Row.Label)=$($Row.Path):$($Row.Threads)"
    $outTsv = Join-Path $outDir "$Variant-iter$Iter-$($Row.Label)-t$($Row.Threads).tsv"
    # Set env var via wrapper for the candidate.
    if ($Variant -eq 'candidate') {
        $env:RVP9_LF_OVERLAP_POOL = '1'
    } else {
        Remove-Item Env:\RVP9_LF_OVERLAP_POOL -ErrorAction SilentlyContinue
    }
    # PowerShell turns native stderr into ErrorRecord objects which
    # halt under $ErrorActionPreference='Stop'. Temporarily switch.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & $Exe --row $rowSpec --out $outTsv --run-index $Iter 2>&1 | Out-String -Stream
    $ErrorActionPreference = $prevEAP
    Remove-Item Env:\RVP9_LF_OVERLAP_POOL -ErrorAction SilentlyContinue
    # Parse output for "  <label> <threads> ... p50= NNNN.Nus"
    foreach ($line in $output) {
        if ($line -match "p50=\s*([\d.]+)us\s+mean=\s*([\d.]+)us") {
            return [pscustomobject]@{
                p50  = [double]$Matches[1]
                mean = [double]$Matches[2]
            }
        }
    }
    throw "Could not parse output for $Variant iter $Iter row $($Row.Label)"
}

Write-Host "Interleaved A/B bench: $Iters iters x $($rows.Count) rows" -ForegroundColor Cyan
for ($i = 1; $i -le $Iters; $i++) {
    Write-Host "  iter $i ..." -ForegroundColor Yellow
    foreach ($row in $rows) {
        $key = "$($row.Label)_t$($row.Threads)"
        # Drift-cancel: alternate A then B order each iter.
        if (($i % 2) -eq 1) {
            $b = Invoke-OneRow -Exe $baselineExe -Row $row -Iter $i -Variant 'baseline'
            $c = Invoke-OneRow -Exe $candidateExe -Row $row -Iter $i -Variant 'candidate'
        } else {
            $c = Invoke-OneRow -Exe $candidateExe -Row $row -Iter $i -Variant 'candidate'
            $b = Invoke-OneRow -Exe $baselineExe -Row $row -Iter $i -Variant 'baseline'
        }
        $results[$key].Baseline += $b
        $results[$key].Candidate += $c
        $delta = (($b.p50 - $c.p50) / $b.p50) * 100
        Write-Host ("    {0,-22} A p50={1,8:F1}us  B p50={2,8:F1}us  delta={3,7:F2}%" -f $key, $b.p50, $c.p50, $delta)
    }
}

function Compute-Stats {
    param([double[]]$Values)
    $sorted = $Values | Sort-Object
    $n = $sorted.Count
    $median = if ($n % 2 -eq 1) { $sorted[[int]($n/2)] } else { ($sorted[[int]($n/2)-1] + $sorted[[int]($n/2)]) / 2 }
    $mean = ($Values | Measure-Object -Average).Average
    $sq = $Values | ForEach-Object { ($_ - $mean) * ($_ - $mean) }
    $stdev = if ($n -gt 1) { [Math]::Sqrt(($sq | Measure-Object -Sum).Sum / ($n - 1)) } else { 0 }
    return [pscustomobject]@{ Median = $median; Mean = $mean; Stdev = $stdev; N = $n }
}

Write-Host ""
Write-Host "=== Results ===" -ForegroundColor Green
$summary = @()
foreach ($row in $rows) {
    $key = "$($row.Label)_t$($row.Threads)"
    $bP50 = $results[$key].Baseline | ForEach-Object { $_.p50 }
    $cP50 = $results[$key].Candidate | ForEach-Object { $_.p50 }
    $bMean = $results[$key].Baseline | ForEach-Object { $_.mean }
    $cMean = $results[$key].Candidate | ForEach-Object { $_.mean }
    $bSt = Compute-Stats $bP50
    $cSt = Compute-Stats $cP50
    $bMs = Compute-Stats $bMean
    $cMs = Compute-Stats $cMean
    $p50Delta = (($bSt.Median - $cSt.Median) / $bSt.Median) * 100
    $meanDelta = (($bMs.Mean - $cMs.Mean) / $bMs.Mean) * 100
    Write-Host ""
    Write-Host "$key  (n=$($bSt.N))"
    Write-Host ("  baseline   p50 median={0,8:F1}us  mean={1,8:F1}us  stdev(p50)={2,5:F1}us" -f $bSt.Median, $bMs.Mean, $bSt.Stdev)
    Write-Host ("  candidate  p50 median={0,8:F1}us  mean={1,8:F1}us  stdev(p50)={2,5:F1}us" -f $cSt.Median, $cMs.Mean, $cSt.Stdev)
    Write-Host ("  delta      p50 = {0,7:F2}%   mean = {1,7:F2}%" -f $p50Delta, $meanDelta) -ForegroundColor Cyan
    $summary += [pscustomobject]@{
        Row = $key
        BaselineP50Median = [Math]::Round($bSt.Median, 1)
        CandidateP50Median = [Math]::Round($cSt.Median, 1)
        BaselineMean = [Math]::Round($bMs.Mean, 1)
        CandidateMean = [Math]::Round($cMs.Mean, 1)
        P50DeltaPct = [Math]::Round($p50Delta, 2)
        MeanDeltaPct = [Math]::Round($meanDelta, 2)
        BaselineP50Stdev = [Math]::Round($bSt.Stdev, 1)
        CandidateP50Stdev = [Math]::Round($cSt.Stdev, 1)
        N = $bSt.N
    }
}

# Write summary CSV.
$summary | Export-Csv -Path (Join-Path $outDir 'summary.csv') -NoTypeInformation
Write-Host ""
Write-Host "Summary CSV: $(Join-Path $outDir 'summary.csv')" -ForegroundColor Green
