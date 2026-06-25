# bench-headtohead.ps1 — Q.1.A orchestrator (Windows host).
#
# Per `wrk_docs/Track_Q/Q.1.A_e2e_perf_HLD.md` §3 + §4 + §7:
#
#   1. Ensure C/D/E .ivf fixtures exist (encode from y4m sources via
#      scripts/encode-corpus.ps1 if missing). HLD §2 + §7.
#   2. Run the rvp9 head-to-head bench binary (single pass — internal
#      warmup + 1000-frame timing window already discipline the rvp9
#      measurement). HLD §3.
#   3. Run `vpxdec --noblit --summary -t N <fixture>` for each row,
#      R=9 repetitions per row (odd, robust median). HLD §3 step 3.
#   4. Join the two result sets on (label, threads).
#   5. Write `bench/results/decode-headtohead.md` (overwritten each run)
#      and append `bench/results/decode-headtohead.csv` (schema-v1
#      trend). HLD §7.
#
# Q.1.A is NOT a hard CI gate (HLD §1, §6). Dev-box runs commit the
# ledger with `unpinned=true` and gate column is `PASS-info`/`FAIL-info`.
#
# Usage:
#   pwsh scripts/bench-headtohead.ps1                # full default matrix, 1 run
#   pwsh scripts/bench-headtohead.ps1 -SkipEncode    # assume C/D/E exist
#   pwsh scripts/bench-headtohead.ps1 -CheckOnly     # report prerequisites only
#   pwsh scripts/bench-headtohead.ps1 -Repetitions 3 # quicker vpxdec pass
#   pwsh scripts/bench-headtohead.ps1 -Runs 5        # 5 rvp9 sweeps for stats
#   pwsh scripts/bench-headtohead.ps1 -Runs 5 -SkipVpxdec  # only rvp9 (faster)
#
# Task C (statistical bench framework): -Runs N produces N CSV rows per
# (fixture, threads) pair, all sharing the same $sweepTimestamp so they
# can be paired with `bench-compare --baseline <ts1> --candidate <ts2>`.
# -SkipVpxdec leaves the vpxdec_* and ratio_mean columns empty — useful
# when measuring rvp9-only deltas (vpxdec is the same reference; you
# only need one sweep of it per session).

param(
    [int] $Repetitions = 9,
    [int] $Runs = 1,
    [switch] $SkipEncode,
    [switch] $SkipRvp9Bench,
    [switch] $SkipVpxdec,
    [switch] $CheckOnly
)

if ($Runs -lt 1) {
    throw "-Runs must be >= 1 (got $Runs)"
}

$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel).Trim()

# --- Paths ---------------------------------------------------------------
$vpxdec     = Join-Path $repo 'build/libvpx/vpxdec.exe'
$vpxenc     = Join-Path $repo 'build/libvpx/vpxenc.exe'
$encodeScript = Join-Path $repo 'scripts/encode-corpus.ps1'
$rvp9bench  = Join-Path $repo 'target/release/decode-headtohead.exe'

$benchOut   = Join-Path $repo 'bench/results'
$rvp9Tsv    = Join-Path $benchOut 'decode-headtohead.rvp9.tsv'
$ledgerMd   = Join-Path $benchOut 'decode-headtohead.md'
$ledgerCsv  = Join-Path $benchOut 'decode-headtohead.csv'

$vectors    = Join-Path $repo 'tests/conformance/vectors'
$fixtures   = Join-Path $repo 'tests/corpus/fixtures'
$y4mDir     = Join-Path $repo 'tests/corpus'

# --- HLD §5 matrix definition --------------------------------------------
# Each entry: label, fixture path (full), threads, optional encode-source.
# `Budget` is the PLAN §2 hard-real-time budget in microseconds, or $null
# for fixtures without a budget.
# `Tiles` is informational (logged in ledger).
$matrix = @(
    @{ Label='quantizer-00';    Path=(Join-Path $vectors 'vp90-2-00-quantizer-00.webm'); Threads=1; Budget=$null;  Tiles=1; Res='?x?'      },
    @{ Label='tile_1x4';        Path=(Join-Path $vectors 'vp90-2-08-tile_1x4.webm');     Threads=1; Budget=$null;  Tiles=4; Res='?x?'      },
    @{ Label='tile_1x4';        Path=(Join-Path $vectors 'vp90-2-08-tile_1x4.webm');     Threads=4; Budget=$null;  Tiles=4; Res='?x?'      },
    @{ Label='testsrc_720p30';  Path=(Join-Path $fixtures 'testsrc_720p30.ivf');         Threads=1; Budget=33000; Tiles=1; Res='1280x720'  },
    # HLD §2 literal: D/E encoded with `--tile-columns=2` (vpxenc log2),
    # i.e. 4 tile columns. The HLD §4 mock-up "tiles=2" is a draft typo;
    # we follow the literal encoder flag here.
    @{ Label='testsrc_1080p30'; Path=(Join-Path $fixtures 'testsrc_1080p30.ivf');        Threads=2; Budget=33000; Tiles=4; Res='1920x1080' },
    @{ Label='testsrc_1080p30'; Path=(Join-Path $fixtures 'testsrc_1080p30.ivf');        Threads=4; Budget=33000; Tiles=4; Res='1920x1080' },
    @{ Label='testsrc_1080p60'; Path=(Join-Path $fixtures 'testsrc_1080p60.ivf');        Threads=4; Budget=16000; Tiles=4; Res='1920x1080' },
    @{ Label='testsrc_1080p60'; Path=(Join-Path $fixtures 'testsrc_1080p60.ivf');        Threads=8; Budget=16000; Tiles=4; Res='1920x1080' },
    @{ Label='foreman-fp1';     Path=(Join-Path $fixtures 'foreman-fp1.ivf');            Threads=1; Budget=$null;  Tiles=1; Res='?x?'      }
)

# --- Step 1: ensure C/D/E exist -----------------------------------------
$encodeJobs = @(
    @{ Src=(Join-Path $y4mDir 'testsrc_720p30.y4m');  Out=(Join-Path $fixtures 'testsrc_720p30.ivf');  ExtraArgs=@('--passes=1','--lag-in-frames=0','--auto-alt-ref=0')                    },
    # HLD §2 literal: `--tile-columns=2` on D/E (4 tile columns).
    @{ Src=(Join-Path $y4mDir 'testsrc_1080p30.y4m'); Out=(Join-Path $fixtures 'testsrc_1080p30.ivf'); ExtraArgs=@('--passes=1','--lag-in-frames=0','--auto-alt-ref=0','--tile-columns=2') },
    @{ Src=(Join-Path $y4mDir 'testsrc_1080p60.y4m'); Out=(Join-Path $fixtures 'testsrc_1080p60.ivf'); ExtraArgs=@('--passes=1','--lag-in-frames=0','--auto-alt-ref=0','--tile-columns=2') }
)

function New-ReadinessItem {
    param(
        [string] $Group,
        [string] $Label,
        [string] $Path,
        [bool] $Required,
        [string] $Detail
    )
    $exists = Test-Path -LiteralPath $Path
    $status = if ($exists) { 'OK' } elseif ($Required) { 'MISSING' } else { 'OPTIONAL' }
    [pscustomobject] @{
        Group = $Group
        Label = $Label
        Path = $Path
        Required = $Required
        Status = $status
        Detail = $Detail
    }
}

function Get-Rvp9TsvPaths {
    $paths = @()
    for ($i = 1; $i -le $Runs; $i++) {
        $paths += if ($Runs -gt 1) {
            Join-Path $benchOut "decode-headtohead.rvp9.r$i.tsv"
        } else {
            $rvp9Tsv
        }
    }
    return $paths
}

function Get-ReadinessItems {
    $items = @()

    $items += New-ReadinessItem -Group 'tools' -Label 'rvp9 decode-headtohead' -Path $rvp9bench `
        -Required:(-not $SkipRvp9Bench) -Detail 'run: cargo build --release -p rvp9-bench --bin decode-headtohead'
    if ($SkipRvp9Bench) {
        foreach ($path in (Get-Rvp9TsvPaths)) {
            $items += New-ReadinessItem -Group 'inputs' -Label (Split-Path -Leaf $path) -Path $path `
                -Required:$true -Detail 'required because -SkipRvp9Bench reuses an existing TSV'
        }
    }
    $items += New-ReadinessItem -Group 'tools' -Label 'libvpx vpxdec' -Path $vpxdec `
        -Required:(-not $SkipVpxdec) -Detail 'build libvpx first, usually via scripts/build-libvpx.sh'

    $missingGeneratedFixtures = @($encodeJobs | Where-Object { -not (Test-Path -LiteralPath $_.Out) })
    $needsEncode = (-not $SkipEncode) -and ($missingGeneratedFixtures.Count -gt 0)
    $items += New-ReadinessItem -Group 'tools' -Label 'libvpx vpxenc' -Path $vpxenc `
        -Required:$needsEncode -Detail 'needed only when generated testsrc fixtures are absent'
    $items += New-ReadinessItem -Group 'scripts' -Label 'encode-corpus.ps1' -Path $encodeScript `
        -Required:$needsEncode -Detail 'encodes generated C/D/E fixtures from raw y4m sources'

    $fixtureRequired = (-not $SkipRvp9Bench) -or (-not $SkipVpxdec)
    foreach ($row in $matrix) {
        $items += New-ReadinessItem -Group 'fixtures' -Label "$($row.Label) t=$($row.Threads)" -Path $row.Path `
            -Required:$fixtureRequired -Detail 'bench matrix input'
    }

    foreach ($job in $encodeJobs) {
        $sourceRequired = (-not $SkipEncode) -and (-not (Test-Path -LiteralPath $job.Out))
        $items += New-ReadinessItem -Group 'sources' -Label (Split-Path -Leaf $job.Src) -Path $job.Src `
            -Required:$sourceRequired -Detail 'raw source for generated fixture'
    }

    return $items
}

function Write-ReadinessReport {
    param([object[]] $Items)

    Write-Host "[check] bench-headtohead readiness"
    foreach ($group in ($Items | Group-Object Group)) {
        Write-Host ""
        Write-Host "[$($group.Name)]"
        foreach ($item in $group.Group) {
            $requirement = if ($item.Required) { 'required' } else { 'optional' }
            Write-Host ("  {0,-8} {1,-42} {2}" -f $item.Status, $item.Label, $item.Path)
            if (-not (Test-Path -LiteralPath $item.Path)) {
                Write-Host ("           ({0}) {1}" -f $requirement, $item.Detail)
            }
        }
    }

    $missing = @($Items | Where-Object { $_.Required -and $_.Status -eq 'MISSING' })
    Write-Host ""
    if ($missing.Count -eq 0) {
        Write-Host "[check] PASS: all required prerequisites are present"
    } else {
        Write-Host "[check] FAIL: $($missing.Count) required prerequisite(s) missing"
    }
    return $missing.Count
}

if ($CheckOnly) {
    $missing = Write-ReadinessReport -Items (Get-ReadinessItems)
    if ($missing -gt 0) { exit 1 }
    exit 0
}

# --- Sanity checks -------------------------------------------------------
if ((-not $SkipVpxdec) -and (-not (Test-Path $vpxdec))) {
    throw "vpxdec.exe not found at $vpxdec — BLOCKER: build libvpx first (scripts/build-libvpx.sh)."
}
if ((-not $SkipRvp9Bench) -and (-not (Test-Path $rvp9bench))) {
    throw "rvp9 bench binary not found at $rvp9bench — run `cargo build --release -p rvp9-bench --bin decode-headtohead` first."
}
New-Item -ItemType Directory -Force $benchOut | Out-Null

if (-not $SkipEncode) {
    foreach ($job in $encodeJobs) {
        if (Test-Path $job.Out) {
            Write-Host "[encode] skip (exists): $($job.Out)"
            continue
        }
        if (-not (Test-Path $job.Src)) {
            Write-Warning "[encode] missing source $($job.Src) — row will be reported as MISSING"
            continue
        }
        Write-Host "[encode] $($job.Src) -> $($job.Out)"
        & pwsh -NoProfile -File $encodeScript -Src $job.Src -Out $job.Out -ExtraArgs $job.ExtraArgs
    }
} else {
    Write-Host "[encode] -SkipEncode set; not encoding C/D/E"
}

# --- Step 2: run rvp9 bench (N times when -Runs > 1) --------------------
# Each iteration writes the TSV to a per-run path so we don't clobber
# samples; the parser below reads them all back. The orchestrator
# captures ONE timestamp at script start ($sweepTimestamp, used for
# every CSV row this invocation appends — N rows per fixture×threads).
$sweepTimestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$rvp9TsvPaths = Get-Rvp9TsvPaths
for ($i = 1; $i -le $Runs; $i++) {
    $perRunTsv = $rvp9TsvPaths[$i - 1]
    if (-not $SkipRvp9Bench) {
        Write-Host ""
        Write-Host "[rvp9] running decode-headtohead (run $i / $Runs)..."
        & $rvp9bench --out $perRunTsv --run-index $i
        if ($LASTEXITCODE -ne 0) {
            throw "rvp9 bench failed (exit $LASTEXITCODE) on run $i"
        }
    } else {
        Write-Host "[rvp9] -SkipRvp9Bench set; assuming $perRunTsv up to date (run $i)"
    }
    if (-not (Test-Path $perRunTsv)) {
        throw "rvp9 TSV missing at $perRunTsv — cannot continue"
    }
}

# --- Step 3: vpxdec wall-time matrix ------------------------------------
# Pre-load each unique fixture into the OS page cache so the first
# vpxdec invocation doesn't pay disk-IO cost (HLD §3 step 1).
$uniqueFixtures = $matrix.Path | Sort-Object -Unique
if (-not $SkipVpxdec) {
    foreach ($f in $uniqueFixtures) {
        if (Test-Path $f) {
            Write-Host "[prefetch] $f"
            $null = [System.IO.File]::ReadAllBytes($f)
        }
    }
}

# `vpxdec --noblit --summary -t N <fixture>` emits to stderr:
#   "<N> decoded frames/<N> showed frames in <us> us (<fps> fps)"
# Parse out total_us and frame_count.
function Invoke-Vpxdec {
    param([string]$Fixture, [int]$Threads)
    # vpxdec writes the summary to stderr; merge with 2>&1 but search
    # ALL lines (not just the last) because PowerShell pipes
    # frequently materialise a trailing empty ErrorRecord.
    $out = & $vpxdec --noblit --summary -t $Threads $Fixture 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "vpxdec failed on $Fixture (t=$Threads): $out"
    }
    foreach ($entry in $out) {
        $s = $entry.ToString()
        if ($s -match '^\s*(\d+)\s+decoded frames/\d+\s+showed frames in\s+(\d+)\s+us') {
            return @{
                DecodedFrames = [int] $Matches[1]
                TotalUs       = [long] $Matches[2]
            }
        }
    }
    $joined = ($out | ForEach-Object { $_.ToString() }) -join ' || '
    throw "could not parse vpxdec summary line: $joined"
}

function Measure-Median {
    param([double[]] $Values)
    if ($Values.Count -eq 0) { return $null }
    $sorted = $Values | Sort-Object
    $mid = [int] [Math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) {
        return $sorted[$mid]
    } else {
        return ($sorted[$mid - 1] + $sorted[$mid]) / 2.0
    }
}

$vpxResults = @{}  # key="label|threads" => @{ MeanUs=...; Frames=...; Reps=...; Missing=... }
if ($SkipVpxdec) {
    Write-Host ""
    Write-Host "[vpxdec] -SkipVpxdec set; leaving vpxdec_* and ratio_mean columns empty"
    foreach ($row in $matrix) {
        $key = "$($row.Label)|$($row.Threads)"
        $vpxResults[$key] = @{ Skipped=$true }
    }
} else {
    Write-Host ""
    Write-Host "[vpxdec] running R=$Repetitions reps over $($matrix.Count) rows..."
    foreach ($row in $matrix) {
        $key = "$($row.Label)|$($row.Threads)"
        if (-not (Test-Path $row.Path)) {
            Write-Warning "[vpxdec] $($row.Label) t=$($row.Threads): fixture missing $($row.Path) — skipping"
            $vpxResults[$key] = @{ Missing=$true }
            continue
        }
        Write-Host "  [vpxdec] $($row.Label) t=$($row.Threads)"
        $perFrameSamples = @()
        $lastFrames = 0
        for ($i = 0; $i -lt $Repetitions; $i++) {
            $r = Invoke-Vpxdec -Fixture $row.Path -Threads $row.Threads
            if ($r.DecodedFrames -lt 1) {
                Write-Warning "  rep ${i}: zero decoded frames — skipping"
                continue
            }
            $lastFrames = $r.DecodedFrames
            $perFrameSamples += [double] $r.TotalUs / [double] $r.DecodedFrames
        }
        if ($perFrameSamples.Count -eq 0) {
            Write-Warning "[vpxdec] $key produced no samples"
            $vpxResults[$key] = @{ Missing=$true }
            continue
        }
        $median = Measure-Median -Values $perFrameSamples
        $vpxResults[$key] = @{
            MeanUs = $median
            Frames = $lastFrames
            Reps   = $perFrameSamples.Count
            Missing = $false
        }
        Write-Host ("    median per-frame mean: {0:F1} us across {1} reps ({2} frames)" -f $median, $perFrameSamples.Count, $lastFrames)
    }
}

# --- Step 4: parse rvp9 TSV(s) ------------------------------------------
# Keyed by "label|threads|runIndex"; the markdown summary below uses the
# last-run entry, while the CSV emitter walks every entry.
$rvp9Rows = @{}
$rvp9Header = @{}
foreach ($tsvPath in $rvp9TsvPaths) {
    foreach ($line in Get-Content -LiteralPath $tsvPath) {
        if ($line.StartsWith('# ')) {
            # `# rvp9-headtohead schema=2 host=... build=... run_index=N`
            foreach ($kv in ($line.Substring(2) -split '\s+')) {
                if ($kv.Contains('=')) {
                    $parts = $kv -split '=', 2
                    $rvp9Header[$parts[0]] = $parts[1]
                }
            }
            continue
        }
        if ($line.StartsWith('label')) { continue }  # column header
        if ([string]::IsNullOrWhiteSpace($line)) { continue }
        $cols = $line -split "`t"
        # Schema v2 = 11 columns (added run_index); v1 = 10 (run_index
        # implied 1). Both accepted so this script can re-parse an old
        # TSV during the v2 rollout if -SkipRvp9Bench is set.
        if ($cols.Count -lt 10) { continue }
        $runIdx = if ($cols.Count -ge 11) { [int] $cols[10] } else { 1 }
        $key = "$($cols[0])|$($cols[2])|$runIdx"
        $rvp9Rows[$key] = @{
            Label       = $cols[0]
            Path        = $cols[1]
            Threads     = [int] $cols[2]
            Width       = [int] $cols[3]
            Height      = [int] $cols[4]
            TimedFrames = [int] $cols[5]
            P50         = [double] $cols[6]
            Mean        = [double] $cols[7]
            P95         = [double] $cols[8]
            P99         = [double] $cols[9]
            RunIndex    = $runIdx
        }
    }
}

# --- Step 5: write ledger.md + append ledger.csv ------------------------
# Display timestamp (human-readable in .md); the CSV uses $sweepTimestamp
# captured at script start so every row in this invocation shares the
# same Unix-seconds value.
$timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ" -AsUTC)
$gitSha    = (git rev-parse --short HEAD).Trim()
# vpxdec has no `--version`; the version banner prints on -v when given
# a valid input. Use the smallest available fixture so the call is cheap.
$verProbe  = if ($SkipVpxdec) { $null } else { ($uniqueFixtures | Where-Object { Test-Path $_ } | Select-Object -First 1) }
if ($SkipVpxdec) {
    $vpxVer = '(libvpx; skipped)'
} elseif ($verProbe) {
    $verLines = & $vpxdec --noblit --summary -v $verProbe 2>&1 | ForEach-Object { $_.ToString() }
    $vpxVer = ($verLines | Where-Object { $_ -match 'Decoder v' } | Select-Object -First 1)
    if (-not $vpxVer) { $vpxVer = '(libvpx; version probe failed)' }
} else {
    $vpxVer = '(libvpx; no fixture for version probe)'
}
$hostFp    = "$($rvp9Header['host'])"
$buildFp   = "$($rvp9Header['build'])"

# Build the markdown table.
$md = @()
$md += "# decode head-to-head — $timestamp"
$md += ""
$md += "host: $hostFp  unpinned  build: release ($buildFp)"
$md += "rvp9: $gitSha  vpxdec: $vpxVer ($vpxdec)"
$md += "reps: R=$Repetitions per vpxdec row (median); rvp9 timing window per HLD §3"
$md += "rvp9 runs: $Runs (summary row = run 1; CSV ledger captures all runs)"
$md += ""
$md += 'note: vpxdec value is per-frame mean from `vpxdec --noblit --summary -t N`,'
$md += '      including construction/teardown overhead (negligible on >=1000-frame'
$md += '      fixtures; up to a few percent on short fixtures — see methodology).'
$md += '      rvp9 mean shown alongside p50 for tail-stability inspection;'
$md += '      ratio compares means (rvp9 mean / vpxdec mean).'
$md += '      gate column is informational on unpinned hosts (HLD §1 + §6).'
$md += '      rvp9 mean for short fixtures (< MIN_TIMED_FRAMES = 1000) is'
$md += '      inflated by repeated keyframe re-decode across replay loops'
$md += '      (steady-state Decoder per HLD §3); p50 is the more stable'
$md += '      number for those rows.'
$md += ""
$md += "| fixture            | res        | tiles | threads | rvp9 p50 us | rvp9 mean us | vpxdec mean us | ratio (rvp9/vpx) | budget us | gate        |"
$md += "|--------------------|------------|-------|---------|-------------|--------------|----------------|------------------|-----------|-------------|"

# Schema-v3 CSV columns: schema v3 inserts `run_index` between `git_sha`
# and `fixture`. Old schema-v2 rows in the file are left untouched
# (append-only ledger).
$csvLines = @()
$csvNeedsHeader = -not (Test-Path $ledgerCsv)
if ($csvNeedsHeader) {
    $csvLines += "schema_version,timestamp,host,unpinned,build_hash,git_sha,run_index,fixture,threads,width,height,timed_frames,rvp9_p50_us,rvp9_mean_us,rvp9_p95_us,rvp9_p99_us,vpxdec_mean_us,vpxdec_reps,vpxdec_frames,ratio_mean,budget_us,gate"
}
$tsUnix = $sweepTimestamp

# Markdown summary uses run 1 for the at-a-glance view (first sweep —
# most representative if -Runs N > 1; deeper analysis goes through
# `bench-compare`).
$summaryRunIndex = 1

foreach ($row in $matrix) {
    $vpx  = $vpxResults["$($row.Label)|$($row.Threads)"]
    $resCol = $row.Res
    $tilesCol = $row.Tiles
    $threadsCol = $row.Threads

    # --- Markdown summary (run 1) ---
    $summaryKey = "$($row.Label)|$($row.Threads)|$summaryRunIndex"
    $rvp9Summary = $rvp9Rows[$summaryKey]
    if (-not $rvp9Summary -or -not $vpx -or $vpx.Missing) {
        if ($null -eq $row.Budget) { $budgetDisp = '-' } else { $budgetDisp = $row.Budget }
        # Suppress vpxdec-MISSING noise in the .md when the user has
        # intentionally skipped that side.
        $vpxCell = if ($vpx -and $vpx.Skipped) { 'SKIPPED' } else { 'MISSING' }
        $ratioCell = if ($vpx -and $vpx.Skipped) { '-' } else { '-' }
        $gateCell = if ($vpx -and $vpx.Skipped) { 'rvp9-only' } else { 'MISSING' }
        $rvp9P50 = if ($rvp9Summary) { ("{0,11:F1}" -f $rvp9Summary.P50) } else { 'MISSING    ' }
        $rvp9Mean = if ($rvp9Summary) { ("{0,12:F1}" -f $rvp9Summary.Mean) } else { 'MISSING     ' }
        $md += "| {0,-18} | {1,-10} | {2,5} | {3,7} | $rvp9P50 | $rvp9Mean | {4,14} | {5,16} | {6,9} | {7,-11} |" -f `
            $row.Label, $resCol, $tilesCol, $threadsCol, $vpxCell, $ratioCell, $budgetDisp, $gateCell
    } else {
        $ratio = $rvp9Summary.Mean / $vpx.MeanUs
        $resCol = "$($rvp9Summary.Width)x$($rvp9Summary.Height)"
        if ($null -ne $row.Budget) {
            if ($rvp9Summary.Mean -le $row.Budget) {
                $gate = 'PASS-info'
            } else {
                $gate = 'FAIL-info'
            }
            $budgetCol = $row.Budget
        } else {
            $gate = 'n/a'
            $budgetCol = '-'
        }
        $md += "| {0,-18} | {1,-10} | {2,5} | {3,7} | {4,11:F1} | {5,12:F1} | {6,14:F1} | {7,16:F2} | {8,9} | {9,-11} |" -f `
            $row.Label, $resCol, $tilesCol, $threadsCol, $rvp9Summary.P50, $rvp9Summary.Mean, $vpx.MeanUs, $ratio, $budgetCol, $gate
    }

    # --- CSV emit (one row per (matrix-row, run_index) pair) ---
    for ($ri = 1; $ri -le $Runs; $ri++) {
        $key = "$($row.Label)|$($row.Threads)|$ri"
        $rvp9 = $rvp9Rows[$key]
        $budgetCell = if ($null -eq $row.Budget) { '' } else { $row.Budget }

        # Defensive: rvp9 row missing for this run? Emit a MISSING row
        # so the CSV reflects the attempt rather than silently dropping.
        if (-not $rvp9) {
            $csvLines += "3,$tsUnix,$hostFp,true,$buildFp,$gitSha,$ri,$($row.Label),$($row.Threads),0,0,0,,,,,,,0,,$($row.Budget),MISSING"
            continue
        }

        if ($vpx -and $vpx.Skipped) {
            # -SkipVpxdec: leave vpxdec_* and ratio_mean blank. Gate is
            # informational only ("rvp9-only").
            $csvLines += ("3,{0},{1},true,{2},{3},{4},{5},{6},{7},{8},{9},{10:F3},{11:F3},{12:F3},{13:F3},,,,,{14},rvp9-only" -f `
                $tsUnix, $hostFp, $buildFp, $gitSha, $ri, $row.Label, $row.Threads, $rvp9.Width, $rvp9.Height, $rvp9.TimedFrames, `
                $rvp9.P50, $rvp9.Mean, $rvp9.P95, $rvp9.P99, $budgetCell)
            continue
        }

        if (-not $vpx -or $vpx.Missing) {
            $csvLines += ("3,{0},{1},true,{2},{3},{4},{5},{6},{7},{8},{9},{10:F3},{11:F3},{12:F3},{13:F3},,,,,{14},MISSING" -f `
                $tsUnix, $hostFp, $buildFp, $gitSha, $ri, $row.Label, $row.Threads, $rvp9.Width, $rvp9.Height, $rvp9.TimedFrames, `
                $rvp9.P50, $rvp9.Mean, $rvp9.P95, $rvp9.P99, $budgetCell)
            continue
        }

        $ratio = $rvp9.Mean / $vpx.MeanUs
        if ($null -ne $row.Budget) {
            $gate = if ($rvp9.Mean -le $row.Budget) { 'PASS-info' } else { 'FAIL-info' }
        } else {
            $gate = 'n/a'
        }
        $csvLines += ("3,{0},{1},true,{2},{3},{4},{5},{6},{7},{8},{9},{10:F3},{11:F3},{12:F3},{13:F3},{14:F3},{15},{16},{17:F4},{18},{19}" -f `
            $tsUnix, $hostFp, $buildFp, $gitSha, $ri, $row.Label, $row.Threads, $rvp9.Width, $rvp9.Height, $rvp9.TimedFrames, `
            $rvp9.P50, $rvp9.Mean, $rvp9.P95, $rvp9.P99, $vpx.MeanUs, $vpx.Reps, $vpx.Frames, $ratio, $budgetCell, $gate)
    }
}

# Methodology footer.
$md += ''
$md += '## Methodology'
$md += ''
$md += '- rvp9 column: `rvp9_bench::throughput::run_one_threaded` with'
$md += '  `Decoder::with_threads(N)`. Timing window covers only'
$md += '  `Decoder::decode_frame`; 30 warmup frames dropped; loop until'
$md += '  >= 1000 timed frames per (fixture, threads). HLD §3.'
$md += "- vpxdec column: ``vpxdec --noblit --summary -t N <fixture>`` x"
$md += "  R=$Repetitions repetitions (odd; robust median). No calibration"
$md += '  constant — short fixtures (< 100 frames) include construction'
$md += '  smear, disclosed in this header per HLD §11.1.'
$md += '- Ratio is rvp9 mean / vpxdec mean (mean-to-mean, apples to apples'
$md += '  with vpxdec which has no per-frame surface).'
$md += '- Gate is informational on unpinned hosts; pinned-host promotion'
$md += '  deferred to the dedicated Linux CI runner (HLD §1 + §6).'
if ($Runs -gt 1) {
    $md += "- Multi-run sweep: -Runs $Runs. The summary table above shows"
    $md += "  run 1 only; the CSV ledger contains all $Runs rows per"
    $md += "  fixture×threads (same `timestamp`, `run_index`=1..$Runs)."
    $md += '  Pair with `bench-compare --baseline <ts1> --candidate <ts2>`'
    $md += '  for Mann-Whitney U significance against a previous sweep.'
}
if ($SkipVpxdec) {
    $md += '- vpxdec sweep skipped (-SkipVpxdec); vpxdec_* and'
    $md += '  ratio_mean columns blank in the CSV for this timestamp.'
}

Set-Content -LiteralPath $ledgerMd -Value ($md -join "`r`n") -Encoding utf8
Add-Content -LiteralPath $ledgerCsv -Value $csvLines -Encoding utf8

Write-Host ""
Write-Host "wrote $ledgerMd"
Write-Host "appended $ledgerCsv"
