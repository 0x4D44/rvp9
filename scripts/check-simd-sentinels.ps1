# G4.1 — source-side aligned-load discipline check (PowerShell port,
# per-function granularity).
#
# Every `unsafe fn` in `crates/rvp9-core/src/dsp/x86_64/*.rs` (the AVX2 /
# SSE2 / SSSE3 / SSE4.1 kernel impls) MUST use unaligned-load intrinsics
# for reads from `dst`. The convention is documented by a grep-able
# sentinel comment of the form:
#
#     // G4: loads from `dst` use _mm256_loadu_si256 (unaligned)
#
# (or the equivalent for SSE2 / SSSE3 — `_mm_loadu_si128` /
# `_mm_loadl_epi64`).
#
# **Per-function check:** for every `unsafe fn <name>(...)` declaration
# under the kernel directory, this script requires a sentinel comment
# either on the line immediately preceding the `unsafe fn` line, or in
# the contiguous block of comment / attribute lines that precedes it.
# This catches the failure mode where a second `unsafe fn` is appended
# to an existing kernel file but the author forgets to add a sentinel
# above it (the original "one sentinel per file" check would pass
# silently).
#
# Usage:
#   pwsh scripts/check-simd-sentinels.ps1

$ErrorActionPreference = 'Stop'

$dir = "crates/rvp9-core/src/dsp/x86_64"
$sentinel = 'G4: loads from `dst` use'
$rc = 0

if (-not (Test-Path $dir)) {
    Write-Error "Directory $dir does not exist"
    exit 2
}

# Match an `unsafe fn <ident>(` declaration at any indent, allowing
# leading `pub` / `pub(crate)` modifiers.
$unsafeFnPattern = '^\s*(pub\s*(\([^)]*\)\s*)?)?unsafe\s+fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\('

# A "preceding block" line is blank, a `//` comment, or an attribute
# (`#[...]` / `#![...]`).
$blockLinePattern = '^\s*(//|#!?\[|$)'

Get-ChildItem -Path $dir -Filter *.rs | Where-Object { $_.Name -ne 'mod.rs' } | ForEach-Object {
    $file = $_.FullName
    $lines = Get-Content -Path $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $unsafeFnPattern) {
            $found = $false
            # Walk backwards through the contiguous comment / attribute /
            # blank block above this line and check for the sentinel.
            for ($j = $i - 1; $j -ge 0; $j--) {
                $prev = $lines[$j]
                if ($prev -notmatch $blockLinePattern) {
                    break
                }
                # Use String.Contains rather than -like: the sentinel
                # contains backticks (`` ` ``) which PowerShell's -like
                # treats as escape characters inside double-quoted
                # patterns.
                if ($prev.Contains($sentinel)) {
                    $found = $true
                    break
                }
            }
            if (-not $found) {
                $trimmed = $lines[$i].TrimStart()
                Write-Output "MISSING sentinel: $($file):$($i + 1) before `"$trimmed`" (expected `"// $sentinel ...`" in the preceding comment block)"
                $rc = 1
            }
        }
    }
}

if ($rc -eq 0) {
    Write-Output "OK: every ``unsafe fn`` under $dir carries a preceding G4 unaligned-load sentinel."
}

exit $rc
