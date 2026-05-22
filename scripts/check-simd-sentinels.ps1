# G4.1 — source-side aligned-load discipline check (PowerShell port).
#
# In every `*.rs` kernel file under `crates/rvp9-core/src/dsp/x86_64`
# (the AVX2 / SSE2 / SSSE3 / SSE4.1 impls, excluding `mod.rs`), reads
# from the `dst` buffer MUST use UNALIGNED load intrinsics
# (`_mm256_loadu_si256` / `_mm_loadu_si128` / `_mm_loadl_epi64`),
# because `dst` is not guaranteed 16/32-byte aligned and an aligned
# load on an unaligned pointer faults.
#
# Rather than mandate a sentinel comment above every `unsafe fn` (most
# kernels are store-only to `dst` and legitimately never load from it),
# this check detects the actual violation directly: it FAILS if any
# kernel file calls an ALIGNED load intrinsic:
#
#     _mm256_load_si256  _mm_load_si128   _mm_load_si64
#     _mm256_load_ps     _mm_load_ps
#     _mm256_load_pd     _mm_load_pd
#
# The safe unaligned variants (`loadu` / `loadl`) are NOT flagged: the
# patterns anchor on the exact intrinsic name with a word boundary so
# `_mm_load_si128` does not match `_mm_loadu_si128` or `_mm_loadl_epi64`.
#
# Usage:
#   pwsh scripts/check-simd-sentinels.ps1

$ErrorActionPreference = 'Stop'

$dir = "crates/rvp9-core/src/dsp/x86_64"
$rc = 0

if (-not (Test-Path $dir)) {
    Write-Error "Directory $dir does not exist"
    exit 2
}

# Aligned-load intrinsics that must never appear in a kernel. Each is a
# whole-word match (`\b` boundary on both sides) so e.g. `_mm_load_si128`
# matches `_mm_load_si128(` but never `_mm_loadu_si128` / `_mm_loadl_epi64`.
$aligned = @(
    '_mm256_load_si256',
    '_mm_load_si128',
    '_mm256_load_ps',
    '_mm_load_ps',
    '_mm256_load_pd',
    '_mm_load_pd',
    '_mm_load_si64'
)
$pattern = '\b(' + ($aligned -join '|') + ')\b'

Get-ChildItem -Path $dir -Filter *.rs | Where-Object { $_.Name -ne 'mod.rs' } | ForEach-Object {
    $file = $_.FullName
    $lines = Get-Content -Path $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $m = [regex]::Match($lines[$i], $pattern)
        if ($m.Success) {
            Write-Output "ALIGNED load: $($file):$($i + 1) uses $($m.Value) (use the unaligned variant: loadu/loadl — `dst` is not guaranteed aligned)"
            $rc = 1
        }
    }
}

if ($rc -eq 0) {
    Write-Output "OK: no aligned-load intrinsics under $dir; all SIMD loads use unaligned (loadu/loadl) variants."
}

exit $rc
