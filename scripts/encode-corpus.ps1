# encode-corpus.ps1 — generate VP9 fixtures from raw y4m sources using the
# in-tree libvpx reference encoder (Wave L L6).
#
# Resolves the long-standing "vpxenc is not on PATH" blocker behind G5.3 /
# G6.1 / G8.1: a working `vpxenc.exe` / `vpxdec.exe` live under
# `build/libvpx/`, and raw `.y4m` sources under `tests/corpus/`. This script
# wraps the proven generation recipe:
#
#   1. vpxenc  <src>.y4m            -> <out>.ivf       (the fixture)
#   2. vpxdec  --i420 -o <pat>.i420 <out>.ivf          (one raw frame/file)
#   3. md5sum  *.i420  (two-space normalised)          -> <out>.ivf.md5
#
# Step 1 alone produces a decode-only fixture (sufficient for the
# decode-throughput bench — G6.1, which times decode and does not bit-check).
# Steps 2-3 additionally emit a per-frame visible-plane MD5 ledger in the
# format the conformance harness expects (`<hex>  <stem>-<W>x<H>-<NNNN>.i420`,
# matching `rvp9_io::md5::md5_of_yv12_visible`, which mirrors
# `vpxdec.c:update_image_md5`). Pass -WithLedger to generate it.
#
# NOTE (Wave L L6 finding): our decoder is bit-exact against the *official*
# WebM-project conformance corpus (201/201, fetched via fetch-vectors.sh) but
# diverges on some features emitted by a default `vpxenc` encode. So a
# self-encoded `.ivf.md5` ledger is a valid libvpx reference, but our decoder
# does not yet pass arbitrary self-encoded conformance vectors — track that as
# a separate decoder-robustness follow-up before relying on -WithLedger for a
# conformance gate. Decode-only bench fixtures (no ledger) are unaffected.
#
# Usage:
#   pwsh scripts/encode-corpus.ps1 -Src tests/corpus/testsrc_720p30.y4m `
#        -Out tests/corpus/fixtures/testsrc_720p30.ivf [-Limit 60] `
#        [-ExtraArgs '--passes=1','--lag-in-frames=0','--auto-alt-ref=0'] `
#        [-WithLedger]

param(
    [Parameter(Mandatory = $true)] [string] $Src,
    [Parameter(Mandatory = $true)] [string] $Out,
    [int] $Limit = 0,
    [string[]] $ExtraArgs = @('--passes=1', '--lag-in-frames=0', '--auto-alt-ref=0'),
    [switch] $WithLedger
)

$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel).Trim()
$vpxenc = Join-Path $repo 'build/libvpx/vpxenc.exe'
$vpxdec = Join-Path $repo 'build/libvpx/vpxdec.exe'

foreach ($tool in @($vpxenc, $vpxdec)) {
    if (-not (Test-Path $tool)) {
        throw "Missing $tool — build libvpx into build/libvpx/ first."
    }
}
if (-not (Test-Path $Src)) { throw "Source not found: $Src" }

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }

# 1. Encode the fixture.
$encArgs = @('--codec=vp9', '--ivf') + $ExtraArgs
if ($Limit -gt 0) { $encArgs += "--limit=$Limit" }
$encArgs += @('-o', $Out, $Src)
Write-Output "vpxenc $($encArgs -join ' ')"
& $vpxenc @encArgs
Write-Output "encoded: $Out ($((Get-Item $Out).Length) bytes)"

if (-not $WithLedger) { return }

# 2-3. Per-frame I420 -> visible-plane MD5 ledger.
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("encode-corpus-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $work | Out-Null
try {
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Out)
    $pattern = Join-Path $work "$stem-%wx%h-%4.i420"
    & $vpxdec --i420 -o $pattern $Out | Out-Null
    $ledger = "$Out.md5"
    Get-ChildItem -Path $work -Filter "$stem-*.i420" | Sort-Object Name | ForEach-Object {
        $hash = (Get-FileHash -Algorithm MD5 $_.FullName).Hash.ToLower()
        "$hash  $($_.Name)"
    } | Set-Content -Path $ledger -Encoding ascii
    Write-Output "ledger: $ledger ($((Get-Content $ledger).Count) frames)"
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}
