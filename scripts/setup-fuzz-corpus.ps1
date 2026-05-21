# scripts/setup-fuzz-corpus.ps1
#
# Populate the rvp9-fuzz seed corpus from the in-repo conformance
# vectors. Both source and destination paths are gitignored — the
# conformance vectors live under tests/conformance/vectors/ and are
# fetched by scripts/fetch-vectors.sh; the fuzz corpus directory is
# excluded under .gitignore to keep the libFuzzer working set out
# of git history.
#
# Per HLD §G9, the seed corpus is "the 14 AOM conformance vectors +
# the G8 in-repo subset". This script copies every .ivf and .webm
# file from tests/conformance/vectors/ into
# crates/rvp9-fuzz/corpus/fuzz_parse_packet/.
#
# Run from the repo root:
#   ./scripts/setup-fuzz-corpus.ps1

$ErrorActionPreference = 'Stop'

$repoRoot = Resolve-Path -Path (Join-Path $PSScriptRoot '..')
$sourceDir = Join-Path $repoRoot 'tests/conformance/vectors'
$destDir   = Join-Path $repoRoot 'crates/rvp9-fuzz/corpus/fuzz_parse_packet'

if (-not (Test-Path -Path $sourceDir)) {
    Write-Error "Conformance vectors not found at $sourceDir. Run ./scripts/fetch-vectors.sh first."
    exit 1
}

if (-not (Test-Path -Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

$copied = 0
Get-ChildItem -Path $sourceDir -File -Recurse |
    Where-Object { $_.Extension -in '.ivf', '.webm' } |
    ForEach-Object {
        Copy-Item -Path $_.FullName -Destination $destDir -Force
        $copied++
    }

Write-Host "Copied $copied seed file(s) to $destDir"
