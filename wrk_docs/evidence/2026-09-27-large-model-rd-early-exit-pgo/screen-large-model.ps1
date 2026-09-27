param([int]$Pairs = 5, [string]$RunLabel = 'large-model-screen-01', [switch]$Libvpx)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$root = Join-Path $repo '.scratch'
$candidateRelative = 'large-model-pgo\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root $candidateRelative
$candidateHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedCandidateHash = '563a9a7e9e40b66ff4d3bcfd456be74d7a70a1a8b43046d96e8c7389f01ccc98'
if ($candidateHash -ne $expectedCandidateHash) {
    throw "Candidate binary hash mismatch: $candidateHash"
}
$source = Join-Path $repo 'wrk_docs\evidence\2026-09-27-early-dual-tx-chroma-proof\screen-early.ps1'
$script = [System.IO.File]::ReadAllText($source)
function Replace-Once([string]$body, [string]$old, [string]$new) {
    $count = [regex]::Matches($body, [regex]::Escape($old)).Count
    if ($count -ne 1) { throw "Expected one occurrence of '$old', found $count" }
    return $body.Replace($old, $new)
}
$script = Replace-Once $script 'D:\worktrees\rvp9-2\20260927-TSK-HUM-validate-early-dual-tx-chroma-proof-on-c\.scratch' $root
$script = Replace-Once $script 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe' 'D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s\.scratch\fresh-integrated-pgo\optimized\release\rvp9enc.exe'
$script = Replace-Once $script 'encoder-pgo-early\optimized\release\rvp9enc.exe' $candidateRelative
$script = Replace-Once $script '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee' '3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad'
& ([scriptblock]::Create($script)) -Pairs $Pairs -RunLabel $RunLabel -Libvpx:$Libvpx
