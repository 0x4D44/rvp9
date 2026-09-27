param([int]$Pairs = 5, [string]$RunLabel = 'in-place-screen-01')
$ErrorActionPreference = 'Stop'
# This exact wrapper was copied to and run from the task worktree's .scratch directory.
$repo = Split-Path $PSScriptRoot -Parent
$root = Join-Path $repo '.scratch'
$candidateRelative = 'in-place-pgo\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root $candidateRelative
$candidateHash = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
$expectedCandidateHash = 'd4c97e0b73227958e623519d4b676fc18c7c3362ed1a149a61795d6bb308e2b4'
if ($candidateHash -ne $expectedCandidateHash) { throw "Candidate binary hash mismatch: $candidateHash" }
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
& ([scriptblock]::Create($script)) -Pairs $Pairs -RunLabel $RunLabel
