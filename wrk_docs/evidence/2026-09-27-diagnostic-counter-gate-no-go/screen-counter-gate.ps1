param([int]$Pairs = 5, [string]$RunLabel = 'counter-gate-screen-01', [switch]$Fresh)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo 'wrk_docs\evidence\2026-09-27-fresh-integrated-pgo-direct\direct-integrated-pgo.ps1'
$script = [System.IO.File]::ReadAllText($source)
function Replace-Once([string]$body, [string]$old, [string]$new) {
    $count = [regex]::Matches($body, [regex]::Escape($old)).Count
    if ($count -ne 1) { throw "Expected one occurrence of '$old', found $count" }
    return $body.Replace($old, $new)
}
$root = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-test-release-diagnostic-counter-gating-o/.scratch'
$control = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch/fresh-integrated-pgo/optimized/release/rvp9enc.exe'
$candidate = if ($Fresh) { Join-Path $root 'counter-fresh-pgo/optimized/release/rvp9enc.exe' } else { Join-Path $root 'counter-pgo-target/release/rvp9enc.exe' }
$candidateHash = if ($Fresh) { '74168ccdDE659A08120DE5F1AC422853C2999D63CED9FAAB337F21990DDB70D7'.ToLowerInvariant() } else { '2f7ae59b4416cf9e9d52fd8516865540a0ad4d376664f76801ee984f8f5151bf' }
$script = Replace-Once $script "`$root = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch'" "`$root = '$root'"
$script = Replace-Once $script "`$control = 'D:/worktrees/rvp9-2/acceptance-d639c265/build/libvpx/vpxenc.exe'" "`$control = '$control'"
$script = Replace-Once $script "`$candidate = 'D:/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch/fresh-integrated-pgo/optimized/release/rvp9enc.exe'" "`$candidate = '$candidate'"
$script = Replace-Once $script "`$expectedControlHash = '88ba702de16af47327490a7d016002fd05169e074c6c6657b9e9ff7a8d4a135d'" "`$expectedControlHash = '3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad'"
$script = Replace-Once $script "`$expectedCandidateHash = '3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad'" "`$expectedCandidateHash = '$candidateHash'"
& ([scriptblock]::Create($script)) -Pairs $Pairs -RunLabel $RunLabel
