param(
    [Parameter(Mandatory=$true)][string]$CandidateHash,
    [int]$Pairs = 30,
    [string]$RunLabel = 'integrated-direct-01'
)
$ErrorActionPreference = 'Stop'
$source = 'D:/worktrees/rvp9-2/20260928-TSK-HUM-integrated-serial-lf-parity-audit/wrk_docs/evidence/2026-09-28-serial-loopfilter-pgo/direct-packet.ps1'
$script = [System.IO.File]::ReadAllText($source)
$oldRoot = 'D:/worktrees/rvp9-2/20260928-TSK-HUM-prototype-serial-tight-loop-filter-x86/.scratch'
$newRoot = 'D:/worktrees/rvp9-2/20260928-TSK-HUM-integrated-serial-lf-parity-audit/.scratch'
$oldHash = '438c94602239f4c143c6efec36dd985feaf4d4e7d9905a629f17f280ca3f5b22'
$oldCandidate = 'serial-lf-packet-pgo/optimized/release/rvp9enc.exe'
$newCandidate = 'integrated-lf-pgo/optimized/release/rvp9enc.exe'
if (-not $script.Contains($oldRoot) -or -not $script.Contains($oldHash) -or -not $script.Contains($oldCandidate)) {
    throw 'Source runner changed; audit substitutions before running'
}
$script = $script.Replace($oldRoot, $newRoot).Replace($oldHash, $CandidateHash).Replace($oldCandidate, $newCandidate)
& ([scriptblock]::Create($script)) -Pairs $Pairs -RunLabel $RunLabel -Libvpx
