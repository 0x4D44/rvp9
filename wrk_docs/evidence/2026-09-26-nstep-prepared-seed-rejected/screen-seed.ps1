param([int]$Pairs = 5, [string]$RunLabel = 'seed-screen-01', [Parameter(Mandatory=$true)][string]$CandidateHash)
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-prepared-initial-seed-re'
$source = Join-Path $worktree 'wrk_docs\evidence\2026-09-26-nstep-duplicate-variance-pgo\screen-nstep.ps1'
$root = Join-Path $worktree '.scratch'
$current = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root 'encoder-pgo-seed\optimized\release\rvp9enc.exe'
$currentHash = '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee'
foreach ($item in @(@{Path=$current;Hash=$currentHash}, @{Path=$candidate;Hash=$CandidateHash})) {
    $actual = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item.Hash) { throw "EXE hash mismatch: $($item.Path) actual=$actual" }
}
if (Test-Path -LiteralPath (Join-Path $root $RunLabel)) { throw "Run root already exists: $RunLabel" }
$scriptText = [IO.File]::ReadAllText($source)
function Replace-Once([string]$Text, [string]$Old, [string]$New) {
    $first = $Text.IndexOf($Old, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $Text.IndexOf($Old, $first + $Old.Length, [StringComparison]::Ordinal) -ge 0) {
        throw "Expected one script anchor: $Old"
    }
    return $Text.Replace($Old, $New)
}
$scriptText = Replace-Once $scriptText `
    "`$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch'" `
    "`$root = '$root'"
$scriptText = Replace-Once $scriptText `
    "`$current = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\encoder-pgo-current\optimized\release\rvp9enc.exe'" `
    "`$current = '$current'"
$scriptText = Replace-Once $scriptText `
    "`$candidate = Join-Path `$root 'encoder-pgo-nstep\optimized\release\rvp9enc.exe'" `
    "`$candidate = '$candidate'"
$scriptText = Replace-Once $scriptText '0fa1028b1310fee93b22731794dcb3fdc14e9b5e05dbe428f3e112c4b30cb395' $currentHash
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($scriptText, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Adapted screen has parse errors: $($parseErrors -join '; ')" }
$scriptPath = Join-Path $root 'adapted-screen-seed.ps1'
if (Test-Path -LiteralPath $scriptPath) {
    if ([IO.File]::ReadAllText($scriptPath) -cne $scriptText) { throw "Adapted script changed: $scriptPath" }
} else {
    [IO.File]::WriteAllText($scriptPath, $scriptText)
}
& powershell -NoProfile -File $scriptPath -Pairs $Pairs -RunLabel $RunLabel
if ($LASTEXITCODE -ne 0) { throw "Paired screen failed: exit=$LASTEXITCODE" }
