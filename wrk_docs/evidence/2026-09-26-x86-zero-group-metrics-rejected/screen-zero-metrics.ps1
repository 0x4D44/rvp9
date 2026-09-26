param(
    [int]$Pairs = 5,
    [string]$RunLabel = 'zero-metrics-screen-01',
    [string]$CandidateHash
)
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-screen-x86-zero-quantized-hadamard-metri'
$root = Join-Path $worktree '.scratch'
$source = Join-Path $worktree 'wrk_docs\evidence\2026-09-26-nstep-duplicate-variance-pgo\screen-nstep.ps1'
$current = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root 'encoder-pgo-zero-quantized\optimized\release\rvp9enc.exe'
$currentHash = '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee'
if ($Pairs -lt 1 -or $Pairs -gt 30) { throw 'Pairs must be 1..30' }
if ($CandidateHash -notmatch '^[0-9a-fA-F]{64}$') { throw 'CandidateHash must be SHA-256' }
if ($RunLabel -notmatch '^[a-zA-Z0-9-]+$') { throw 'RunLabel must be simple and unique' }
if (Test-Path -LiteralPath (Join-Path $root $RunLabel)) { throw "Run root already exists: $RunLabel" }
foreach ($item in @(@{Path=$current;Hash=$currentHash}, @{Path=$candidate;Hash=$CandidateHash})) {
    $actual = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item.Hash.ToLowerInvariant()) { throw "EXE hash mismatch: $($item.Path) actual=$actual" }
}
function Replace-Once([string]$Text, [string]$Old, [string]$New) {
    $first = $Text.IndexOf($Old, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $Text.IndexOf($Old, $first + $Old.Length, [StringComparison]::Ordinal) -ge 0) {
        throw "Expected one script anchor: $Old"
    }
    return $Text.Replace($Old, $New)
}
$text = [IO.File]::ReadAllText($source)
$text = Replace-Once $text `
    "`$root = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch'" `
    "`$root = '$root'"
$text = Replace-Once $text `
    "`$current = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\encoder-pgo-current\optimized\release\rvp9enc.exe'" `
    "`$current = '$current'"
$text = Replace-Once $text `
    "`$candidate = Join-Path `$root 'encoder-pgo-nstep\optimized\release\rvp9enc.exe'" `
    "`$candidate = '$candidate'"
$text = Replace-Once $text '0fa1028b1310fee93b22731794dcb3fdc14e9b5e05dbe428f3e112c4b30cb395' $currentHash
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Adapted screen parse errors: $($parseErrors -join '; ')" }
$scriptPath = Join-Path $root 'adapted-screen-zero-metrics.ps1'
if (Test-Path -LiteralPath $scriptPath) {
    if ([IO.File]::ReadAllText($scriptPath) -cne $text) { throw "Adapted script changed: $scriptPath" }
} else {
    [IO.File]::WriteAllText($scriptPath, $text)
}
& powershell -NoProfile -File $scriptPath -Pairs $Pairs -RunLabel $RunLabel
if ($LASTEXITCODE -ne 0) { throw "Paired screen failed: exit=$LASTEXITCODE" }
