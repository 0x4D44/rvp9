param(
    [int]$Pairs = 5,
    [string]$RunLabel = 'residual-screen-01',
    [string]$ExpectedCandidateSha256,
    [switch]$Libvpx
)
$ErrorActionPreference = 'Stop'
if ($Pairs -le 0 -or $ExpectedCandidateSha256 -notmatch '^[0-9a-fA-F]{64}$') {
    throw 'Positive pairs and the exact candidate SHA-256 are required'
}

$worktree = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-validate-chroma-residual-reuse-on-crucib'
$source = Join-Path $worktree 'wrk_docs\evidence\2026-09-27-early-dual-tx-chroma-proof\screen-early.ps1'
$candidate = Join-Path $worktree '.scratch\encoder-pgo-residual\optimized\release\rvp9enc.exe'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing reviewed screen source: $source" }
if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { throw "Missing candidate: $candidate" }
$actual = (Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $ExpectedCandidateSha256.ToLowerInvariant()) {
    throw "Candidate hash mismatch: expected=$ExpectedCandidateSha256 actual=$actual"
}

$screenText = [IO.File]::ReadAllText($source, [Text.Encoding]::UTF8)
function Replace-Once([string]$Old, [string]$New, [string]$Name) {
    $first = $script:screenText.IndexOf($Old, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $script:screenText.IndexOf($Old, $first + $Old.Length, [StringComparison]::Ordinal) -ge 0) {
        throw "Expected exactly one $Name anchor in $source"
    }
    $script:screenText = $script:screenText.Replace($Old, $New)
}

Replace-Once 'D:\worktrees\rvp9-2\20260927-TSK-HUM-validate-early-dual-tx-chroma-proof-on-c\.scratch' (Join-Path $worktree '.scratch') 'run root'
Replace-Once 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe' 'D:\worktrees\rvp9-2\20260927-TSK-HUM-profile-integrated-early-chroma-parity\.scratch\encoder-pgo-control\optimized\release\rvp9enc.exe' 'integrated control'
Replace-Once 'encoder-pgo-early\optimized\release\rvp9enc.exe' 'encoder-pgo-residual\optimized\release\rvp9enc.exe' 'candidate binary'
Replace-Once '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee' 'ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2' 'control hash'

$tokens = $null
$errors = $null
[void][Management.Automation.Language.Parser]::ParseInput($screenText, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw "Adapted screen parse errors: $($errors -join '; ')" }
$screen = [scriptblock]::Create($screenText)
if ($Libvpx) {
    . $screen -Pairs $Pairs -RunLabel $RunLabel -Libvpx
} else {
    . $screen -Pairs $Pairs -RunLabel $RunLabel
}
