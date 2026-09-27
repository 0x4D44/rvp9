param(
    [ValidateSet('ducks', 'tree', 'town')][string[]]$Clips = @('ducks', 'tree', 'town'),
    [switch]$SmokeOnly
)
$ErrorActionPreference = 'Stop'

$worktree = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-profile-integrated-early-chroma-parity'
$source = Join-Path $worktree 'wrk_docs\evidence\2026-09-25-unit1434-three-clip-ip-attribution\capture-clip.ps1'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing reviewed capture source: $source" }
$captureText = [IO.File]::ReadAllText($source, [Text.Encoding]::UTF8)

function Replace-Once([string]$Old, [string]$New, [string]$Name) {
    $first = $script:captureText.IndexOf($Old, [StringComparison]::Ordinal)
    if ($first -lt 0 -or $script:captureText.IndexOf($Old, $first + $Old.Length, [StringComparison]::Ordinal) -ge 0) {
        throw "Expected exactly one $Name anchor in $source"
    }
    $script:captureText = $script:captureText.Replace($Old, $New)
}

Replace-Once 'D:\worktrees\rvp9-2\20260925-TSK-HUM-unit1434-integrated-ip-profile' $worktree 'worktree'
Replace-Once '88644d86641f99a4291c21842bdc17aec9baee8e' 'c717a24ad0e336c0249459d43223aee859755f0d' 'integrated HEAD'
Replace-Once '.scratch\target\release\rvp9enc.exe' '.scratch\encoder-pgo-control\optimized\release\rvp9enc.exe' 'production PGO EXE path'
Replace-Once '1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a' 'ce0b5bc008ab9700abacad99ea3445691e1a3ca8b5f114b9ce682e15d27730f2' 'production PGO EXE hash'
Replace-Once '27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4' '7dbf06fcda772d00e99403a86ab4811e20d2a96607392e527c18e5cf251c832d' 'paired PDB hash'
Replace-Once "'.scratch\unit1434-ip-profile'" "'.scratch\after-early-chroma-pgo-ip'" 'artifact root'

$tokens = $null
$errors = $null
[void][Management.Automation.Language.Parser]::ParseInput($captureText, [ref]$tokens, [ref]$errors)
if ($errors.Count -ne 0) { throw "Adapted capture parse errors: $($errors -join '; ')" }
$capture = [scriptblock]::Create($captureText)
foreach ($clip in $Clips) {
    if ($SmokeOnly) {
        . $capture -ClipName $clip -SmokeOnly
    } else {
        . $capture -ClipName $clip
    }
}
