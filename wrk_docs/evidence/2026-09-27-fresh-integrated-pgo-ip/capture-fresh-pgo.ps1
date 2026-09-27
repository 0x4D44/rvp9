param(
    [ValidateSet('ducks', 'tree', 'town')][string[]]$Clips = @('ducks', 'tree', 'town'),
    [switch]$SmokeOnly
)
$ErrorActionPreference = 'Stop'

$worktree = 'D:\worktrees\rvp9-2\20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s'
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
Replace-Once '88644d86641f99a4291c21842bdc17aec9baee8e' '75c23fe224eedb26c9a014040c42b67e3a1b5fa1' 'integrated HEAD'
Replace-Once '.scratch\target\release\rvp9enc.exe' '.scratch\fresh-integrated-pgo\optimized\release\rvp9enc.exe' 'production PGO EXE path'
Replace-Once '1b64fc5f3de6dc3cfe9d486f6b2515be7a44354eeb0b025ca28e6e9903d1470a' '3c294166213bce7dc707734bf6fbe86e2387b23ef533c16ed16acf4780aaf3ad' 'production PGO EXE hash'
Replace-Once '27c0b186f4a2a5b2edd19a665ccf8cc1bff1786c90df3a906ed9a1a6e58f72b4' 'c41525ee2ccd56d6bda7e4e9e0880452437d5ac7981bc2af4f1b7c34a71943f8' 'paired PDB hash'
Replace-Once "'.scratch\unit1434-ip-profile'" "'.scratch\fresh-integrated-pgo-ip'" 'artifact root'

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
