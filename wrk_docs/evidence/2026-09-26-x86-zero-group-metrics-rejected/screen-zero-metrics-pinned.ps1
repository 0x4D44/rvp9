param(
    [int]$Pairs = 15,
    [string]$RunLabel = 'zero-metrics-pinned-01',
    [string]$CandidateHash
)
$ErrorActionPreference = 'Stop'
$worktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-screen-x86-zero-quantized-hadamard-metri'
$root = Join-Path $worktree '.scratch'
$source = Join-Path $worktree 'wrk_docs\evidence\2026-09-26-one-filter-result-forwarding\result-forward-screen.ps1'
$control = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score\.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe'
$candidate = Join-Path $root 'encoder-pgo-zero-quantized\optimized\release\rvp9enc.exe'
$controlHash = '3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee'
if ($Pairs -lt 1 -or $Pairs -gt 30) { throw 'Pairs must be 1..30' }
if ($CandidateHash -notmatch '^[0-9a-fA-F]{64}$') { throw 'CandidateHash must be SHA-256' }
if ($RunLabel -notmatch '^[a-zA-Z0-9-]+$') { throw 'RunLabel must be simple and unique' }
if (Test-Path -LiteralPath (Join-Path $root $RunLabel)) { throw "Run root already exists: $RunLabel" }
foreach ($item in @(@{Path=$control;Hash=$controlHash}, @{Path=$candidate;Hash=$CandidateHash})) {
    $actual = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item.Hash.ToLowerInvariant()) { throw "EXE hash mismatch: $($item.Path) actual=$actual" }
}
$text = [IO.File]::ReadAllText($source)
$old = "`$control = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-profile-integrated-subpel-pass\.scratch\target\release\rvp9enc.exe'"
$new = "`$control = '$control'"
$first = $text.IndexOf($old, [StringComparison]::Ordinal)
if ($first -lt 0 -or $text.IndexOf($old, $first + $old.Length, [StringComparison]::Ordinal) -ge 0) {
    throw 'Expected exactly one control-path anchor'
}
$text = $text.Replace($old, $new)
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Adapted screen parse errors: $($parseErrors -join '; ')" }
$scriptPath = Join-Path $root 'adapted-screen-zero-metrics-pinned.ps1'
if (Test-Path -LiteralPath $scriptPath) {
    if ([IO.File]::ReadAllText($scriptPath) -cne $text) { throw "Adapted script changed: $scriptPath" }
} else {
    [IO.File]::WriteAllText($scriptPath, $text)
}
& powershell -NoProfile -File $scriptPath -Pairs $Pairs -RunLabel $RunLabel -AffinityMask 1 -CandidatePath $candidate -RootPath $root
if ($LASTEXITCODE -ne 0) { throw "Paired screen failed: exit=$LASTEXITCODE" }
