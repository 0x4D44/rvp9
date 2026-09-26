param(
    [ValidateRange(1, 99)] [int]$StartAt = 2,
    [ValidateRange(1, 99)] [int]$Count = 9
)

$ErrorActionPreference = 'Stop'
$source = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\extract-repeats-ducks.ps1'
$expectedHash = '61a57516dd8def85a82630f1d8d7e414953d496b131e8fc901f2028c94651fa7'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing extraction source: $source" }
$actualHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $expectedHash) { throw "Extraction source SHA-256 mismatch: $actualHash" }
$text = [IO.File]::ReadAllText($source, [Text.Encoding]::UTF8)
$old = 'repeated-ducks-ip'
if (([regex]::Matches($text, [regex]::Escape($old))).Count -ne 1) {
    throw 'Expected exactly one repeated-ducks-ip path in extraction source.'
}
$text = $text.Replace($old, 'repeated-tree-ip')
$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Adapted extraction parse errors: $($parseErrors -join '; ')" }
& ([scriptblock]::Create($text)) -StartAt $StartAt -Count $Count
