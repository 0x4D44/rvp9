param(
    [Parameter(Mandatory = $true)] [ValidateRange(1, 99)] [int]$Count,
    [ValidateRange(1, 99)] [int]$StartAt = 1,
    [switch]$Preflight
)

$ErrorActionPreference = 'Stop'
$source = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\capture-repeats-ducks.ps1'
$expectedSourceHash = 'f988743b398123143cef975219147a9a89a014a22483e341881e1d9ed54aed5f'
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing capture source: $source" }
$actualSourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSourceHash -ne $expectedSourceHash) { throw "Capture source SHA-256 mismatch: $actualSourceHash" }
$text = [IO.File]::ReadAllText($source, [Text.Encoding]::UTF8)

function Replace-Exactly([string]$Old, [string]$New, [int]$CountExpected) {
    $count = ([regex]::Matches($script:text, [regex]::Escape($Old))).Count
    if ($count -ne $CountExpected) { throw "Expected $CountExpected occurrences of $Old; found $count" }
    $script:text = $script:text.Replace($Old, $New)
}

Replace-Exactly 'ducks_take_off_420_1080p30_60f.y4m' 'in_to_tree_420_1080p30_60f.y4m' 1
Replace-Exactly '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7' 'ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791' 1
Replace-Exactly '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41' 1
Replace-Exactly 'repeated-ducks-ip' 'repeated-tree-ip' 2
Replace-Exactly 'Ducks input' 'Tree input' 1
Replace-Exactly '-ClipName ducks' '-ClipName tree' 1

$tokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "Adapted Tree capture parse errors: $($parseErrors -join '; ')" }

if ($Preflight) {
    & ([scriptblock]::Create($text)) -Count $Count -StartAt $StartAt -Preflight
    return
}

# The capture runner uses $script:logPath. Give it its own script scope: an
# invoked ScriptBlock would resolve that variable in this wrapper instead.
$adaptedPath = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch\capture-tree-adapted.ps1'
$bytes = [Text.UTF8Encoding]::new($false).GetBytes($text)
$sha = [Security.Cryptography.SHA256]::Create()
try { $expectedAdaptedHash = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant() }
finally { $sha.Dispose() }
if (Test-Path -LiteralPath $adaptedPath) {
    $actualAdaptedHash = (Get-FileHash -LiteralPath $adaptedPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualAdaptedHash -ne $expectedAdaptedHash) { throw "Adapted script SHA-256 mismatch: $actualAdaptedHash" }
} else {
    $stream = [IO.File]::Open($adaptedPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
}
& $adaptedPath -Count $Count -StartAt $StartAt
