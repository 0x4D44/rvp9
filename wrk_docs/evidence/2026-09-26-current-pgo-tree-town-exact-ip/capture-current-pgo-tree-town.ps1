# Run on Crucible with Windows PowerShell 5.1 from the diagnostic worktree:
#   powershell -NoProfile -File .\.scratch\capture-current-pgo-tree-town.ps1 -Preflight
#   powershell -NoProfile -File .\.scratch\capture-current-pgo-tree-town.ps1
# Defaults to three fresh-process Tree captures and three Town captures.
# After an interruption, use -Clip tree|town -StartAt N -Count N with unused run IDs.
param(
    [ValidateSet('tree', 'town', 'both')][string]$Clip = 'both',
    [ValidateRange(1, 99)][int]$Count = 3,
    [ValidateRange(1, 99)][int]$StartAt = 1,
    [switch]$Preflight,
    [switch]$ParseOnly,
    [string]$SourcePath
)

$ErrorActionPreference = 'Stop'
$diagnosticWorktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-capture-current-pgo-tree-and-town-exact'
$imageWorktree = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-prototype-nstep-duplicate-variance-score'
$diagnosticScratch = "$diagnosticWorktree\.scratch"
$inputRoot = 'D:\worktrees\rvp9-2\acceptance-d639c265\target\encoder-pgo\inputs'
if (-not $SourcePath) {
    $SourcePath = "$diagnosticWorktree\wrk_docs\evidence\2026-09-26-pgo-nstep-ducks-exact-ip\capture-pgo-nstep-ducks.ps1"
}

$specs = @{
    tree = [pscustomobject]@{
        File = 'in_to_tree_420_1080p30_60f.y4m'
        InputSha256 = 'ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791'
        OutputSha256 = '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'
    }
    town = [pscustomobject]@{
        File = 'old_town_cross_420_1080p30_60f.y4m'
        InputSha256 = 'b738b8068d1200f8a3119c72f1edc844424d3fe96d8b28acb9af11497ed7c909'
        OutputSha256 = '4c5e8ffa47656109df76e05ac0165a0509366dd2453d06ea690f403fdd65027f'
    }
}

function Get-Sha256([byte[]]$Bytes) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return [BitConverter]::ToString($sha.ComputeHash($Bytes)).Replace('-', '').ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Replace-Exactly([string]$Old, [string]$New, [string]$Name) {
    $count = [regex]::Matches($script:text, [regex]::Escape($Old)).Count
    if ($count -ne 1) {
        throw "Expected exactly one $Name anchor in $SourcePath; found $count"
    }
    $script:text = $script:text.Replace($Old, $New)
}

if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
    throw "Committed Ducks capture wrapper is missing: $SourcePath"
}
$source = [IO.File]::ReadAllText($SourcePath, [Text.Encoding]::UTF8)
$normalizedSource = $source.Replace([string]::Concat([char]13, [char]10), [string][char]10)
$sourceHash = Get-Sha256 ([Text.UTF8Encoding]::new($false).GetBytes($normalizedSource))
if ($sourceHash -ne '92ddf7ffbcfea87a4ff2d23fa5c5def0ed5fbaae1a71e50c2335043624b460fc') {
    throw "Committed Ducks capture wrapper SHA-256 mismatch: $sourceHash"
}

$selected = if ($Clip -eq 'both') { @('tree', 'town') } else { @($Clip) }
$adapted = @()
foreach ($name in $selected) {
    $spec = $specs[$name]
    $script:text = $normalizedSource
    $captureRoot = "$diagnosticScratch\pgo-current-$name-ip"
    $sourceCapture = "$diagnosticWorktree\wrk_docs\evidence\2026-09-25-unit1434-three-clip-ip-attribution\capture-clip.ps1"

    Replace-Exactly '$scratchPath = Join-Path $worktree ''.scratch''' ('$scratchPath = ''{0}''' -f $diagnosticScratch) 'diagnostic scratch'
    Replace-Exactly '$sourcePath = Join-Path $worktree ''wrk_docs\evidence\2026-09-25-unit1434-three-clip-ip-attribution\capture-clip.ps1''' ('$sourcePath = ''{0}''' -f $sourceCapture) 'capture source path'
    Replace-Exactly '$artifactRoot = Join-Path $scratchPath ''pgo-nstep-ducks-ip''' ('$artifactRoot = Join-Path $scratchPath ''pgo-current-{0}-ip''' -f $name) 'artifact root'
    Replace-Exactly '$encoderPath = Join-Path $scratchPath ''encoder-pgo-nstep\optimized\release\rvp9enc.exe''' '$encoderPath = Join-Path $worktree ''.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.exe''' 'EXE path'
    Replace-Exactly '$pdbPath = Join-Path $scratchPath ''encoder-pgo-nstep\optimized\release\rvp9enc.pdb''' '$pdbPath = Join-Path $worktree ''.scratch\encoder-pgo-nstep\optimized\release\rvp9enc.pdb''' 'PDB path'
    Replace-Exactly 'ducks_take_off_420_1080p30_60f.y4m' $spec.File 'input file'
    Replace-Exactly '6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7' $spec.InputSha256 'input hash'
    Replace-Exactly '8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089' $spec.OutputSha256 'IVF hash'
    Replace-Exactly 'Ducks input' ('{0} input' -f $name) 'input label'
    Replace-Exactly '-ClipName ducks' ('-ClipName {0}' -f $name) 'clip selection'

    # The nested reviewed capture runner uses the image worktree for HEAD and
    # EXE/PDB checks, while this replacement writes traces in diagnostic scratch.
    $oldRoot = [string]::Concat([char]39, [char]36, 'artifactRoot = Join-Path ', [char]36,
        'worktree ', [char]39, [char]39, '.scratch\pgo-nstep-ducks-ip', [char]39, [char]39, [char]39)
    $newRoot = [string]::Concat([char]39, [char]36, 'artifactRoot = ', [char]39, [char]39,
        $captureRoot, [char]39, [char]39, [char]39)
    Replace-Exactly $oldRoot $newRoot 'nested capture artifact root'

    $tokens = $null
    $parseErrors = $null
    [void][System.Management.Automation.Language.Parser]::ParseInput($script:text, [ref]$tokens, [ref]$parseErrors)
    if ($parseErrors.Count -ne 0) {
        throw "Adapted $name capture has PowerShell parse errors: $($parseErrors -join '; ')"
    }
    $adapted += [pscustomobject]@{
        Name = $name
        Text = $script:text
        Path = "$diagnosticScratch\capture-current-pgo-$name-adapted.ps1"
    }
}

if ($ParseOnly) {
    Write-Output "PARSE_OK clips=$($selected -join ',') source_sha256=$sourceHash"
    return
}
if ($env:OS -ne 'Windows_NT') { throw 'Run captures on Crucible Windows.' }
if (-not (Test-Path -LiteralPath $diagnosticScratch -PathType Container) -or
    ((Get-Item -LiteralPath $diagnosticScratch).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw "Diagnostic .scratch is missing or a reparse point: $diagnosticScratch"
}

# Preflight both clips before any capture. The reviewed wrapper checks WPR idle,
# exact image/input hashes, unique run paths, and the nested script's syntax.
foreach ($item in $adapted) {
    & ([scriptblock]::Create($item.Text)) -Count $Count -StartAt $StartAt -Preflight
}
if ($Preflight) {
    Write-Output 'PREFLIGHT_COMPLETE capture_started=false artifacts_created=false'
    return
}

foreach ($item in $adapted) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($item.Text)
    $expected = Get-Sha256 $bytes
    if (Test-Path -LiteralPath $item.Path) {
        $actual = (Get-FileHash -LiteralPath $item.Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "Refusing to replace a different adapted capture script: $($item.Path)"
        }
    } else {
        $stream = [IO.File]::Open($item.Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try { $stream.Write($bytes, 0, $bytes.Length) } finally { $stream.Dispose() }
    }
    # Invoke as a file to preserve the nested capture runner's script:logPath scope.
    & $item.Path -Count $Count -StartAt $StartAt
}
