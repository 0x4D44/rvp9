#requires -Version 7
<#
.SYNOPSIS
    Fetch the G8 real-world corpus sources into tests/corpus/sources/.

.DESCRIPTION
    Windows / PowerShell companion to scripts/fetch-corpus.sh (POSIX).
    Downloads raw .y4m clips from Xiph.org's Derf collection (public,
    CC-licensed) into the gitignored tests/corpus/sources/ landing zone.

    Per HLD V6 §G8 (decoder-hardening) the full corpus is too large to
    commit (several hundred MB per clip). Skip-cleans if files are
    already present.

    Stage 0 seeds 720p clips. Larger clips (1080p, 4K) are added when
    a follow-up needs them.

    Blender Open-Movies sources and self-recorded sources are NOT
    fetched by this script — they require manual acquisition with the
    correct licensing breadcrumb. See `tests/corpus/sources/README.md`.

.PARAMETER WhatIf
    Print the planned downloads without fetching anything. Mirrors the
    standard PowerShell -WhatIf semantics.

.EXAMPLE
    pwsh ./scripts/fetch-corpus.ps1

.EXAMPLE
    pwsh ./scripts/fetch-corpus.ps1 -WhatIf

.NOTES
    Currently a real fetch script for the Derf subset. The encode step
    (`vpxenc` → IVF + .md5) is deferred to scripts/encode-corpus.ps1
    (lands as part of G8.1 once vpxenc is on PATH).
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param()

$ErrorActionPreference = 'Stop'

# Resolve repo root via `git rev-parse`; fall back to two-levels-up
# from this script if git is unavailable.
$repoRoot = $null
try {
    $repoRoot = (& git rev-parse --show-toplevel 2>$null).Trim()
} catch {
    # Ignore — handled below.
}
if (-not $repoRoot) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
}

$destDir = Join-Path $repoRoot 'tests/corpus/sources'
if (-not (Test-Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
}

$xiphBase = 'https://media.xiph.org/video/derf/y4m'

# Stage-0 source-to-local filename map — same as scripts/fetch-corpus.sh.
$clips = @(
    [pscustomobject]@{ Source = '720p50_parkrun_ter.y4m'; Local = 'parkrun_ter_720p50.y4m' },
    [pscustomobject]@{ Source = 'in_to_tree_420_720p50.y4m'; Local = 'in_to_tree_420_720p50.y4m' },
    [pscustomobject]@{ Source = 'old_town_cross_420_720p50.y4m'; Local = 'old_town_cross_420_720p50.y4m' },
    [pscustomobject]@{ Source = 'ducks_take_off_420_720p50.y4m'; Local = 'ducks_take_off_420_720p50.y4m' }
)

$fetched = 0
$skipped = 0
$failed = 0

foreach ($clip in $clips) {
    $target = Join-Path $destDir $clip.Local
    $url = "$xiphBase/$($clip.Source)"

    if ((Test-Path $target) -and ((Get-Item $target).Length -gt 0)) {
        Write-Host "[skip] $($clip.Local) (already present)"
        $skipped++
        continue
    }

    if ($PSCmdlet.ShouldProcess($target, "Download $url")) {
        Write-Host "[fetch] $($clip.Local) (several hundred MB; large download)"
        try {
            # Invoke-WebRequest streams to disk; Progress UI off for
            # large downloads (massive speedup on PowerShell 5/7).
            $oldProgress = $ProgressPreference
            $ProgressPreference = 'SilentlyContinue'
            try {
                Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -MaximumRetryCount 3 -RetryIntervalSec 5
            } finally {
                $ProgressPreference = $oldProgress
            }
            Write-Host "  done."
            $fetched++
        } catch {
            Write-Warning "  FAILED: $($_.Exception.Message)"
            if (Test-Path $target) {
                Remove-Item $target -Force -ErrorAction SilentlyContinue
            }
            $failed++
        }
    } else {
        Write-Host "[plan] $($clip.Local) <- $url"
    }
}

Write-Host ''
Write-Host "Corpus: $fetched fetched, $skipped already present, $failed failed."
Write-Host "Stored in: $destDir"

if ($failed -gt 0) {
    exit 1
}
