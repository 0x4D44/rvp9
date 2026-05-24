# encode-corpus.ps1 — generate VP9 fixtures from raw y4m sources using the
# in-tree libvpx reference encoder (Wave L L6).
#
# Resolves the long-standing "vpxenc is not on PATH" blocker behind G5.3 /
# G6.1 / G8.1: a working `vpxenc.exe` / `vpxdec.exe` live under
# `build/libvpx/`, and raw `.y4m` sources under `tests/corpus/`. This script
# wraps the proven generation recipe:
#
#   1. vpxenc  <src>.y4m            -> <out>.ivf       (the fixture)
#   2. vpxdec  --i<420|422|440|444> -o <pat>.iNNN <out>.ivf  (one frame/file)
#   3. md5sum  *.iNNN  (two-space normalised)         -> <out>.ivf.md5
#
# Step 1 alone produces a decode-only fixture (sufficient for the
# decode-throughput bench — G6.1, which times decode and does not bit-check).
# Steps 2-3 additionally emit a per-frame visible-plane MD5 ledger in the
# format the conformance harness expects (`<hex>  <stem>-<W>x<H>-<NNNN>.iNNN`,
# matching `rvp9_io::md5::md5_of_yv12_visible`, which mirrors
# `vpxdec.c:update_image_md5`). Pass -WithLedger to generate it.
#
# Track T-2 extension: -Profile1Format {420|422|440|444} drives
#   --profile={0,1} + --i{420,422,440,444} + ledger extension iNNN.
# Plus passthroughs -TileColumns / -FrameParallel / -KfMaxDist /
# -LagInFrames / -AqMode / -Sharpness for feature-flag variations.
#
# For Profile-1 (non-420), the source MUST be a y4m with matching chroma
# (foreman_c444.y4m / foreman_c422.y4m), OR a raw .yuv where the script
# auto-derives width/height/fps and passes --i440 + -w/-h/--fps.
#
# Y4M format does NOT support yuv440p (libvpx y4minput.c rejects it), so
# 4:4:0 sources are stored as raw .yuv files (per-frame Y..U..V planes,
# Y at WxH, U,V at Wx(H/2)). Filenames must encode WxH e.g.
# `foreman_c440.yuv` paired with -Width/-Height/-FpsNum/-FpsDen flags.
#
# NOTE (Wave L L6 finding): our decoder is bit-exact against the *official*
# WebM-project conformance corpus (201/201, fetched via fetch-vectors.sh) but
# diverges on some features emitted by a default `vpxenc` encode. So a
# self-encoded `.ivf.md5` ledger is a valid libvpx reference, but our decoder
# does not yet pass arbitrary self-encoded conformance vectors — track that as
# a separate decoder-robustness follow-up before relying on -WithLedger for a
# conformance gate. Decode-only bench fixtures (no ledger) are unaffected.
#
# Usage:
#   pwsh scripts/encode-corpus.ps1 -Src tests/corpus/testsrc_720p30.y4m `
#        -Out tests/corpus/fixtures/testsrc_720p30.ivf [-Limit 60] `
#        [-ExtraArgs '--passes=1','--lag-in-frames=0','--auto-alt-ref=0'] `
#        [-WithLedger]
#
#   pwsh scripts/encode-corpus.ps1 -Src tests/corpus/profile1/foreman_c444.y4m `
#        -Out tests/corpus/profile1/foreman_c444_kfonly.ivf -Limit 10 `
#        -Profile1Format 444 -KfMaxDist 1 -LagInFrames 0 -WithLedger

param(
    [Parameter(Mandatory = $true)] [string] $Src,
    [Parameter(Mandatory = $true)] [string] $Out,
    [int] $Limit = 0,
    [string[]] $ExtraArgs = @('--passes=1', '--lag-in-frames=0', '--auto-alt-ref=0'),
    [switch] $WithLedger,
    # Track T-2: Profile-1 chroma format. 420 = Profile-0 (default).
    [ValidateSet('420','422','440','444')]
    [string] $Profile1Format = '420',
    # Feature-flag passthroughs (negative = "do not pass the flag").
    [int] $TileColumns = -1,
    [int] $FrameParallel = -1,
    [int] $KfMaxDist = -1,
    [int] $LagInFrames = -1,
    [int] $AqMode = -1,
    [int] $Sharpness = -1,
    # Raw-yuv input (required when feeding a .yuv for I440 since y4m
    # cannot encode yuv440p).
    [int] $Width = 0,
    [int] $Height = 0,
    [int] $FpsNum = 30000,
    [int] $FpsDen = 1001
)

$ErrorActionPreference = 'Stop'
$repo = (git rev-parse --show-toplevel).Trim()
$vpxenc = Join-Path $repo 'build/libvpx/vpxenc.exe'
$vpxdec = Join-Path $repo 'build/libvpx/vpxdec.exe'

foreach ($tool in @($vpxenc, $vpxdec)) {
    if (-not (Test-Path $tool)) {
        throw "Missing $tool — build libvpx into build/libvpx/ first."
    }
}
if (-not (Test-Path $Src)) { throw "Source not found: $Src" }

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }

# Map chroma format -> (vpxenc input flag, --profile=, ledger extension).
switch ($Profile1Format) {
    '420' { $inFlag = '--i420'; $profile = '0'; $ledgerExt = 'i420' }
    '422' { $inFlag = '--i422'; $profile = '1'; $ledgerExt = 'i422' }
    '440' { $inFlag = '--i440'; $profile = '1'; $ledgerExt = 'i440' }
    '444' { $inFlag = '--i444'; $profile = '1'; $ledgerExt = 'i444' }
}

# 1. Encode the fixture.
$encArgs = @('--codec=vp9', '--ivf', "--profile=$profile", $inFlag) + $ExtraArgs

# Raw yuv input needs -w/-h/--fps (y4m carries them in-band; raw doesn't).
$isRaw = $Src.ToLower().EndsWith('.yuv')
if ($isRaw) {
    if ($Width -le 0 -or $Height -le 0) {
        throw "Raw .yuv input requires -Width and -Height."
    }
    $encArgs += @("-w", "$Width", "-h", "$Height", "--fps=$FpsNum/$FpsDen")
}

# Feature-flag passthroughs (only emit when set).
if ($TileColumns   -ge 0) { $encArgs += "--tile-columns=$TileColumns" }
if ($FrameParallel -ge 0) { $encArgs += "--frame-parallel=$FrameParallel" }
if ($KfMaxDist     -ge 0) { $encArgs += "--kf-max-dist=$KfMaxDist" }
if ($LagInFrames   -ge 0) { $encArgs += "--lag-in-frames=$LagInFrames" }
if ($AqMode        -ge 0) { $encArgs += "--aq-mode=$AqMode" }
if ($Sharpness     -ge 0) { $encArgs += "--sharpness=$Sharpness" }

if ($Limit -gt 0) { $encArgs += "--limit=$Limit" }
$encArgs += @('-o', $Out, $Src)
Write-Output "vpxenc $($encArgs -join ' ')"
& $vpxenc @encArgs
if ($LASTEXITCODE -ne 0) { throw "vpxenc failed with exit $LASTEXITCODE" }
Write-Output "encoded: $Out ($((Get-Item $Out).Length) bytes)"

if (-not $WithLedger) { return }

# 2-3. Per-frame visible-plane MD5 ledger (chroma-aware).
#
# vpxdec's `-o <pattern>` per-frame splitting only works for --yv12 /
# --i420 (its own help text). For non-420 chroma we use --rawvideo into
# one concatenated stream and split in PowerShell (mirrors what
# oracle-ledger-ffmpeg.ps1 does on the ffmpeg side).
$stemBase = [System.IO.Path]::GetFileName($Out)  # full filename (e.g. foo.ivf) — matches existing ledger stem.
$ledger = "$Out.md5"

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("encode-corpus-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $work | Out-Null
try {
    if ($ledgerExt -eq 'i420') {
        # Profile-0 fast path: per-frame split done by vpxdec itself.
        $pattern = Join-Path $work "$stemBase-%wx%h-%4.i420"
        & $vpxdec --i420 -o $pattern $Out | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "vpxdec failed with exit $LASTEXITCODE" }
        Get-ChildItem -Path $work -Filter "$stemBase-*.i420" | Sort-Object Name | ForEach-Object {
            $hash = (Get-FileHash -Algorithm MD5 $_.FullName).Hash.ToLower()
            "$hash  $($_.Name)"
        } | Set-Content -Path $ledger -Encoding ascii
    } else {
        # Profile-1 (--i422 / --i440 / --i444): decode to a single raw
        # stream and split per-frame in PowerShell. The frame layout
        # matches `md5_of_yv12_visible`: Y at WxH, U,V at chroma dims.
        # Use ffprobe on the encoded ivf to get the output width/height
        # (vpxdec respects the bitstream's chroma subsampling exposed
        # via --rawvideo, with planar order Y..U..V).
        $probe = & ffprobe -v error -select_streams v:0 -show_entries stream=width,height,pix_fmt -of default=nw=1 $Out
        $probeMap = @{}
        foreach ($l in $probe) {
            $kv = $l -split '=', 2
            if ($kv.Count -eq 2) { $probeMap[$kv[0]] = $kv[1] }
        }
        $w  = [int]$probeMap['width']
        $ht = [int]$probeMap['height']
        switch ($ledgerExt) {
            'i422' { $cw = $w/2; $ch = $ht }
            'i440' { $cw = $w;   $ch = $ht/2 }
            'i444' { $cw = $w;   $ch = $ht }
        }
        $framelen = $w*$ht + 2 * $cw * $ch

        $raw = Join-Path $work 'all.yuv'
        & $vpxdec --rawvideo -o $raw $Out | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "vpxdec --rawvideo failed with exit $LASTEXITCODE" }
        $bytes = (Get-Item $raw).Length
        if ($bytes % $framelen -ne 0) { throw "raw $bytes not divisible by framelen $framelen" }
        $frames = [int]($bytes / $framelen)

        $list = New-Object System.Collections.Generic.List[string]
        $fs = [System.IO.File]::OpenRead($raw)
        try {
            $buf = New-Object byte[] $framelen
            for ($i = 1; $i -le $frames; $i++) {
                [void]$fs.Read($buf, 0, $framelen)
                $m = [System.Security.Cryptography.MD5]::Create()
                $hash = ($m.ComputeHash($buf) | ForEach-Object { $_.ToString('x2') }) -join ''
                $m.Dispose()
                $name = "{0}-{1}x{2}-{3:D4}.{4}" -f $stemBase, $w, $ht, $i, $ledgerExt
                $list.Add("$hash  $name")
            }
        } finally { $fs.Dispose() }
        Set-Content -Path $ledger -Value $list -Encoding ascii
    }
    Write-Output "ledger: $ledger ($((Get-Content $ledger).Count) frames)"
}
finally {
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

# Wave O: also emit the ffmpeg-native ledger and cross-check it agrees
# with vpxdec, so harness tests using `read_triangulated_ledger` find
# both files and the oracles are verified independent. Best-effort — if
# ffmpeg isn't on PATH, warn rather than fail (some sandboxes don't have
# it, and the vpxdec ledger alone is still useful for decode-only work).
$ffScript = Join-Path $repo 'scripts/oracle-ledger-ffmpeg.ps1'
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    & pwsh -NoProfile -File $ffScript -Vector $Out -Verify
    if ($LASTEXITCODE -ne 0) { throw "oracle-ledger-ffmpeg.ps1 -Verify failed with exit $LASTEXITCODE — ORACLE DISAGREEMENT, do not adopt this vector" }
} else {
    Write-Warning "ffmpeg not on PATH — skipped ffmpeg cross-check. Run later with: pwsh $ffScript -Vector $Out -Verify"
}
