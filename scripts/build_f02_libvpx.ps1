# F.0.2 — patched libvpx build driver (HLD §3.3 revision 5).
#
# Applies third_party/libvpx-patches/f02_oracle.patch to the libvpx
# checkout at third_party/libvpx, runs configure + make, and emits the
# patched `vpxenc.exe` at third_party/libvpx/build_f02/vpxenc.exe so
# the f02-block-oracle binary can find it without any extra config.
#
# Abort conditions (HLD §3.3 revision 5):
#   (a) third_party/libvpx is not a git checkout — patch can't apply.
#   (b) the libvpx commit doesn't match the patch's version-pin
#       header (`# libvpx commit: <hash> (v<tag>)` on line 1).
#   (c) the patch fails `git apply --check`.
#   (d) the build emits no `vpxenc.exe` (build silently broke).
#
# Idempotent: if the patch is already applied (verified by running
# `git apply -R --check`), skip the apply step. The script does NOT
# auto-revert if a build fails — the developer is expected to inspect
# the failure and re-run after fixing.
#
# Build environment: see scripts/build-libvpx.sh — the Windows path
# wants MSYS2's `make`, `gcc`, `yasm`, `perl`. If those are absent the
# script aborts loudly with the exact command the developer must run
# to install / set up the environment.

[CmdletBinding()]
param(
    [switch] $SkipBuild,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
# Native commands (git, make, bash) are checked via $LASTEXITCODE below.
# Disabling strict native-error preference so probes like `git apply -R
# --check` can return non-zero ("not applied") without aborting.
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = (Resolve-Path -Path (Join-Path $PSScriptRoot '..')).Path
$libvpxDir = Join-Path $repoRoot 'third_party/libvpx'
$patchPath = Join-Path $repoRoot 'third_party/libvpx-patches/f02_oracle.patch'
$buildDir  = Join-Path $libvpxDir 'build_f02'
$expectedBinary = Join-Path $buildDir 'vpxenc.exe'

# (a) — libvpx must be a git checkout under third_party/libvpx.
if (-not (Test-Path -LiteralPath (Join-Path $libvpxDir '.git'))) {
    # `.git` can be a file (submodule gitdir pointer) — check for either.
    if (-not (Test-Path -LiteralPath (Join-Path $libvpxDir '.git'))) {
        Write-Host ("ERROR: third_party/libvpx is not initialised. Run " +
                    "`git submodule update --init third_party/libvpx` first.") `
            -ForegroundColor Red
        exit 1
    }
}
if (-not (Test-Path -LiteralPath $patchPath)) {
    Write-Host "ERROR: patch file not found at $patchPath" -ForegroundColor Red
    exit 1
}

Push-Location $libvpxDir
try {
    # (b) — version-pin check. Read the patch's first-line header,
    # extract the expected commit hash, and compare against
    # `git rev-parse HEAD`.
    $patchHeader = Get-Content -LiteralPath $patchPath -TotalCount 1
    if ($patchHeader -notmatch '^#\s*libvpx commit:\s*([0-9a-fA-F]{7,40})\s*\(([^)]+)\)\s*$') {
        Write-Host ("ERROR: patch header on line 1 doesn't match the " +
                    "expected `# libvpx commit: <hash> (<tag>)` form. " +
                    "Got:`n  $patchHeader") -ForegroundColor Red
        exit 1
    }
    $expectedCommit = $Matches[1].ToLower()
    $expectedTag = $Matches[2]
    $actualCommit = (& git rev-parse HEAD).Trim().ToLower()
    if (-not $actualCommit.StartsWith($expectedCommit)) {
        Write-Host "ERROR: libvpx commit mismatch." -ForegroundColor Red
        Write-Host "  expected (per patch header): $expectedCommit ($expectedTag)" -ForegroundColor Red
        Write-Host "  found    (third_party/libvpx HEAD): $actualCommit" -ForegroundColor Red
        Write-Host "Re-base the patch against the current libvpx HEAD, or" -ForegroundColor Red
        Write-Host "checkout the pinned commit in third_party/libvpx." -ForegroundColor Red
        exit 1
    }
    Write-Host "OK: libvpx commit $actualCommit matches patch pin $expectedCommit ($expectedTag)."

    # Is the patch already applied? `git apply -R --check` succeeds
    # iff the reverse-apply would clean — i.e. the patch's additions
    # are present in the tree. A non-zero exit just means "not
    # already applied".
    & git apply --unidiff-zero -R --check $patchPath 2>$null
    $alreadyApplied = ($LASTEXITCODE -eq 0)
    $global:LASTEXITCODE = 0

    if ($alreadyApplied -and -not $Force) {
        Write-Host "OK: patch already applied (git apply -R --check succeeded). Use -Force to re-apply."
    } else {
        if ($alreadyApplied -and $Force) {
            Write-Host "Reverting previously-applied patch (per -Force)..."
            & git apply --unidiff-zero -R $patchPath
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: git apply -R failed; the working tree is dirty." -ForegroundColor Red
                exit 1
            }
        }
        # (c) — patch must apply cleanly. Use `--check` first so we
        # don't leave the tree half-applied on a malformed patch.
        & git apply --unidiff-zero --check $patchPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: patch failed `git apply --check`." -ForegroundColor Red
            Write-Host "  Patch: $patchPath" -ForegroundColor Red
            Write-Host "  Inspect rejection list with: cd third_party/libvpx; git apply --reject ../libvpx-patches/f02_oracle.patch" -ForegroundColor Red
            exit 1
        }
        & git apply --unidiff-zero $patchPath
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: git apply failed (post-check)." -ForegroundColor Red
            exit 1
        }
        Write-Host "OK: F.0.2 oracle patch applied to third_party/libvpx."
    }

    if ($SkipBuild) {
        Write-Host "SkipBuild set — stopping after patch apply. Build manually:" -ForegroundColor Yellow
        Write-Host "  Re-use scripts/build-libvpx.sh from a Git Bash / MSYS2 shell," -ForegroundColor Yellow
        Write-Host "  pointing BUILD_DIR at $buildDir." -ForegroundColor Yellow
        return
    }

    # The libvpx Makefile build is driven by ./configure + GNU make;
    # PowerShell can't drive that natively on Windows. We delegate to
    # the existing `scripts/build-libvpx.sh` if MSYS2 is available;
    # otherwise we print the manual command and exit with a non-zero
    # code so callers know the build isn't done.
    $bashExe = $null
    foreach ($candidate in @(
        'C:\Program Files\Git\bin\bash.exe',
        'C:\Program Files\Git\usr\bin\bash.exe',
        'C:\msys64\usr\bin\bash.exe'
    )) {
        if (Test-Path -LiteralPath $candidate) {
            $bashExe = $candidate
            break
        }
    }
    if ($null -eq $bashExe) {
        Write-Host "WARN: no MSYS2/Git Bash detected." -ForegroundColor Yellow
        Write-Host "Manual build steps (from a Git Bash or MSYS2 mingw64 shell):" -ForegroundColor Yellow
        Write-Host "  cd $libvpxDir" -ForegroundColor Yellow
        Write-Host "  mkdir -p build_f02 && cd build_f02" -ForegroundColor Yellow
        Write-Host "  ../configure --disable-vp8 --enable-vp9 --disable-docs \\" -ForegroundColor Yellow
        Write-Host "      --disable-examples --disable-unit-tests" -ForegroundColor Yellow
        Write-Host "  make -j8 vpxenc" -ForegroundColor Yellow
        Write-Host "Re-run this script with -SkipBuild to verify the patch state without building." -ForegroundColor Yellow
        exit 2
    }
    if (-not (Test-Path -LiteralPath $buildDir)) {
        New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    }
    # Invoke the build in MSYS2 bash. The libvpx `configure` script
    # expects a POSIX environment.
    $configCmd = (
        "cd '$($buildDir -replace '\\','/')' && " +
        "'../configure' --disable-vp8 --enable-vp9 --disable-docs " +
        "--disable-examples --disable-unit-tests && " +
        'make -j8 vpxenc'
    )
    Write-Host "Building patched vpxenc via $bashExe..."
    & $bashExe -lc $configCmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: libvpx build failed (exit $LASTEXITCODE)." -ForegroundColor Red
        Write-Host "  Re-run manually from a Git Bash shell to see the full output:" -ForegroundColor Red
        Write-Host "    cd $buildDir; $configCmd" -ForegroundColor Red
        exit 1
    }

    # (d) — must produce vpxenc.exe at the expected path.
    if (-not (Test-Path -LiteralPath $expectedBinary)) {
        Write-Host "ERROR: build completed but $expectedBinary does NOT exist." -ForegroundColor Red
        Write-Host "  Check the make output above for silent build failures." -ForegroundColor Red
        exit 1
    }
    Write-Host "OK: patched vpxenc built at $expectedBinary."
    Write-Host "Smoke test: VPX_F02_ORACLE_OUT=/tmp/f02_test.jsonl $expectedBinary --help" -ForegroundColor Green
}
finally {
    Pop-Location
}
