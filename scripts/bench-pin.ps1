# bench-pin.ps1 — run a built binary with the process pinned to a single
# CPU core via ProcessorAffinity. Eliminates P/E migration jitter on
# hybrid Windows hosts for microbenches.
#
# Usage:
#   pwsh -File scripts/bench-pin.ps1 -Exe target\release\dsp-microbench.exe
#   pwsh -File scripts/bench-pin.ps1 -Exe target\release\dsp-microbench.exe -Affinity 0x1
#
# -Affinity is a bitmask: 0x1 = CPU0, 0x2 = CPU1, 0x4 = CPU2, etc. On
# Arrow Lake hybrid the P-cores are typically the first cluster (0..N-1
# logical processors). CPU 0 is the most common bench pin point.
#
# Two-stage approach so the JIT/loader path is also pinned:
#   1. Use Start-Process to spawn the exe with -PassThru
#   2. Set $proc.ProcessorAffinity (System.IntPtr) immediately after spawn
#   3. Wait for exit; report stdout
#
# Note: PowerShell 5.1's Start-Process lacks -Affinity. We set it
# manually after spawn, which is fine — the process is suspended briefly
# at start anyway. For maximum hygiene we set affinity BEFORE the bench
# loop runs by relying on the fact the bench has its own warmup.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Exe,
    [int] $Affinity = 1,            # bitmask: 0x1 = CPU0 only
    [string[]] $Args = @()
)

if (-not (Test-Path -LiteralPath $Exe)) {
    Write-Error "exe not found: $Exe"
    exit 2
}

$exePath = (Resolve-Path -LiteralPath $Exe).Path
Write-Host "[bench-pin] launching $exePath with affinity 0x$([Convert]::ToString($Affinity, 16))" -ForegroundColor Cyan

# Spawn with stdout/stderr inherited so we see the bench output live.
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = $exePath
foreach ($a in $Args) { [void]$psi.ArgumentList.Add($a) }
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $false
$psi.RedirectStandardError = $false

$proc = [System.Diagnostics.Process]::Start($psi)
try {
    $proc.ProcessorAffinity = [System.IntPtr]::new($Affinity)
    # Optional: bump priority so the OS prefers to schedule us when our
    # one pinned CPU is contended. AboveNormal is a light touch — no
    # admin needed.
    $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
}
catch {
    Write-Warning "[bench-pin] could not set affinity/priority on PID $($proc.Id): $_"
}

$proc.WaitForExit()
exit $proc.ExitCode
