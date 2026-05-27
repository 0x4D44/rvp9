# run-kernel-audit.ps1 — head-to-head DSP kernel bench, rvp9 vs libvpx.
#
# Runs the two pinned micro-bench harnesses N times each, keeps the min
# ns/iter per (kernel, size), joins the two sides, verifies bit-exactness
# via FNV-1a 64-bit hashes of inputs and outputs, and writes a CSV +
# prints a ratio table. A hash divergence breaks the apples-to-apples
# guarantee → the script aborts with a nonzero exit.
#
# Output:
#   - CSV at bench/results/kernel-audit-<utc-stamp>.csv
#   - ratio table on stdout
#
# Usage (from anywhere):
#   pwsh -File scripts/run-kernel-audit.ps1
#   pwsh -File scripts/run-kernel-audit.ps1 -Affinity 0x1 -Runs 5
#
# Prereqs:
#   cargo build --release -p rvp9-bench --bin dsp-microbench
#   make -C tools/libvpx_kernel_bench   (MSYS2 mingw64 shell; see Makefile)
#
# Implementation notes (hard-won — see notes/decoder-libvpx-kernel-audit.md):
#   - Use $psi.Arguments (string), NOT $psi.ArgumentList — New-Object does
#     not populate the lazy ArgumentList collection, so Add()ed args never
#     reach the child process.
#   - Do not return arrays from helper functions with the `,$arr` idiom;
#     it double-wraps under foreach. This script keeps all collection
#     handling inline and imperative for that reason.

[CmdletBinding()]
param(
    [int]    $Affinity = 1,
    [int]    $Runs     = 5,
    [string] $RvpExe   = "target/release/dsp-microbench.exe",
    [string] $VpxExe   = "tools/libvpx_kernel_bench/libvpx_kernel_bench.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
Set-Location $repoRoot

if (-not (Test-Path $RvpExe)) {
    Write-Error "rvp9 bench not built: $RvpExe`n  cargo build --release -p rvp9-bench --bin dsp-microbench"
}
if (-not (Test-Path $VpxExe)) {
    Write-Error "libvpx bench not built: $VpxExe`n  make -C tools/libvpx_kernel_bench"
}

function Get-PinnedStdout {
    param([string] $Exe, [string] $ArgStr, [int] $Affinity)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = (Resolve-Path -LiteralPath $Exe).Path
    if ($ArgStr) { $psi.Arguments = $ArgStr }   # NB: NOT ArgumentList
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError  = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    try {
        $proc.ProcessorAffinity = [System.IntPtr]::new($Affinity)
        $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal
    } catch {
        Write-Warning "could not pin PID $($proc.Id): $_"
    }
    $out = $proc.StandardOutput.ReadToEnd()
    $err = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) { Write-Error "$Exe exited $($proc.ExitCode): $err" }
    return $out
}

# Accumulate min-ns row per "kernel|size" into a hashtable, in place.
function Add-MinRows {
    param([hashtable] $Into, [string] $Stdout)
    foreach ($ln in ($Stdout -split "`r?`n")) {
        if ($ln -match "^(rvp9|libvpx),([^,]+),([^,]+),([0-9.]+),([0-9a-f]+),([0-9a-f]+)$") {
            $key = "$($Matches[2])|$($Matches[3])"
            $ns  = [double]$Matches[4]
            if (-not $Into.ContainsKey($key) -or $ns -lt $Into[$key].Ns) {
                $Into[$key] = [pscustomobject]@{
                    Kernel = $Matches[2]; Size = $Matches[3]; Ns = $ns
                    InputHash = $Matches[5]; OutputHash = $Matches[6]
                }
            }
        }
    }
}

Write-Host "[kernel-audit] affinity 0x$([Convert]::ToString($Affinity,16)), $Runs run(s)/side" -ForegroundColor Cyan
Write-Host "[kernel-audit] rvp9:   $RvpExe"
Write-Host "[kernel-audit] libvpx: $VpxExe"

$rvp = @{}
$vpx = @{}
for ($i = 0; $i -lt $Runs; ++$i) {
    Add-MinRows $rvp (Get-PinnedStdout $RvpExe "vs_libvpx" $Affinity)
    Add-MinRows $vpx (Get-PinnedStdout $VpxExe ""          $Affinity)
}

$utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH-mm-ssZ")
$csvPath = "bench/results/kernel-audit-$utc.csv"
New-Item -ItemType Directory -Force -Path (Split-Path $csvPath) | Out-Null

$lines = @("kernel,size,rvp9_ns,libvpx_ns,ratio,input_match,output_match")
$report = @()
$abort = $null
foreach ($key in ($rvp.Keys | Sort-Object)) {
    if (-not $vpx.ContainsKey($key)) { Write-Warning "no libvpx row for $key"; continue }
    $r = $rvp[$key]; $v = $vpx[$key]
    $im = ($r.InputHash  -eq $v.InputHash)
    $om = ($r.OutputHash -eq $v.OutputHash)
    $ratio = $r.Ns / $v.Ns
    $lines  += "{0},{1},{2:F2},{3:F2},{4:F2},{5},{6}" -f $r.Kernel,$r.Size,$r.Ns,$v.Ns,$ratio,$im,$om
    $report += [pscustomobject]@{
        Kernel=$r.Kernel; Size=$r.Size; Rvp9Ns=[Math]::Round($r.Ns,2)
        VpxNs=[Math]::Round($v.Ns,2); Ratio=[Math]::Round($ratio,2)
        InMatch=$im; OutMatch=$om
    }
    if (-not $im -or -not $om) {
        $abort = "bit-exactness FAILED on $($r.Kernel) ($($r.Size)): in=$im out=$om"
    }
}

$lines | Set-Content -LiteralPath $csvPath -Encoding ascii
Write-Host ""
$report | Format-Table -AutoSize | Out-String | Write-Host
Write-Host "[kernel-audit] CSV -> $csvPath"
if ($abort) { Write-Error $abort }
