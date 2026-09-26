param(
    [ValidateRange(1, 99)] [int]$StartAt = 2,
    [ValidateRange(1, 99)] [int]$Count = 9
)

$ErrorActionPreference = 'Stop'
$profileScratch = 'D:\worktrees\rvp9-2\20260926-TSK-HUM-crucible-profile-after-zero-eob\.scratch'
$root = Join-Path $profileScratch 'repeated-tree-ip'
$aggregator = Join-Path $profileScratch 'aggregate-samples.py'
$output = Join-Path $root 'aggregate'
$expectedIvf = '5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41'
if ($StartAt + $Count - 1 -gt 99) { throw 'Run names end at run-99.' }
if (-not (Test-Path -LiteralPath $aggregator -PathType Leaf)) { throw "Missing aggregator: $aggregator" }
$aggregatorHash = (Get-FileHash -LiteralPath $aggregator -Algorithm SHA256).Hash.ToLowerInvariant()
if ($aggregatorHash -ne 'dcc4fcb8d90472e37368237787c00afa3227dc95487e473327ffad7758562f9b') {
    throw "Aggregator SHA-256 mismatch: $aggregatorHash"
}
if (Test-Path -LiteralPath $output) { throw "Refusing existing aggregate: $output" }

$argsList = @($aggregator, '--expected-runs', "$Count", '--start-at', "$StartAt")
for ($number = $StartAt; $number -lt $StartAt + $Count; $number++) {
    $name = 'run-{0:D2}' -f $number
    $run = Join-Path $root $name
    $verification = Join-Path $run 'verification.txt'
    $extracted = Join-Path $run 'extracted'
    if (-not (Test-Path -LiteralPath $verification -PathType Leaf) -or
        -not (Test-Path -LiteralPath $extracted -PathType Container)) {
        throw "Incomplete run: $name"
    }
    $record = [IO.File]::ReadAllText($verification)
    if ($record -notmatch [regex]::Escape("OUTPUT_SHA256=$expectedIvf") -or
        $record -notmatch '(?m)^LOST_BUFFERS=0\r?$' -or
        $record -notmatch '(?m)^LOST_EVENTS=0\r?$') {
        throw "Missing exact output or zero-loss proof: $name"
    }
    $argsList += @('--run', "$name=$extracted")
}
New-Item -ItemType Directory -Path $output -ErrorAction Stop | Out-Null
$argsList += @('--out-dir', $output)
& deltic timeout 600 python @argsList
if ($LASTEXITCODE -ne 0) { throw "Aggregate failed: exit=$LASTEXITCODE" }
