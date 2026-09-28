$ErrorActionPreference = 'Stop'
$bash = 'C:\Program Files\Git\bin\bash.exe'
$root = '/d/worktrees/rvp9-2/20260928-TSK-HUM-prototype-serial-tight-loop-filter-x86'
$training = '/d/worktrees/rvp9-2/20260927-TSK-HUM-fresh-integrated-production-pgo-direct-s/.scratch/canonical-corpus'
$command = "cd '$root' && scripts/build-encoder-pgo.sh --build-root '$root/.scratch/serial-lf-packet-pgo' --training-dir '$training' --frames 60"
& $bash -lc $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
