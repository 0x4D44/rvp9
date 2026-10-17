set -euo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
r=/c/Users/marti/AppData/Local/Temp/deltic-scratch/rvp9-native-replay474-9dee6be6
printf 'source_head='; git rev-parse HEAD
printf 'source_status='; git status --porcelain
printf 'sealed_manifest_begin\n'; cat "$r/production-speed/manifest.txt"; printf 'sealed_manifest_end\n'
printf 'timing_paths_begin\n'; find "$r/production-speed" -maxdepth 2 -type d; printf 'timing_paths_end\n'
printf 'quiet_state_begin\n'; deltic quiet status --json
