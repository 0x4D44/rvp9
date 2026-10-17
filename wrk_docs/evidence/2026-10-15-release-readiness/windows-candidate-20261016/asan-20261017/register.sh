set -euo pipefail
cd /d/worktrees/rvp9-2/20261015-TSK-HUM-final-release-native-correctness-qualifi
[ "$(git rev-parse HEAD)" = 474dfd99a673df85f750136f7552ac156b7a24e9 ]
[ -z "$(git status --porcelain)" ]
deltic scratch new --label rvp9-native-asan474 --json
