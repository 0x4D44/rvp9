# Q.1.A decoder-throughput input preflight

Checked at 2026-10-17 15:28 UTC against current product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`:

| Worktree | HEAD | Raw Y4M files | Encoded IVF files |
| --- | --- | --- | --- |
| `/Users/md/language/rvp9` | `fb5c0c91` | all absent | all absent |
| `/Users/md/worktrees/rvp9/20261015-TSK-HUM-prepare-final-release-readiness-evidence` | `d38f8e92` (evidence branch) | all absent | all absent |
| `/Users/md/worktrees/rvp9/20261017-TSK-HUM-requalify-post-fix-candidate` | `fb5c0c91` | all absent | all absent |

The six checked paths in each worktree are:

- `tests/corpus/testsrc_720p30.y4m`, `testsrc_1080p30.y4m`, `testsrc_1080p60.y4m`
- `tests/corpus/fixtures/testsrc_720p30.ivf`, `testsrc_1080p30.ivf`, `testsrc_1080p60.ivf`

The Q.1.A HLD identifies the raw Y4M paths and points to `scripts/encode-corpus.ps1` for encoding, but the inspected HLD and benchmark orchestrator do not define or hash a recipe for generating those raw sources. `-SkipEncode` therefore cannot run the complete nine-row matrix, and the default Windows orchestrator cannot prepare rows C–E from the absent inputs. This is a missing-input prerequisite, not a decoder result. No benchmark or fixture generation was started, and no synthetic replacement is proposed.

The evidence worktree is not an exact-source build tree: compared with `origin/main`, its CLI and encoder source files differ. Do not use its existing `target` binaries as current-candidate evidence. The exact-source candidate worktree was checked separately and is also missing the six paths above.
