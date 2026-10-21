# Independent Windows verification: RVP9-BUG-FLU-00058

Verified integrated fix `d872b7216cb7c2ca8f23e494b1ade99b202e22e9` on CRUCIBLE, verification lane base `a0b538c7`. The verifier did not implement the fix. The normal local claim succeeded with owner run `verify-20261021T072658Z-fd362310`; the remote claim first failed because its GitHub token was unavailable. No identity override or credential change was used.

The original timestamp-clock mismatch is resolved in real Windows output. Native pinned libvpx-vs-libvpx selfcomparison encoded a generated 128×64 clip, exactly 60 frames at 30fps, at four bitrates. All eight role streams use the genuine millisecond IVF clock. Selfcomparison returned 0% BD-rate; an independent PowerShell audit counted 60 packets and checked every CSV bitrate against `file_bytes * 8 / 1000 / (60 / 30)`, passing all eight points.

Fails-before proof changed only the duration expression back to `frames * clock_scale / clock_rate` in the verifier's own worktree. The rebuilt mutant still returned 0% selfcomparison, showing why that check alone is insufficient. The same eight-point independent audit failed its named `BITRATE_AUDIT_ASSERTION`, including 1,683.6kbps observed versus 50.508kbps expected at the first point. Raw failure and per-point JSON preserve all eight failures.

The source was restored from captured bytes using `WriteAllBytes`, rebuilt, and the same native selfcomparison plus audit reran green. Source/captured SHA256 hashes match and scoped Git diff is empty. All 19 cross-platform bdrate unit tests pass after restoration. The newly added file-based unit and `bdrate_fps` integration suite are Unix-only: the attempted specific Windows filter selected zero tests, explicitly not accepted as red proof. The native runtime audit provides the selected assertion proof instead.

No source fixes, commits or code integration were performed by this verifier. Verification summary records executable hashes, owner run and counts; copy-map binds remote raw bytes to normalized local logs. All streams remain in registered scratch `C:\Users\marti\AppData\Local\Temp\deltic-scratch\bug58-windows-verify-3b31d84b`.

Verdict: `deltic bugs verify --close` succeeded, transitioning Fixed → Closed and clearing the verifier claim. A fresh full 27-curve artifact-bound quality campaign is running with the repaired release helper; invalid prior rows are not resumed.

The fresh corrected quality campaign also independently audited all 12 real role rates for both original Tree and Ducks scenes. All 24 rates matched bytes/2seconds, and all 24 newly encoded stream hashes match retained pre-fix streams exactly. This repairs measurement of the original observations without claiming a codec change. See `two-scene-real-rate-audit.json` (PowerShell nested arrays appear under `rates.value`).
