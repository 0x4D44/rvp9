# Windows repaired-oracle beta campaign

Fresh unchanged beta-CBR profile execution at source `a0b538c7` containing integrated bitrate fix `d872b7216cb7c2ca8f23e494b1ade99b202e22e9`. Existing staged portable CLI binaries remain from `15bf7162c9fb92a5165fab652460bd42da03a6e6`; the checked-in artifact guard verified all guarded encoder/decoder shipped paths are unchanged. Helpers other than rebuilt bdrate and the profile runner retain previous hashes.

This is a fresh 27-curve campaign, including all three clips at all three targets and CPU6/7/8. No invalid pre-fix quality rows were resumed. The repaired release helper was built before this run. Quality passed with exit 0: all 27 curves/clip checks and nine aggregate layer gates passed; 162 candidate streams completed 486 decoder runs, and the AQ oracle passed with activation evidence. No rows were reused.

Registered output root: `C:\Users\marti\AppData\Local\Temp\deltic-scratch\bug58-windows-verify-3b31d84b\quality-fixed`. Every candidate stream is cross-decoded by actual staged rvp9dec, pinned vpxdec and ffmpeg under the checked-in contract. Independent native rate verification and old-duration assertion proof reside in sibling `bug58-verification/`.
