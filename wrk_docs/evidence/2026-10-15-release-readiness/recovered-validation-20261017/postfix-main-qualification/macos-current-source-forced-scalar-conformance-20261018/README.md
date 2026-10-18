# macOS AArch64 conformance dispatch recheck

At exact revision d5947f2eccba2c1dd75e2f235df095d75d7af9fb on flux (macOS 26.6.2, arm64), the official conformance runner passed all 307 pinned vectors and 4,816 frames at one and two threads.

Both commands set RVP9_FORCE_SCALAR=1. That variable does not force scalar dispatch on AArch64: force_scalar_env is compiled only for x86_64, while AArch64Caps::detect reads the host NEON and dotprod features in crates/rvp9-core/src/dsp/caps.rs:85-123. These are native AArch64 conformance runs, not scalar runs. They confirm the current-candidate manifestation of [RVP9-BUG-FLU-00043](../../../../../../bugs/RVP9-BUG-FLU-00043.md), which remains Open.

The runner was built with Rust 1.95.0 using the locked release command. The fetch log records SHA-1 verification of all 307 vector files and their 307 MD5 ledgers against pinned libvpx commit 12f3a2ac603e8f10742105519e0cd03c3b8f71dd. The standard runner checks the decoded-output MD5 ledgers and does not invoke vpxdec.

See qualification.json for source, commands, counts, and artifact hashes. The raw fetch, build, and conformance logs are stored beside it. The per-thread conformance log hashes match the earlier native Mac logs exactly; the rebuilt runner has a current-revision SHA-256 recorded in qualification.json.
