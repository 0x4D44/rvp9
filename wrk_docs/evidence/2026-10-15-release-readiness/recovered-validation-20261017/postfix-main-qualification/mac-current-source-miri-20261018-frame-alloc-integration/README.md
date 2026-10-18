# `rvp9-core` `frame_alloc` Miri integration target

The exact-source Apple Silicon Miri attempt exited 101 while compiling `rvp9-core` as a dependency, before the integration-test binary ran. The compiler reported eight E0425 errors for missing scalar transform helpers in `transforms/forward.rs` and `transforms/inverse.rs`; no test started, passed, or failed. This is the same dependent-library `cfg(miri)` failure tracked by [RVP9-BUG-FLU-00037](../../../../../../bugs/RVP9-BUG-FLU-00037.md), not a new defect.

The target contains three tests: two reject oversized custom-border allocations, and one accepts an ordinary custom border. The passing `rvp9-core --lib` Miri suite compiles its library with `cfg(test)`; it does not cover this integration target's non-test library build.

The exact command, source and toolchain identity, raw compiler output, exit status, and artifact hashes are preserved in this bundle. The failure confirms that the three integration tests remain without Miri coverage.
