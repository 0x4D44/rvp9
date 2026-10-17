# Declared task-local cargo-fuzz0.13.2 setup

One authorized exact-version locked install with normal public registry access succeeded (exit0), using isolated registered-scratch CARGO_HOME, --root/tools, separate tool-target and jobs2. Explicit resulting executable --version exited0 and reported cargo-fuzz0.13.2. Exact command/env/toolchain/source/exit/binary SHA-256 are in run.json. Original raw output is cargo-install.raw.txt.

Tool-source .crate SHA-256 equals the exact0.13.2 sparse-index checksum. Extracted source manifest, lockfile and VCS identity are preserved; tool-source-identity.json records their hashes and all task-local downloaded source archive hashes. Generated archives/builds/driver remain in registered scratch, not committed evidence. No global install, PATH/config edit or project dependency addition. Workspace/fuzz manifests and lockfile hashes are identical before/after; source-delta proof is captured separately.

Earlier offline resolution failure remains immutable in macos-fuzz-tool-rebuild. This setup result supersedes driver availability as a blocker, but proves no fuzz-target build, sanitizer instrumentation or runtime coverage. No fuzz target built/run; reported to lead before further work. No source edits, staging or commits. Files are final; integrity.json hashes each other compact evidence file.
