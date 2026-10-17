# Task-local offline cargo-fuzz tool rebuild

One authorized attempt to rebuild the existing declared cargo-fuzz tool at exact version0.13.2, using offline/locked Cargo, a new registered scratch --root/tools, separate --target-dir and jobs2. No global install or PATH/config change; no project dependency addition; no fuzz target built or run.

Terminal exit101 after1.7seconds: `error: could not find cargo-fuzz in registry crates-io with version =0.13.2` (original diagnostic retains Cargo quoting in cargo-install.raw.txt). The local cache inspection found no exact cargo-fuzz0.13.2 archive, source manifest or source lock at the inspected registry cache/source locations. This establishes that this offline attempt cannot resolve the declared tool; it does not establish absence from online registries. Transitive cache sufficiency was not reached.

Stopped at the first actual diagnostic, with no online retry/download/install workaround. Therefore no resulting executable/version/hash exists to verify, and no bounded fuzz run is claimed. Project and fuzz Cargo.toml/Cargo.lock hashes are identical before and after. Exact source/toolchain/command/env/cache inspection and exit are in run.json; original raw output and hashes are preserved. Files are final; integrity.json hashes every other compact file. No source edits, staging or commits.
