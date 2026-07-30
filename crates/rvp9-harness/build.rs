//! Marks fetched conformance vectors so corpus-dependent tests report
//! as ignored when their external oracle data is absent.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rvp9_conformance_vectors)");

    let workspace_root = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    )
    .parent()
    .and_then(|path| path.parent())
    .expect("harness crate is two levels below the workspace root")
    .to_owned();
    let vectors_dir = workspace_root.join("tests/conformance/vectors");
    println!("cargo::rerun-if-changed={}", vectors_dir.display());

    let has_vector = std::fs::read_dir(vectors_dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            entry
                .path()
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "ivf" | "webm"))
        })
    });
    if has_vector {
        println!("cargo::rustc-cfg=rvp9_conformance_vectors");
    }
}
