//! Marks fetched conformance vectors so captured-vector tests are visibly
//! ignored when their external oracle data is absent.

use std::path::PathBuf;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rvp9_conformance_vectors)");

    let workspace_root = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"),
    )
    .parent()
    .and_then(|path| path.parent())
    .expect("core crate is two levels below the workspace root")
    .to_owned();
    let vector = workspace_root
        .join("tests/conformance/vectors")
        .join("vp90-2-05-resize.ivf");
    println!("cargo::rerun-if-changed={}", vector.display());

    if vector.is_file() {
        println!("cargo::rustc-cfg=rvp9_conformance_vectors");
    }
}
