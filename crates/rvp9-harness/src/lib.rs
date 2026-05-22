//! rvp9-harness: shared utilities for the test-harness binaries.
//!
//! See `src/bin/conformance.rs` and `src/bin/diff_bitstream.rs`.

#![allow(missing_docs)]

pub mod bdrate;
pub mod md5file;
pub mod y4m;

// Container (IVF/WebM) and MD5 helpers moved to the shared `rvp9-io`
// crate so the shipped CLIs can use them without depending on this
// test-harness crate. Re-exported here to keep every existing call site
// (`rvp9_harness::container::*`, `rvp9_harness::{md5, hex16,
// md5_of_file, md5_of_yv12_visible}`, plus `crate::hex16` inside
// `md5file`) compiling unchanged.
pub use rvp9_io::container;
pub use rvp9_io::md5::{hex16, md5, md5_of_file, md5_of_yv12_visible};
