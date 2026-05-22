//! rvp9-io: shared byte-level I/O for the rvp9 toolchain.
//!
//! Container readers/writers and digest helpers that are needed by both
//! the shipped CLIs (`rvp9enc` / `rvp9dec` in `rvp9-cli`) and the test
//! harness (`rvp9-harness`). Keeping them here avoids making the
//! tooling/test crate a runtime dependency of the binaries.
//!
//! - [`container`] — IVF read/write + WebM (Matroska/EBML) read.
//! - [`y4m`] — Y4M (YUV4MPEG2) 4:2:0 reader and writer.
//! - [`md5`] — RFC 1321 MD5 + the `vpxdec --md5` visible-plane digest.

pub mod container;
pub mod md5;
pub mod y4m;
