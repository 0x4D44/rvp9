#[path="/Users/md/worktrees/rvp9/20261015-TSK-HUM-prepare-final-release-readiness-evidence/crates/rvp9-io/src/container.rs"]
mod container;
fn main(){ let bytes=[0x18,0x53,0x80,0x67,0x81,0x18,0x53,0x80,0x67,0xff]; println!("input={bytes:02x?}"); println!("memory={:?}",std::panic::catch_unwind(||container::read_webm_packets(&bytes))); }
