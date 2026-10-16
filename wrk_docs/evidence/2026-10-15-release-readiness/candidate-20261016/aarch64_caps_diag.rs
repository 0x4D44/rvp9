#![allow(dead_code)]

#[path = "../../crates/rvp9-core/src/dsp/caps.rs"]
mod caps;

fn main() {
    println!("AArch64Caps::detect() = {:?}", caps::AArch64Caps::detect());
}
