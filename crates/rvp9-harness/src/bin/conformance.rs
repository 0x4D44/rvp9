//! VP9 conformance test runner.
//!
//! For each test vector under `--vectors-dir`, decodes with the configured
//! decoder using `--md5`. Reports pass/fail counts.
//!
//! **Stage 0:** the success criterion is that vpxdec decodes every vector
//! to completion (exit 0). The `.md5` file that ships with each vector
//! contains *per-frame* MD5s (one MD5 per decoded frame, with the
//! resolution embedded in the filename: `<name>-WxH-NNNN.i420`). Full
//! per-frame comparison is Stage 3 territory — when we ship our own
//! decoder, we'll parse those files properly and compare frame-by-frame.
//!
//! Vector layout (libvpx convention):
//!   <vectors_dir>/<name>.webm  (or .ivf)
//!   <vectors_dir>/<name>.<ext>.md5   <- per-frame YUV MD5s
//!
//! Usage:
//!   cargo run --release --bin conformance -- \
//!       --decoder      build/libvpx/vpxdec.exe \
//!       --vectors-dir  tests/conformance/vectors

use std::path::{Path, PathBuf};
use std::process::Command;

struct Args {
    decoder: PathBuf,
    vectors_dir: PathBuf,
    limit: Option<usize>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .and_then(|i| argv.get(i + 1).cloned())
    };
    Args {
        decoder: get("--decoder")
            .map(PathBuf::from)
            .expect("--decoder required"),
        vectors_dir: get("--vectors-dir")
            .map(PathBuf::from)
            .expect("--vectors-dir required"),
        limit: get("--limit").map(|s| s.parse().expect("--limit usize")),
    }
}

/// Stage 0 semantic: decoder exits cleanly. We use `--md5` to force the
/// decoder to actually walk every frame (no `--noblit`), without writing
/// hundreds of megabytes of YUV to disk.
fn run_one(decoder: &Path, vector: &Path) -> Result<(), String> {
    let output = Command::new(decoder)
        .args(["--codec=vp9", "--md5"])
        .arg(vector)
        .output()
        .map_err(|e| format!("invoke: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "decoder failed ({}): {}",
            output.status,
            stderr.lines().last().unwrap_or("").trim()
        ));
    }
    Ok(())
}

fn main() {
    let args = parse_args();

    let mut vectors: Vec<PathBuf> = match std::fs::read_dir(&args.vectors_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|x| matches!(x, "ivf" | "webm"))
            })
            .collect(),
        Err(e) => {
            eprintln!(
                "could not list vectors in {}: {e}",
                args.vectors_dir.display()
            );
            std::process::exit(2);
        }
    };
    vectors.sort();
    if let Some(n) = args.limit {
        vectors.truncate(n);
    }

    if vectors.is_empty() {
        println!("no .ivf vectors found in {}", args.vectors_dir.display());
        println!("(Stage 0: run scripts/fetch-vectors.ps1 to populate)");
        std::process::exit(0);
    }

    let mut passed = 0usize;
    let mut failed = 0usize;

    println!("running {} vector(s) against {}", vectors.len(), args.decoder.display());
    println!();

    for vector in &vectors {
        let name = vector.file_name().unwrap().to_string_lossy().to_string();
        match run_one(&args.decoder, vector) {
            Ok(()) => {
                println!("  PASS {name}");
                passed += 1;
            }
            Err(e) => {
                println!("  FAIL {name}  {e}");
                failed += 1;
            }
        }
    }

    println!();
    println!("Pass: {passed}   Fail: {failed}");

    if failed > 0 {
        std::process::exit(1);
    }
}
