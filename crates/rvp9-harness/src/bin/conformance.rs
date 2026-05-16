//! VP9 conformance test runner.
//!
//! For each test vector under `--vectors-dir`, decodes with the configured
//! decoder and checks the output YUV MD5 against the reference `.md5` that
//! ships with the vector. Reports pass/fail counts and a per-vector log.
//!
//! In Stage 0 this runs against `libvpx vpxdec` as a sanity check that our
//! oracle (the test vectors + the reference decoder) works. In Stage 3 we
//! re-point `--decoder` at `rvp9dec` to gate the decoder out of scope.
//!
//! Vector layout (libvpx convention):
//!   <vectors_dir>/<name>.ivf
//!   <vectors_dir>/<name>.ivf.md5    <- reference YUV MD5
//!
//! Usage:
//!   cargo run --release --bin conformance -- \
//!       --decoder      D:/language/libvpx-stock/vpxdec.exe \
//!       --vectors-dir  tests/conformance/vectors

use std::path::{Path, PathBuf};
use std::process::Command;

use rvp9_harness::{hex16, md5_of_file};

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

fn run_one(decoder: &Path, ivf: &Path) -> Result<String, String> {
    let yuv_out = std::env::temp_dir().join(format!(
        "rvp9_conf_{}.yuv",
        ivf.file_stem().unwrap().to_string_lossy()
    ));
    let status = Command::new(decoder)
        .args(["--codec=vp9", "-o"])
        .arg(&yuv_out)
        .arg(ivf)
        .status()
        .map_err(|e| format!("invoke: {e}"))?;
    if !status.success() {
        return Err(format!("decoder failed with {status}"));
    }
    let digest = md5_of_file(&yuv_out).map_err(|e| format!("md5: {e}"))?;
    let _ = std::fs::remove_file(&yuv_out);
    Ok(hex16(digest))
}

fn read_reference_md5(ivf: &Path) -> std::io::Result<String> {
    let md5_path = ivf.with_extension("ivf.md5");
    let md5_path = if md5_path.exists() {
        md5_path
    } else {
        let mut p = ivf.to_path_buf();
        p.as_mut_os_string().push(".md5");
        p
    };
    let content = std::fs::read_to_string(md5_path)?;
    // libvpx format: "<md5> *<filename>"
    Ok(content
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_lowercase())
}

fn main() {
    let args = parse_args();

    let mut vectors: Vec<PathBuf> = match std::fs::read_dir(&args.vectors_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "ivf"))
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
    let mut skipped = 0usize;

    println!("running {} vector(s) against {}", vectors.len(), args.decoder.display());
    println!();

    for ivf in &vectors {
        let name = ivf.file_name().unwrap().to_string_lossy().to_string();
        let reference = match read_reference_md5(ivf) {
            Ok(s) if !s.is_empty() => s,
            _ => {
                println!("  SKIP {name} (no reference .md5)");
                skipped += 1;
                continue;
            }
        };
        match run_one(&args.decoder, ivf) {
            Ok(got) if got == reference => {
                println!("  PASS {name}");
                passed += 1;
            }
            Ok(got) => {
                println!("  FAIL {name}  got {got}  want {reference}");
                failed += 1;
            }
            Err(e) => {
                println!("  ERR  {name}  {e}");
                failed += 1;
            }
        }
    }

    println!();
    println!("Pass: {passed}   Fail: {failed}   Skip: {skipped}");

    if failed > 0 {
        std::process::exit(1);
    }
}
