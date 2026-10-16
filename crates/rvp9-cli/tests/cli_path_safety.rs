//! Destination alias regressions use only files owned by each test.

mod common;
#[path = "../src/path_guard.rs"]
mod path_guard;

use common::CommandExt;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
    entries: RefCell<Vec<(PathBuf, bool)>>,
}

impl Fixture {
    fn new() -> Self {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        std::fs::create_dir_all(&scratch).expect("create worktree scratch");
        loop {
            let root = scratch.join(format!(
                "cli_path_safety_{}_{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&root) {
                Ok(()) => {
                    return Self {
                        root,
                        entries: RefCell::new(Vec::new()),
                    };
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create owned fixture: {error}"),
            }
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        let path = self.root.join(name);
        self.entries.borrow_mut().push((path.clone(), false));
        path
    }

    fn raw(&self) -> PathBuf {
        let raw = self.path("input.yuv");
        // Two frames make truncation observable independently of encoder success.
        std::fs::write(&raw, vec![128; 64 * 64 * 3]).expect("write owned raw fixture");
        raw
    }

    fn ivf(&self) -> PathBuf {
        let raw = self.raw();
        let ivf = self.path("input.ivf");
        let result = encode(&raw, &ivf, None, &[]);
        assert!(
            result.status.success(),
            "fixture encode: {}",
            stderr(&result)
        );
        ivf
    }

    #[cfg(unix)]
    fn directory(&self, name: &str) -> PathBuf {
        let path = self.root.join(name);
        std::fs::create_dir(&path).expect("create owned directory");
        self.entries.borrow_mut().push((path.clone(), true));
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        // Every entry was allocated inside this uniquely created fixture.
        for (path, directory) in self.entries.get_mut().iter().rev() {
            if *directory {
                let _ = std::fs::remove_dir(path);
            } else {
                let _ = std::fs::remove_file(path);
            }
        }
        let _ = std::fs::remove_dir(&self.root);
    }
}

fn stderr(result: &Output) -> String {
    String::from_utf8_lossy(&result.stderr).into_owned()
}

fn encode(input: &Path, output: &Path, csv: Option<&Path>, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rvp9enc"));
    command.args([
        "-w",
        "64",
        "-h",
        "64",
        "--cpu-used=8",
        "--threads=1",
        "--lag-in-frames=0",
    ]);
    command.args(extra).arg("-o").arg(output);
    if let Some(csv) = csv {
        command.arg("--frame-time-csv").arg(csv);
    }
    command.arg(input).bounded_output().expect("run encoder")
}

fn decode(input: &Path, output: Option<&Path>, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_rvp9dec"));
    command.args(["--threads=1"]).args(extra);
    if let Some(output) = output {
        command.arg("-o").arg(output);
    }
    command.arg(input).bounded_output().expect("run decoder")
}

fn rejects_alias(result: &Output, roles: &[&str]) {
    let message = stderr(result);
    assert!(
        !result.status.success(),
        "aliased roles must fail; stderr: {message}"
    );
    assert!(
        message.contains("alias"),
        "must diagnose destination alias: {message}"
    );
    for role in roles {
        assert!(
            message.to_ascii_lowercase().contains(role),
            "must name {role} role: {message}"
        );
    }
}

#[derive(Clone, Copy)]
enum Alias {
    Equal,
    Hardlink,
    #[cfg(unix)]
    Symlink,
}

fn alias(fixture: &Fixture, target: &Path, name: &str, kind: Alias) -> PathBuf {
    match kind {
        Alias::Equal => target.to_owned(),
        Alias::Hardlink => {
            let path = fixture.path(name);
            std::fs::hard_link(target, &path).expect("link owned fixture");
            path
        }
        #[cfg(unix)]
        Alias::Symlink => {
            let path = fixture.path(name);
            std::os::unix::fs::symlink(target, &path).expect("symlink owned fixture");
            path
        }
    }
}

fn encoder_aliases(kind: Alias, pair: usize) {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let original_input = std::fs::read(&input).unwrap();
    let mut output = fixture.path("output.ivf");
    let mut csv = fixture.path("timings.csv");
    let roles = match pair {
        0 => {
            output = alias(&fixture, &input, "output-alias.ivf", kind);
            vec!["input", "output"]
        }
        1 => {
            csv = alias(&fixture, &input, "csv-alias.csv", kind);
            vec!["input", "csv"]
        }
        _ => {
            std::fs::write(&output, b"existing completed output").unwrap();
            csv = alias(&fixture, &output, "csv-alias.csv", kind);
            vec!["output", "csv"]
        }
    };
    let original_output = std::fs::read(&output).ok();
    let original_csv = std::fs::read(&csv).ok();
    let result = encode(&input, &output, Some(&csv), &[]);
    // Check preservation even when the old binary exits successfully.
    assert!(
        std::fs::read(&input).unwrap() == original_input,
        "input bytes changed for pair {pair}"
    );
    assert!(
        std::fs::read(&output).ok() == original_output,
        "output changed/created for pair {pair}"
    );
    assert!(
        std::fs::read(&csv).ok() == original_csv,
        "CSV changed/created for pair {pair}"
    );
    rejects_alias(&result, &roles);
}

macro_rules! encoder_alias_test {
    ($name:ident, $kind:ident, $pair:expr) => {
        #[test]
        fn $name() {
            encoder_aliases(Alias::$kind, $pair);
        }
    };
}

encoder_alias_test!(encoder_equal_input_output_preserved, Equal, 0);
encoder_alias_test!(encoder_equal_input_csv_preserved, Equal, 1);
encoder_alias_test!(encoder_equal_output_csv_preserved, Equal, 2);
encoder_alias_test!(encoder_hardlinked_input_output_preserved, Hardlink, 0);
encoder_alias_test!(encoder_hardlinked_input_csv_preserved, Hardlink, 1);
encoder_alias_test!(encoder_hardlinked_output_csv_preserved, Hardlink, 2);
#[cfg(unix)]
encoder_alias_test!(encoder_symlinked_input_output_preserved, Symlink, 0);
#[cfg(unix)]
encoder_alias_test!(encoder_symlinked_input_csv_preserved, Symlink, 1);
#[cfg(unix)]
encoder_alias_test!(encoder_symlinked_output_csv_preserved, Symlink, 2);

fn decoder_aliases(kind: Alias, extra: &[&str]) {
    let fixture = Fixture::new();
    let input = fixture.ivf();
    let original = std::fs::read(&input).unwrap();
    let output = alias(&fixture, &input, "decoded-alias.yuv", kind);
    let result = decode(&input, Some(&output), extra);
    assert!(
        std::fs::read(&input).unwrap() == original,
        "decoder changed input for {extra:?}"
    );
    rejects_alias(&result, &["input", "output"]);
}

macro_rules! decoder_alias_test {
    ($name:ident, $kind:ident, $extra:expr) => {
        #[test]
        fn $name() {
            decoder_aliases(Alias::$kind, $extra);
        }
    };
}

decoder_alias_test!(decoder_equal_pixels_preserved, Equal, &[]);
decoder_alias_test!(decoder_equal_md5_preserved, Equal, &["--md5"]);
decoder_alias_test!(
    decoder_equal_noblit_md5_preserved,
    Equal,
    &["--noblit", "--framemd5"]
);
decoder_alias_test!(decoder_hardlinked_pixels_preserved, Hardlink, &[]);
decoder_alias_test!(decoder_hardlinked_md5_preserved, Hardlink, &["--md5"]);
decoder_alias_test!(
    decoder_hardlinked_noblit_md5_preserved,
    Hardlink,
    &["--noblit", "--framemd5"]
);
#[cfg(unix)]
decoder_alias_test!(decoder_symlinked_pixels_preserved, Symlink, &[]);
#[cfg(unix)]
decoder_alias_test!(decoder_symlinked_md5_preserved, Symlink, &["--md5"]);
#[cfg(unix)]
decoder_alias_test!(
    decoder_symlinked_noblit_md5_preserved,
    Symlink,
    &["--noblit", "--framemd5"]
);

#[test]
fn missing_output_and_csv_equal_roles_are_rejected_before_creation() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let output = fixture.path("not-created.ivf");
    let result = encode(&input, &output, Some(&output), &[]);
    assert!(!output.exists(), "aliased output must never be created");
    rejects_alias(&result, &["output", "csv"]);
}

#[test]
fn distinct_roles_stdout_and_ignored_decoder_output_succeed() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let output = fixture.path("distinct.ivf");
    let csv = fixture.path("distinct.csv");
    std::fs::write(&output, b"old independent output").unwrap();
    std::fs::write(&csv, b"old independent CSV").unwrap();
    let keep = alias(
        &fixture,
        &output,
        "independent-hardlink.ivf",
        Alias::Hardlink,
    );
    let encoded = encode(&input, &output, Some(&csv), &[]);
    assert!(
        encoded.status.success(),
        "distinct roles: {}",
        stderr(&encoded)
    );
    assert!(std::fs::read(&output).unwrap().starts_with(b"DKIF"));
    assert_eq!(
        std::fs::read(&keep).unwrap(),
        std::fs::read(&output).unwrap()
    );
    assert!(
        std::fs::read_to_string(&csv)
            .unwrap()
            .starts_with("frame_index,")
    );
    let original = std::fs::read(&output).unwrap();
    let ignored = Command::new(env!("CARGO_BIN_EXE_rvp9dec"))
        .args(["--threads=1", "--noblit", "-o"])
        .arg(&output)
        .arg(&output)
        .bounded_status()
        .expect("decode with ignored output");
    assert!(ignored.success(), "ignored aliased output must succeed");
    assert_eq!(std::fs::read(&output).unwrap(), original);
    let stdout = decode(&output, None, &["--md5"]);
    assert!(stdout.status.success(), "stdout: {}", stderr(&stdout));
    assert_eq!(String::from_utf8(stdout.stdout).unwrap().lines().count(), 2);
}

#[cfg(unix)]
#[test]
fn missing_output_and_csv_through_symlink_parent_are_rejected() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let directory = fixture.directory("real");
    let parent_alias = alias(&fixture, &directory, "parent-alias", Alias::Symlink);
    let output = directory.join("future.ivf");
    fixture.entries.borrow_mut().push((output.clone(), false));
    let csv = parent_alias.join("future.ivf");
    let result = encode(&input, &output, Some(&csv), &[]);
    assert!(
        !std::fs::read(&output).is_ok_and(|bytes| !bytes.is_empty()),
        "aliased future output must never receive bytes"
    );
    rejects_alias(&result, &["output", "csv"]);
}

#[cfg(unix)]
#[test]
fn missing_output_and_csv_through_dangling_symlink_are_rejected() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let output = fixture.path("future.ivf");
    let csv = alias(&fixture, &output, "dangling.csv", Alias::Symlink);
    let result = encode(&input, &output, Some(&csv), &[]);
    assert!(
        !std::fs::read(&output).is_ok_and(|bytes| !bytes.is_empty()),
        "dangling alias target must never receive bytes"
    );
    rejects_alias(&result, &["output", "csv"]);
}

fn encoder_branch_alias_before_input_parsing(flags: &[&str]) {
    let fixture = Fixture::new();
    let input = fixture.path("invalid.yuv");
    let original = b"not a complete raw frame";
    std::fs::write(&input, original).unwrap();
    let result = encode(&input, &input, None, flags);
    assert_eq!(std::fs::read(&input).unwrap(), original);
    rejects_alias(&result, &["input", "output"]);
}

#[test]
fn encoder_two_pass_alias_is_rejected_before_input_parsing() {
    encoder_branch_alias_before_input_parsing(&["--passes=2"]);
}

#[test]
fn encoder_spatial_alias_is_rejected_before_input_parsing() {
    encoder_branch_alias_before_input_parsing(&["--spatial-layers=2"]);
}

#[cfg(unix)]
#[test]
fn missing_output_and_csv_through_relative_symlink_chain_are_rejected() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let output = fixture.path("future.ivf");
    alias(
        &fixture,
        Path::new("future.ivf"),
        "first.csv",
        Alias::Symlink,
    );
    let csv = alias(
        &fixture,
        Path::new("first.csv"),
        "second.csv",
        Alias::Symlink,
    );
    let result = encode(&input, &output, Some(&csv), &[]);
    assert!(
        !std::fs::read(&output).is_ok_and(|bytes| !bytes.is_empty()),
        "relative alias target must never receive bytes"
    );
    rejects_alias(&result, &["output", "csv"]);
}

#[test]
fn guarded_open_rechecks_identity_before_truncating() {
    let fixture = Fixture::new();
    let input = fixture.raw();
    let output = fixture.path("late-hardlink.ivf");
    let original = std::fs::read(&input).unwrap();
    let mut guard = path_guard::PathGuard::new(&[("input", &input), ("output", &output)]).unwrap();
    // A path that was missing during preflight now aliases the captured input.
    std::fs::hard_link(&input, &output).unwrap();
    let result = guard.create("output");
    assert!(
        std::fs::read(&input).unwrap() == original,
        "guard truncated before comparing handle identity"
    );
    let message = result.unwrap_err();
    assert!(message.contains("alias") && message.contains("input") && message.contains("output"));
}

fn future_filesystem_alias(first: &str, second: &str) {
    let fixture = Fixture::new();
    let probe_first = fixture.path(&format!("probe-{first}"));
    let probe_second = fixture.path(&format!("probe-{second}"));
    std::fs::write(&probe_first, b"filesystem alias probe").unwrap();
    if std::fs::read(&probe_second).ok().as_deref() != Some(b"filesystem alias probe") {
        eprintln!("SKIP: filesystem keeps {first:?} and {second:?} distinct");
        return;
    }
    let input = fixture.raw();
    let original = std::fs::read(&input).unwrap();
    let output = fixture.path(first);
    let csv = fixture.path(second);
    let result = encode(&input, &output, Some(&csv), &[]);
    assert!(
        std::fs::read(&input).unwrap() == original,
        "filesystem alias changed input"
    );
    assert!(
        !std::fs::read(&output).is_ok_and(|bytes| !bytes.is_empty()),
        "future filesystem alias received bytes before rejection"
    );
    rejects_alias(&result, &["output", "csv"]);
}

#[test]
fn future_case_alias_is_rejected_when_filesystem_folds_case() {
    future_filesystem_alias("future.ivf", "FUTURE.IVF");
}

#[test]
fn future_unicode_alias_is_rejected_when_filesystem_normalizes_names() {
    future_filesystem_alias("café.ivf", "cafe\u{301}.ivf");
}

#[cfg(any(unix, windows))]
#[test]
fn nonregular_output_sinks_accept_encoder_and_decoder_writes() {
    let fixture = Fixture::new();
    let input = fixture.ivf();
    let raw = fixture.root.join("input.yuv");
    #[cfg(unix)]
    let null = Path::new("/dev/null");
    #[cfg(windows)]
    let null = Path::new("NUL");
    let encoded = encode(&raw, null, None, &[]);
    assert!(
        encoded.status.success(),
        "encoding to null: {}",
        stderr(&encoded)
    );
    let decoded = decode(&input, Some(null), &[]);
    assert!(
        decoded.status.success(),
        "decoding to null: {}",
        stderr(&decoded)
    );
}
