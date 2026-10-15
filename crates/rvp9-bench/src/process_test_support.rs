//! Fake tools execute a Rust test child, so timeout tests never need real codecs.
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

pub const DEADLINE: Duration = Duration::from_secs(1);
pub const Y4M: &[u8] = b"YUV4MPEG2 W2 H2 F30:1 Ip A1:1 C420jpeg\nFRAME\nabcdef";
static NEXT: AtomicU64 = AtomicU64::new(0);

pub struct Fixture {
    pub dir: PathBuf,
    pub tool: PathBuf,
}
impl Fixture {
    pub fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        std::fs::create_dir_all(&root).unwrap();
        let dir = root.join(format!(
            "bench-process-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        let tool = dir.join("fake-tool");
        let executable = std::env::current_exe().unwrap();
        let quote = |text: &str| format!("'{}'", text.replace('\'', "'\\''"));
        std::fs::write(&tool, format!("#!/bin/sh\nBENCH_FAKE_CHILD=1\nexport BENCH_FAKE_CHILD\nif [ \"$1\" = '-o' ]; then BENCH_FAKE_OUTPUT=\"$2\"; export BENCH_FAKE_OUTPUT; fi\nexec {} --exact process_test_support::fake_child --nocapture\n", quote(executable.to_str().unwrap()))).unwrap();
        std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self { dir, tool }
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        for entry in std::fs::read_dir(&self.dir).unwrap() {
            std::fs::remove_file(entry.unwrap().path()).unwrap();
        }
        std::fs::remove_dir(&self.dir).unwrap();
    }
}

#[test]
fn fake_child() {
    if std::env::var_os("BENCH_FAKE_CHILD").is_none() {
        return;
    }
    let mut input = Vec::new();
    std::io::stdin().read_to_end(&mut input).unwrap();
    assert!(input.is_empty(), "external tool stdin must be closed");
    println!("fake stdout diagnostic pid={}", std::process::id());
    eprintln!("fake stderr diagnostic");
    std::io::stdout().flush().unwrap();
    std::io::stderr().flush().unwrap();
    if std::env::var_os("BENCH_FAKE_LOUD").is_some() {
        std::io::stdout()
            .write_all(&vec![b'x'; 256 * 1024])
            .unwrap();
        std::io::stderr()
            .write_all(&vec![b'y'; 256 * 1024])
            .unwrap();
    }
    std::thread::sleep(Duration::from_secs(2));
    if let Some(path) = std::env::var_os("BENCH_FAKE_OUTPUT") {
        std::fs::write(path, Y4M).unwrap();
    }
    if std::env::var_os("BENCH_FAKE_FAIL").is_some() {
        std::process::exit(7);
    }
}
