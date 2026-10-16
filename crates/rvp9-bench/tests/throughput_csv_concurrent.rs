use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use rvp9_bench::Stats;
use rvp9_bench::throughput::{self, CsvRow, FixtureResult};

const HEADER: &str = "schema_version,timestamp_unix,host,unpinned,build_profile_hash,fixture,timed_frames,width,height,p50_us,p95_us,p99_us\n";
const CHILD_MODE: &str = "RVP9_THROUGHPUT_CSV_CHILD_MODE";
const CHILD_CSV: &str = "RVP9_THROUGHPUT_CSV_CHILD_CSV";
const CHILD_READY: &str = "RVP9_THROUGHPUT_CSV_CHILD_READY";
const CHILD_GO: &str = "RVP9_THROUGHPUT_CSV_CHILD_GO";
const CHILD_WORKER: &str = "RVP9_THROUGHPUT_CSV_CHILD_WORKER";

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        fs::create_dir_all(&scratch).expect("create task scratch directory");

        for _ in 0..100 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = scratch.join(format!(
                "throughput-csv-concurrent-{}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory {}: {error}", path.display()),
            }
        }
        panic!("could not allocate a unique throughput CSV test directory");
    }

    fn child_tmpdir(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).expect("create child TMPDIR");
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);

impl ChildGuard {
    fn wait(&mut self, timeout: Duration) -> Result<ExitStatus, String> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.0.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {}
                Err(error) => return Err(format!("poll child process: {error}")),
            }
            if Instant::now() >= deadline {
                let _ = self.0.kill();
                let _ = self.0.wait();
                return Err(format!("child process timed out after {timeout:?}"));
            }
            thread::sleep(Duration::from_millis(5));
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        match self.0.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
    }
}

fn spawn_writer(
    scratch: &TestDirectory,
    mode: &str,
    csv: &Path,
    worker: usize,
    ready: &Path,
    go: Option<&Path>,
) -> ChildGuard {
    let child_tmp = scratch.child_tmpdir(&format!("tmp-{worker}"));
    let mut command = Command::new(std::env::current_exe().expect("current test executable"));
    command
        .args([
            "--exact",
            "append_csv_child_process",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .env(CHILD_MODE, mode)
        .env(CHILD_CSV, csv)
        .env(CHILD_READY, ready)
        .env(CHILD_WORKER, worker.to_string())
        .env("TMPDIR", &child_tmp)
        .env("TMP", &child_tmp)
        .env("TEMP", &child_tmp);
    if let Some(go) = go {
        command.env(CHILD_GO, go);
    }
    ChildGuard(command.spawn().expect("spawn bounded CSV writer child"))
}

fn wait_for_file(child: &mut ChildGuard, path: &Path, timeout: Duration) -> Result<(), String> {
    let deadline = Instant::now() + timeout;
    loop {
        if path.exists() {
            return Ok(());
        }
        match child.0.try_wait() {
            Ok(Some(status)) => {
                return Err(format!("child exited before creating marker: {status}"));
            }
            Ok(None) => {}
            Err(error) => return Err(format!("poll child while waiting for marker: {error}")),
        }
        if Instant::now() >= deadline {
            return Err(format!("timed out waiting for marker {}", path.display()));
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn wait_for_unchanged_csv(lock_holder: &File, initial_len: u64, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    loop {
        if lock_holder.metadata().map(|metadata| metadata.len()).ok() != Some(initial_len) {
            return false;
        }
        if Instant::now() >= deadline {
            return true;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn writer_batch(worker: usize) -> (u64, String, bool, String, Vec<FixtureResult>) {
    let timestamp = 50_000 + worker as u64;
    let host = format!("writer-{worker}, \"host\"");
    let unpinned = worker % 2 == 1;
    let build_profile_hash = format!("build-{worker}\nprofile,\"release\"");
    let results = (0..4)
        .map(|row| {
            let p50 = (worker * 10 + row + 1) as f64;
            FixtureResult {
                fixture: format!("writer-{worker}-row-{row}\nquoted,\"fixture\""),
                timed_frames: 1_000 + worker * 10 + row,
                width: 640 + worker as u32,
                height: 480 + row as u32,
                stats: Stats {
                    n: 1_000,
                    min: p50 - 1.0,
                    max: p50 + 3.0,
                    mean: p50 + 1.0,
                    p50,
                    p90: p50 + 1.0,
                    p95: p50 + 2.0,
                    p99: p50 + 3.0,
                },
            }
        })
        .collect();
    (timestamp, host, unpinned, build_profile_hash, results)
}

fn expected_rows(workers: usize) -> BTreeMap<String, (f64, f64, f64)> {
    let mut expected = BTreeMap::new();
    for worker in 0..workers {
        let (_, _, _, _, results) = writer_batch(worker);
        for result in results {
            expected.insert(
                result.fixture,
                (result.stats.p50, result.stats.p95, result.stats.p99),
            );
        }
    }
    expected
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RowIdentity {
    timestamp: u64,
    host: String,
    unpinned: bool,
    build_profile_hash: String,
    fixture: String,
}

fn identity(row: &CsvRow) -> RowIdentity {
    RowIdentity {
        timestamp: row.timestamp_unix,
        host: row.host.clone(),
        unpinned: row.unpinned,
        build_profile_hash: row.build_profile_hash.clone(),
        fixture: row.fixture.clone(),
    }
}

fn expected_identities(workers: usize) -> BTreeSet<RowIdentity> {
    let mut expected = BTreeSet::new();
    for worker in 0..workers {
        let (timestamp, host, unpinned, build_profile_hash, results) = writer_batch(worker);
        for result in results {
            expected.insert(RowIdentity {
                timestamp,
                host: host.clone(),
                unpinned,
                build_profile_hash: build_profile_hash.clone(),
                fixture: result.fixture,
            });
        }
    }
    expected
}

fn assert_complete_batches(csv: &Path, workers: usize) {
    let contents = fs::read_to_string(csv).expect("read concurrent throughput CSV");
    assert_eq!(
        contents.matches(HEADER).count(),
        1,
        "the CSV must contain one header"
    );

    let rows = throughput::read_csv(csv).expect("parse concurrent throughput CSV");
    assert_eq!(rows.len(), workers * 4, "every complete row must parse");
    assert_eq!(
        rows.iter().map(identity).collect::<BTreeSet<_>>(),
        expected_identities(workers),
        "writer identity and quoted or multiline fields must survive"
    );

    let expected_metrics = expected_rows(workers);
    for row in &rows {
        let (p50, p95, p99) = expected_metrics
            .get(&row.fixture)
            .expect("each parsed fixture must be expected");
        assert_eq!((row.p50_us, row.p95_us, row.p99_us), (*p50, *p95, *p99));
    }

    let mut emitted_workers = BTreeSet::new();
    let mut last_worker = None;
    for row in &rows {
        let worker = row.timestamp_unix;
        if last_worker != Some(worker) {
            assert!(
                emitted_workers.insert(worker),
                "each multirow append batch must remain contiguous: {rows:?}"
            );
            last_worker = Some(worker);
        }
    }
    assert_eq!(emitted_workers.len(), workers);
}

#[test]
fn append_blocks_until_another_process_releases_the_file_lock() {
    const OBSERVATION_WINDOW: Duration = Duration::from_secs(1);
    const CHILD_TIMEOUT: Duration = Duration::from_secs(10);

    let scratch = TestDirectory::new();
    let csv = scratch.0.join("existing.csv");
    fs::write(&csv, HEADER).expect("write existing CSV header");
    let initial_len = fs::metadata(&csv).expect("read initial CSV metadata").len();
    let lock_holder = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&csv)
        .expect("open lock holder file");
    lock_holder.lock().expect("hold exclusive CSV lock");

    let ready = scratch.0.join("writer.ready");
    let mut child = spawn_writer(&scratch, "locked", &csv, 0, &ready, None);
    let ready_result = wait_for_file(&mut child, &ready, CHILD_TIMEOUT);
    let unchanged_while_locked = if ready_result.is_ok() {
        wait_for_unchanged_csv(&lock_holder, initial_len, OBSERVATION_WINDOW)
    } else {
        false
    };

    drop(lock_holder);
    let child_status = child.wait(CHILD_TIMEOUT);
    let rows = throughput::read_csv(&csv).expect("read appended row after lock release");

    assert!(ready_result.is_ok(), "child readiness: {ready_result:?}");
    assert!(
        unchanged_while_locked,
        "append must remain blocked while another process holds the CSV lock"
    );
    assert!(child_status.is_ok(), "child completion: {child_status:?}");
    assert!(child_status.unwrap().success(), "writer child must succeed");
    assert_eq!(
        rows.len(),
        4,
        "the complete child batch must follow release"
    );
    assert_eq!(
        rows.iter().map(identity).collect::<BTreeSet<_>>(),
        expected_identities(1),
        "all rows must be present after the lock is released"
    );
}

#[test]
fn concurrent_first_writers_preserve_one_header_and_complete_batches() {
    const WORKERS: usize = 8;
    const OBSERVATION_WINDOW: Duration = Duration::from_secs(1);
    const CHILD_TIMEOUT: Duration = Duration::from_secs(15);

    let scratch = TestDirectory::new();
    let csv = scratch.0.join("empty.csv");
    fs::write(&csv, b"").expect("create empty CSV for first writers");
    let initial_len = fs::metadata(&csv).expect("read empty CSV metadata").len();
    let lock_holder = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&csv)
        .expect("open first-writer lock holder");
    lock_holder
        .lock()
        .expect("hold exclusive first-writer lock");

    let go = scratch.0.join("go");
    let mut children = Vec::new();
    for worker in 0..WORKERS {
        let ready = scratch.0.join(format!("writer-{worker}.ready"));
        children.push(spawn_writer(
            &scratch,
            "barrier",
            &csv,
            worker,
            &ready,
            Some(&go),
        ));
    }

    let mut readiness = Vec::new();
    for (worker, child) in children.iter_mut().enumerate() {
        let ready = scratch.0.join(format!("writer-{worker}.ready"));
        readiness.push(wait_for_file(child, &ready, CHILD_TIMEOUT));
    }
    fs::write(&go, b"append").expect("release first-writer barrier");
    let mut append_attempts = Vec::new();
    for (worker, child) in children.iter_mut().enumerate() {
        let ready = scratch.0.join(format!("writer-{worker}.attempting"));
        append_attempts.push(wait_for_file(child, &ready, CHILD_TIMEOUT));
    }
    let unchanged_while_locked =
        if readiness.iter().all(Result::is_ok) && append_attempts.iter().all(Result::is_ok) {
            wait_for_unchanged_csv(&lock_holder, initial_len, OBSERVATION_WINDOW)
        } else {
            false
        };

    drop(lock_holder);
    let statuses = children
        .iter_mut()
        .map(|child| child.wait(CHILD_TIMEOUT))
        .collect::<Vec<_>>();
    let contents = fs::read_to_string(&csv).expect("read first-writer CSV");

    assert!(
        readiness.iter().all(Result::is_ok),
        "all children must reach the start barrier: {readiness:?}"
    );
    assert!(
        append_attempts.iter().all(Result::is_ok),
        "all writers must attempt append while the lock is held: {append_attempts:?}"
    );
    assert!(
        unchanged_while_locked,
        "first writers must wait for the held lock before initializing the CSV"
    );
    assert!(
        statuses
            .iter()
            .all(|status| matches!(status, Ok(status) if status.success())),
        "all child writers must complete: {statuses:?}"
    );
    assert_complete_batches(Path::new(&csv), WORKERS);
    assert!(
        contents.starts_with(HEADER),
        "header must be the first record"
    );
}

#[test]
#[ignore = "launched as a child process by the concurrency regression tests"]
fn append_csv_child_process() {
    let Ok(mode) = std::env::var(CHILD_MODE) else {
        return;
    };
    let csv = PathBuf::from(std::env::var_os(CHILD_CSV).expect("child CSV path"));
    let ready = PathBuf::from(std::env::var_os(CHILD_READY).expect("child ready marker"));
    let worker = std::env::var(CHILD_WORKER)
        .expect("child worker ID")
        .parse::<usize>()
        .expect("numeric child worker ID");

    if mode == "barrier" {
        let go = PathBuf::from(std::env::var_os(CHILD_GO).expect("child go marker"));
        fs::write(&ready, b"ready").expect("write child ready marker");
        let deadline = Instant::now() + Duration::from_secs(10);
        while !go.exists() {
            assert!(
                Instant::now() < deadline,
                "parent did not release child barrier"
            );
            thread::sleep(Duration::from_millis(5));
        }
    } else if mode == "locked" {
        fs::write(&ready, b"attempting").expect("write child append marker");
    } else {
        panic!("unknown child mode: {mode}");
    }

    if mode == "barrier" {
        let attempting = ready.with_extension("attempting");
        fs::write(&attempting, b"attempting").expect("write child append marker");
    }
    let (timestamp, host, unpinned, build_profile_hash, results) = writer_batch(worker);
    throughput::append_csv(
        &csv,
        timestamp,
        &host,
        unpinned,
        &build_profile_hash,
        &results,
    )
    .expect("append child CSV batch");
}
