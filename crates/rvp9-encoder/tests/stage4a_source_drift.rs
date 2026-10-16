#![allow(clippy::print_stderr)]

//! Regression coverage for Stage 4a's final ignored-corpus validation.

mod common;

use common::stage4a_product::source_hash_validation_reasons;

#[test]
fn final_source_hash_detects_drift_and_pinned_hash_mismatch() {
    let reasons = source_hash_validation_reasons(Some("start-hash"), "changed-hash", "pinned-hash");

    assert_eq!(
        reasons,
        [
            "SOURCE_SHA256_CHANGED_DURING_RUN:start-hash_TO_changed-hash",
            "SOURCE_SHA256_FINAL_MISMATCH:changed-hash_EXPECTED_pinned-hash",
        ],
    );
}

#[test]
fn final_source_hash_accepts_unchanged_pinned_input() {
    let reasons = source_hash_validation_reasons(Some("pinned-hash"), "pinned-hash", "pinned-hash");

    assert!(reasons.is_empty());
}

use common::stage4a_product::{
    ProductReport, ProductRunRole, finish_and_write_report, new_product_report, source_tree_hash,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static FIXTURE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

const INITIAL_SOURCE: &[u8] = b"tiny initial ignored corpus\n";
const CHANGED_SOURCE: &[u8] = b"tiny changed ignored corpus\n";

struct SourceFixture {
    root: PathBuf,
    report: ProductReport,
}

impl SourceFixture {
    fn new() -> Self {
        let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.scratch");
        fs::create_dir_all(&scratch).expect("create task scratch");
        let root = loop {
            let sequence = FIXTURE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let root = scratch.join(format!("stage4a-source-{}-{sequence}", std::process::id()));
            match fs::create_dir(&root) {
                Ok(()) => break root,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create unique fixture root: {error}"),
            }
        };
        // Own cleanup from the first successful creation, including setup failures.
        let mut fixture = Self {
            report: new_product_report(
                &root,
                &root.join("report.json"),
                ProductRunRole::StrictBaseline,
                None,
                String::new(),
            ),
            root,
        };
        fs::create_dir_all(fixture.root.join("tests/corpus")).unwrap();
        fs::write(fixture.root.join(".gitignore"), "tests/corpus/\n").unwrap();
        fs::write(fixture.root.join("tests/fixture.txt"), "tracked fixture\n").unwrap();
        fs::write(fixture.source(), INITIAL_SOURCE).unwrap();
        fixture.git(&["init"]);
        fixture.git(&["add", ".gitignore", "tests/fixture.txt"]);
        fs::write(
            fixture.root.join(".git/fixture-message"),
            "test fixture\n\nCo-Authored-By: GPT-6.1 Sol <noreply@openai.com>\n",
        )
        .unwrap();
        fixture.git(&[
            "-c",
            "user.name=0x4D44",
            "-c",
            "user.email=martingdavidson@gmail.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-F",
            ".git/fixture-message",
        ]);
        fixture.report.git_sha = fixture.git(&["rev-parse", "HEAD"]);
        fixture.report.source_tree_hash = Some(source_tree_hash(&fixture.root).unwrap());
        fixture.report.source_sha256 = Some(digest(INITIAL_SOURCE));
        fixture.report.source_expected_sha256 = digest(INITIAL_SOURCE);
        fixture
    }

    fn git(&self, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.root)
            .stdin(Stdio::null())
            .output()
            .expect("run fixture git");
        assert!(
            output.status.success(),
            "fixture git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn source(&self) -> PathBuf {
        self.root.join("tests/corpus/foreman_cif.y4m")
    }

    fn finish(&mut self) -> Value {
        let path = self.root.join("report.json");
        finish_and_write_report(&self.root, &path, &mut self.report);
        let json: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(
            json["validation_clean"], true,
            "source-only change must leave validation tree clean"
        );
        json
    }
}

impl Drop for SourceFixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).expect("remove owned source fixture");
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn assert_source_failure(json: &Value, expected_reason: &str) {
    assert_eq!(
        json["overall_status"], "FAIL",
        "final source validation must block PASS"
    );
    assert!(
        json["status_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason
                .as_str()
                .is_some_and(|reason| reason.starts_with(expected_reason))),
        "missing {expected_reason} in {json}"
    );
    assert_eq!(
        json["source_sha256"],
        digest(INITIAL_SOURCE),
        "retain run-start corpus fingerprint"
    );
}

#[test]
fn final_report_rejects_changed_ignored_source() {
    let mut fixture = SourceFixture::new();
    fs::write(fixture.source(), CHANGED_SOURCE).unwrap();
    let json = fixture.finish();
    assert_source_failure(&json, "SOURCE_SHA256_CHANGED_DURING_RUN:");
    assert_source_failure(&json, "SOURCE_SHA256_FINAL_MISMATCH:");
    assert_eq!(json["source_final_sha256"], digest(CHANGED_SOURCE));
}

#[test]
fn final_report_rejects_missing_ignored_source() {
    let mut fixture = SourceFixture::new();
    fs::rename(fixture.source(), fixture.root.join("saved-source.y4m")).unwrap();
    let json = fixture.finish();
    assert_source_failure(&json, "SOURCE_SHA256_FINAL_FAILED:");
    assert!(json["source_final_sha256"].is_null());
}

#[test]
fn final_report_rejects_unchanged_source_with_wrong_pin() {
    let mut fixture = SourceFixture::new();
    fixture.report.source_expected_sha256 = digest(CHANGED_SOURCE);
    let json = fixture.finish();
    assert_source_failure(&json, "SOURCE_SHA256_FINAL_MISMATCH:");
    assert_eq!(json["status_reasons"].as_array().unwrap().len(), 1);
    assert_eq!(json["source_final_sha256"], digest(INITIAL_SOURCE));
}

#[test]
fn final_report_passes_unchanged_pinned_source() {
    let mut fixture = SourceFixture::new();
    let json = fixture.finish();
    assert_eq!(
        json["overall_status"], "PASS",
        "unchanged pinned input must pass finalization"
    );
    assert_eq!(json["status_reasons"], serde_json::json!([]));
    assert_eq!(json["source_sha256"], digest(INITIAL_SOURCE));
    assert_eq!(json["source_final_sha256"], digest(INITIAL_SOURCE));
}
