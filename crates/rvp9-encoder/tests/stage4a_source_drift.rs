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
