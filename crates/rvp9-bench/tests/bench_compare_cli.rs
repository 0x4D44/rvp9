use std::process::{Command, Output};

const HEADER: &str =
    "schema,timestamp,c2,c3,c4,git_sha,run_index,fixture,threads,c9,c10,c11,rvp9_p50_us\n";

fn compare(csv_body: &str, baseline: u64, candidate: u64, suffix: &str) -> Output {
    let temp = std::env::temp_dir();
    let unique = format!("{}-{suffix}", std::process::id());
    let csv = temp.join(format!("rvp9-bench-compare-{unique}.csv"));
    let report = temp.join(format!("rvp9-bench-compare-{unique}.md"));
    std::fs::write(&csv, format!("{HEADER}{csv_body}")).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bench-compare"))
        .args(["--baseline", &baseline.to_string()])
        .args(["--candidate", &candidate.to_string()])
        .args(["--csv", csv.to_str().unwrap()])
        .args(["--out", report.to_str().unwrap()])
        .output()
        .unwrap();

    std::fs::remove_file(csv).unwrap();
    if report.exists() {
        std::fs::remove_file(report).unwrap();
    }
    output
}

fn assert_failed(output: &Output) {
    assert!(
        !output.status.success(),
        "comparison unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn timestamps_without_matching_rows_fail() {
    let output = compare(
        "3,1,x,x,x,abc,1,clip,1,x,x,x,1000\n\
         3,2,x,x,x,def,1,clip,1,x,x,x,900\n",
        10,
        20,
        "unmatched",
    );
    assert_failed(&output);
}

#[test]
fn one_sided_group_fails() {
    let output = compare("3,10,x,x,x,abc,1,clip,1,x,x,x,1000\n", 10, 20, "one-sided");
    assert_failed(&output);
}

#[test]
fn malformed_current_schema_row_fails() {
    let output = compare(
        "3,not-a-timestamp,x,x,x,abc,1,clip,1,x,x,x,1000\n",
        10,
        20,
        "malformed",
    );
    assert_failed(&output);
}

#[test]
fn paired_group_passes_while_future_schema_is_ignored() {
    let output = compare(
        "4,this,row,is,intentionally,not,schema,three\n\
         3,10,x,x,x,abc,1,clip,1,x,x,x,1000\n\
         3,20,x,x,x,def,1,clip,1,x,x,x,900\n",
        10,
        20,
        "paired",
    );
    assert!(
        output.status.success(),
        "valid comparison failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
