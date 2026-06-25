#![allow(dead_code)]

use rvp9_core::BitDepth;
use rvp9_core::mode_decision::nonrd_partition::NonRdSplitStats;
use rvp9_decoder::{Decoder, ParsedFrame};
use rvp9_encoder::{
    AqMode, Deadline, EncoderConfig, RcMode, Stage4aProductSplitOverride, StreamEncoder,
    YuvFrameRef,
};
use rvp9_harness::bdrate::{bdrate, psnr_delta_summary};
use rvp9_harness::container::{IvfHeader, write_ivf};
use rvp9_harness::y4m::{YuvFrameBuf, read_cropped as y4m_read_cropped};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const W: u32 = 256;
const H: u32 = 256;
const N_FRAMES: usize = 64;
const FOREMAN_REL: &str = "tests/corpus/foreman_cif.y4m";
const FOREMAN_SHA256: &str = "b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b";
const FOREMAN_BYTES: u64 = 45_621_044;
const LIBVPX_URL: &str = "https://chromium.googlesource.com/webm/libvpx";
const LIBVPX_TAG: &str = "v1.14.1";
const LIBVPX_COMMIT: &str = "12f3a2ac603e8f10742105519e0cd03c3b8f71dd";
const STAGE4A_TARGETS_BPS: [u32; 7] = [25_000, 50_000, 75_000, 100_000, 150_000, 250_000, 400_000];
const VALIDATION_PATHS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    ".cargo",
    "crates",
    "scripts",
    "tests",
];
const FORBIDDEN_PRODUCT_STATUSES: &[&str] =
    &["SKIPPED", "DEFERRED", "PARTIAL", "MISSING", "UNVERIFIED"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProductRunRole {
    PreenablePreflight,
    StrictBaseline,
    Candidate,
}

impl ProductRunRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::PreenablePreflight => "preenable-preflight",
            Self::StrictBaseline => "strict-baseline",
            Self::Candidate => "candidate",
        }
    }

    fn split_policy(self) -> &'static str {
        match self {
            Self::Candidate => "Slack1Over8",
            Self::PreenablePreflight | Self::StrictBaseline => "Strict",
        }
    }

    fn split_override(self) -> Stage4aProductSplitOverride {
        match self {
            Self::Candidate => Stage4aProductSplitOverride::Auto,
            Self::PreenablePreflight | Self::StrictBaseline => {
                Stage4aProductSplitOverride::ForceStrict
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ApprovedToolManifest {
    schema: u32,
    approved_tools: Vec<ApprovedToolRow>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ApprovedToolRow {
    platform: String,
    target_triple: String,
    libvpx_url: String,
    libvpx_tag: String,
    libvpx_commit: String,
    build_script: String,
    configure_args: Vec<String>,
    vpxenc_sha256: String,
    vpxdec_sha256: String,
    notes: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductReport {
    pub run_role: String,
    pub overall_status: String,
    pub status_reasons: Vec<String>,
    pub split_policy: String,
    pub split_override: String,
    pub git_sha: String,
    pub worktree_path: String,
    pub validation_clean: bool,
    pub validation_clean_paths: Vec<String>,
    pub tracked_diff_empty: bool,
    pub staged_diff_empty: bool,
    pub untracked_validation_files: Vec<String>,
    pub ignored_validation_files: Vec<String>,
    pub allowed_ignored_inputs: Vec<AllowedIgnoredInput>,
    pub source_tree_hash: Option<String>,
    pub host_id: String,
    pub utc_started: String,
    pub source_path: String,
    pub source_expected_sha256: String,
    pub source_expected_bytes: u64,
    pub source_sha256: Option<String>,
    pub source_bytes: Option<u64>,
    pub product_config_hash: Option<String>,
    pub libvpx_gitlink_commit: Option<String>,
    pub approved_tool_manifest_path: String,
    pub approved_tool_manifest_sha256: Option<String>,
    pub approved_tool: Option<ApprovedToolRow>,
    pub vpxenc: Option<ToolFingerprint>,
    pub vpxdec: Option<ToolFingerprint>,
    pub ffmpeg: Option<ToolFingerprint>,
    pub encoder_config: ProductEncoderConfigReport,
    pub targets: Vec<ProductTargetReport>,
    pub bd_rate_percent: Option<f64>,
    pub matched_psnr_delta_mean_db: Option<f64>,
    pub matched_psnr_delta_count_in_range: usize,
    pub bd_psnr_overlap_db: Option<f64>,
    pub timing_median_ms: Option<f64>,
    pub block_stats_250kbps_json: Option<String>,
    pub block_stats_250kbps_summary: Option<ProductBlockStatsSummary>,
    pub diff_json_artifacts: Vec<String>,
    pub baseline_report_path: Option<String>,
    pub baseline_report_sha256: Option<String>,
    pub baseline_git_sha: Option<String>,
    pub baseline_host_id: Option<String>,
    pub baseline_product_config_hash: Option<String>,
    pub baseline_source_sha256: Option<String>,
    pub baseline_source_tree_hash: Option<String>,
    pub report_path: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AllowedIgnoredInput {
    path: String,
    sha256: String,
    role: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductEncoderConfigReport {
    width: u32,
    height: u32,
    frames: usize,
    target_bitrates_bps: Vec<u32>,
    aq_mode: String,
    rc_mode: String,
    deadline: String,
    speed: i32,
    lag_in_frames: u32,
    enable_alt_ref: bool,
    min_qindex: u8,
    max_qindex: u8,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolFingerprint {
    path: String,
    sha256: String,
    probe_command: String,
    probe_status: Option<i32>,
    normalized_probe_sha256: String,
    normalized_probe_first_lines: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProductTargetReport {
    pub target_bitrate_bps: u32,
    pub rvp9: Option<RdPointReport>,
    pub libvpx: Option<RdPointReport>,
    pub rvp9_ivf_path: Option<String>,
    pub libvpx_ivf_path: Option<String>,
    pub split_stats: ProductSplitStats,
    pub timing_ms: Vec<f64>,
    pub timing_median_ms: Option<f64>,
    pub decoder_verdicts: DecoderVerdicts,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProductBlockStatsSummary {
    pub frames_requested: usize,
    pub rvp9_frames: usize,
    pub libvpx_frames: Option<usize>,
    pub achieved_kbps: f64,
    pub decode_failed: usize,
    pub partition_area_pct: Vec<ProductBlockArea>,
    pub mode_area_pct: ProductModeArea,
    pub skip_pct: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProductBlockArea {
    pub block: String,
    pub count: u64,
    pub area_pct: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ProductModeArea {
    pub intra: f64,
    pub nearest_mv: f64,
    pub near_mv: f64,
    pub zero_mv: f64,
    pub new_mv: f64,
    pub ref_last: f64,
    pub ref_golden: f64,
    pub ref_altref: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ProductBlockStatsArtifact {
    git_sha: String,
    overall_status: String,
    run_role: String,
    split_policy: String,
    validation_clean: bool,
    ignored_validation_files: Vec<String>,
    allowed_ignored_inputs: Vec<AllowedIgnoredInput>,
    source_tree_hash: Option<String>,
    source_sha256: Option<String>,
    product_config_hash: Option<String>,
    target_bitrate_bps: u32,
    frames_requested: usize,
    rvp9_frames: usize,
    libvpx_frames: Option<usize>,
    achieved_kbps: f64,
    decode_failed: usize,
    partition_area_pct: Vec<ProductBlockArea>,
    mode_area_pct: ProductModeArea,
    skip_pct: f64,
}

#[derive(Debug, Deserialize)]
struct E1IvfJsonReport {
    frames_requested: usize,
    rvp9: E1AggregateJson,
    libvpx: Option<E1AggregateJson>,
}

#[derive(Debug, Deserialize)]
struct E1AggregateJson {
    frames: usize,
    decode_failed: usize,
    achieved_kbps: f64,
    partition_area_pct: Vec<ProductBlockArea>,
    mode_area_pct: ProductModeArea,
    skip_pct: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct RdPointReport {
    pub bitrate_kbps: f64,
    pub psnr_db: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DecoderVerdicts {
    pub rvp9_decoder: String,
    pub vpxdec: String,
    pub ffmpeg: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ProductSplitStats {
    pub block16_attempts: u32,
    pub block16_breakout_suppressed: u32,
    pub block16_feasible: u32,
    pub block16_infeasible_or_early: u32,
    pub block16_child_infeasible: u32,
    pub block16_partial_rd_early_out: u32,
    pub block16_partial_early_out_after_child: [u32; 4],
    pub block16_split_beats_none: u32,
    pub block16_split_loses_to_none: u32,
    pub block16_split_committed: u32,
    pub block16_rect_overrides_split: u32,
    pub block16_no_none: u32,
    pub block16_slack_would_continue: u32,
    pub block16_slack_completed: u32,
    pub block16_slack_split_eligible: u32,
    pub block16_slack_split_committed: u32,
    pub block16_slack_rect_override: u32,
    pub block16_slack_rejected: u32,
    pub block16_margin_sum: i64,
    pub block16_margin_min: Option<i64>,
    pub block16_margin_max: Option<i64>,
}

impl From<NonRdSplitStats> for ProductSplitStats {
    fn from(stats: NonRdSplitStats) -> Self {
        Self {
            block16_attempts: stats.block16_attempts,
            block16_breakout_suppressed: stats.block16_breakout_suppressed,
            block16_feasible: stats.block16_feasible,
            block16_infeasible_or_early: stats.block16_infeasible_or_early,
            block16_child_infeasible: stats.block16_child_infeasible,
            block16_partial_rd_early_out: stats.block16_partial_rd_early_out,
            block16_partial_early_out_after_child: stats.block16_partial_early_out_after_child,
            block16_split_beats_none: stats.block16_split_beats_none,
            block16_split_loses_to_none: stats.block16_split_loses_to_none,
            block16_split_committed: stats.block16_split_committed,
            block16_rect_overrides_split: stats.block16_rect_overrides_split,
            block16_no_none: stats.block16_no_none,
            block16_slack_would_continue: stats.block16_slack_would_continue,
            block16_slack_completed: stats.block16_slack_completed,
            block16_slack_split_eligible: stats.block16_slack_split_eligible,
            block16_slack_split_committed: stats.block16_slack_split_committed,
            block16_slack_rect_override: stats.block16_slack_rect_override,
            block16_slack_rejected: stats.block16_slack_rejected,
            block16_margin_sum: stats.block16_margin_sum,
            block16_margin_min: stats.block16_margin_min,
            block16_margin_max: stats.block16_margin_max,
        }
    }
}

struct ProductContext {
    root: PathBuf,
    run_dir: PathBuf,
    source: PathBuf,
    vpxenc: PathBuf,
    vpxdec: PathBuf,
    ffmpeg: PathBuf,
}

struct ResolvedTools {
    vpxenc: PathBuf,
    vpxdec: PathBuf,
    ffmpeg: PathBuf,
    approved_tool: ApprovedToolRow,
    manifest_sha256: String,
    vpxenc_fp: ToolFingerprint,
    vpxdec_fp: ToolFingerprint,
    ffmpeg_fp: ToolFingerprint,
}

pub fn run_stage4a_product_preenable_preflight() -> ProductReport {
    run_stage4a_product_report(ProductRunRole::PreenablePreflight, None)
}

pub fn run_stage4a_product_strict_baseline() -> ProductReport {
    run_stage4a_product_report(ProductRunRole::StrictBaseline, None)
}

pub fn run_stage4a_product_candidate_with_fresh_baseline() -> (ProductReport, ProductReport) {
    let baseline = run_stage4a_product_report(ProductRunRole::StrictBaseline, None);
    let candidate = run_stage4a_product_report(ProductRunRole::Candidate, Some(&baseline));
    (baseline, candidate)
}

pub fn assert_product_report_pass(report: &ProductReport) {
    let json =
        serde_json::to_string_pretty(report).expect("serialize product report for assertion");
    assert_eq!(
        report.overall_status,
        "PASS",
        "Stage 4a product report failed: {}\nReport:\n{}",
        report.status_reasons.join(", "),
        json
    );
    for forbidden in FORBIDDEN_PRODUCT_STATUSES {
        assert!(
            !json.contains(forbidden),
            "product report must not contain forbidden status token {forbidden}\nReport:\n{json}"
        );
    }
}

#[allow(dead_code)]
pub fn assert_product_cross_decoder_pass(report: &ProductReport) {
    assert_product_report_pass(report);
    for target in &report.targets {
        assert_eq!(target.decoder_verdicts.rvp9_decoder, "PASS");
        assert_eq!(target.decoder_verdicts.vpxdec, "BYTE-EXACT");
        assert!(
            matches!(
                target.decoder_verdicts.ffmpeg.as_str(),
                "BYTE-EXACT" | "PSNR-GATE"
            ),
            "ffmpeg product verdict must pass, got {}",
            target.decoder_verdicts.ffmpeg
        );
    }
}

#[allow(dead_code)]
pub fn assert_product_timing_pass(baseline: &ProductReport, candidate: &ProductReport) {
    assert_product_report_pass(baseline);
    assert_product_report_pass(candidate);
    let strict_ms = baseline
        .timing_median_ms
        .expect("strict baseline timing median must be present");
    let candidate_ms = candidate
        .timing_median_ms
        .expect("candidate timing median must be present");
    assert!(
        candidate_ms <= strict_ms * 1.15,
        "candidate timing median {candidate_ms:.3} ms regressed >15% from strict {strict_ms:.3} ms"
    );
}

fn run_stage4a_product_report(
    role: ProductRunRole,
    baseline: Option<&ProductReport>,
) -> ProductReport {
    let root = repo_root();
    let git_sha = git_stdout(&root, &["rev-parse", "HEAD"]).unwrap_or_else(|_| "UNKNOWN".into());
    let short_sha = git_sha.chars().take(12).collect::<String>();
    let now = unix_seconds();
    let run_stamp = unix_nanos();
    let run_dir = root.join("target").join("stage4a_product").join(format!(
        "{}-{short_sha}-{run_stamp}-p{}",
        role.as_str(),
        std::process::id()
    ));
    fs::create_dir_all(run_dir.parent().expect("product run dir has parent"))
        .expect("create Stage 4a product root");
    fs::create_dir(&run_dir).expect("create unique Stage 4a product run directory");
    let report_path = run_dir.join("stage4a_product_report.json");
    let manifest_path = root.join("tests/reference/stage4a_libvpx_tools.json");
    let mut report = ProductReport {
        run_role: role.as_str().to_string(),
        overall_status: "FAIL".into(),
        status_reasons: Vec::new(),
        split_policy: role.split_policy().into(),
        split_override: format!("{:?}", role.split_override()),
        git_sha,
        worktree_path: root.display().to_string(),
        validation_clean: false,
        validation_clean_paths: VALIDATION_PATHS.iter().map(|p| (*p).to_string()).collect(),
        tracked_diff_empty: false,
        staged_diff_empty: false,
        untracked_validation_files: Vec::new(),
        ignored_validation_files: Vec::new(),
        allowed_ignored_inputs: Vec::new(),
        source_tree_hash: None,
        host_id: host_id(),
        utc_started: format!("{now}"),
        source_path: FOREMAN_REL.into(),
        source_expected_sha256: FOREMAN_SHA256.into(),
        source_expected_bytes: FOREMAN_BYTES,
        source_sha256: None,
        source_bytes: None,
        product_config_hash: None,
        libvpx_gitlink_commit: None,
        approved_tool_manifest_path: rel_display(&root, &manifest_path),
        approved_tool_manifest_sha256: None,
        approved_tool: None,
        vpxenc: None,
        vpxdec: None,
        ffmpeg: None,
        encoder_config: encoder_config_report(),
        targets: Vec::new(),
        bd_rate_percent: None,
        matched_psnr_delta_mean_db: None,
        matched_psnr_delta_count_in_range: 0,
        bd_psnr_overlap_db: None,
        timing_median_ms: None,
        block_stats_250kbps_json: None,
        block_stats_250kbps_summary: None,
        diff_json_artifacts: Vec::new(),
        baseline_report_path: baseline.map(|b| b.report_path.clone()),
        baseline_report_sha256: None,
        baseline_git_sha: baseline.map(|b| b.git_sha.clone()),
        baseline_host_id: baseline.map(|b| b.host_id.clone()),
        baseline_product_config_hash: baseline.and_then(|b| b.product_config_hash.clone()),
        baseline_source_sha256: baseline.and_then(|b| b.source_sha256.clone()),
        baseline_source_tree_hash: baseline.and_then(|b| b.source_tree_hash.clone()),
        report_path: rel_display(&root, &report_path),
    };

    if let Some(baseline) = baseline {
        let baseline_report_path = Path::new(&baseline.report_path)
            .is_absolute()
            .then_some(PathBuf::from(&baseline.report_path))
            .unwrap_or_else(|| root.join(&baseline.report_path));
        match sha256_file(&baseline_report_path) {
            Ok(hash) => report.baseline_report_sha256 = Some(hash),
            Err(e) => report
                .status_reasons
                .push(format!("BASELINE_REPORT_HASH_FAILED:{e}")),
        }
    }

    let context = match build_context(&root, &run_dir, &manifest_path, &mut report) {
        Some(ctx) => ctx,
        None => {
            finish_and_write_report(&root, &report_path, &mut report);
            return report;
        }
    };

    let (frames, framerate) = match y4m_read_cropped(&context.source, W, H, N_FRAMES) {
        Ok((frames, framerate)) if frames.len() == N_FRAMES => (frames, framerate),
        Ok((frames, _)) => {
            report
                .status_reasons
                .push(format!("SOURCE_SHORT_READ:{}_OF_{N_FRAMES}", frames.len()));
            finish_and_write_report(&root, &report_path, &mut report);
            return report;
        }
        Err(e) => {
            report
                .status_reasons
                .push(format!("SOURCE_READ_FAILED:{e}"));
            finish_and_write_report(&root, &report_path, &mut report);
            return report;
        }
    };

    for &target_bitrate_bps in &STAGE4A_TARGETS_BPS {
        match run_target(role, &context, framerate, target_bitrate_bps, &frames) {
            Ok(target) => report.targets.push(target),
            Err(e) => report
                .status_reasons
                .push(format!("TARGET_{}BPS_FAILED:{e}", target_bitrate_bps)),
        }
    }

    run_product_artifacts(&context, &mut report);
    evaluate_quality_and_split(role, baseline, &mut report);
    finish_and_write_report(&root, &report_path, &mut report);
    report
}

fn build_context(
    root: &Path,
    run_dir: &Path,
    manifest_path: &Path,
    report: &mut ProductReport,
) -> Option<ProductContext> {
    reject_rvp9_env(report);
    validate_worktree(root, report);
    validate_libvpx_gitlink(root, report);

    let source = root.join(FOREMAN_REL);
    if source.is_file() {
        match fs::metadata(&source) {
            Ok(md) => {
                report.source_bytes = Some(md.len());
                if md.len() != FOREMAN_BYTES {
                    report.status_reasons.push(format!(
                        "SOURCE_BYTES_MISMATCH:{}_EXPECTED_{FOREMAN_BYTES}",
                        md.len()
                    ));
                }
            }
            Err(e) => report
                .status_reasons
                .push(format!("SOURCE_METADATA_FAILED:{e}")),
        }
        match sha256_file(&source) {
            Ok(hash) => {
                if hash != FOREMAN_SHA256 {
                    report.status_reasons.push("SOURCE_SHA256_MISMATCH".into());
                }
                report.source_sha256 = Some(hash);
            }
            Err(e) => report
                .status_reasons
                .push(format!("SOURCE_HASH_FAILED:{e}")),
        }
    } else {
        report
            .status_reasons
            .push("MISSING_SOURCE_FOREMAN_CIF".into());
    }

    match source_tree_hash(root) {
        Ok(hash) => {
            report.source_tree_hash = Some(hash);
        }
        Err(e) => {
            report
                .status_reasons
                .push(format!("SOURCE_TREE_HASH_FAILED:{e}"));
        }
    }

    let tools = resolve_product_tools(root, manifest_path, report);
    let Some(tools) = tools else {
        return None;
    };

    report.approved_tool_manifest_sha256 = Some(tools.manifest_sha256.clone());
    report.approved_tool = Some(tools.approved_tool.clone());
    report.vpxenc = Some(tools.vpxenc_fp.clone());
    report.vpxdec = Some(tools.vpxdec_fp.clone());
    report.ffmpeg = Some(tools.ffmpeg_fp.clone());

    if !report.status_reasons.is_empty() {
        return None;
    }

    let product_config_hash = product_config_hash(&tools);
    report.product_config_hash = Some(product_config_hash.clone());

    Some(ProductContext {
        root: root.to_path_buf(),
        run_dir: run_dir.to_path_buf(),
        source,
        vpxenc: tools.vpxenc,
        vpxdec: tools.vpxdec,
        ffmpeg: tools.ffmpeg,
    })
}

fn run_target(
    role: ProductRunRole,
    ctx: &ProductContext,
    framerate: f64,
    target_bitrate_bps: u32,
    frames: &[YuvFrameBuf],
) -> Result<ProductTargetReport, String> {
    let rvp9_ivf = ctx
        .run_dir
        .join(format!("rvp9_{}kbps.ivf", target_bitrate_bps / 1000));
    let libvpx_ivf = ctx
        .run_dir
        .join(format!("libvpx_{}kbps.ivf", target_bitrate_bps / 1000));
    let rvp9 = encode_rvp9_target(role, framerate, target_bitrate_bps, frames, &rvp9_ivf)?;
    let libvpx = encode_libvpx_target(ctx, framerate, target_bitrate_bps, frames, &libvpx_ivf)?;

    let mut decoder_verdicts = rvp9.decoder_verdicts;
    let rvp9_yuv = rvp9.decoded_i420;
    let vpx_yuv_path = ctx
        .run_dir
        .join(format!("rvp9_{}kbps_vpxdec.yuv", target_bitrate_bps / 1000));
    match run_yuv_decoder(&ctx.vpxdec, &rvp9_ivf, &vpx_yuv_path) {
        Ok(vpx_yuv) if vpx_yuv == rvp9_yuv => decoder_verdicts.vpxdec = "BYTE-EXACT".into(),
        Ok(vpx_yuv) => {
            decoder_verdicts.vpxdec = format!("FAIL-SIZE-OR-BYTES:{}", vpx_yuv.len());
        }
        Err(e) => decoder_verdicts.vpxdec = format!("FAIL:{e}"),
    }
    let _ = fs::remove_file(&vpx_yuv_path);

    let ff_yuv_path = ctx
        .run_dir
        .join(format!("rvp9_{}kbps_ffmpeg.yuv", target_bitrate_bps / 1000));
    match run_ffmpeg_decoder(&ctx.ffmpeg, &rvp9_ivf, &ff_yuv_path) {
        Ok(ff_yuv) if ff_yuv == rvp9_yuv => decoder_verdicts.ffmpeg = "BYTE-EXACT".into(),
        Ok(ff_yuv) => {
            if ff_yuv.len() != rvp9_yuv.len() {
                decoder_verdicts.ffmpeg =
                    format!("FAIL-SIZE:{}_EXPECTED_{}", ff_yuv.len(), rvp9_yuv.len());
            } else {
                let min_psnr = min_frame_y_psnr(&rvp9_yuv, &ff_yuv, W as usize, H as usize);
                if min_psnr >= 40.0 {
                    decoder_verdicts.ffmpeg = "PSNR-GATE".into();
                } else {
                    decoder_verdicts.ffmpeg = format!("FAIL-PSNR:{min_psnr:.2}");
                }
            }
        }
        Err(e) => decoder_verdicts.ffmpeg = format!("FAIL:{e}"),
    }
    let _ = fs::remove_file(&ff_yuv_path);

    Ok(ProductTargetReport {
        target_bitrate_bps,
        rvp9: Some(rvp9.point),
        libvpx: Some(libvpx),
        rvp9_ivf_path: Some(rel_display(&ctx.root, &rvp9_ivf)),
        libvpx_ivf_path: Some(rel_display(&ctx.root, &libvpx_ivf)),
        split_stats: rvp9.split_stats,
        timing_ms: rvp9.timing_ms,
        timing_median_ms: rvp9.timing_median_ms,
        decoder_verdicts,
    })
}

struct Rvp9TargetResult {
    point: RdPointReport,
    split_stats: ProductSplitStats,
    timing_ms: Vec<f64>,
    timing_median_ms: Option<f64>,
    decoded_i420: Vec<u8>,
    decoder_verdicts: DecoderVerdicts,
}

fn encode_rvp9_target(
    role: ProductRunRole,
    framerate: f64,
    target_bitrate_bps: u32,
    frames: &[YuvFrameBuf],
    ivf_path: &Path,
) -> Result<Rvp9TargetResult, String> {
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps,
        gop_length: frames.len() as u32 + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: BitDepth::Bd8,
        lag_in_frames: 0,
        enable_alt_ref: false,
        rc_mode: RcMode::Vbr,
        cq_level: 0,
        temporal_layers: 1,
        enable_tpl: false,
        stage4a_product_split_override: role.split_override(),
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).map_err(|e| format!("StreamEncoder::new:{e}"))?;
    let mut packets = Vec::with_capacity(frames.len());
    let mut timing_ms = Vec::with_capacity(frames.len());
    let mut total_bits = 0u64;
    for (y, u, v) in frames {
        let start = Instant::now();
        let pkt = enc
            .encode_frame(&yuv_view(y, u, v, W as usize))
            .map_err(|e| format!("encode_frame:{e}"))?;
        timing_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        total_bits += u64::from(pkt.actual_bits);
        packets.push(pkt.data);
    }
    write_packets_as_ivf(ivf_path, framerate, &packets)?;
    let split_stats = enc.diag_take_stage4a_split_stats().into();
    let decoded = decode_rvp9_packets(&packets, frames)?;
    let point = RdPointReport {
        bitrate_kbps: total_bits as f64 / frames.len() as f64 * framerate / 1000.0,
        psnr_db: decoded.psnr_sum / frames.len() as f64,
    };
    let mut sorted = timing_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Rvp9TargetResult {
        point,
        split_stats,
        timing_ms,
        timing_median_ms: sorted.get(sorted.len() / 2).copied(),
        decoded_i420: decoded.i420,
        decoder_verdicts: DecoderVerdicts {
            rvp9_decoder: "PASS".into(),
            vpxdec: "UNVERIFIED".into(),
            ffmpeg: "UNVERIFIED".into(),
        },
    })
}

fn encode_libvpx_target(
    ctx: &ProductContext,
    framerate: f64,
    target_bitrate_bps: u32,
    frames: &[YuvFrameBuf],
    ivf_path: &Path,
) -> Result<RdPointReport, String> {
    let raw_path = ctx
        .run_dir
        .join(format!("raw_{}kbps.yuv", target_bitrate_bps / 1000));
    write_i420_raw(&raw_path, frames).map_err(|e| format!("write raw:{e}"))?;
    let out = Command::new(&ctx.vpxenc)
        .args([
            "--codec=vp9",
            "--ivf",
            "--rt",
            "--cpu-used=7",
            "--tune-content=default",
            "--lag-in-frames=0",
            "--passes=1",
            "--i420",
        ])
        .arg(format!("--width={W}"))
        .arg(format!("--height={H}"))
        .arg(format!("--fps={}/1", framerate.round() as u32))
        .arg(format!("--target-bitrate={}", target_bitrate_bps / 1000))
        .arg("--aq-mode=0")
        .arg("-o")
        .arg(ivf_path)
        .arg(&raw_path)
        .output()
        .map_err(|e| format!("spawn vpxenc:{e}"))?;
    let _ = fs::remove_file(&raw_path);
    if !out.status.success() {
        return Err(format!(
            "vpxenc failed:{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let dec_raw_path = ctx
        .run_dir
        .join(format!("libvpx_{}kbps_dec.yuv", target_bitrate_bps / 1000));
    let decoded = run_yuv_decoder(&ctx.vpxdec, ivf_path, &dec_raw_path)?;
    let frame_size = (W * H + 2 * (W / 2) * (H / 2)) as usize;
    if decoded.len() < frame_size * frames.len() {
        return Err(format!(
            "vpxdec output too small:{} < {}",
            decoded.len(),
            frame_size * frames.len()
        ));
    }
    let y_size = (W * H) as usize;
    let mut psnr_sum = 0.0;
    for (i, (orig_y, _, _)) in frames.iter().enumerate() {
        let dec_y = &decoded[i * frame_size..i * frame_size + y_size];
        psnr_sum += psnr_db(orig_y, dec_y);
    }
    let ivf_bytes = fs::metadata(ivf_path)
        .map_err(|e| format!("libvpx ivf metadata:{e}"))?
        .len();
    let _ = fs::remove_file(&dec_raw_path);
    Ok(RdPointReport {
        bitrate_kbps: ivf_bytes as f64 * 8.0 / frames.len() as f64 * framerate / 1000.0,
        psnr_db: psnr_sum / frames.len() as f64,
    })
}

struct DecodedRvp9 {
    i420: Vec<u8>,
    psnr_sum: f64,
}

fn decode_rvp9_packets(packets: &[Vec<u8>], frames: &[YuvFrameBuf]) -> Result<DecodedRvp9, String> {
    let mut dec = Decoder::new();
    let mut i420 = Vec::with_capacity((W as usize) * (H as usize) * 3 / 2 * packets.len());
    let mut psnr_sum = 0.0;
    for (packet, (orig_y, _, _)) in packets.iter().zip(frames.iter()) {
        let parsed = decode_packet(&mut dec, packet)?;
        if parsed.len() != 1 {
            return Err(format!("unexpected parsed frame count {}", parsed.len()));
        }
        let buf = dec
            .decode_frame(&parsed[0])
            .map_err(|e| format!("rvp9 decode_frame:{e}"))?;
        let yp = buf.y_plane();
        let mut decoded_y = vec![0u8; (W * H) as usize];
        for r in 0..H as usize {
            decoded_y[r * W as usize..(r + 1) * W as usize]
                .copy_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
        }
        psnr_sum += psnr_db(orig_y, &decoded_y);
        i420.extend_from_slice(&yv12_to_i420_bytes(&buf));
    }
    Ok(DecodedRvp9 { i420, psnr_sum })
}

fn decode_packet<'a>(dec: &mut Decoder, pkt: &'a [u8]) -> Result<Vec<ParsedFrame<'a>>, String> {
    let parsed = dec
        .parse_packet(pkt)
        .map_err(|e| format!("parse_packet:{e}"))?;
    Ok(parsed
        .frames()
        .iter()
        .map(|pf| ParsedFrame {
            header: pf.header.clone(),
            header_end_byte: pf.header_end_byte,
            full_frame_data: pf.full_frame_data,
        })
        .collect())
}

fn yv12_to_i420_bytes(buf: &rvp9_core::frame::Yv12Buffer) -> Vec<u8> {
    let yp = buf.y_plane();
    let up = buf.u_plane();
    let vp = buf.v_plane();
    let mut out = Vec::with_capacity((W as usize) * (H as usize) * 3 / 2);
    for r in 0..H as usize {
        out.extend_from_slice(&yp.data()[r * yp.stride()..r * yp.stride() + W as usize]);
    }
    let uv_w = (W / 2) as usize;
    let uv_h = (H / 2) as usize;
    for r in 0..uv_h {
        out.extend_from_slice(&up.data()[r * up.stride()..r * up.stride() + uv_w]);
    }
    for r in 0..uv_h {
        out.extend_from_slice(&vp.data()[r * vp.stride()..r * vp.stride() + uv_w]);
    }
    out
}

fn evaluate_quality_and_split(
    role: ProductRunRole,
    baseline: Option<&ProductReport>,
    report: &mut ProductReport,
) {
    if report.targets.len() != STAGE4A_TARGETS_BPS.len() {
        report.status_reasons.push("INCOMPLETE_TARGET_SET".into());
        return;
    }
    for target in &report.targets {
        if target.decoder_verdicts.rvp9_decoder != "PASS" {
            report
                .status_reasons
                .push(format!("RVP9_DECODER_FAIL_{}", target.target_bitrate_bps));
        }
        if target.decoder_verdicts.vpxdec != "BYTE-EXACT" {
            report
                .status_reasons
                .push(format!("VPXDEC_FAIL_{}", target.target_bitrate_bps));
        }
        if !matches!(
            target.decoder_verdicts.ffmpeg.as_str(),
            "BYTE-EXACT" | "PSNR-GATE"
        ) {
            report
                .status_reasons
                .push(format!("FFMPEG_FAIL_{}", target.target_bitrate_bps));
        }
    }

    let rvp9_curve: Vec<(f64, f64)> = report
        .targets
        .iter()
        .filter_map(|t| t.rvp9.map(|p| (p.bitrate_kbps, p.psnr_db)))
        .collect();
    let libvpx_curve: Vec<(f64, f64)> = report
        .targets
        .iter()
        .filter_map(|t| t.libvpx.map(|p| (p.bitrate_kbps, p.psnr_db)))
        .collect();
    if rvp9_curve.len() >= 2 && libvpx_curve.len() >= 2 {
        let bd = bdrate(&libvpx_curve, &rvp9_curve);
        if bd.is_finite() {
            report.bd_rate_percent = Some(bd);
        } else {
            report.status_reasons.push("BD_RATE_NONFINITE".into());
        }
        if let Some(summary) = psnr_delta_summary(&libvpx_curve, &rvp9_curve) {
            report.matched_psnr_delta_mean_db = Some(summary.mean_delta_db);
            report.matched_psnr_delta_count_in_range = summary.count_in_range;
            if summary.count_in_range < 3 {
                report
                    .status_reasons
                    .push("PSNR_DELTA_IN_RANGE_LT_3".into());
            }
            if summary.mean_delta_db > 1.0 {
                report.status_reasons.push(format!(
                    "PSNR_DELTA_MEAN_GT_1DB:{:.3}",
                    summary.mean_delta_db
                ));
            }
        } else {
            report.status_reasons.push("PSNR_DELTA_UNCOMPUTABLE".into());
        }
        let overlap = psnr_overlap_db(&rvp9_curve, &libvpx_curve);
        report.bd_psnr_overlap_db = Some(overlap);
        if overlap < 0.50 {
            report
                .status_reasons
                .push(format!("BD_PSNR_OVERLAP_LT_0_5DB:{overlap:.3}"));
        }
        if !is_monotonic_psnr(&rvp9_curve) {
            report.status_reasons.push("RVP9_CURVE_NONMONOTONIC".into());
        }
        if !is_monotonic_psnr(&libvpx_curve) {
            report
                .status_reasons
                .push("LIBVPX_CURVE_NONMONOTONIC".into());
        }
        if let Some(bd) = report.bd_rate_percent {
            if bd > 2.0 {
                report
                    .status_reasons
                    .push(format!("BD_RATE_GT_2PCT:{bd:.3}"));
            }
            if matches!(role, ProductRunRole::Candidate) && bd > -2.99 {
                report
                    .status_reasons
                    .push(format!("BD_RATE_REGRESSED_VS_BASELINE:{bd:.3}"));
            }
        }
    } else {
        report.status_reasons.push("RD_CURVES_INCOMPLETE".into());
    }

    let mut all_times: Vec<f64> = report
        .targets
        .iter()
        .flat_map(|t| t.timing_ms.iter().copied())
        .collect();
    all_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    report.timing_median_ms = all_times.get(all_times.len() / 2).copied();

    match role {
        ProductRunRole::PreenablePreflight => evaluate_preenable_materiality(report),
        ProductRunRole::StrictBaseline => {}
        ProductRunRole::Candidate => evaluate_candidate_pairing(baseline, report),
    }
}

fn evaluate_preenable_materiality(report: &mut ProductReport) {
    let total = aggregate_split_stats(report);
    if total.block16_attempts == 0 {
        report.status_reasons.push("NO_BLOCK16_ATTEMPTS".into());
    }
    if total.block16_attempts > 0
        && total.block16_child_infeasible.saturating_mul(100) > total.block16_attempts * 5
    {
        report
            .status_reasons
            .push("BLOCK16_CHILD_INFEASIBLE_GT_5PCT".into());
    }
    let slack_threshold = 200.max(total.block16_partial_rd_early_out / 50);
    if total.block16_slack_would_continue < slack_threshold {
        report.status_reasons.push(format!(
            "SLACK_WOULD_CONTINUE_BELOW_MATERIALITY:{}_LT_{slack_threshold}",
            total.block16_slack_would_continue
        ));
    }
    let targets_with_slack = report
        .targets
        .iter()
        .filter(|t| t.split_stats.block16_slack_would_continue > 0)
        .count();
    if targets_with_slack < 2 {
        report.status_reasons.push(format!(
            "SLACK_WOULD_CONTINUE_TARGETS_LT_2:{targets_with_slack}"
        ));
    }
}

fn evaluate_candidate_pairing(baseline: Option<&ProductReport>, report: &mut ProductReport) {
    let Some(baseline) = baseline else {
        report
            .status_reasons
            .push("MISSING_STRICT_BASELINE_PAIR".into());
        return;
    };
    if baseline.overall_status != "PASS" {
        report
            .status_reasons
            .push("BASELINE_REPORT_NOT_PASS".into());
    }
    if !baseline.validation_clean {
        report
            .status_reasons
            .push("BASELINE_VALIDATION_NOT_CLEAN".into());
    }
    if baseline_contains_forbidden_status(baseline) {
        report
            .status_reasons
            .push("BASELINE_CONTAINS_FORBIDDEN_STATUS".into());
    }
    if baseline.run_role != "strict-baseline" || baseline.split_policy != "Strict" {
        report
            .status_reasons
            .push("INVALID_BASELINE_ROLE_OR_POLICY".into());
    }
    if baseline.split_override != "ForceStrict" {
        report
            .status_reasons
            .push("INVALID_BASELINE_SPLIT_OVERRIDE".into());
    }
    if report.baseline_report_sha256.is_none() {
        report
            .status_reasons
            .push("BASELINE_REPORT_HASH_MISSING".into());
    }
    if baseline.git_sha != report.git_sha {
        report
            .status_reasons
            .push("BASELINE_GIT_SHA_MISMATCH".into());
    }
    if baseline.host_id != report.host_id {
        report
            .status_reasons
            .push("BASELINE_HOST_ID_MISMATCH".into());
    }
    if baseline.product_config_hash != report.product_config_hash {
        report
            .status_reasons
            .push("BASELINE_PRODUCT_CONFIG_HASH_MISMATCH".into());
    }
    if baseline.source_sha256 != report.source_sha256 {
        report
            .status_reasons
            .push("BASELINE_SOURCE_SHA256_MISMATCH".into());
    }
    if baseline.source_tree_hash != report.source_tree_hash {
        report
            .status_reasons
            .push("BASELINE_SOURCE_TREE_HASH_MISMATCH".into());
    }
    if !tool_fingerprints_match(&baseline.vpxenc, &report.vpxenc) {
        report
            .status_reasons
            .push("BASELINE_VPXENC_FINGERPRINT_MISMATCH".into());
    }
    if !tool_fingerprints_match(&baseline.vpxdec, &report.vpxdec) {
        report
            .status_reasons
            .push("BASELINE_VPXDEC_FINGERPRINT_MISMATCH".into());
    }
    if !tool_fingerprints_match(&baseline.ffmpeg, &report.ffmpeg) {
        report
            .status_reasons
            .push("BASELINE_FFMPEG_FINGERPRINT_MISMATCH".into());
    }
    evaluate_block_stats_pairing(baseline, report);
    if !report
        .targets
        .iter()
        .any(|t| t.split_stats.block16_slack_split_committed > 0)
    {
        report
            .status_reasons
            .push("CANDIDATE_DID_NOT_COMMIT_SLACK_SPLIT".into());
    }
    if let (Some(strict_ms), Some(candidate_ms)) =
        (baseline.timing_median_ms, report.timing_median_ms)
    {
        if candidate_ms > strict_ms * 1.15 {
            report.status_reasons.push(format!(
                "TIMING_REGRESSION_GT_15PCT:{candidate_ms:.3}_VS_{strict_ms:.3}"
            ));
        }
    }
}

fn baseline_contains_forbidden_status(report: &ProductReport) -> bool {
    let json = serde_json::to_string(report).expect("serialize baseline report");
    FORBIDDEN_PRODUCT_STATUSES
        .iter()
        .any(|status| json.contains(status))
}

fn tool_fingerprints_match(a: &Option<ToolFingerprint>, b: &Option<ToolFingerprint>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => {
            a.sha256 == b.sha256
                && a.normalized_probe_sha256 == b.normalized_probe_sha256
                && a.probe_command == b.probe_command
        }
        _ => false,
    }
}

fn evaluate_block_stats_pairing(baseline: &ProductReport, report: &mut ProductReport) {
    let (Some(strict), Some(candidate)) = (
        baseline.block_stats_250kbps_summary.as_ref(),
        report.block_stats_250kbps_summary.as_ref(),
    ) else {
        report
            .status_reasons
            .push("BLOCK_STATS_PAIR_MISSING".into());
        return;
    };
    if strict.decode_failed != 0 || candidate.decode_failed != 0 {
        report
            .status_reasons
            .push("BLOCK_STATS_DECODE_FAILED".into());
    }
    if !block_stats_have_exact_frame_counts(strict)
        || !block_stats_have_exact_frame_counts(candidate)
    {
        report
            .status_reasons
            .push("BLOCK_STATS_FRAME_COUNT_MISMATCH".into());
    }
    let strict_small = block_area_pct(strict, "8x8")
        + block_area_pct(strict, "8x16")
        + block_area_pct(strict, "16x8");
    let candidate_small = block_area_pct(candidate, "8x8")
        + block_area_pct(candidate, "8x16")
        + block_area_pct(candidate, "16x8");
    if candidate_small + 2.0 < strict_small {
        report.status_reasons.push(format!(
            "BLOCK_STATS_SMALL_AREA_REGRESSED:{candidate_small:.3}_LT_{strict_small:.3}"
        ));
    }
    let strict_16x16 = block_area_pct(strict, "16x16");
    let candidate_16x16 = block_area_pct(candidate, "16x16");
    if candidate_16x16 > strict_16x16 + 3.0 {
        report.status_reasons.push(format!(
            "BLOCK_STATS_16X16_AREA_REGRESSED:{candidate_16x16:.3}_GT_{strict_16x16:.3}"
        ));
    }
}

fn block_area_pct(summary: &ProductBlockStatsSummary, block: &str) -> f64 {
    summary
        .partition_area_pct
        .iter()
        .find(|entry| entry.block == block)
        .map_or(0.0, |entry| entry.area_pct)
}

fn block_stats_have_exact_frame_counts(summary: &ProductBlockStatsSummary) -> bool {
    summary.rvp9_frames == summary.frames_requested
        && summary
            .libvpx_frames
            .is_some_and(|frames| frames == summary.frames_requested)
}

fn run_product_artifacts(ctx: &ProductContext, report: &mut ProductReport) {
    let Some(target_250) = report
        .targets
        .iter()
        .find(|t| t.target_bitrate_bps == 250_000)
    else {
        report.status_reasons.push("MISSING_250KBPS_TARGET".into());
        return;
    };
    if let (Some(rvp9), Some(libvpx)) = (&target_250.rvp9_ivf_path, &target_250.libvpx_ivf_path) {
        let json_path = ctx.run_dir.join("block_stats_250kbps.json");
        let e1_json_path = ctx
            .run_dir
            .join(format!("block_stats_250kbps_e1_{}.json", report.run_role));
        let md_path = ctx
            .run_dir
            .join(format!("block_stats_250kbps_{}.md", report.run_role));
        let status = run_cargo(
            &ctx.root,
            &[
                "run",
                "--release",
                "-p",
                "rvp9-harness",
                "--bin",
                "e1_block_stats",
                "--",
                "--rvp9-ivf",
                rvp9,
                "--libvpx-ivf",
                libvpx,
                "--frames",
                "64",
                "--fps",
                "30",
                "--out",
                &rel_display(&ctx.root, &md_path),
                "--json-out",
                &rel_display(&ctx.root, &e1_json_path),
            ],
        );
        match status {
            Ok(out) if out.status.success() => {
                match write_product_block_stats(&e1_json_path, &json_path, report) {
                    Ok(summary) => {
                        if summary.decode_failed != 0 {
                            report
                                .status_reasons
                                .push("BLOCK_STATS_DECODE_FAILED".into());
                        }
                        report.block_stats_250kbps_summary = Some(summary);
                        report.block_stats_250kbps_json = Some(rel_display(&ctx.root, &json_path));
                    }
                    Err(e) => report
                        .status_reasons
                        .push(format!("BLOCK_STATS_PARSE_FAILED:{e}")),
                }
            }
            Ok(out) => report.status_reasons.push(format!(
                "BLOCK_STATS_FAILED:{}",
                String::from_utf8_lossy(&out.stderr)
            )),
            Err(e) => report
                .status_reasons
                .push(format!("BLOCK_STATS_SPAWN_FAILED:{e}")),
        }
    }

    for target in [100_000_u32, 150_000, 250_000] {
        let Some(target_report) = report
            .targets
            .iter()
            .find(|t| t.target_bitrate_bps == target)
        else {
            report
                .status_reasons
                .push(format!("DIFF_TARGET_{target}_MISSING"));
            continue;
        };
        let (Some(rvp9), Some(libvpx)) =
            (&target_report.rvp9_ivf_path, &target_report.libvpx_ivf_path)
        else {
            report
                .status_reasons
                .push(format!("DIFF_TARGET_{target}_IVF_MISSING"));
            continue;
        };
        let diff_path = ctx.run_dir.join(format!("diff_{}kbps.json", target / 1000));
        match run_cargo(
            &ctx.root,
            &[
                "run",
                "--release",
                "-p",
                "rvp9-harness",
                "--bin",
                "diff-bitstream",
                "--",
                "--a",
                libvpx,
                "--b",
                rvp9,
                "--level",
                "4",
                "--format",
                "json",
                "--verbose",
                "first",
                "--max-frames",
                "64",
            ],
        ) {
            Ok(out) if out.status.success() && !out.stdout.is_empty() => {
                if !diff_json_lines_are_valid(&out.stdout) {
                    report
                        .status_reasons
                        .push(format!("DIFF_ARTIFACT_INVALID_JSON_{target}"));
                    continue;
                }
                fs::write(&diff_path, out.stdout).expect("write product diff JSON");
                report
                    .diff_json_artifacts
                    .push(rel_display(&ctx.root, &diff_path));
            }
            Ok(out) => report.status_reasons.push(format!(
                "DIFF_ARTIFACT_FAILED_{target}:{}",
                String::from_utf8_lossy(&out.stderr)
            )),
            Err(e) => report
                .status_reasons
                .push(format!("DIFF_ARTIFACT_SPAWN_FAILED_{target}:{e}")),
        }
    }
}

fn write_product_block_stats(
    e1_json_path: &Path,
    product_json_path: &Path,
    report: &ProductReport,
) -> Result<ProductBlockStatsSummary, String> {
    let bytes = fs::read(e1_json_path).map_err(|e| format!("read E1 block JSON:{e}"))?;
    let e1: E1IvfJsonReport =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse E1 block JSON:{e}"))?;
    let decode_failed =
        e1.rvp9.decode_failed + e1.libvpx.as_ref().map_or(0, |libvpx| libvpx.decode_failed);
    let libvpx_frames = e1.libvpx.as_ref().map(|libvpx| libvpx.frames);
    let summary = ProductBlockStatsSummary {
        frames_requested: e1.frames_requested,
        rvp9_frames: e1.rvp9.frames,
        libvpx_frames,
        achieved_kbps: e1.rvp9.achieved_kbps,
        decode_failed,
        partition_area_pct: e1.rvp9.partition_area_pct,
        mode_area_pct: e1.rvp9.mode_area_pct,
        skip_pct: e1.rvp9.skip_pct,
    };
    let artifact = ProductBlockStatsArtifact {
        git_sha: report.git_sha.clone(),
        overall_status: if summary.decode_failed == 0 {
            "PASS".into()
        } else {
            "FAIL".into()
        },
        run_role: report.run_role.clone(),
        split_policy: report.split_policy.clone(),
        validation_clean: validation_clean_now(report),
        ignored_validation_files: report.ignored_validation_files.clone(),
        allowed_ignored_inputs: report.allowed_ignored_inputs.clone(),
        source_tree_hash: report.source_tree_hash.clone(),
        source_sha256: report.source_sha256.clone(),
        product_config_hash: report.product_config_hash.clone(),
        target_bitrate_bps: 250_000,
        frames_requested: summary.frames_requested,
        rvp9_frames: summary.rvp9_frames,
        libvpx_frames: summary.libvpx_frames,
        achieved_kbps: summary.achieved_kbps,
        decode_failed: summary.decode_failed,
        partition_area_pct: summary.partition_area_pct.clone(),
        mode_area_pct: summary.mode_area_pct.clone(),
        skip_pct: summary.skip_pct,
    };
    let json = serde_json::to_vec_pretty(&artifact)
        .map_err(|e| format!("serialize product block stats:{e}"))?;
    fs::write(product_json_path, json).map_err(|e| format!("write product block stats:{e}"))?;
    Ok(summary)
}

fn diff_json_lines_are_valid(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(bytes);
    let mut saw_summary = false;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return false;
        };
        if value.get("event").and_then(serde_json::Value::as_str) == Some("summary") {
            saw_summary = true;
        }
    }
    saw_summary
}

fn finish_and_write_report(root: &Path, path: &Path, report: &mut ProductReport) {
    refresh_final_validation(root, report);
    report.validation_clean = validation_clean_now(report);
    if !report.validation_clean
        && !report
            .status_reasons
            .iter()
            .any(|r| r == "VALIDATION_DIRTY")
    {
        report.status_reasons.push("VALIDATION_DIRTY".into());
    }
    if report.status_reasons.is_empty() {
        report.overall_status = "PASS".into();
    }
    let json = serde_json::to_vec_pretty(report).expect("serialize product report");
    fs::write(path, json).expect("write product report");
    eprintln!("Stage 4a product report: {}", path.display());
    eprintln!("Stage 4a product status: {}", report.overall_status);
    if !report.status_reasons.is_empty() {
        eprintln!("Stage 4a product reasons: {:?}", report.status_reasons);
    }
}

fn validation_clean_now(report: &ProductReport) -> bool {
    report.tracked_diff_empty
        && report.staged_diff_empty
        && report.untracked_validation_files.is_empty()
        && report.ignored_validation_files.len() == report.allowed_ignored_inputs.len()
}

fn refresh_final_validation(root: &Path, report: &mut ProductReport) {
    match git_stdout(root, &["rev-parse", "HEAD"]) {
        Ok(current_sha) if current_sha == report.git_sha => {}
        Ok(current_sha) => report.status_reasons.push(format!(
            "GIT_SHA_CHANGED_DURING_RUN:{}_TO_{current_sha}",
            report.git_sha
        )),
        Err(e) => report
            .status_reasons
            .push(format!("GIT_SHA_FINAL_PROBE_FAILED:{e}")),
    }
    let original_source_tree_hash = report.source_tree_hash.clone();
    validate_worktree(root, report);
    match source_tree_hash(root) {
        Ok(current_hash) if Some(current_hash.clone()) == original_source_tree_hash => {
            report.source_tree_hash = Some(current_hash);
        }
        Ok(current_hash) => {
            report.status_reasons.push(format!(
                "SOURCE_TREE_HASH_CHANGED_DURING_RUN:{}_TO_{current_hash}",
                original_source_tree_hash.unwrap_or_else(|| "NONE".into())
            ));
            report.source_tree_hash = Some(current_hash);
        }
        Err(e) => report
            .status_reasons
            .push(format!("SOURCE_TREE_HASH_FINAL_FAILED:{e}")),
    }
}

fn reject_rvp9_env(report: &mut ProductReport) {
    for (key, _) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if key.to_ascii_uppercase().starts_with("RVP9_") {
            report
                .status_reasons
                .push(format!("FORBIDDEN_ENV_VAR:{key}"));
        }
    }
}

fn validate_worktree(root: &Path, report: &mut ProductReport) {
    report.untracked_validation_files.clear();
    report.ignored_validation_files.clear();
    report.allowed_ignored_inputs.clear();
    report.tracked_diff_empty =
        git_status(root, &git_args_with_paths(&["diff", "--quiet"])).unwrap_or(false);
    report.staged_diff_empty =
        git_status(root, &git_args_with_paths(&["diff", "--cached", "--quiet"])).unwrap_or(false);
    report.untracked_validation_files = git_lines(
        root,
        &git_args_with_paths(&["ls-files", "--others", "--exclude-standard"]),
    )
    .unwrap_or_else(|e| vec![format!("GIT_FAILED:{e}")]);
    let ignored = git_lines(
        root,
        &git_args_with_paths(&["ls-files", "--others", "--ignored", "--exclude-standard"]),
    )
    .unwrap_or_else(|e| vec![format!("GIT_FAILED:{e}")]);
    for path in ignored {
        let normalized = normalize_rel_path(&path);
        if normalized == "tests/corpus/foreman_cif.y4m" {
            let full = root.join(&normalized);
            if let Ok(sha256) = sha256_file(&full) {
                report.allowed_ignored_inputs.push(AllowedIgnoredInput {
                    path: normalized.clone(),
                    sha256,
                    role: "stage4a-product-source".into(),
                });
            }
        }
        report.ignored_validation_files.push(normalized);
    }
}

fn validate_libvpx_gitlink(root: &Path, report: &mut ProductReport) {
    match git_stdout(root, &["ls-tree", "HEAD", "third_party/libvpx"]) {
        Ok(line) => {
            let commit = line.split_whitespace().nth(2).map(str::to_string);
            report.libvpx_gitlink_commit = commit.clone();
            if commit.as_deref() != Some(LIBVPX_COMMIT) {
                report.status_reasons.push("LIBVPX_GITLINK_MISMATCH".into());
            }
        }
        Err(e) => report
            .status_reasons
            .push(format!("LIBVPX_GITLINK_PROBE_FAILED:{e}")),
    }
}

fn resolve_product_tools(
    root: &Path,
    manifest_path: &Path,
    report: &mut ProductReport,
) -> Option<ResolvedTools> {
    let manifest_bytes = match fs::read(manifest_path) {
        Ok(bytes) => bytes,
        Err(e) => {
            report
                .status_reasons
                .push(format!("TOOL_MANIFEST_MISSING:{e}"));
            return None;
        }
    };
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    report.approved_tool_manifest_sha256 = Some(manifest_sha256.clone());
    let manifest: ApprovedToolManifest = match serde_json::from_slice(&manifest_bytes) {
        Ok(manifest) => manifest,
        Err(e) => {
            report
                .status_reasons
                .push(format!("TOOL_MANIFEST_PARSE_FAILED:{e}"));
            return None;
        }
    };
    if manifest.schema != 1 {
        report.status_reasons.push(format!(
            "TOOL_MANIFEST_SCHEMA_UNSUPPORTED:{}",
            manifest.schema
        ));
    }

    let vpx_pairs = resolve_vpx_tool_pairs(root);
    let ffmpeg = resolve_binary(root, "ffmpeg");
    if vpx_pairs.pairs.is_empty() && !vpx_pairs.saw_vpxenc {
        report.status_reasons.push("MISSING_VPXENC".into());
    }
    if vpx_pairs.pairs.is_empty() && vpx_pairs.saw_vpxenc {
        report.status_reasons.push("MISSING_SIBLING_VPXDEC".into());
    }
    if ffmpeg.is_none() {
        report.status_reasons.push("MISSING_FFMPEG".into());
    }
    if vpx_pairs.pairs.is_empty() {
        return None;
    }
    let Some(ffmpeg) = ffmpeg else {
        return None;
    };

    let mut approved_pair = None;
    for (vpxenc, vpxdec) in vpx_pairs.pairs {
        let vpxenc_sha = match sha256_file(&vpxenc) {
            Ok(hash) => hash,
            Err(e) => {
                report
                    .status_reasons
                    .push(format!("VPXENC_HASH_FAILED:{}:{e}", vpxenc.display()));
                continue;
            }
        };
        let vpxdec_sha = match sha256_file(&vpxdec) {
            Ok(hash) => hash,
            Err(e) => {
                report
                    .status_reasons
                    .push(format!("VPXDEC_HASH_FAILED:{}:{e}", vpxdec.display()));
                continue;
            }
        };
        if let Some(row) = manifest.approved_tools.iter().find(|row| {
            row.vpxenc_sha256.eq_ignore_ascii_case(&vpxenc_sha)
                && row.vpxdec_sha256.eq_ignore_ascii_case(&vpxdec_sha)
                && row.libvpx_url == LIBVPX_URL
                && row.libvpx_tag == LIBVPX_TAG
                && row.libvpx_commit == LIBVPX_COMMIT
        }) {
            approved_pair = Some((vpxenc, vpxdec, row.clone()));
            break;
        }
    }
    let Some((vpxenc, vpxdec, row)) = approved_pair else {
        report
            .status_reasons
            .push("UNAPPROVED_LIBVPX_TOOL_PAIR".into());
        return None;
    };

    let vpxenc_fp = fingerprint_tool(&vpxenc, &["--help"], "vpxenc --help")
        .map_err(|e| {
            report
                .status_reasons
                .push(format!("VPXENC_PROBE_FAILED:{e}"))
        })
        .ok()?;
    let vpxdec_fp = fingerprint_tool(&vpxdec, &["--help"], "vpxdec --help")
        .map_err(|e| {
            report
                .status_reasons
                .push(format!("VPXDEC_PROBE_FAILED:{e}"))
        })
        .ok()?;
    let ffmpeg_fp = fingerprint_tool(&ffmpeg, &["-version"], "ffmpeg -version")
        .map_err(|e| {
            report
                .status_reasons
                .push(format!("FFMPEG_PROBE_FAILED:{e}"))
        })
        .ok()?;

    Some(ResolvedTools {
        vpxenc,
        vpxdec,
        ffmpeg,
        approved_tool: row,
        manifest_sha256,
        vpxenc_fp,
        vpxdec_fp,
        ffmpeg_fp,
    })
}

fn product_config_hash(tools: &ResolvedTools) -> String {
    let config = serde_json::json!({
        "source": {
            "path": FOREMAN_REL,
            "expected_sha256": FOREMAN_SHA256,
            "expected_bytes": FOREMAN_BYTES,
            "width": W,
            "height": H,
            "frames": N_FRAMES,
        },
        "target_bitrates_bps": STAGE4A_TARGETS_BPS,
        "encoder": encoder_config_report(),
        "libvpx": {
            "url": LIBVPX_URL,
            "tag": LIBVPX_TAG,
            "commit": LIBVPX_COMMIT,
            "configure_args": tools.approved_tool.configure_args,
            "vpxenc_sha256": tools.vpxenc_fp.sha256,
            "vpxdec_sha256": tools.vpxdec_fp.sha256,
        },
        "ffmpeg": {
            "sha256": tools.ffmpeg_fp.sha256,
            "probe_sha256": tools.ffmpeg_fp.normalized_probe_sha256,
        },
        "validation_allowlist": [
            "tests/corpus/foreman_cif.y4m"
        ],
    });
    sha256_bytes(&serde_json::to_vec(&config).expect("serialize product config"))
}

fn encoder_config_report() -> ProductEncoderConfigReport {
    ProductEncoderConfigReport {
        width: W,
        height: H,
        frames: N_FRAMES,
        target_bitrates_bps: STAGE4A_TARGETS_BPS.to_vec(),
        aq_mode: "Off".into(),
        rc_mode: "Vbr".into(),
        deadline: "Realtime".into(),
        speed: 7,
        lag_in_frames: 0,
        enable_alt_ref: false,
        min_qindex: 4,
        max_qindex: 255,
    }
}

fn write_packets_as_ivf(path: &Path, framerate: f64, packets: &[Vec<u8>]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|e| format!("create ivf:{e}"))?;
    let header = IvfHeader::vp9(W as u16, H as u16, framerate.round() as u32, 1);
    let refs: Vec<&[u8]> = packets.iter().map(Vec::as_slice).collect();
    write_ivf(&mut file, &header, &refs).map_err(|e| format!("write ivf:{e}"))
}

fn write_i420_raw(path: &Path, frames: &[YuvFrameBuf]) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    for (y, u, v) in frames {
        file.write_all(y)?;
        file.write_all(u)?;
        file.write_all(v)?;
    }
    Ok(())
}

fn yuv_view<'a>(y: &'a [u8], u: &'a [u8], v: &'a [u8], w: usize) -> YuvFrameRef<'a> {
    YuvFrameRef {
        y,
        u,
        v,
        y_stride: w,
        u_stride: w / 2,
        v_stride: w / 2,
    }
}

fn run_yuv_decoder(decoder: &Path, ivf: &Path, out: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new(decoder)
        .args(["--i420", "-o"])
        .arg(out)
        .arg(ivf)
        .output()
        .map_err(|e| format!("spawn decoder:{e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    fs::read(out).map_err(|e| format!("read decoded yuv:{e}"))
}

fn run_ffmpeg_decoder(ffmpeg: &Path, ivf: &Path, out: &Path) -> Result<Vec<u8>, String> {
    let output = Command::new(ffmpeg)
        .args(["-y", "-i"])
        .arg(ivf)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(out)
        .output()
        .map_err(|e| format!("spawn ffmpeg:{e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    fs::read(out).map_err(|e| format!("read ffmpeg yuv:{e}"))
}

fn run_cargo(root: &Path, args: &[&str]) -> std::io::Result<Output> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    Command::new(cargo).current_dir(root).args(args).output()
}

fn min_frame_y_psnr(a: &[u8], b: &[u8], w: usize, h: usize) -> f64 {
    let frame = w * h * 3 / 2;
    let y_size = w * h;
    let n = a.len().min(b.len()) / frame;
    (0..n)
        .map(|i| {
            let offset = i * frame;
            psnr_db(&a[offset..offset + y_size], &b[offset..offset + y_size])
        })
        .fold(f64::INFINITY, f64::min)
}

fn psnr_db(orig: &[u8], recon: &[u8]) -> f64 {
    if orig.len() != recon.len() {
        return 0.0;
    }
    let mut sse = 0u64;
    for (a, b) in orig.iter().zip(recon.iter()) {
        let d = i32::from(*a) - i32::from(*b);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 100.0;
    }
    let mse = sse as f64 / orig.len() as f64;
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn psnr_overlap_db(a: &[(f64, f64)], b: &[(f64, f64)]) -> f64 {
    let a_min = a.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let a_max = a.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let b_min = b.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let b_max = b.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    a_max.min(b_max) - a_min.max(b_min)
}

fn is_monotonic_psnr(curve: &[(f64, f64)]) -> bool {
    curve.windows(2).all(|w| w[1].1 >= w[0].1 - 1.5)
}

fn aggregate_split_stats(report: &ProductReport) -> ProductSplitStats {
    let mut out = ProductSplitStats::default();
    for stats in report.targets.iter().map(|t| &t.split_stats) {
        out.block16_attempts += stats.block16_attempts;
        out.block16_breakout_suppressed += stats.block16_breakout_suppressed;
        out.block16_feasible += stats.block16_feasible;
        out.block16_infeasible_or_early += stats.block16_infeasible_or_early;
        out.block16_child_infeasible += stats.block16_child_infeasible;
        out.block16_partial_rd_early_out += stats.block16_partial_rd_early_out;
        for (a, b) in out
            .block16_partial_early_out_after_child
            .iter_mut()
            .zip(stats.block16_partial_early_out_after_child)
        {
            *a += b;
        }
        out.block16_split_beats_none += stats.block16_split_beats_none;
        out.block16_split_loses_to_none += stats.block16_split_loses_to_none;
        out.block16_split_committed += stats.block16_split_committed;
        out.block16_rect_overrides_split += stats.block16_rect_overrides_split;
        out.block16_no_none += stats.block16_no_none;
        out.block16_slack_would_continue += stats.block16_slack_would_continue;
        out.block16_slack_completed += stats.block16_slack_completed;
        out.block16_slack_split_eligible += stats.block16_slack_split_eligible;
        out.block16_slack_split_committed += stats.block16_slack_split_committed;
        out.block16_slack_rect_override += stats.block16_slack_rect_override;
        out.block16_slack_rejected += stats.block16_slack_rejected;
        out.block16_margin_sum += stats.block16_margin_sum;
        out.block16_margin_min = match (out.block16_margin_min, stats.block16_margin_min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (None, Some(b)) => Some(b),
            (a, None) => a,
        };
        out.block16_margin_max = match (out.block16_margin_max, stats.block16_margin_max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (None, Some(b)) => Some(b),
            (a, None) => a,
        };
    }
    out
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonicalize repo root")
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before UNIX epoch")
        .as_secs()
}

fn unix_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before UNIX epoch")
        .as_nanos()
}

fn host_id() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown-host".into())
}

fn resolve_binary(root: &Path, stem: &str) -> Option<PathBuf> {
    let tool_dir = root.join("build/libvpx-win-tools-nodeps");
    for candidate in tool_candidates(&tool_dir, stem) {
        if candidate.is_file() {
            return candidate.canonicalize().ok().or(Some(candidate));
        }
    }
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for candidate in tool_candidates(&dir, stem) {
            if candidate.is_file() {
                return candidate.canonicalize().ok().or(Some(candidate));
            }
        }
    }
    None
}

struct VpxToolSearch {
    pairs: Vec<(PathBuf, PathBuf)>,
    saw_vpxenc: bool,
}

fn resolve_vpx_tool_pairs(root: &Path) -> VpxToolSearch {
    let mut dirs = vec![root.join("build/libvpx-win-tools-nodeps")];
    if let Some(path) = std::env::var_os("PATH") {
        dirs.extend(std::env::split_paths(&path));
    }
    let mut search = VpxToolSearch {
        pairs: Vec::new(),
        saw_vpxenc: false,
    };
    for dir in dirs {
        for vpxenc in tool_candidates(&dir, "vpxenc") {
            if !vpxenc.is_file() {
                continue;
            }
            search.saw_vpxenc = true;
            let vpxenc = vpxenc.canonicalize().ok().unwrap_or(vpxenc);
            if let Some(vpxdec) = sibling_tool(&vpxenc, "vpxdec") {
                let vpxdec = vpxdec.canonicalize().ok().unwrap_or(vpxdec);
                search.pairs.push((vpxenc.clone(), vpxdec));
            }
        }
    }
    search
}

fn sibling_tool(tool: &Path, stem: &str) -> Option<PathBuf> {
    let dir = tool.parent()?;
    tool_candidates(dir, stem)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

fn tool_candidates(dir: &Path, stem: &str) -> Vec<PathBuf> {
    if cfg!(windows) {
        vec![dir.join(format!("{stem}.exe")), dir.join(stem)]
    } else {
        vec![dir.join(stem), dir.join(format!("{stem}.exe"))]
    }
}

fn fingerprint_tool(
    path: &Path,
    args: &[&str],
    command_label: &str,
) -> Result<ToolFingerprint, String> {
    let sha256 = sha256_file(path)?;
    let output = Command::new(path)
        .args(args)
        .output()
        .map_err(|e| format!("spawn probe:{e}"))?;
    let mut probe = Vec::new();
    probe.extend_from_slice(&output.stdout);
    probe.extend_from_slice(&output.stderr);
    let normalized = normalize_probe_text(&String::from_utf8_lossy(&probe));
    let normalized_probe_sha256 = sha256_bytes(normalized.as_bytes());
    let normalized_probe_first_lines = normalized.lines().take(20).map(str::to_string).collect();
    if !output.status.success() {
        return Err(format!(
            "probe failed with status {:?}",
            output.status.code()
        ));
    }
    Ok(ToolFingerprint {
        path: path.display().to_string(),
        sha256,
        probe_command: command_label.into(),
        probe_status: output.status.code(),
        normalized_probe_sha256,
        normalized_probe_first_lines,
    })
}

fn normalize_probe_text(text: &str) -> String {
    let text = text.replace("\r\n", "\n");
    text.trim_end_matches(|c: char| c.is_whitespace())
        .to_string()
}

fn source_tree_hash(root: &Path) -> Result<String, String> {
    let mut paths = git_lines(root, &git_args_with_paths(&["ls-files"]))?;
    paths.sort();
    let mut hasher = Sha256::new();
    for rel in paths {
        let rel = normalize_rel_path(&rel);
        let bytes = fs::read(root.join(&rel)).map_err(|e| format!("read {rel}:{e}"))?;
        hasher.update(rel.as_bytes());
        hasher.update([0]);
        hasher.update(bytes.len().to_string().as_bytes());
        hasher.update([0]);
        hasher.update(&bytes);
        hasher.update([0]);
    }
    Ok(hex_digest(hasher.finalize().as_slice()))
}

fn git_args_with_paths(prefix: &[&str]) -> Vec<String> {
    let mut args: Vec<String> = prefix.iter().map(|s| (*s).to_string()).collect();
    args.push("--".into());
    args.extend(VALIDATION_PATHS.iter().map(|s| (*s).to_string()));
    args
}

fn git_status(root: &Path, args: &[String]) -> Result<bool, String> {
    Ok(Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| format!("spawn git:{e}"))?
        .status
        .success())
}

fn git_lines(root: &Path, args: &[String]) -> Result<Vec<String>, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| format!("spawn git:{e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(normalize_rel_path)
        .collect())
}

fn git_stdout(root: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(args)
        .output()
        .map_err(|e| format!("spawn git:{e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).to_string());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn normalize_rel_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn rel_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
        .replace('\\', "/")
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}:{e}", path.display()))?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(out, "{b:02x}");
    }
    out
}
