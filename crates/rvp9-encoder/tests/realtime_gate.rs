// Integration test: stderr IS the output mechanism (the harness's job
// is to print a measurement report). The workspace's `print_stderr =
// "deny"` lint targets library code; for this binary that lint is the
// wrong rule.
#![allow(clippy::print_stderr)]

//! RC.3 — 720p30 frame-time oracle (Stage 4a exit gate #1).
//!
//! Gated by `#[ignore]` so it doesn't run on every `cargo test`. Invoke
//! explicitly, **in release mode**:
//!
//! ```text
//! cargo test --release --test realtime_gate -- --ignored --nocapture
//! ```
//!
//! ## Why `--release`
//!
//! Frame-time measurement in debug builds is meaningless — `rvp9-core`
//! and `rvp9-encoder` rely heavily on inlining and bounds-check
//! elision for performance. A debug-mode run is typically 30-100×
//! slower than release, and the 33 ms budget makes no sense against
//! debug timings. **The assertion is calibrated against `--release`
//! timings only.**
//!
//! ## What this measures
//!
//! Stage 4a exit gate #1 from `PLAN.md`:
//!
//! > 720p30 real-time: ≤33 ms/frame median on a single modern x86_64
//! > core.
//!
//! The harness:
//!
//! 1. Locates `tests/corpus/testsrc_720p30.y4m` via
//!    [`rvp9_harness::y4m::probe_testsrc_720p`]. If absent, prints a
//!    skip message and returns successfully.
//! 2. Reads 30 frames at the encoder's coded resolution
//!    (`1280 × 704`, see below) via
//!    [`rvp9_harness::y4m::read_cropped`].
//! 3. For each frame: calls
//!    [`rvp9_encoder::StreamEncoder::encode_frame`] and records the
//!    wall-clock elapsed time via `std::time::Instant`.
//! 4. Computes the **median** frame time over frames `5..N` (the first
//!    5 frames are warm-up — they trigger one-shot allocations and the
//!    page-fault path on the reconstruction buffers, neither of which
//!    is representative of steady-state cost).
//! 5. Reports the 5-number summary to stderr.
//! 6. Asserts `median ≤ 33 ms`. The test FAILS if the gate misses; the
//!    diagnostic explains what to do next.
//!
//! ## Coded resolution: 1280 × 704, not 1280 × 720
//!
//! `StreamEncoder` requires both `width` and `height` to be positive
//! multiples of 64 (the VP9 super-block size). 720 ÷ 64 = 11.25 — not
//! a multiple. The closest 64-multiple ≤ 720 is **704**, so we
//! centre-crop 16 rows (8 top + 8 bottom) from the 720p source. This
//! preserves the FULL 1280-wide luma plus 97.8 % of the rows; the
//! frame-time measurement is for all practical purposes equivalent to
//! a "true 720p" encode (the workload is dominated by the per-block
//! kernels, which are independent of the trimmed boundary rows).
//!
//! When the encoder grows arbitrary-resolution support (cropping
//! within a 64-multiple coded frame, per VP9's `render_width`/
//! `render_height` mechanism), this test should re-target 1280 × 720
//! directly. Until then, 1280 × 704 is the honest measurement.
//!
//! ## What to do if the gate FAILS
//!
//! Do not try to optimize from here — that's outside this test's
//! scope. Instead:
//!
//! 1. Read the printed 5-number summary; understand by how much we
//!    miss.
//! 2. Profile via:
//!    ```text
//!    cargo flamegraph -p rvp9-encoder --test realtime_gate -- \
//!        --ignored
//!    ```
//!    to identify the hot kernels.
//! 3. Open a SIMD-pilot ticket (the pattern is in `G4.1` — see the
//!    `idct8x8` AVX2 work).

use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};
use rvp9_harness::y4m::{probe_testsrc_720p, read_cropped};
use std::time::{Duration, Instant};

/// Coded luma width — equal to 720p's 1280.
const W: u32 = 1280;
/// Coded luma height. 720 is NOT a multiple of 64 (VP9 super-block);
/// 704 is the closest 64-multiple ≤ 720 (see the module docs).
const H: u32 = 704;
/// Total frames to read + encode.
const N_FRAMES: usize = 30;
/// Frames to skip as warm-up before computing the median. The first
/// few frames trigger one-shot allocations and the page-fault path on
/// the reconstruction buffers; they are not representative of
/// steady-state cost.
const WARMUP_FRAMES: usize = 5;
/// Stage 4a budget: 33 ms = 1 / 30 fps. The literal exit-gate
/// threshold from `PLAN.md` Stage 4a §exit criteria item 1.
const BUDGET_NS: u128 = 33_000_000;

/// Build a tight-stride `YuvFrameRef` over `(y, u, v)` planes.
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

/// Format a duration in milliseconds with two decimals.
fn fmt_ms(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1_000.0)
}

/// Return the 5-number summary `(min, p25, median, p75, max)` over a
/// pre-sorted slice. Uses lower-index quantile selection
/// (`samples[(N × q).floor()]`) — sufficient for a ~25-sample frame
/// time distribution where the next-tick interpolation noise dwarfs
/// quantile-method precision.
fn five_number_summary(sorted: &[Duration]) -> (Duration, Duration, Duration, Duration, Duration) {
    assert!(!sorted.is_empty(), "five_number_summary needs ≥ 1 sample");
    let n = sorted.len();
    let pick = |q: f64| sorted[((n as f64 * q) as usize).min(n - 1)];
    let median_idx = n / 2;
    (sorted[0], pick(0.25), sorted[median_idx], pick(0.75), sorted[n - 1])
}

/// RC.3 — 720p30 single-core real-time frame-time gate.
///
/// Stage 4a exit criterion #1: `≤33 ms/frame median on a single
/// modern x86_64 core`. Skip-cleans when the y4m corpus is unavailable
/// (the typical CI environment); otherwise enforces the budget via
/// `assert!`.
#[test]
#[ignore = "RC.3 — 720p30 frame-time exit gate; run with `--release --ignored`"]
fn realtime_gate_720p30() {
    // ---------------------------------------------------------------
    // Step 1: locate the y4m corpus, or skip-clean.
    // ---------------------------------------------------------------
    let Some(path) = probe_testsrc_720p() else {
        eprintln!(
            "[SKIPPED: no testsrc corpus available — fetch via \
             scripts/fetch-corpus.{{ps1,sh}}]"
        );
        return;
    };

    eprintln!("RC.3 — 720p30 Real-Time Frame-Time Gate");
    eprintln!("========================================");
    eprintln!("Source: {}", path.display());
    eprintln!("Coded resolution: {W} × {H} (centre-cropped 1280×720 → {W}×{H})");
    eprintln!("Frames: {N_FRAMES}    Warm-up: {WARMUP_FRAMES}    Budget: 33.00 ms (30 fps)");
    eprintln!();

    // ---------------------------------------------------------------
    // Step 2: read N frames at 1280×704 (centre-crop of 1280×720).
    // ---------------------------------------------------------------
    let (frames, framerate) = match read_cropped(&path, W, H, N_FRAMES) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "[SKIPPED: y4m read failed ({e}) — fetch via \
                 scripts/fetch-corpus.{{ps1,sh}}]"
            );
            return;
        }
    };
    if frames.len() < N_FRAMES {
        eprintln!(
            "[SKIPPED: only got {}/{N_FRAMES} frames from corpus — \
             fetch a longer testsrc via scripts/fetch-corpus.{{ps1,sh}}]",
            frames.len()
        );
        return;
    }

    // ---------------------------------------------------------------
    // Step 3: build the encoder; configure for RT-style operation.
    //
    // Target bitrate is set to 2 Mbps — a representative WebRTC
    // 720p30 operating point. AQ is OFF (Stage 4a baseline gate; AQ
    // is Stage 4b). GOP length matches `N_FRAMES + 1` so the run
    // contains exactly one keyframe at frame 0 (the worst frame for
    // the timing, but still included in the measurement after warm-up).
    // ---------------------------------------------------------------
    let cfg = EncoderConfig {
        width: W,
        height: H,
        framerate,
        target_bitrate_bps: 2_000_000,
        gop_length: (N_FRAMES as u32) + 1,
        aq_mode: AqMode::Off,
        use_prev_frame_mvs: false,
        speed: 7,
        deadline: rvp9_encoder::Deadline::Realtime,
        noise_sensitivity: 0,
        bit_depth: rvp9_core::BitDepth::Bd8,
        lag_in_frames: 0,
        min_qindex: 4,
        max_qindex: 255,
    };
    let mut enc = StreamEncoder::new(cfg).expect("StreamEncoder::new");

    // ---------------------------------------------------------------
    // Step 4: encode each frame, recording wall-clock elapsed.
    // ---------------------------------------------------------------
    let mut elapsed: Vec<Duration> = Vec::with_capacity(N_FRAMES);
    for (i, (y, u, v)) in frames.iter().enumerate() {
        let view = yuv_view(y, u, v, W as usize);
        let t0 = Instant::now();
        let _pkt = enc.encode_frame(&view).expect("encode_frame");
        let dt = t0.elapsed();
        elapsed.push(dt);
        eprintln!(
            "  frame {i:>2}: {}{}",
            fmt_ms(dt),
            if i < WARMUP_FRAMES { "  [warm-up]" } else { "" }
        );
    }
    eprintln!();

    // ---------------------------------------------------------------
    // Step 5: 5-number summary over post-warmup samples.
    // ---------------------------------------------------------------
    let mut sorted: Vec<Duration> = elapsed[WARMUP_FRAMES..].to_vec();
    sorted.sort();
    let (mn, p25, median, p75, mx) = five_number_summary(&sorted);
    eprintln!(
        "Timing summary (over frames {WARMUP_FRAMES}..{N_FRAMES}, n={}):",
        sorted.len()
    );
    eprintln!("  min   : {}", fmt_ms(mn));
    eprintln!("  p25   : {}", fmt_ms(p25));
    eprintln!("  median: {}", fmt_ms(median));
    eprintln!("  p75   : {}", fmt_ms(p75));
    eprintln!("  max   : {}", fmt_ms(mx));
    eprintln!();

    // ---------------------------------------------------------------
    // Step 6: assert the gate.
    // ---------------------------------------------------------------
    let median_ns = median.as_nanos();
    let median_ms = median.as_secs_f64() * 1_000.0;
    if median_ns <= BUDGET_NS {
        eprintln!(
            "Stage 4a RT gate (median ≤ 33 ms): PASS — median {median_ms:.2} ms ≤ 33.00 ms"
        );
    } else {
        let over_ms = (median_ns as i128 - BUDGET_NS as i128) as f64 / 1_000_000.0;
        eprintln!(
            "Stage 4a RT gate (median ≤ 33 ms): FAIL by {over_ms:.2} ms — \
             median {median_ms:.2} ms > 33.00 ms"
        );
        eprintln!(
            "Stage 4a RT gate FAILS by {over_ms:.2}ms. Profile via \
             `cargo flamegraph -p rvp9-encoder --test realtime_gate -- \
             --ignored` to identify hot kernels."
        );
    }

    assert!(
        median_ns <= BUDGET_NS,
        "RC.3 Stage 4a RT gate: median frame time {median_ms:.2} ms > 33.00 ms budget",
    );
}
