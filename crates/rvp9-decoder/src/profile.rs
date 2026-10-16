//! T-5 per-tile-dispatch profiling instrumentation.
//!
//! This module is compiled only under the crate's `profile-tiles`
//! cargo feature. With the feature off (default), the whole module
//! vanishes at parse time — every use site in `lib.rs` is also
//! `#[cfg(feature = "profile-tiles")]`-gated. Conformance and bench
//! numbers are unchanged in the default build.
//!
//! ## What this module provides
//!
//! - [`now_cycles`] — `unsafe`-free `#[inline(always)]` wrapper around
//!   `core::arch::x86_64::_rdtsc()`. The whole module is gated on
//!   `cfg(target_arch = "x86_64")` by the lib.rs `mod` declaration so
//!   non-x86 hosts skip the instrumentation entirely.
//! - [`PerTileTiming`] — three rdtsc-delta fields (lock_in, decode,
//!   lock_out). One per tile slot under the feature. Workers write
//!   their own slot; main thread reads after scope return.
//! - [`PerFrameTiming`] — main-thread phase deltas plus a `Vec` of
//!   per-tile timings and a wall-time sanity column.
//! - [`emit_row`] — append one CSV row per frame to the path in
//!   `RVP9_PROFILE_TILES_OUT`. No I/O if the env var is unset.
//! - [`estimate_cpu_mhz`] — 10 ms rdtsc-vs-`Instant` calibration so the
//!   analysis tool can convert cycles → microseconds.
//!
//! See `wrk_docs/Track_T/T5_tile_dispatch_profile_HLD.md` §3 for the
//! design rationale (rdtsc over QPC, feature flag over env-var gate,
//! per-tile in-slot writes vs. thread-local discovery).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// CSV schema version. Bump whenever the column list changes. The
/// analysis tool checks column 0 before parsing the rest.
pub const CSV_SCHEMA_VERSION: u32 = 1;

/// Per-tile cycle deltas for one frame, written by the worker before
/// it releases its slot. `Copy` so the main thread can read it after
/// the scope returns without taking ownership.
#[derive(Default, Copy, Clone, Debug)]
pub struct PerTileTiming {
    /// Cycles spent acquiring the six per-slot mutexes at job start.
    pub lock_in: u64,
    /// Cycles spent inside `decode_tile_intra` / `decode_tile_inter`
    /// (the actual useful work).
    pub decode: u64,
    /// Cycles spent restoring the six per-slot mutexes at job end.
    pub lock_out: u64,
}

/// Per-frame, main-thread-owned timing bucket. One per
/// `decode_frame` call. After all phases land we hand this to
/// [`emit_row`] which appends a CSV row.
#[derive(Default, Clone, Debug)]
pub struct PerFrameTiming {
    /// Cycles for pre-scope serial setup: per_tile_bounds compute,
    /// scratch pool take, per-tile slot Vec allocation.
    pub setup: u64,
    /// Cycles for the `worker_pool.scope` submit-loop on main (between
    /// scope-open and last `scope.submit` return).
    pub submit: u64,
    /// Cycles main blocks waiting for all workers (scope return after
    /// the last submit). Captures the join barrier.
    pub join: u64,
    /// Cycles for the post-scope serial merge: counts, pixel
    /// rectangles, seg_ids, mv_grid.
    pub merge: u64,
    /// Cycles for `build_frame_mi_grid`.
    pub grid_stitch: u64,
    /// Cycles for `apply_loop_filter_to_frame` (single-threaded).
    pub loopfilter: u64,
    /// Cycles for `extend_frame_borders` (single-threaded).
    pub border: u64,
    /// Per-tile cycle deltas, one entry per tile.
    pub per_tile: Vec<PerTileTiming>,
    /// Total frame decode wall-time (QPC via `Instant`) for sanity
    /// checking — the sum of phases should agree to within ±2%.
    pub frame_total_wall_us: f64,
}

/// Read the invariant TSC. Returns a raw cycle count; conversion to
/// microseconds is the caller's job (via [`estimate_cpu_mhz`]).
#[inline(always)]
pub fn now_cycles() -> u64 {
    // SAFETY: `_rdtsc` is unconditionally safe to execute in user-mode
    // on every x86_64 CPU; no privilege or feature gate beyond
    // `target_arch = "x86_64"` (which the parent `mod profile` in
    // `lib.rs` already enforces via the `#[cfg]` on the declaration).
    // The intrinsic is marked `unsafe` only because the SDM allows
    // OSes to disable user-mode rdtsc via CR4.TSD; that knob is off
    // on every supported OS/host configuration. See HLD §3.1 and §R3
    // in the risk register.
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// One-shot estimate of `cycles_per_microsecond` for the current CPU.
/// Spins for ~10 ms reading rdtsc against `Instant::now()`. Cached so
/// subsequent calls are free.
///
/// Returns the ratio `cycles / microsecond` (i.e. effective CPU MHz).
/// On a 3 GHz CPU this returns ~3000.0.
pub fn estimate_cpu_mhz() -> f64 {
    static CACHED: OnceLock<f64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // 10 ms is plenty for a stable estimate; on a 3 GHz CPU that
        // is ~30M cycles, dwarfing the ~20 ns QPC noise floor.
        let calibration = std::time::Duration::from_millis(10);
        let t_start = Instant::now();
        let c_start = now_cycles();
        while t_start.elapsed() < calibration {
            std::hint::spin_loop();
        }
        let c_end = now_cycles();
        let elapsed_us = t_start.elapsed().as_secs_f64() * 1e6;
        let cycles = c_end.wrapping_sub(c_start) as f64;
        if elapsed_us > 0.0 {
            cycles / elapsed_us
        } else {
            // Fallback to a sensible default if QPC misbehaves.
            3000.0
        }
    })
}

/// Holds the resolved CSV output path on first use, plus a `Mutex`
/// over the open file handle.
struct CsvSink {
    file: Mutex<std::fs::File>,
    header_written: AtomicBool,
}

static ROWS_WRITTEN: AtomicU64 = AtomicU64::new(0);
static WRITE_FAILED: AtomicBool = AtomicBool::new(false);

/// Snapshot the process-wide successful row count and latched output failure.
pub fn output_status() -> crate::TileProfileOutputStatus {
    crate::TileProfileOutputStatus {
        rows_written: ROWS_WRITTEN.load(Ordering::Acquire),
        write_failed: WRITE_FAILED.load(Ordering::Acquire),
    }
}

/// Lazy initialiser for the CSV sink. Returns `None` if
/// `RVP9_PROFILE_TILES_OUT` is unset (no I/O path), else opens
/// (creating + truncating) the file once.
fn sink() -> Option<&'static CsvSink> {
    static SINK: OnceLock<Option<CsvSink>> = OnceLock::new();
    SINK.get_or_init(|| {
        let path = std::env::var("RVP9_PROFILE_TILES_OUT").ok()?;
        if path.is_empty() {
            return None;
        }
        if let Some(parent) = std::path::Path::new(&path).parent()
            && !parent.as_os_str().is_empty()
            && !parent.exists()
            && std::fs::create_dir_all(parent).is_err()
        {
            WRITE_FAILED.store(true, Ordering::Release);
            return None;
        }
        let file = match std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
        {
            Ok(file) => file,
            Err(_) => {
                WRITE_FAILED.store(true, Ordering::Release);
                return None;
            }
        };
        Some(CsvSink {
            file: Mutex::new(file),
            header_written: AtomicBool::new(false),
        })
    })
    .as_ref()
}

/// Global frame counter; bumped once per [`emit_row`] call. Used to
/// stamp a stable `frame_idx` into each row regardless of which
/// thread or decoder instance emitted it.
static GLOBAL_FRAME_IDX: AtomicU64 = AtomicU64::new(0);

/// Maximum per-tile columns to emit. Beyond this we truncate. Four
/// tiles is the canonical tile_1x4 fixture; eight covers tile_1x8.
const MAX_TILES_IN_CSV: usize = 8;

fn write_header(file: &mut impl std::io::Write) -> std::io::Result<()> {
    let mut header = String::from(
        "schema,frame_idx,num_tiles,num_threads,wall_us,setup,submit,join,merge,grid_stitch,loopfilter,border",
    );
    for t in 0..MAX_TILES_IN_CSV {
        header.push_str(&format!(",tile{t}_lockin,tile{t}_decode,tile{t}_lockout"));
    }
    header.push('\n');
    file.write_all(header.as_bytes())
}

// The failure latch prevents a partial header or row from being followed by
// apparently valid evidence. Called while holding the sink's file lock.
fn write_row(
    file: &mut impl std::io::Write,
    header_written: &AtomicBool,
    failed: &AtomicBool,
    rows_written: &AtomicU64,
    row: &[u8],
) {
    if failed.load(Ordering::Acquire) {
        return;
    }
    if !header_written.load(Ordering::Relaxed) {
        if write_header(file).is_err() {
            failed.store(true, Ordering::Release);
            return;
        }
        header_written.store(true, Ordering::Relaxed);
    }
    if file.write_all(row).is_err() {
        failed.store(true, Ordering::Release);
        return;
    }
    rows_written.fetch_add(1, Ordering::Release);
}

/// Append one CSV row for the given per-frame timing snapshot. No-op
/// if `RVP9_PROFILE_TILES_OUT` is unset.
///
/// `frame_idx` is ignored — replaced with a monotonically-increasing
/// process-wide counter so concurrent decoders cannot collide. (The
/// parameter is kept to match the HLD signature for callers that
/// want to pass their own index.)
pub fn emit_row(t: &PerFrameTiming, frame_idx: u64) {
    let _ = frame_idx; // counter overrides; kept for HLD signature parity
    let Some(sink) = sink() else { return };
    let idx = GLOBAL_FRAME_IDX.fetch_add(1, Ordering::Relaxed);
    let num_tiles = t.per_tile.len();
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let mut row = format!(
        "{schema},{idx},{nt},{th},{wall:.3},{setup},{submit},{join},{merge},{grid},{lf},{border}",
        schema = CSV_SCHEMA_VERSION,
        idx = idx,
        nt = num_tiles,
        th = num_threads,
        wall = t.frame_total_wall_us,
        setup = t.setup,
        submit = t.submit,
        join = t.join,
        merge = t.merge,
        grid = t.grid_stitch,
        lf = t.loopfilter,
        border = t.border,
    );
    for slot in 0..MAX_TILES_IN_CSV {
        if slot < num_tiles {
            let pt = t.per_tile[slot];
            row.push_str(&format!(",{},{},{}", pt.lock_in, pt.decode, pt.lock_out));
        } else {
            row.push_str(",0,0,0");
        }
    }
    row.push('\n');
    if let Ok(mut file) = sink.file.lock() {
        write_row(
            &mut *file,
            &sink.header_written,
            &WRITE_FAILED,
            &ROWS_WRITTEN,
            row.as_bytes(),
        );
    } else {
        WRITE_FAILED.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ShortWriter {
        remaining: usize,
        bytes: Vec<u8>,
    }
    impl std::io::Write for ShortWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let n = self.remaining.min(buf.len());
            self.bytes.extend_from_slice(&buf[..n]);
            self.remaining -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn profile_output_counts_only_complete_rows_and_latches_partial_writes() {
        let mut header = Vec::new();
        write_header(&mut header).unwrap();
        let row = b"profile row\n";
        for capacity in [
            0,
            5,
            header.len(),
            header.len() + 3,
            header.len() + row.len(),
        ] {
            let mut writer = ShortWriter {
                remaining: capacity,
                bytes: Vec::new(),
            };
            let header_written = AtomicBool::new(false);
            let failed = AtomicBool::new(false);
            let rows = AtomicU64::new(0);
            write_row(&mut writer, &header_written, &failed, &rows, row);
            let complete = capacity == header.len() + row.len();
            assert_eq!(rows.load(Ordering::Relaxed), u64::from(complete));
            assert_eq!(failed.load(Ordering::Relaxed), !complete);
            assert_eq!(
                header_written.load(Ordering::Relaxed),
                capacity >= header.len()
            );
            let written = writer.bytes.len();
            writer.remaining = usize::MAX;
            write_row(&mut writer, &header_written, &failed, &rows, row);
            if !complete {
                assert_eq!(
                    writer.bytes.len(),
                    written,
                    "failed sink must remain failed"
                );
                assert_eq!(rows.load(Ordering::Relaxed), 0);
            } else {
                assert_eq!(rows.load(Ordering::Relaxed), 2);
            }
        }
    }
}
