//! Process-wide cache of `RVP9_TRACE_*` / `RVP9_DEBUG_*` env vars
//! consumed by the per-block / per-transform tracing breadcrumbs that
//! live in the decoder hot path.
//!
//! ## Why this exists (R.1)
//!
//! Every site that previously did `std::env::var("RVP9_TRACE_XYZ")` was
//! paying a real syscall (`GetEnvironmentVariableW` on Windows) on every
//! mi-block / every transform / every plane iteration. For a 1080p
//! keyframe with ~30k tx-block iterations × 3 planes that adds up to
//! 100k+ syscalls/frame across the various trace sites — easily tens of
//! milliseconds per frame of pure "is the flag set?" overhead, in
//! production builds where none of the flags are ever set.
//!
//! The fix is to snap the environment exactly once, the first time any
//! decoder code asks for a flag value. Subsequent calls are a single
//! atomic load (via [`std::sync::OnceLock`]) + a struct-field access.
//!
//! ## Trade-off — stale cache
//!
//! Setting an `RVP9_*` env var **after** the first decoder API call has
//! no effect — by the time the test/binary reaches `set_var`, the
//! `OnceLock` has already been initialised from the pre-set environment.
//! This is acceptable because:
//!
//! - Production never sets these vars; the cache is all-defaults and
//!   identical to "env::var returned Err".
//! - Tests that exercise the flags `std::env::set_var(...)` **before**
//!   constructing the [`crate::Decoder`] — the OnceLock initialises on
//!   first decode after that, so the test still sees the trace fire.
//! - "Set env mid-decode" was always racy and untested.
//!
//! See HLD `wrk_docs/Track_R/R.1_env_var_hot_path_HLD.md` §3.

use std::sync::OnceLock;

/// Snapshot of every `RVP9_TRACE_* / RVP9_DEBUG_*` env var read by the
/// decoder hot path. Built once at first access via [`trace_flags`].
///
/// Field names are kebab→snake_case of the env-var suffix. Boolean
/// fields are pre-resolved (`is_ok_and(|v| v == "1")` or `is_ok()`
/// depending on the site's prior semantics — these are preserved
/// exactly). Filter-string fields stay `Option<String>` so the call
/// site can do whatever parsing it already did.
#[derive(Debug, Default)]
pub(crate) struct TraceFlags {
    // -- tile.rs --
    /// `RVP9_TRACE_KFSB=1` — keyframe SB-walk cursor trace.
    pub kfsb: bool,
    /// `RVP9_TRACE_KFPARTITION` (any value) — keyframe partition trace.
    pub kfpartition: bool,
    /// `RVP9_DEBUG_BOOLCURSOR=1` — bool-coder cursor pre-trace.
    pub debug_boolcursor: bool,
    /// `RVP9_TRACE_KFLEAF=1` — per-leaf keyframe trace.
    pub kfleaf: bool,
    /// `RVP9_TRACE_INTRA_AT="plane,x,y"` — positional intra-pred trace.
    pub intra_at: Option<String>,
    /// `RVP9_TRACE_PARTITION` (any value) — inter-partition trace.
    pub partition: bool,
    /// `RVP9_TRACE_INTERMI=1` — per-block INTERMI trace.
    pub intermi: bool,
    /// `RVP9_TRACE_FRAME=<u32>` — limit INTERMI trace to one frame_id.
    pub trace_frame: Option<u32>,
    /// `RVP9_TRACE_SEG_ID` (any value) — segment-id resolution dump.
    pub seg_id: bool,
    /// `RVP9_TRACE_ISINTER="mi_row,mi_col"` — pre-skip bool-reader dump.
    pub isinter: Option<String>,
    /// `RVP9_TRACE_INTER_MV` (any value) — inter-block MV dump.
    pub inter_mv: bool,

    // -- detokenize.rs --
    /// `RVP9_DEBUG_DQCOEFS` value (`"1" / "2" / "tx8" / "all"`).
    /// Stored as parsed enum-y string so call sites preserve their
    /// exact prior `== "1"` / `== "tx8"` checks via [`Self::dqcoefs_eq`].
    pub debug_dqcoefs: Option<String>,
    /// `RVP9_DEBUG_DQCOEFS_AT` — `"x4,y4"` filter for the tx8 dump.
    pub debug_dqcoefs_at: Option<String>,
    /// `RVP9_DEBUG_BUFFER_DUMP` — path for tile-buffer dump.
    pub debug_buffer_dump: Option<String>,

    // -- reconstruct_inter.rs --
    /// `RVP9_TRACE_MC` (any value) — per-MC dump base toggle.
    pub trace_mc: bool,
    /// `RVP9_TRACE_MC_FILTER="row,colmin,colmax"` — scope filter.
    pub trace_mc_filter: Option<String>,
    /// `RVP9_TRACE_MC_SRC` (any value) — source-data dump.
    pub trace_mc_src: bool,
}

impl TraceFlags {
    /// Sample `std::env` once. Called exactly once per process by the
    /// `OnceLock::get_or_init` callback in [`trace_flags`].
    fn read_env() -> Self {
        // Local helpers to keep the per-field lines short.
        let get = |k: &str| std::env::var(k).ok();
        let any = |k: &str| std::env::var(k).is_ok();
        let eq1 = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");

        Self {
            kfsb: eq1("RVP9_TRACE_KFSB"),
            kfpartition: any("RVP9_TRACE_KFPARTITION"),
            debug_boolcursor: eq1("RVP9_DEBUG_BOOLCURSOR"),
            kfleaf: eq1("RVP9_TRACE_KFLEAF"),
            intra_at: get("RVP9_TRACE_INTRA_AT"),
            partition: any("RVP9_TRACE_PARTITION"),
            intermi: get("RVP9_TRACE_INTERMI").as_deref() == Some("1"),
            trace_frame: get("RVP9_TRACE_FRAME").and_then(|s| s.parse::<u32>().ok()),
            seg_id: any("RVP9_TRACE_SEG_ID"),
            isinter: get("RVP9_TRACE_ISINTER"),
            inter_mv: any("RVP9_TRACE_INTER_MV"),
            debug_dqcoefs: get("RVP9_DEBUG_DQCOEFS"),
            debug_dqcoefs_at: get("RVP9_DEBUG_DQCOEFS_AT"),
            debug_buffer_dump: get("RVP9_DEBUG_BUFFER_DUMP"),
            trace_mc: any("RVP9_TRACE_MC"),
            trace_mc_filter: get("RVP9_TRACE_MC_FILTER"),
            trace_mc_src: any("RVP9_TRACE_MC_SRC"),
        }
    }

    /// Convenience: `RVP9_DEBUG_DQCOEFS == "<wanted>"`. Replaces the
    /// pre-fix `std::env::var("RVP9_DEBUG_DQCOEFS").is_ok_and(|v| v == "<wanted>")`
    /// pattern.
    #[inline]
    pub(crate) fn dqcoefs_eq(&self, wanted: &str) -> bool {
        self.debug_dqcoefs.as_deref() == Some(wanted)
    }
}

static TRACE_FLAGS: OnceLock<TraceFlags> = OnceLock::new();

/// Process-wide accessor for the [`TraceFlags`] snapshot. First call
/// reads the environment; every subsequent call is a single atomic
/// load + pointer return.
#[inline]
pub(crate) fn trace_flags() -> &'static TraceFlags {
    TRACE_FLAGS.get_or_init(TraceFlags::read_env)
}
