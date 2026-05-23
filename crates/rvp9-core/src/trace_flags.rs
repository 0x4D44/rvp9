//! Process-wide cache of `RVP9_TRACE_*` / `RVP9_DEBUG_*` env vars
//! consumed by core hot-path kernels (intra-prediction, entropy
//! detokenize).
//!
//! Companion to `rvp9_decoder::trace_flags`. Both crates need their own
//! cache because `rvp9-core` is the lower-level dependency — it can't
//! reach into the decoder. The split is mechanical; flag semantics are
//! identical to the pre-fix `std::env::var(...)` checks they replace.
//!
//! See the decoder-side module + `wrk_docs/Track_R/R.1_env_var_hot_path_HLD.md`
//! for the design rationale (Windows `GetEnvironmentVariableW` is a
//! syscall; per-block trace gates were burning >tens of ms/frame).
//!
//! The cache snaps the environment exactly once at first access. Tests
//! that need a flag set must do so before exercising the kernels.

use std::sync::OnceLock;

/// Core-side env-var snapshot.
#[derive(Debug, Default)]
pub(crate) struct TraceFlags {
    /// `RVP9_TRACE_INTRA` — `"all"` or `"4x4"` (only inspected in
    /// `intra_pred::intra_predict`). `None` ⇒ trace disabled.
    pub intra: Option<String>,
    /// `RVP9_DEBUG_REFCOEFS` — `"1"` (diff-check only) / `"trace"`
    /// (diff-check + per-call dump).
    pub debug_refcoefs: Option<String>,
    /// `RVP9_DEBUG_SCAN_WRITE=1` — M.17 scan-write trace for Tx8x8.
    pub debug_scan_write: bool,
}

impl TraceFlags {
    fn read_env() -> Self {
        Self {
            intra: std::env::var("RVP9_TRACE_INTRA").ok(),
            debug_refcoefs: std::env::var("RVP9_DEBUG_REFCOEFS").ok(),
            debug_scan_write: std::env::var("RVP9_DEBUG_SCAN_WRITE")
                .is_ok_and(|v| v == "1"),
        }
    }
}

static TRACE_FLAGS: OnceLock<TraceFlags> = OnceLock::new();

/// Process-wide accessor for the core-side trace-flag snapshot.
#[inline]
pub(crate) fn trace_flags() -> &'static TraceFlags {
    TRACE_FLAGS.get_or_init(TraceFlags::read_env)
}
