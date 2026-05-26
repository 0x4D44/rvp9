//! Wave 7 — ML-driven partition heuristics.
//!
//! Foundation tier (W7.5 / W7.6 / W7.7) ports libvpx's neural-net and
//! linear-model partition pruners. The modules ship standalone — no
//! encoder integration in this batch; later waves will wire them into
//! the partition driver.
//!
//! ## libvpx mapping
//!
//! | rvp9 module      | libvpx source                                |
//! |------------------|----------------------------------------------|
//! | [`nn_predict`]   | `vp9/encoder/vp9_encodeframe.c:3111-3153` (runtime) + `vp9/encoder/vp9_partition_models.h:18-974` (13 NN configs + linear tables) |
//! | [`pruners`]      | `vp9/encoder/vp9_encodeframe.c:3155-3427` (`ml_pruning_partition`, `ml_predict_breakout`, `ml_prune_rect_partition`) |
//! | [`var_partition`]| `vp9/encoder/vp9_encodeframe.c:3429-3618` (`simple_motion_search` + `ml_predict_var_rd_partitioning` + `pred_buffer`) |
//!
//! ## Scope
//!
//! - **8-bit only.** HBD variants are explicitly out of scope per the
//!   Wave 7 charter.
//! - **Scalar reference.** SIMD parity lands in a later wave.
//! - **Standalone.** No `MACROBLOCK` / `VP9_COMP` / `PC_TREE` plumbing
//!   — the ports take plain Rust feature structs. Production callers
//!   are responsible for wiring real encoder state.
//! - **Faithful port.** All 13 NN weight/bias tables are transcribed
//!   verbatim from libvpx `vp9_partition_models.h`; the runtime
//!   matches `nn_predict` line-for-line. Citations live at each port site.

pub mod nn_predict;
pub mod pruners;
pub mod var_partition;

pub use nn_predict::{nn_predict, NnConfig, NN_MAX_HIDDEN_LAYERS, NN_MAX_NODES_PER_LAYER};
pub use pruners::{
    ml_predict_breakout, ml_prune_rect_partition, ml_pruning_partition, BreakoutInput,
    PruningDecision, PruningPartitionInput, RectPruneDecision, RectPruneInput,
};
pub use var_partition::{
    ml_predict_var_rd_partitioning, new_pred_buffer, predict_partition_size, simple_motion_search,
    MotionSearchOps, Mv, PlaneView, PlaneViewMut, PredBuffer, VarPartDecision, VarPartInput,
};
