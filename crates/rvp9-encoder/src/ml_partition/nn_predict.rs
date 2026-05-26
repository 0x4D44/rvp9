//! W7.5 — Neural-net partition predictor runtime + 13 NN configs.
//!
//! ## libvpx references
//!
//! - **Runtime — `nn_predict`** — `vp9/encoder/vp9_encodeframe.c:3111-3153`
//!   (commit `12f3a2ac6`). A tiny fully-connected float-MLP evaluator:
//!   ReLU on hidden layers, linear on the output layer. Used by the ML
//!   partition pruners.
//! - **13 NN configs** — `vp9/encoder/vp9_partition_models.h:18-974`.
//!   Each config is a static const set of weight + bias float arrays
//!   plus the layout (`num_inputs`, `num_outputs`, `num_hidden_layers`,
//!   `num_hidden_nodes`). Specific table line ranges are cited at each
//!   `NnConfig::new(...)` site below.
//!
//! ## Verbatim port
//!
//! Every float in this module matches its libvpx counterpart bit-for-bit.
//! Re-ordering / re-formatting was done only where Rust syntax demanded
//! it; no values were dropped, rounded, or recomputed.
//!
//! ## Scope
//!
//! - 8-bit only (HBD skipped — see Wave 7 charter).
//! - Scalar reference — SIMD parity lands later.
//! - Standalone — modules ship without encoder integration; the
//!   downstream W7.6 / W7.7 pruners call into the runtime / tables here.

#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

/// Maximum hidden layers supported by an [`NnConfig`].
/// Mirrors libvpx `NN_MAX_HIDDEN_LAYERS` (`vp9_partition_models.h:18`).
pub const NN_MAX_HIDDEN_LAYERS: usize = 10;

/// Maximum nodes per hidden layer.
/// Mirrors libvpx `NN_MAX_NODES_PER_LAYER` (`vp9_partition_models.h:19`).
pub const NN_MAX_NODES_PER_LAYER: usize = 128;

/// Neural-net model configuration — the Rust counterpart of libvpx
/// `NN_CONFIG` (`vp9_partition_models.h:24-34`).
///
/// Each weight slice is the row-major flattened weight matrix for one
/// layer: `weights[layer]` has length `num_inputs(layer) *
/// num_outputs(layer)`, indexed `node * num_inputs + i`. Bias slices
/// have length `num_outputs(layer)`. Slices indexed 0..num_hidden_layers
/// are hidden layers; index `num_hidden_layers` is the output layer.
#[derive(Debug, Clone, Copy)]
pub struct NnConfig {
    /// Number of input features.
    pub num_inputs: usize,
    /// Number of output nodes.
    pub num_outputs: usize,
    /// Number of hidden layers (≤ [`NN_MAX_HIDDEN_LAYERS`]).
    pub num_hidden_layers: usize,
    /// Number of nodes for each hidden layer. Only the first
    /// `num_hidden_layers` entries are meaningful.
    pub num_hidden_nodes: [usize; NN_MAX_HIDDEN_LAYERS],
    /// Weight slices, indexed by layer. Length =
    /// `num_hidden_layers + 1`. Use empty slice (`&[]`) for any unused
    /// trailing entry — only entries 0..=num_hidden_layers are read.
    pub weights: [&'static [f32]; NN_MAX_HIDDEN_LAYERS + 1],
    /// Bias slices, indexed by layer. Length = `num_hidden_layers + 1`.
    pub bias: [&'static [f32]; NN_MAX_HIDDEN_LAYERS + 1],
}

/// Predict an output vector from a feature vector using `nn_config`.
///
/// libvpx `nn_predict` (`vp9_encodeframe.c:3111-3153`). ReLU on hidden
/// layers, linear on the output. Two `NN_MAX_NODES_PER_LAYER`-sized
/// scratch buffers are flip-flopped between layers — identical to the
/// libvpx control flow.
///
/// # Panics
/// - if `features.len() != nn_config.num_inputs`,
/// - if `output.len() < nn_config.num_outputs`,
/// - if `nn_config.num_hidden_layers > NN_MAX_HIDDEN_LAYERS`,
/// - if any hidden layer has `>= NN_MAX_NODES_PER_LAYER` nodes
///   (libvpx asserts the same).
pub fn nn_predict(features: &[f32], nn_config: &NnConfig, output: &mut [f32]) {
    assert!(
        features.len() == nn_config.num_inputs,
        "nn_predict: features.len ({}) != num_inputs ({})",
        features.len(),
        nn_config.num_inputs
    );
    assert!(
        output.len() >= nn_config.num_outputs,
        "nn_predict: output.len ({}) < num_outputs ({})",
        output.len(),
        nn_config.num_outputs
    );
    assert!(
        nn_config.num_hidden_layers <= NN_MAX_HIDDEN_LAYERS,
        "nn_predict: num_hidden_layers ({}) > NN_MAX_HIDDEN_LAYERS",
        nn_config.num_hidden_layers
    );

    let num_layers = nn_config.num_hidden_layers;
    // Flip-flop scratch buffers — libvpx `float buf[2][NN_MAX_NODES_PER_LAYER]`.
    let mut buf = [[0.0f32; NN_MAX_NODES_PER_LAYER]; 2];
    let mut buf_index: usize = 0;
    let mut num_input_nodes = nn_config.num_inputs;

    // Use a small heap-free copy of the current input on the first iteration:
    // libvpx points `input_nodes` at the original `features` slice for layer 0
    // then at `buf[1-buf_index]` thereafter. We model this with an enum-less
    // approach: stash a slice reference.
    let mut current_input: &[f32] = features;

    // Propagate hidden layers.
    for layer in 0..num_layers {
        let weights = nn_config.weights[layer];
        let bias = nn_config.bias[layer];
        let num_output_nodes = nn_config.num_hidden_nodes[layer];
        assert!(
            num_output_nodes < NN_MAX_NODES_PER_LAYER,
            "nn_predict: hidden layer {layer} has {num_output_nodes} nodes (>= NN_MAX_NODES_PER_LAYER)"
        );

        // Slice the destination buffer to the active half.
        let (dst, _) = buf.split_at_mut(1);
        // We only need the slice corresponding to `buf_index`. Borrow checker:
        // take a mutable reference into `buf[buf_index]`.
        // (We already grabbed `dst` only to silence the borrow-checker — drop it.)
        let _ = dst;

        // Compute outputs.
        // libvpx walks `weights += num_input_nodes` per node; mirror that.
        for node in 0..num_output_nodes {
            let mut val: f32 = 0.0;
            let w_off = node * num_input_nodes;
            for i in 0..num_input_nodes {
                val += weights[w_off + i] * current_input[i];
            }
            val += bias[node];
            // ReLU.
            if val < 0.0 {
                val = 0.0;
            }
            buf[buf_index][node] = val;
        }
        num_input_nodes = num_output_nodes;
        // Rebind the input slice to the freshly-written buffer.
        // Borrow-checker dance: extend the lifetime by copying through a raw
        // index — safe because `buf` lives for the rest of the call.
        current_input = unsafe {
            // SAFETY: We immediately re-borrow a slice of `buf[buf_index]`
            // and only read it during the next iteration; `buf` is on the
            // stack and outlives `current_input`. The previous mutable
            // borrow ended at the loop tail.
            let ptr = buf[buf_index].as_ptr();
            core::slice::from_raw_parts(ptr, num_input_nodes)
        };
        buf_index = 1 - buf_index;
    }

    // Final output layer — linear (no ReLU).
    let weights = nn_config.weights[num_layers];
    let bias = nn_config.bias[num_layers];
    for node in 0..nn_config.num_outputs {
        let mut val: f32 = 0.0;
        let w_off = node * num_input_nodes;
        for i in 0..num_input_nodes {
            val += weights[w_off + i] * current_input[i];
        }
        output[node] = val + bias[node];
    }
}

// ============================================================================
// Linear partition-search breakout weights
// libvpx `vp9_partition_models.h:36-241`
// ============================================================================
// Layout in libvpx: `[RESOLUTION_CTX=2][Q_CTX=3][FEATURES+1=5]`.
// We flatten to `[2][3][5]` Rust arrays to match the C indexing
// `weights[resolution_ctx][q_ctx][...]`.

/// libvpx `vp9_partition_breakout_weights_64` (`vp9_partition_models.h:40-88`).
pub const PARTITION_BREAKOUT_WEIGHTS_64: [[[f32; 5]; 3]; 2] = [
    [
        [-0.016673, -0.001025, -0.000032, 0.000833, 1.94261885 - 2.1],
        [-0.160867, -0.002101, 0.000011, 0.002448, 1.65738142 - 2.5],
        [-0.628934, -0.011459, -0.000009, 0.013833, 1.47982645 - 1.6],
    ],
    [
        [-0.064309, -0.006121, 0.000232, 0.005778, 0.7989465 - 5.0],
        [-0.314957, -0.009346, -0.000225, 0.010072, 2.80695581 - 5.5],
        [-0.635535, -0.015135, 0.000091, 0.015247, 2.90381241 - 5.0],
    ],
];

/// libvpx `vp9_partition_breakout_weights_32` (`vp9_partition_models.h:90-138`).
pub const PARTITION_BREAKOUT_WEIGHTS_32: [[[f32; 5]; 3]; 2] = [
    [
        [-0.010554, -0.003081, -0.000134, 0.004491, 1.68445992 - 3.5],
        [-0.051489, -0.007609, 0.000016, 0.009792, 1.28089404 - 2.5],
        [-0.163097, -0.013081, 0.000022, 0.019006, 1.36129403 - 3.2],
    ],
    [
        [-0.024629, -0.006492, -0.000254, 0.004895, 1.27919173 - 4.5],
        [-0.083936, -0.009827, -0.000200, 0.010399, 2.73731065 - 4.5],
        [-0.279052, -0.013334, 0.000289, 0.023203, 2.43595719 - 3.5],
    ],
];

/// libvpx `vp9_partition_breakout_weights_16` (`vp9_partition_models.h:140-188`).
pub const PARTITION_BREAKOUT_WEIGHTS_16: [[[f32; 5]; 3]; 2] = [
    [
        [-0.013154, -0.002404, -0.000977, 0.008450, 2.57404566 - 5.5],
        [-0.019146, -0.004018, 0.000064, 0.008187, 2.15043926 - 2.5],
        [-0.075755, -0.010858, 0.000030, 0.024505, 2.06848121 - 2.5],
    ],
    [
        [-0.007636, -0.002751, -0.000682, 0.005968, 0.19225763 - 4.5],
        [-0.047306, -0.009113, -0.000518, 0.016007, 2.61068869 - 4.0],
        [-0.069336, -0.010448, -0.001120, 0.023083, 1.47591054 - 5.5],
    ],
];

/// libvpx `vp9_partition_breakout_weights_8` (`vp9_partition_models.h:190-238`).
pub const PARTITION_BREAKOUT_WEIGHTS_8: [[[f32; 5]; 3]; 2] = [
    [
        [-0.011807, -0.009873, -0.000931, 0.034768, 1.32254851 - 2.0],
        [-0.003861, -0.002701, 0.000100, 0.013876, 1.96755111 - 1.5],
        [-0.013522, -0.008677, -0.000562, 0.034468, 1.53440356 - 1.5],
    ],
    [
        [-0.003221, -0.002125, 0.000993, 0.012768, 0.03541421 - 2.0],
        [-0.006069, -0.007335, 0.000229, 0.026104, 0.17135315 - 1.5],
        [-0.039894, -0.011419, 0.000070, 0.061817, 0.6739977 - 1.5],
    ],
];

// ============================================================================
// Rect-partition pruning NN configs (1 hidden layer, 8 features, 4 outputs).
// libvpx `vp9_partition_models.h:243-473`.
// ============================================================================

// ---- rect_part 16 — libvpx `vp9_partition_models.h:247-314` ------------------

const RECT_PART_NN_WEIGHTS_16_LAYER0: [f32; 8 * 16] = [
    -0.432522, 0.133070, -0.169187, 0.768340, 0.891228, 0.554458, 0.356000, 0.403621, 0.809165,
    0.778214, -0.520357, 0.301451, -0.386972, -0.314402, 0.021878, 1.148746, -0.462258, -0.175524,
    -0.344589, -0.475159, -0.232322, 0.471147, -0.489948, 0.467740, -0.391550, 0.208601, 0.054138,
    0.076859, -0.309497, -0.095927, 0.225917, 0.011582, -0.520730, -0.585497, 0.174036, 0.072521,
    0.120771, -0.517234, -0.581908, -0.034003, -0.694722, -0.364368, 0.290584, 0.038373, 0.685654,
    0.394019, 0.759667, 1.257502, -0.610516, -0.185434, 0.211997, -0.172458, 0.044605, 0.145316,
    -0.182525, -0.147376, 0.578742, 0.312412, -0.446135, -0.389112, 0.454033, 0.260490, 0.664285,
    0.395856, -0.231827, 0.215228, 0.014856, -0.395462, 0.479646, -0.391445, -0.357788, 0.166238,
    -0.056818, -0.027783, 0.060880, -1.604710, 0.531268, 0.282184, 0.714944, 0.093523, -0.218312,
    -0.095546, -0.285621, -0.190871, -0.448340, -0.016611, 0.413913, -0.286720, -0.158828,
    -0.092635, -0.279551, 0.166509, -0.088162, 0.446543, -0.276830, -0.065642, -0.176346,
    -0.984754, 0.338738, 0.403809, 0.738065, 1.154439, 0.750764, 0.770959, -0.269403, 0.295651,
    -0.331858, 0.367144, 0.279279, 0.157419, -0.348227, -0.168608, -0.956000, -0.647136, 0.250516,
    0.858084, 0.809802, 0.492408, 0.804841, 0.282802, 0.079395, -0.291771, -0.024382, -1.615880,
    -0.445166, -0.407335, -0.483044, 0.141126,
];

const RECT_PART_NN_BIAS_16_LAYER0: [f32; 16] = [
    0.275384, -0.053745, 0.000000, 0.000000, -0.178103, 0.513965, -0.161352, 0.228551, 0.000000,
    1.013712, 0.000000, 0.000000, -1.144009, -0.000006, -0.241727, 2.048764,
];

const RECT_PART_NN_WEIGHTS_16_LAYER1: [f32; 16 * 4] = [
    -1.435278, 2.204691, -0.410718, 0.202708, 0.109208, 1.059142, -0.306360, 0.845906, 0.489654,
    -1.121915, -0.169133, -0.003385, 0.660590, -0.018711, 1.227158, -2.967504, 1.407345, -1.293243,
    -0.386921, 0.300492, 0.338824, -0.083250, -0.069454, -1.001827, -0.327891, 0.899353, 0.367397,
    -0.118601, -0.171936, -0.420646, -0.803319, 2.029634, 0.940268, -0.664484, 0.339916, 0.315944,
    0.157374, -0.402482, -0.491695, 0.595827, 0.015031, 0.255887, -0.466327, -0.212598, 0.136485,
    0.033363, -0.796921, 1.414304, -0.282185, -2.673571, -0.280994, 0.382658, -0.350902, 0.227926,
    0.062602, -1.000199, 0.433731, 1.176439, -0.163216, -0.229015, -0.640098, -0.438852, -0.947700,
    2.203434,
];

const RECT_PART_NN_BIAS_16_LAYER1: [f32; 4] = [-0.875510, 0.982408, 0.560854, -0.415209];

/// libvpx `vp9_rect_part_nnconfig_16` (`vp9_partition_models.h:299-314`).
pub const RECT_PART_NN_CONFIG_16: NnConfig = NnConfig {
    num_inputs: 8,
    num_outputs: 4,
    num_hidden_layers: 1,
    num_hidden_nodes: [16, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &RECT_PART_NN_WEIGHTS_16_LAYER0,
        &RECT_PART_NN_WEIGHTS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &RECT_PART_NN_BIAS_16_LAYER0,
        &RECT_PART_NN_BIAS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- rect_part 32 — libvpx `vp9_partition_models.h:316-383` ------------------

const RECT_PART_NN_WEIGHTS_32_LAYER0: [f32; 8 * 16] = [
    -0.147312, -0.753248, 0.540206, 0.661415, 0.484117, -0.341609, 0.016183, 0.064177, 0.781580,
    0.902232, -0.505342, 0.325183, -0.231072, -0.120107, -0.076216, 0.120038, 0.403695, -0.463301,
    -0.192158, 0.407442, 0.106633, 1.072371, -0.446779, 0.467353, 0.318812, -0.505996, -0.008768,
    -0.239598, 0.085480, 0.284640, -0.365045, -0.048083, -0.112090, -0.067089, 0.304138, -0.228809,
    0.383651, -0.196882, 0.477039, -0.217978, -0.506931, -0.125675, 0.050456, 1.086598, 0.732128,
    0.326941, 0.103952, 0.121769, -0.154487, -0.255514, 0.030591, -0.382797, -0.019981, -0.326570,
    0.149691, -0.435633, -0.070795, 0.167691, 0.251413, -0.153405, 0.160347, 0.455107, -0.968580,
    -0.575879, 0.623115, -0.069793, -0.379768, -0.965807, -0.062057, 0.071312, 0.457098, 0.350372,
    -0.460659, -0.985393, 0.359963, -0.093677, 0.404272, -0.326896, -0.277752, 0.609322, -0.114193,
    -0.230701, 0.089208, 0.645381, 0.494485, 0.467876, -0.166187, 0.251044, -0.394661, 0.192895,
    -0.344777, -0.041893, -0.111163, 0.066347, 0.378158, -0.455465, 0.339839, -0.418207, -0.356515,
    -0.227536, -0.211091, -0.122945, 0.361772, -0.338095, 0.004564, -0.398510, 0.060876, -2.132504,
    -0.086776, -0.029166, 0.039241, 0.222534, -0.188565, -0.288792, -0.160789, -0.123905, 0.397916,
    -0.063779, 0.167210, -0.445004, 0.056889, 0.207280, 0.000101, 0.384507, -1.721239, -2.036402,
    -2.084403, -2.060483,
];

const RECT_PART_NN_BIAS_32_LAYER0: [f32; 16] = [
    -0.859251, -0.109938, 0.091838, 0.187817, -0.728265, 0.253080, 0.000000, -0.357195, -0.031290,
    -1.373237, -0.761086, 0.000000, -0.024504, 1.765711, 0.000000, 1.505390,
];

const RECT_PART_NN_WEIGHTS_32_LAYER1: [f32; 16 * 4] = [
    0.680940, 1.367178, 0.403075, 0.029957, 0.500917, 1.407776, -0.354002, 0.011667, 1.663767,
    0.959155, 0.428323, -0.205345, -0.081850, -3.920103, -0.243802, -4.253933, -0.034020, -1.361057,
    0.128236, -0.138422, -0.025790, -0.563518, -0.148715, -0.344381, -1.677389, -0.868332,
    -0.063792, 0.052052, 0.359591, 2.739808, -0.414304, 3.036597, -0.075368, -1.019680, 0.642501,
    0.209779, -0.374539, -0.718294, -0.116616, -0.043212, -1.787809, -0.773262, 0.068734, 0.508309,
    0.099334, 1.802239, -0.333538, 2.708645, -0.447682, -2.355555, -0.506674, -0.061028, -0.310305,
    -0.375475, 0.194572, 0.431788, -0.789624, -0.031962, 0.358353, 0.382937, 0.232002, 2.321813,
    -0.037523, 2.104652,
];

const RECT_PART_NN_BIAS_32_LAYER1: [f32; 4] = [-0.693383, 0.773661, 0.426878, -0.070619];

/// libvpx `vp9_rect_part_nnconfig_32` (`vp9_partition_models.h:368-383`).
pub const RECT_PART_NN_CONFIG_32: NnConfig = NnConfig {
    num_inputs: 8,
    num_outputs: 4,
    num_hidden_layers: 1,
    num_hidden_nodes: [16, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &RECT_PART_NN_WEIGHTS_32_LAYER0,
        &RECT_PART_NN_WEIGHTS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &RECT_PART_NN_BIAS_32_LAYER0,
        &RECT_PART_NN_BIAS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- rect_part 64 — libvpx `vp9_partition_models.h:386-470` (NODES=24) -------

const RECT_PART_NN_WEIGHTS_64_LAYER0: [f32; 8 * 24] = [
    0.024671, -0.220610, -0.284362, -0.069556, -0.315700, 0.187861, 0.139782, 0.063110, 0.796561,
    0.172868, -0.662194, -1.393074, 0.085003, 0.393381, 0.358477, -0.187268, -0.370745, 0.218287,
    0.027271, -0.254089, -0.048236, -0.459137, 0.253171, 0.122598, -0.550107, -0.568456, 0.159866,
    -0.246534, 0.096384, -0.255460, 0.077864, -0.334837, 0.026921, -0.697252, 0.345262, 1.343578,
    0.815984, 1.118211, 1.574016, 0.578476, -0.285967, -0.508672, 0.118137, 0.037695, 1.540510,
    1.256648, 1.163819, 1.172027, 0.661551, -0.111980, -0.434204, -0.894217, 0.570524, 0.050292,
    -0.113680, 0.000784, -0.211554, -0.369394, 0.158306, -0.512505, -0.238696, 0.091498, -0.448490,
    -0.491268, -0.353112, -0.303315, -0.428438, 0.127998, -0.406790, -0.401786, -0.279888,
    -0.384223, 0.026100, 0.041621, -0.315818, -0.087888, 0.353497, 0.163123, -0.380128, -0.090334,
    -0.216647, -0.117849, -0.173502, 0.301871, 0.070854, 0.114627, -0.050545, -0.160381, 0.595294,
    0.492696, -0.453858, -1.154139, 0.126000, 0.034550, 0.456665, -0.236618, -0.112640, 0.050759,
    -0.449162, 0.110059, 0.147116, 0.249358, -0.049894, 0.063351, -0.004467, 0.057242, -0.482015,
    -0.174335, -0.085617, -0.333808, -0.358440, -0.069006, 0.099260, -1.243430, -0.052963, 0.112088,
    -2.661115, -2.445893, -2.688174, -2.624232, 0.030494, 0.161311, 0.012136, 0.207564, -2.776856,
    -2.791940, -2.623962, -2.918820, 1.231619, -0.376692, -0.698078, 0.110336, -0.285378, 0.258367,
    -0.180159, -0.376608, -0.034348, -0.130206, 0.160020, 0.852977, 0.580573, 1.450782, 1.357596,
    0.787382, -0.544004, -0.014795, 0.032121, -0.557696, 0.159994, -0.540908, 0.180380, -0.398045,
    0.705095, 0.515103, -0.511521, -1.271374, -0.231019, 0.423647, 0.064907, -0.255338, -0.877748,
    -0.667205, 0.267847, 0.135229, 0.617844, 1.349849, 1.012623, 0.730506, -0.078571, 0.058401,
    0.053221, -2.426146, -0.098808, -0.138508, -0.153299, 0.149116, -0.444243, 0.301807, 0.065066,
    0.092929, -0.372784, -0.095540, 0.192269, 0.237894, 0.080228, -0.214074, -0.011426, -2.352367,
    -0.085394, -0.190361, -0.001177, 0.089197,
];

const RECT_PART_NN_BIAS_64_LAYER0: [f32; 24] = [
    0.000000, -0.057652, -0.175413, -0.175389, -1.084097, -1.423801, -0.076307, -0.193803, 0.000000,
    -0.066474, -0.050318, -0.019832, -0.038814, -0.144184, 2.652451, 2.415006, 0.197464, -0.729842,
    -0.173774, 0.239171, 0.486425, 2.463304, -0.175279, 2.352637,
];

const RECT_PART_NN_WEIGHTS_64_LAYER1: [f32; 24 * 4] = [
    -0.063237, 1.925696, -0.182145, -0.226687, 0.602941, -0.941140, 0.814598, -0.117063, 0.282988,
    0.066369, 0.096951, 1.049735, -0.188188, -0.281227, -4.836746, -5.047797, 0.892358, 0.417145,
    -0.279849, 1.335945, 0.660338, -2.757938, -0.115714, -1.862183, -0.045980, -1.597624, -0.586822,
    -0.615589, -0.330537, 1.068496, -0.167290, 0.141290, -0.112100, 0.232761, 0.252307, -0.399653,
    0.353118, 0.241583, 2.635241, 4.026119, -1.137327, -0.052446, -0.139814, -1.104256, -0.759391,
    2.508457, -0.526297, 2.095348, -0.444473, -1.090452, 0.584122, 0.468729, -0.368865, 1.041425,
    -1.079504, 0.348837, 0.390091, 0.416191, 0.212906, -0.660255, 0.053630, 0.209476, 3.595525,
    2.257293, -0.514030, 0.074203, -0.375862, -1.998307, -0.930310, 1.866686, -0.247137, 1.087789,
    0.100186, 0.298150, 0.165265, 0.050478, 0.249167, 0.371789, -0.294497, 0.202954, 0.037310,
    0.193159, 0.161551, 0.301597, 0.299286, 0.185946, 0.822976, 2.066130, -1.724588, 0.055977,
    -0.330747, -0.067747, -0.475801, 1.555958, -0.025808, -0.081516,
];

const RECT_PART_NN_BIAS_64_LAYER1: [f32; 4] = [-0.090723, 0.894968, 0.844754, -3.496194];

/// libvpx `vp9_rect_part_nnconfig_64` (`vp9_partition_models.h:455-470`).
pub const RECT_PART_NN_CONFIG_64: NnConfig = NnConfig {
    num_inputs: 8,
    num_outputs: 4,
    num_hidden_layers: 1,
    num_hidden_nodes: [24, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &RECT_PART_NN_WEIGHTS_64_LAYER0,
        &RECT_PART_NN_WEIGHTS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &RECT_PART_NN_BIAS_64_LAYER0,
        &RECT_PART_NN_BIAS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ============================================================================
// Partition-pruning NN configs (7 features, 1 output, 1 hidden layer w/ 8 nodes).
// libvpx `vp9_partition_models.h:475-607`.
// ============================================================================

// ---- partition 64x64 — libvpx `vp9_partition_models.h:477-519` ---------------

const PARTITION_NN_WEIGHTS_64X64_LAYER0: [f32; 7 * 8] = [
    -3.571348, 0.014835, -3.255393, -0.098090, -0.013120, 0.000221, 0.056273, 0.190179, -0.268130,
    -1.828242, -0.010655, 0.937244, -0.435120, 0.512125, 1.610679, 0.190816, -0.799075, -0.377348,
    -0.144232, 0.614383, -0.980388, 1.754150, -0.185603, -0.061854, -0.807172, 1.240177, 1.419531,
    -0.438544, -5.980774, 0.139045, -0.032359, -0.068887, -1.237918, 0.115706, 0.003164, 2.924212,
    1.246838, -0.035833, 0.810011, -0.805894, 0.010966, 0.076463, -4.226380, -2.437764, -0.010619,
    -0.020935, -0.451494, 0.300079, -0.168961, -3.326450, -2.731094, 0.002518, 0.018840, -1.656815,
    0.068039, 0.010586,
];

const PARTITION_NN_BIAS_64X64_LAYER0: [f32; 8] = [
    -3.469882, 0.683989, 0.194010, 0.313782, -3.153335, 2.245849, -1.946190, -3.740020,
];

const PARTITION_NN_WEIGHTS_64X64_LAYER1: [f32; 8] = [
    -8.058566, 0.108306, -0.280620, -0.818823, -6.445117, 0.865364, -1.127127, -8.808660,
];

const PARTITION_NN_BIAS_64X64_LAYER1: [f32; 1] = [6.46909416];

/// libvpx `vp9_partition_nnconfig_64x64` (`vp9_partition_models.h:504-519`).
pub const PARTITION_NN_CONFIG_64X64: NnConfig = NnConfig {
    num_inputs: 7,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PARTITION_NN_WEIGHTS_64X64_LAYER0,
        &PARTITION_NN_WEIGHTS_64X64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PARTITION_NN_BIAS_64X64_LAYER0,
        &PARTITION_NN_BIAS_64X64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- partition 32x32 — libvpx `vp9_partition_models.h:521-563` ---------------

const PARTITION_NN_WEIGHTS_32X32_LAYER0: [f32; 7 * 8] = [
    -0.295437, -4.002648, -0.205399, -0.060919, 0.708037, 0.027221, -0.039137, -0.907724,
    -3.151662, 0.007106, 0.018726, -0.534928, 0.022744, 0.000159, -1.717189, -3.229031, -0.027311,
    0.269863, -0.400747, -0.394366, -0.108878, 0.603027, 0.455369, -0.197170, 1.241746, -1.347820,
    -0.575636, -0.462879, -2.296426, 0.196696, -0.138347, -0.030754, -0.200774, 0.453795, 0.055625,
    -3.163116, -0.091003, -0.027028, -0.042984, -0.605185, 0.143240, -0.036439, -0.801228, 0.313409,
    -0.159942, 0.031267, 0.886454, -1.531644, -0.089655, 0.037683, -0.163441, -0.130454, -0.058344,
    0.060011, 0.275387, 1.552226,
];

const PARTITION_NN_BIAS_32X32_LAYER0: [f32; 8] = [
    -0.838372, -2.609089, -0.055763, 1.329485, -1.297638, -2.636622, -0.826909, 1.012644,
];

const PARTITION_NN_WEIGHTS_32X32_LAYER1: [f32; 8] = [
    -1.792632, -7.322353, -0.683386, 0.676564, -1.488118, -7.527719, 1.240163, 0.614309,
];

const PARTITION_NN_BIAS_32X32_LAYER1: [f32; 1] = [4.97422546];

/// libvpx `vp9_partition_nnconfig_32x32` (`vp9_partition_models.h:548-563`).
pub const PARTITION_NN_CONFIG_32X32: NnConfig = NnConfig {
    num_inputs: 7,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PARTITION_NN_WEIGHTS_32X32_LAYER0,
        &PARTITION_NN_WEIGHTS_32X32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PARTITION_NN_BIAS_32X32_LAYER0,
        &PARTITION_NN_BIAS_32X32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- partition 16x16 — libvpx `vp9_partition_models.h:565-607` ---------------

const PARTITION_NN_WEIGHTS_16X16_LAYER0: [f32; 7 * 8] = [
    -1.717673, -4.718130, -0.125725, -0.183427, -0.511764, 0.035328, 0.130891, -3.096753, 0.174968,
    -0.188769, -0.640796, 1.305661, 1.700638, -0.073806, -4.006781, -1.630999, -0.064863, -0.086410,
    -0.148617, 0.172733, -0.018619, 2.152595, 0.778405, -0.156455, 0.612995, -0.467878, 0.152022,
    -0.236183, 0.339635, -0.087119, -3.196610, -1.080401, -0.637704, -0.059974, 1.706298, -0.793705,
    -6.399260, 0.010624, -0.064199, -0.650621, 0.338087, -0.001531, 1.023655, -3.700272, -0.055281,
    -0.386884, 0.375504, -0.898678, 0.281156, -0.314611, 0.863354, -0.040582, -0.145019, 0.029329,
    -2.197880, -0.108733,
];

const PARTITION_NN_BIAS_16X16_LAYER0: [f32; 8] = [
    0.411516, -2.143737, -3.693192, 2.123142, -1.356910, -3.561016, -0.765045, -2.417082,
];

const PARTITION_NN_WEIGHTS_16X16_LAYER1: [f32; 8] = [
    -0.619755, -2.202391, -4.337171, 0.611319, 0.377677, -4.998723, -1.052235, 1.949922,
];

const PARTITION_NN_BIAS_16X16_LAYER1: [f32; 1] = [3.20981717];

/// libvpx `vp9_partition_nnconfig_16x16` (`vp9_partition_models.h:592-607`).
pub const PARTITION_NN_CONFIG_16X16: NnConfig = NnConfig {
    num_inputs: 7,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PARTITION_NN_WEIGHTS_16X16_LAYER0,
        &PARTITION_NN_WEIGHTS_16X16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PARTITION_NN_BIAS_16X16_LAYER0,
        &PARTITION_NN_BIAS_16X16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ============================================================================
// Var-part NN configs (6 features, 1 output, 1 hidden layer w/ 8 nodes).
// libvpx `vp9_partition_models.h:610-735`.
// ============================================================================

// ---- var_part 64 — libvpx `vp9_partition_models.h:611-651` -------------------

const VAR_PART_NN_WEIGHTS_64_LAYER0: [f32; 6 * 8] = [
    -0.249572, 0.205532, -2.175608, 1.094836, -2.986370, 0.193160, -0.143823, 0.378511, -1.997788,
    -2.166866, -1.930158, -1.202127, -0.611875, -0.506422, -0.432487, 0.071205, 0.578172, -0.154285,
    -0.051830, 0.331681, -1.457177, -2.443546, -2.000302, -1.389283, 0.372084, -0.464917, 2.265235,
    2.385787, 2.312722, 2.127868, -0.403963, -0.177860, -0.436751, -0.560539, 0.254903, 0.193976,
    -0.305611, 0.256632, 0.309388, -0.437439, 1.702640, -5.007069, -0.323450, 0.294227, 1.267193,
    1.056601, 0.387181, -0.191215,
];

const VAR_PART_NN_BIAS_64_LAYER0: [f32; 8] = [
    -0.044396, -0.938166, 0.000000, -0.916375, 1.242299, 0.000000, -0.405734, 0.014206,
];

const VAR_PART_NN_WEIGHTS_64_LAYER1: [f32; 8] = [
    1.635945, 0.979557, 0.455315, 1.197199, -2.251024, -0.464953, 1.378676, -0.111927,
];

const VAR_PART_NN_BIAS_64_LAYER1: [f32; 1] = [-0.37972447];

/// libvpx `vp9_var_part_nnconfig_64` (`vp9_partition_models.h:636-651`).
pub const VAR_PART_NN_CONFIG_64: NnConfig = NnConfig {
    num_inputs: 6,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &VAR_PART_NN_WEIGHTS_64_LAYER0,
        &VAR_PART_NN_WEIGHTS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &VAR_PART_NN_BIAS_64_LAYER0,
        &VAR_PART_NN_BIAS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- var_part 32 — libvpx `vp9_partition_models.h:653-693` -------------------

const VAR_PART_NN_WEIGHTS_32_LAYER0: [f32; 6 * 8] = [
    0.067243, -0.083598, -2.191159, 2.726434, -3.324013, 3.477977, 0.323736, -0.510199, 2.960693,
    2.937661, 2.888476, 2.938315, -0.307602, -0.503353, -0.080725, -0.473909, -0.417162, 0.457089,
    0.665153, -0.273210, 0.028279, 0.972220, -0.445596, 1.756611, -0.177892, -0.091758, 0.436661,
    -0.521506, 0.133786, 0.266743, 0.637367, -0.160084, -1.396269, 1.020841, -1.112971, 0.919496,
    -0.235883, 0.651954, 0.109061, -0.429463, 0.740839, -0.962060, 0.299519, -0.386298, 1.550231,
    2.464915, 1.311969, 2.561612,
];

const VAR_PART_NN_BIAS_32_LAYER0: [f32; 8] = [
    0.368242, 0.736617, 0.000000, 0.757287, 0.000000, 0.613248, -0.776390, 0.928497,
];

const VAR_PART_NN_WEIGHTS_32_LAYER1: [f32; 8] = [
    0.939884, -2.420850, -0.410489, -0.186690, 0.063287, -0.522011, 0.484527, -0.639625,
];

const VAR_PART_NN_BIAS_32_LAYER1: [f32; 1] = [-0.6455006];

/// libvpx `vp9_var_part_nnconfig_32` (`vp9_partition_models.h:678-693`).
pub const VAR_PART_NN_CONFIG_32: NnConfig = NnConfig {
    num_inputs: 6,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &VAR_PART_NN_WEIGHTS_32_LAYER0,
        &VAR_PART_NN_WEIGHTS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &VAR_PART_NN_BIAS_32_LAYER0,
        &VAR_PART_NN_BIAS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- var_part 16 — libvpx `vp9_partition_models.h:695-735` -------------------

const VAR_PART_NN_WEIGHTS_16_LAYER0: [f32; 6 * 8] = [
    0.742567, -0.580624, -0.244528, 0.331661, -0.113949, -0.559295, -0.386061, 0.438653, 1.467463,
    0.211589, 0.513972, 1.067855, -0.876679, 0.088560, -0.687483, -0.380304, -0.016412, 0.146380,
    0.015318, 0.000351, -2.764887, 3.269717, 2.752428, -2.236754, 0.561539, -0.852050, -0.084667,
    0.202057, 0.197049, 0.364922, -0.463801, 0.431790, 1.872096, -0.091887, -0.055034, 2.443492,
    -0.156958, -0.189571, -0.542424, -0.589804, -0.354422, 0.401605, 0.642021, -0.875117, 2.040794,
    1.921070, 1.792413, 1.839727,
];

const VAR_PART_NN_BIAS_16_LAYER0: [f32; 8] = [
    2.901234, -1.940932, -0.198970, -0.406524, 0.059422, -1.879207, -0.232340, 2.979821,
];

const VAR_PART_NN_WEIGHTS_16_LAYER1: [f32; 8] = [
    -0.528731, 0.375234, -0.088422, 0.668629, 0.870449, 0.578735, 0.546103, -1.957207,
];

const VAR_PART_NN_BIAS_16_LAYER1: [f32; 1] = [-1.95769405];

/// libvpx `vp9_var_part_nnconfig_16` (`vp9_partition_models.h:720-735`).
pub const VAR_PART_NN_CONFIG_16: NnConfig = NnConfig {
    num_inputs: 6,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &VAR_PART_NN_WEIGHTS_16_LAYER0,
        &VAR_PART_NN_WEIGHTS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &VAR_PART_NN_BIAS_16_LAYER0,
        &VAR_PART_NN_BIAS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ============================================================================
// Part-split NN configs (12 features, 1 output, 1 hidden layer w/ 8 nodes).
// libvpx `vp9_partition_models.h:738-939`.
// ============================================================================

// ---- part_split 64 — libvpx `vp9_partition_models.h:741-789` -----------------

const PART_SPLIT_NN_WEIGHTS_64_LAYER0: [f32; 12 * 8] = [
    -0.609728, -0.409099, -0.472449, 0.183769, -0.457740, 0.081089, 0.171003, 0.578696, -0.019043,
    -0.856142, 0.557369, -1.779424, -0.274044, -0.320632, -0.392531, -0.359462, -0.404106,
    -0.288357, 0.200620, 0.038013, -0.430093, 0.235083, -0.487442, 0.424814, -0.232758, -0.442943,
    0.229397, -0.540301, -0.648421, -0.649747, -0.171638, 0.603824, 0.468497, -0.421580, 0.178840,
    -0.533838, -0.029471, -0.076296, 0.197426, -0.187908, -0.003950, -0.065740, 0.085165, -0.039674,
    -5.640702, 1.909538, -1.434604, 3.294606, -0.788812, 0.196864, 0.057012, -0.019757, 0.336233,
    0.075378, 0.081503, 0.491864, -1.899470, -1.764173, -1.888137, -1.762343, 0.845542, 0.202285,
    0.381948, -0.150996, 0.556893, -0.305354, 0.561482, -0.021974, -0.703117, 0.268638, -0.665736,
    1.191005, -0.081568, -0.115653, 0.272029, -0.140074, 0.072683, 0.092651, -0.472287, -0.055790,
    -0.434425, 0.352055, 0.048246, 0.372865, 0.111499, -0.338304, 0.739133, 0.156519, -0.594644,
    0.137295, 0.613350, -0.165102, -1.003731, 0.043070, -0.887896, -0.174202,
];

const PART_SPLIT_NN_BIAS_64_LAYER0: [f32; 8] = [
    1.182714, 0.000000, 0.902019, 0.953115, -1.372486, -1.288740, -0.155144, -3.041362,
];

const PART_SPLIT_NN_WEIGHTS_64_LAYER1: [f32; 8] = [
    0.841214, 0.456016, 0.869270, 1.692999, -1.700494, -0.911761, 0.030111, -1.447548,
];

const PART_SPLIT_NN_BIAS_64_LAYER1: [f32; 1] = [1.17782545];

/// libvpx `vp9_part_split_nnconfig_64` (`vp9_partition_models.h:774-789`).
pub const PART_SPLIT_NN_CONFIG_64: NnConfig = NnConfig {
    num_inputs: 12,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PART_SPLIT_NN_WEIGHTS_64_LAYER0,
        &PART_SPLIT_NN_WEIGHTS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PART_SPLIT_NN_BIAS_64_LAYER0,
        &PART_SPLIT_NN_BIAS_64_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- part_split 32 — libvpx `vp9_partition_models.h:791-839` -----------------

const PART_SPLIT_NN_WEIGHTS_32_LAYER0: [f32; 12 * 8] = [
    -0.105488, -0.218662, 0.010980, -0.226979, 0.028076, 0.743430, 0.789266, 0.031907, -1.464200,
    0.222336, -1.068493, -0.052712, -0.176181, -0.102654, -0.973932, -0.182637, -0.198000, 0.335977,
    0.271346, 0.133005, 1.674203, 0.689567, 0.657133, 0.283524, 0.115529, 0.738327, 0.317184,
    -0.179736, 0.403691, 0.679350, 0.048925, 0.271338, -1.538921, -0.900737, -1.377845, 0.084245,
    0.803122, -0.107806, 0.103045, -0.023335, -0.098116, -0.127809, 0.037665, -0.523225, 1.622185,
    1.903999, 1.358889, 1.680785, 0.027743, 0.117906, -0.158810, 0.057775, 0.168257, 0.062414,
    0.086228, -0.087381, -3.066082, 3.021855, -4.092155, 2.550104, -0.230022, -0.207445, -0.000347,
    0.034042, 0.097057, 0.220088, -0.228841, -0.029405, -1.507174, -1.455184, 2.624904, 2.643355,
    0.319912, 0.585531, -1.018225, -0.699606, 1.026490, 0.169952, -0.093579, -0.142352, -0.107256,
    0.059598, 0.043190, 0.507543, -0.138617, 0.030197, 0.059574, -0.634051, -0.586724, -0.148020,
    -0.334380, 0.459547, 1.620600, 0.496850, 0.639480, -0.465715,
];

const PART_SPLIT_NN_BIAS_32_LAYER0: [f32; 8] = [
    -1.125885, 0.753197, -0.825808, 0.004839, 0.583920, 0.718062, 0.976741, 0.796188,
];

const PART_SPLIT_NN_WEIGHTS_32_LAYER1: [f32; 8] = [
    -0.458745, 0.724624, -0.479720, -2.199872, 1.162661, 1.194153, -0.716896, 0.824080,
];

const PART_SPLIT_NN_BIAS_32_LAYER1: [f32; 1] = [0.71644074];

/// libvpx `vp9_part_split_nnconfig_32` (`vp9_partition_models.h:824-839`).
pub const PART_SPLIT_NN_CONFIG_32: NnConfig = NnConfig {
    num_inputs: 12,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PART_SPLIT_NN_WEIGHTS_32_LAYER0,
        &PART_SPLIT_NN_WEIGHTS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PART_SPLIT_NN_BIAS_32_LAYER0,
        &PART_SPLIT_NN_BIAS_32_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- part_split 16 — libvpx `vp9_partition_models.h:841-889` -----------------

const PART_SPLIT_NN_WEIGHTS_16_LAYER0: [f32; 12 * 8] = [
    -0.003629, -0.046852, 0.220428, -0.033042, 0.049365, 0.112818, -0.306149, -0.005872, 1.066947,
    -2.290226, 2.159505, -0.618714, -0.213294, 0.451372, -0.199459, 0.223730, -0.321709, 0.063364,
    0.148704, -0.293371, 0.077225, -0.421947, -0.515543, -0.240975, -0.418516, 1.036523, -0.009165,
    0.032484, 1.086549, 0.220322, -0.247585, -0.221232, -0.225050, 0.993051, 0.285907, 1.308846,
    0.707456, 0.335152, 0.234556, 0.264590, -0.078033, 0.542226, 0.057777, 0.163471, 0.039245,
    -0.725960, 0.963780, -0.972001, 0.252237, -0.192745, -0.836571, -0.460539, -0.528713, -0.160198,
    -0.621108, 0.486405, -0.221923, 1.519426, -0.857871, 0.411595, 0.947188, 0.203339, 0.174526,
    0.016382, 0.256879, 0.049818, 0.057836, -0.659096, 0.459894, 0.174695, 0.379359, 0.062530,
    -0.210201, -0.355788, -0.208432, -0.401723, -0.115373, 0.191336, -0.109342, 0.002455, -0.078746,
    -0.391871, 0.149892, -0.239615, -0.520709, 0.118568, -0.437975, 0.118116, -0.565426, -0.206446,
    0.113407, 0.558894, 0.534627, 1.154350, -0.116833, 1.723311,
];

const PART_SPLIT_NN_BIAS_16_LAYER0: [f32; 8] = [
    0.013109, -0.034341, 0.679845, -0.035781, -0.104183, 0.098055, -0.041130, 0.160107,
];

const PART_SPLIT_NN_WEIGHTS_16_LAYER1: [f32; 8] = [
    1.499564, -0.403259, 1.366532, -0.469868, 0.482227, -2.076697, 0.527691, 0.540495,
];

const PART_SPLIT_NN_BIAS_16_LAYER1: [f32; 1] = [0.01134653];

/// libvpx `vp9_part_split_nnconfig_16` (`vp9_partition_models.h:874-889`).
pub const PART_SPLIT_NN_CONFIG_16: NnConfig = NnConfig {
    num_inputs: 12,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PART_SPLIT_NN_WEIGHTS_16_LAYER0,
        &PART_SPLIT_NN_WEIGHTS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PART_SPLIT_NN_BIAS_16_LAYER0,
        &PART_SPLIT_NN_BIAS_16_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ---- part_split 8 — libvpx `vp9_partition_models.h:891-939` ------------------

const PART_SPLIT_NN_WEIGHTS_8_LAYER0: [f32; 12 * 8] = [
    -0.668875, -0.159078, -0.062663, -0.483785, -0.146814, -0.608975, -0.589145, 0.203704,
    -0.051007, -0.113769, -0.477511, -0.122603, -1.329890, 1.403386, 0.199636, -0.161139, 2.182090,
    -0.014307, 0.015755, -0.208468, 0.884353, 0.815920, 0.632464, 0.838225, 1.369483, -0.029068,
    0.570213, -0.573546, 0.029617, 0.562054, -0.653093, -0.211910, -0.661013, -0.384418, -0.574038,
    -0.510069, 0.173047, -0.274231, -1.044008, -0.422040, -0.810296, 0.144069, -0.406704, 0.411230,
    -0.144023, 0.745651, -0.595091, 0.111787, 0.840651, 0.030123, -0.242155, 0.101486, -0.017889,
    -0.254467, -0.285407, -0.076675, -0.549542, -0.013544, -0.686566, -0.755150, 1.623949,
    -0.286369, 0.170976, 0.016442, -0.598353, -0.038540, 0.202597, -0.933582, 0.599510, 0.362273,
    0.577722, 0.477603, 0.767097, 0.431532, 0.457034, 0.223279, 0.381349, 0.033777, 0.423923,
    -0.664762, 0.385662, 0.075744, 0.182681, 0.024118, 0.319408, -0.528864, 0.976537, -0.305971,
    -0.189380, -0.241689, -1.318092, 0.088647, -0.109030, -0.945654, 1.082797, 0.184564,
];

const PART_SPLIT_NN_BIAS_8_LAYER0: [f32; 8] = [
    -0.237472, 2.051396, 0.297062, -0.730194, 0.060472, -0.565959, 0.560869, -0.395448,
];

const PART_SPLIT_NN_WEIGHTS_8_LAYER1: [f32; 8] = [
    0.568121, 1.575915, -0.544309, 0.751595, -0.117911, -1.340730, -0.739671, 0.661216,
];

const PART_SPLIT_NN_BIAS_8_LAYER1: [f32; 1] = [-0.63375306];

/// libvpx `vp9_part_split_nnconfig_8` (`vp9_partition_models.h:924-939`).
pub const PART_SPLIT_NN_CONFIG_8: NnConfig = NnConfig {
    num_inputs: 12,
    num_outputs: 1,
    num_hidden_layers: 1,
    num_hidden_nodes: [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    weights: [
        &PART_SPLIT_NN_WEIGHTS_8_LAYER0,
        &PART_SPLIT_NN_WEIGHTS_8_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
    bias: [
        &PART_SPLIT_NN_BIAS_8_LAYER0,
        &PART_SPLIT_NN_BIAS_8_LAYER1,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    ],
};

// ============================================================================
// Linear partition-pruning tables (feature normalisation + linear weights).
// libvpx `vp9_partition_models.h:944-969`.
// ============================================================================

/// libvpx `vp9_partition_feature_mean[24]` (`vp9_partition_models.h:945-952`).
/// 3 block sizes × 8 entries each (matches partition pruner offsets
/// 0 / 8 / 16 for 64×64 / 32×32 / 16×16 — see W7.6 `ml_pruning_partition`).
pub const PARTITION_FEATURE_MEAN: [f32; 24] = [
    303501.697372, 3042630.372158, 24.694696, 1.392182, 689.413511, 162.027012, 1.478213, 0.0,
    135382.260230, 912738.513263, 28.845217, 1.515230, 544.158492, 131.807995, 1.436863, 0.0,
    43682.377587, 208131.711766, 28.084737, 1.356677, 138.254122, 119.522553, 1.252322, 0.0,
];

/// libvpx `vp9_partition_feature_std[24]` (`vp9_partition_models.h:954-961`).
pub const PARTITION_FEATURE_STD: [f32; 24] = [
    673689.212982, 5996652.516628, 0.024449, 1.989792, 985.880847, 0.014638, 2.001898, 0.0,
    208798.775332, 1812548.443284, 0.018693, 1.838009, 396.986910, 0.015657, 1.332541, 0.0,
    55888.847031, 448587.962714, 0.017900, 1.904776, 98.652832, 0.016598, 1.320992, 0.0,
];

/// libvpx `vp9_partition_linear_weights[24]` (`vp9_partition_models.h:963-969`).
/// Each block-size lane is 8 entries (7 feature weights + bias).
pub const PARTITION_LINEAR_WEIGHTS: [f32; 24] = [
    0.111736, 0.289977, 0.042219, 0.204765, 0.120410, -0.143863, 0.282376, 0.847811, 0.637161,
    0.131570, 0.018636, 0.202134, 0.112797, 0.028162, 0.182450, 1.124367, 0.386133, 0.083700,
    0.050028, 0.150873, 0.061119, 0.109318, 0.127255, 0.625211,
];

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// W7.5 — `nn_predict` runtime smoke. Feeds an all-zero feature
    /// vector through `PARTITION_NN_CONFIG_64X64`. With zero features
    /// the hidden ReLU outputs reduce to `max(0, bias[node])`, and the
    /// output is a deterministic linear combination — checked finite +
    /// matches a hand-computed expectation.
    ///
    /// libvpx reference: `vp9_encodeframe.c:3111-3153` (`nn_predict`).
    #[test]
    fn w7_5_nn_predict_runtime_matches_libvpx_reference_inputs() {
        let features = [0.0f32; 7];
        let mut output = [0.0f32; 1];
        nn_predict(&features, &PARTITION_NN_CONFIG_64X64, &mut output);

        // With zero inputs, hidden layer outputs = max(0, bias).
        // Then output = sum(w1_i * relu(bias0_i)) + bias1[0].
        let mut expected = PARTITION_NN_BIAS_64X64_LAYER1[0];
        for i in 0..8 {
            let h = PARTITION_NN_BIAS_64X64_LAYER0[i].max(0.0);
            expected += PARTITION_NN_WEIGHTS_64X64_LAYER1[i] * h;
        }
        assert!(
            (output[0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            output[0]
        );
        assert!(output[0].is_finite());
    }

    /// W7.5 — Each of the 13 NN configs has the libvpx-cited dimensions.
    ///
    /// libvpx reference: `vp9_partition_models.h:243-939`.
    #[test]
    fn w7_5_nn_partition_config_tables_have_expected_dimensions() {
        // Helper checks: weight slice length matches num_inputs * num_hidden_nodes[0],
        // and layer1 weights match num_hidden_nodes[0] * num_outputs.
        let check = |cfg: &NnConfig, name: &str| {
            assert_eq!(cfg.num_hidden_layers, 1, "{name}: hidden layers");
            let h = cfg.num_hidden_nodes[0];
            assert_eq!(
                cfg.weights[0].len(),
                cfg.num_inputs * h,
                "{name}: layer0 weight size",
            );
            assert_eq!(cfg.bias[0].len(), h, "{name}: layer0 bias size");
            assert_eq!(
                cfg.weights[1].len(),
                h * cfg.num_outputs,
                "{name}: layer1 weight size",
            );
            assert_eq!(cfg.bias[1].len(), cfg.num_outputs, "{name}: layer1 bias size");
        };

        // 3 rect_part configs (8 features → 4 outputs).
        for (cfg, name, hidden) in [
            (&RECT_PART_NN_CONFIG_16, "rect_part_16", 16),
            (&RECT_PART_NN_CONFIG_32, "rect_part_32", 16),
            (&RECT_PART_NN_CONFIG_64, "rect_part_64", 24),
        ] {
            assert_eq!(cfg.num_inputs, 8);
            assert_eq!(cfg.num_outputs, 4);
            assert_eq!(cfg.num_hidden_nodes[0], hidden);
            check(cfg, name);
        }

        // 3 partition_nn configs (7 → 1, hidden=8).
        for (cfg, name) in [
            (&PARTITION_NN_CONFIG_64X64, "partition_64x64"),
            (&PARTITION_NN_CONFIG_32X32, "partition_32x32"),
            (&PARTITION_NN_CONFIG_16X16, "partition_16x16"),
        ] {
            assert_eq!(cfg.num_inputs, 7);
            assert_eq!(cfg.num_outputs, 1);
            assert_eq!(cfg.num_hidden_nodes[0], 8);
            check(cfg, name);
        }

        // 3 var_part configs (6 → 1, hidden=8).
        for (cfg, name) in [
            (&VAR_PART_NN_CONFIG_64, "var_part_64"),
            (&VAR_PART_NN_CONFIG_32, "var_part_32"),
            (&VAR_PART_NN_CONFIG_16, "var_part_16"),
        ] {
            assert_eq!(cfg.num_inputs, 6);
            assert_eq!(cfg.num_outputs, 1);
            assert_eq!(cfg.num_hidden_nodes[0], 8);
            check(cfg, name);
        }

        // 4 part_split configs (12 → 1, hidden=8).
        for (cfg, name) in [
            (&PART_SPLIT_NN_CONFIG_64, "part_split_64"),
            (&PART_SPLIT_NN_CONFIG_32, "part_split_32"),
            (&PART_SPLIT_NN_CONFIG_16, "part_split_16"),
            (&PART_SPLIT_NN_CONFIG_8, "part_split_8"),
        ] {
            assert_eq!(cfg.num_inputs, 12);
            assert_eq!(cfg.num_outputs, 1);
            assert_eq!(cfg.num_hidden_nodes[0], 8);
            check(cfg, name);
        }

        // Linear partition tables (24 entries each, 3 lanes × 8).
        assert_eq!(PARTITION_FEATURE_MEAN.len(), 24);
        assert_eq!(PARTITION_FEATURE_STD.len(), 24);
        assert_eq!(PARTITION_LINEAR_WEIGHTS.len(), 24);
        // Breakout linear tables: 2 resolution × 3 q × 5 (features + bias).
        for tbl in [
            &PARTITION_BREAKOUT_WEIGHTS_64,
            &PARTITION_BREAKOUT_WEIGHTS_32,
            &PARTITION_BREAKOUT_WEIGHTS_16,
            &PARTITION_BREAKOUT_WEIGHTS_8,
        ] {
            assert_eq!(tbl.len(), 2);
            for row in tbl.iter() {
                assert_eq!(row.len(), 3);
                for cell in row.iter() {
                    assert_eq!(cell.len(), 5);
                }
            }
        }
    }
}
