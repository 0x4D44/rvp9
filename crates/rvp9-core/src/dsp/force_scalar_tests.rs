//! Cross-process routing oracle. Trace the actual SIMD entries, because equal
//! scalar/SIMD output alone cannot prove that the scalar implementation ran.

use std::cell::Cell;
use std::process::{Command, Stdio};
use std::sync::{Arc, Barrier};

const KERNELS: &[&str] = &[
    "fdct8x8",
    "fdct16x16",
    "hadamard_8x8",
    "hadamard_16x16",
    "hadamard_32x32",
    "hadamard_8x8_subtract",
    "hadamard_16x16_subtract",
    "hadamard_32x32_subtract",
    "vpx_int_pro_row_sse2",
    "vpx_int_pro_row_clamped_sse2",
    "int_pro_64x64_fused_sse2",
    "vpx_vector_var_sse2",
    "vp9_quantize_fp_i16_metrics_sse2",
    "vp9_quantize_fp_i16_rate_metrics_sse2",
    "vp9_quantize_fp_32x32_i16_metrics_sse2",
    "vp9_quantize_fp_32x32_i16_rate_metrics_sse2",
    "subtract_square_sse2",
    "subtract_square_sse2_with_nonzero",
    "sse_block_sse2",
    "minmax_8x8_sse2",
    "avg_diff_8x8_grid_64x64_sse2",
];

thread_local! {
    static ENTRIES: Cell<u64> = const { Cell::new(0) };
}

pub(crate) fn record(name: &str) {
    let index = KERNELS
        .iter()
        .position(|&kernel| kernel == name)
        .expect("instrumented kernel must have an oracle bit");
    ENTRIES.with(|entries| entries.set(entries.get() | (1 << index)));
}

pub(crate) fn reset() {
    ENTRIES.with(|entries| entries.set(0));
}

pub(crate) fn snapshot() -> u64 {
    ENTRIES.with(Cell::get)
}

fn exercise_wrappers() -> u64 {
    reset();
    crate::transforms::forward::exercise_force_scalar_paths();
    crate::mode_decision::hadamard::exercise_force_scalar_paths();
    crate::mv_search::int_pro_me::exercise_force_scalar_paths();
    crate::mode_decision::inter::exercise_force_scalar_paths();
    crate::mode_decision::partition::exercise_force_scalar_paths();

    let source = [19u8; 64];
    let predictor = [7u8; 64];
    let mut residual = [0i16; 64];
    super::accelerated::subtract_square_sse2(&source, 8, &predictor, 8, &mut residual, 8);
    assert_eq!(residual, [12; 64]);
    assert!(super::accelerated::subtract_square_sse2_with_nonzero(
        &source,
        8,
        &predictor,
        8,
        &mut residual,
        8,
    ));
    assert_eq!(residual, [12; 64]);
    snapshot()
}

fn assert_routes(observed: u64, forced: bool) {
    let expected = if forced {
        0
    } else {
        (1u64 << KERNELS.len()) - 1
    };
    let differences: Vec<_> = KERNELS
        .iter()
        .enumerate()
        .filter(|(index, _)| (observed ^ expected) & (1u64 << *index) != 0)
        .map(|(_, &name)| name)
        .collect();
    assert_eq!(
        observed, expected,
        "forced={forced}: unexpected SIMD entry bits for {differences:?}"
    );
    assert_eq!(super::get_ref().caps.sse2, !forced);
}

#[test]
fn force_scalar_subprocess_child() {
    if std::env::var_os("RVP9_SCALAR_ROUTING_CHILD").is_none() {
        return;
    }
    let forced = std::env::var_os("RVP9_FORCE_SCALAR").is_some_and(|value| value == "1");

    // Nothing above initialises DSP. Race the first use from several callers,
    // collecting each caller's own trace so TLS cannot hide a worker bypass.
    let barrier = Arc::new(Barrier::new(4));
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                exercise_wrappers()
            })
        })
        .collect();
    for worker in workers {
        assert_routes(worker.join().expect("routing worker must complete"), forced);
    }
    // Also prove that ordinary calls after initialisation retain that routing.
    assert_routes(exercise_wrappers(), forced);
}

#[test]
fn force_scalar_subprocess_routes() {
    let executable = std::env::current_exe().expect("unit test executable");
    for value in [None, Some("1"), Some("0"), Some("true"), Some("01")] {
        let mut child = Command::new(&executable);
        child
            .args([
                "--exact",
                "dsp::force_scalar_tests::force_scalar_subprocess_child",
                "--test-threads=1",
            ])
            .stdin(Stdio::null())
            .env("RVP9_SCALAR_ROUTING_CHILD", "1")
            .env_remove("RVP9_FORCE_SCALAR");
        if let Some(value) = value {
            child.env("RVP9_FORCE_SCALAR", value);
        }
        let output = child.output().expect("spawn isolated routing test");
        assert!(
            output.status.success(),
            "RVP9_FORCE_SCALAR={value:?}: child failed\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
