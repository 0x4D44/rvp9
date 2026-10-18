//! Cross-process proof that the Arm force-scalar hook reaches table and
//! direct decoder-transform entry points without changing process env in a
//! parallel unit-test process.

use std::process::{Command, Stdio};

use super::{SimdDispatch, caps::AArch64Caps};

const CAPS_CHILD: &str = "RVP9_AARCH64_FORCE_SCALAR_CHILD";
const IDCT_CHILD: &str = "RVP9_AARCH64_IDCT16_TRACE_CHILD";

fn run_child(test_name: &str, marker: &str, value: Option<&str>) -> String {
    let executable = std::env::current_exe().expect("unit test executable");
    let mut child = Command::new(executable);
    child
        .args(["--exact", test_name, "--test-threads=1"])
        .stdin(Stdio::null())
        .env(marker, "1")
        .env_remove("RVP9_FORCE_SCALAR");
    if let Some(value) = value {
        child.env("RVP9_FORCE_SCALAR", value);
    }
    let output = child.output().expect("spawn isolated AArch64 routing test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "RVP9_FORCE_SCALAR={value:?}: child failed\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("running 1 test") && stdout.contains("1 passed; 0 failed"),
        "RVP9_FORCE_SCALAR={value:?}: child did not execute exactly one test\nstdout: {stdout}\nstderr: {stderr}"
    );
    format!("{stdout}{stderr}")
}

#[test]
fn force_scalar_aarch64_subprocess_child() {
    if std::env::var_os(CAPS_CHILD).is_none() {
        return;
    }

    let forced = std::env::var_os("RVP9_FORCE_SCALAR").is_some_and(|value| value == "1");
    let native_caps = AArch64Caps {
        neon: std::arch::is_aarch64_feature_detected!("neon"),
        dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
    };
    assert!(native_caps.neon, "NEON is mandatory on native AArch64");
    let expected_caps = if forced {
        AArch64Caps::default()
    } else {
        native_caps
    };

    assert_eq!(
        AArch64Caps::detect(),
        expected_caps,
        "AArch64Caps::detect must honor only the exact value 1"
    );
    let dispatch = super::get_ref();
    assert_eq!(dispatch.caps, expected_caps);

    let scalar_dispatch = SimdDispatch::build(AArch64Caps::default());
    let native_dispatch = SimdDispatch::build(native_caps);
    let expected_v = if forced {
        scalar_dispatch.v_predictor()
    } else {
        native_dispatch.v_predictor()
    };
    let expected_h = if forced {
        scalar_dispatch.h_predictor()
    } else {
        native_dispatch.h_predictor()
    };
    let expected_idct8 = if forced {
        scalar_dispatch.idct8x8_add()
    } else {
        native_dispatch.idct8x8_add()
    };
    let expected_iht8 = if forced {
        scalar_dispatch.iht8x8_add()
    } else {
        native_dispatch.iht8x8_add()
    };
    let expected_loopfilter = if forced {
        scalar_dispatch.lpf_vertical_4()
    } else {
        native_dispatch.lpf_vertical_4()
    };
    let expected_convolve = if forced {
        scalar_dispatch.convolve8_horiz()
    } else {
        native_dispatch.convolve8_horiz()
    };
    assert_eq!(dispatch.v_predictor() as usize, expected_v as usize);
    assert_eq!(dispatch.h_predictor() as usize, expected_h as usize);
    assert_eq!(dispatch.idct8x8_add() as usize, expected_idct8 as usize);
    assert_eq!(dispatch.iht8x8_add() as usize, expected_iht8 as usize);
    assert_eq!(
        dispatch.lpf_vertical_4() as usize,
        expected_loopfilter as usize
    );
    assert_eq!(
        dispatch.convolve8_horiz() as usize,
        expected_convolve as usize
    );
    if forced {
        assert_eq!(
            dispatch.v_predictor() as usize,
            scalar_dispatch.v_predictor() as usize,
            "forced-scalar AArch64 table must install the scalar V predictor"
        );
        assert_eq!(
            dispatch.h_predictor() as usize,
            scalar_dispatch.h_predictor() as usize,
            "forced-scalar AArch64 table must install the scalar H predictor"
        );
        assert_eq!(
            dispatch.idct8x8_add() as usize,
            crate::transforms::inverse::idct8x8_64_add as *const () as usize,
            "forced-scalar AArch64 table must install the scalar IDCT8 kernel"
        );
    }
}

#[test]
fn force_scalar_aarch64_subprocess_routes() {
    for value in [None, Some("1"), Some("0"), Some("true"), Some("01")] {
        run_child(
            "dsp::aarch64_force_scalar_tests::force_scalar_aarch64_subprocess_child",
            CAPS_CHILD,
            value,
        );
    }
}

#[test]
fn idct16x16_neon_trace_subprocess_child() {
    if std::env::var_os(IDCT_CHILD).is_none() {
        return;
    }

    let forced = std::env::var_os("RVP9_FORCE_SCALAR").is_some_and(|value| value == "1");
    let input = [0i32; 256];
    let mut dest = [128u8; 256];
    crate::transforms::inverse::reset_idct16x16_neon_entry_trace();

    // Exercise the public transform wrappers directly, outside the dispatch
    // table. Output parity alone cannot prove these calls avoided NEON.
    crate::transforms::inverse::idct16x16_256_add(&input, &mut dest, 16);
    crate::transforms::inverse::idct16x16_10_add(&input, &mut dest, 16);
    crate::transforms::inverse::idct16x16_38_add(&input, &mut dest, 16);

    // The table points these slots at the same public wrapper functions on
    // AArch64; exercise each getter route as well.
    let dispatch = super::get_ref();
    (dispatch.idct16x16_add())(&input, &mut dest, 16);
    (dispatch.idct16x16_10_add())(&input, &mut dest, 16);
    (dispatch.idct16x16_38_add())(&input, &mut dest, 16);

    let expected = if forced { [0, 0, 0] } else { [2, 2, 2] };
    assert_eq!(
        crate::transforms::inverse::idct16x16_neon_entry_trace(),
        expected,
        "direct and dispatch IDCT16 EOB buckets must use NEON iff force-scalar is off"
    );
}

#[test]
fn idct16x16_neon_trace_subprocess_routes() {
    for value in [None, Some("1")] {
        run_child(
            "dsp::aarch64_force_scalar_tests::idct16x16_neon_trace_subprocess_child",
            IDCT_CHILD,
            value,
        );
    }
}
