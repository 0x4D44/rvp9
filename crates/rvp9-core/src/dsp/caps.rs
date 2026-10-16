//! Runtime CPU capability detection.
//!
//! Per HLD-simd §2.1: stable Rust intrinsics-detection macros
//! (`is_x86_feature_detected!`, `std::arch::is_aarch64_feature_detected!`)
//! return whether the *running* CPU supports a given ISA, independently of
//! how the binary was compiled. We ship one binary supporting all the
//! per-arch ISA tiers we target; the runtime check picks the best at boot.
//!
//! ## v1 ISA scope
//!
//! - **x86_64**: scalar, SSE2, SSSE3, SSE4.1, AVX2.
//! - **aarch64**: scalar, NEON, NEON dot-product.
//! - **other** (wasm32, riscv64, etc.): scalar only.
//!
//! Per HLD-simd §2.2 (R-09 init invariant): caps are detected exactly once
//! during `dispatch::init()` and embedded in the resulting `&'static
//! SimdDispatch` table. Workers never re-detect; they read pre-resolved
//! function pointers.

/// x86_64 capability set. Populated by [`detect`] on x86_64 targets.
///
/// SSSE3 and SSE4.1 are listed separately even though SSE4.1 implies SSSE3
/// because libvpx ships some kernels with SSSE3-only specialisation (e.g.
/// the `vpx_sad_skip` family) and others SSE4.1-only (e.g. some intra
/// predictors). We mirror that granularity.
///
/// `bmi2` is independent of the SSE/AVX feature tiers. It is exposed
/// separately so future scalar-bit-manipulation kernels (e.g. `pdep`,
/// `pext`, `bzhi`, `shrx`/`shlx`) can opt in without dragging an AVX2
/// requirement along. Per Track T T-3 measurement slice: cap detection
/// is plumbed first; consumers (e.g. a future `BoolReader::read`
/// renormaliser variant) land later only if a microbench shows a real,
/// non-noise win over the existing `leading_zeros`/`shlx` codegen.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct X86Caps {
    /// SSE2. Available on every x86_64 CPU.
    pub sse2: bool,
    /// SSSE3 (Supplemental SSE3) — `pshufb`, `palignr`, etc.
    pub ssse3: bool,
    /// SSE4.1 — `pblendw`, `pmovzxbw`, etc.
    pub sse4_1: bool,
    /// AVX2 — 256-bit integer SIMD.
    pub avx2: bool,
    /// BMI2 — `pdep`, `pext`, `bzhi`, `shrx`/`shlx`, `mulx`. Independent
    /// of the SSE/AVX tier (introduced with Haswell alongside AVX2 but
    /// promoted to its own CPUID bit). Currently detected-only; no
    /// kernel selects on it yet (Track T T-3 baseline).
    pub bmi2: bool,
}

#[cfg(target_arch = "x86_64")]
impl X86Caps {
    /// Detect the running CPU's x86_64 ISA support. Stable since Rust 1.27.
    ///
    /// **Byte-identity oracle hook** (HLD-simd-speed §8.2): when the env var
    /// `RVP9_FORCE_SCALAR=1` is set, every ISA tier is masked to `false`,
    /// forcing every `SimdDispatch` slot to its scalar reference arm. Because
    /// `init()`/`get()`/`get_ref()` all build the process-wide table from
    /// `SimdDispatch::build(detect())`, masking here makes the *entire* table
    /// scalar — with no change to `build`'s visibility. This lets the
    /// cross-process oracle encode the same input twice (forced-scalar vs
    /// auto/AVX2) and byte-diff the bitstreams to prove no decision moved.
    /// Encoder wrappers outside the table also consult its cached capabilities
    /// before selecting x86 kernels. Explicit ISA entry points used by kernel
    /// tests and microbenchmarks retain their explicit selection semantics.
    ///
    /// Zero-cost when unset: the env read happens once at first dispatch
    /// `get()` (the table is built once via the process-wide `OnceLock`).
    pub fn detect() -> Self {
        if force_scalar_env() {
            // All-false caps → every dispatch slot resolves to its scalar arm.
            return Self::default();
        }
        Self {
            sse2: std::is_x86_feature_detected!("sse2"),
            ssse3: std::is_x86_feature_detected!("ssse3"),
            sse4_1: std::is_x86_feature_detected!("sse4.1"),
            avx2: std::is_x86_feature_detected!("avx2"),
            bmi2: std::is_x86_feature_detected!("bmi2"),
        }
    }
}

/// Returns `true` iff `RVP9_FORCE_SCALAR=1` is set in the environment.
///
/// Drives the forced-scalar leg of the cross-process byte-identity oracle
/// (HLD-simd-speed §8.2). Only the exact value `"1"` enables it; any other
/// value (or absence) leaves SIMD detection untouched.
#[cfg(target_arch = "x86_64")]
fn force_scalar_env() -> bool {
    std::env::var_os("RVP9_FORCE_SCALAR").is_some_and(|v| v == "1")
}

/// aarch64 capability set. Populated by [`detect`] on aarch64 targets.
///
/// NEON is mandatory on every aarch64 we ship; the explicit check guards
/// against truly exotic builds and documents the dependency.
#[cfg(target_arch = "aarch64")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AArch64Caps {
    /// NEON SIMD. Effectively always true on aarch64.
    pub neon: bool,
    /// Armv8.2-A integer dot-product instructions.
    pub dotprod: bool,
}

#[cfg(target_arch = "aarch64")]
impl AArch64Caps {
    /// Detect the running CPU's aarch64 ISA support.
    pub fn detect() -> Self {
        #[cfg(miri)]
        {
            // Miri cannot execute AArch64 SIMD intrinsics. Keep its runtime
            // dispatch on the scalar oracle while still checking every
            // non-SIMD path on native AArch64 hosts.
            Self::default()
        }
        #[cfg(not(miri))]
        {
            Self {
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            }
        }
    }
}

/// Stub for "other targets" (wasm32, riscv64, ppc64le, ...). Forces every
/// dispatch slot to the scalar fallback path.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScalarOnlyCaps;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl ScalarOnlyCaps {
    /// "Detect" — always returns `Self`.
    pub fn detect() -> Self {
        Self
    }
}

/// The platform-appropriate capability type at the current build target.
/// Re-exported so callers don't need their own `#[cfg(target_arch)]`.
#[cfg(target_arch = "x86_64")]
pub type Caps = X86Caps;
/// The platform-appropriate capability type at the current build target.
#[cfg(target_arch = "aarch64")]
pub type Caps = AArch64Caps;
/// The platform-appropriate capability type at the current build target.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type Caps = ScalarOnlyCaps;

/// Detect the running CPU's capabilities. Equivalent to `Caps::detect()`
/// but works without naming the platform type.
#[inline]
pub fn detect() -> Caps {
    Caps::detect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On any tier-1 x86_64 dev box SSE2 is universally available — every
    /// x86_64 CPU has it. This is a sanity test for the detection wiring.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_sse2_always_present() {
        let caps = X86Caps::detect();
        assert!(caps.sse2, "x86_64 CPUs always support SSE2");
    }

    /// BMI2 cap is plumbed as a stable bool. Cannot assert it's `true`
    /// (pre-Haswell x86_64 CPUs and many older Atoms lack it); this just
    /// pins the field's existence and that detection is deterministic
    /// across calls.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_bmi2_field_is_stable() {
        let a = X86Caps::detect().bmi2;
        let b = X86Caps::detect().bmi2;
        assert_eq!(a, b, "bmi2 detection must be deterministic");
    }

    /// On aarch64 NEON is mandatory.
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    #[test]
    fn aarch64_neon_always_present() {
        let caps = AArch64Caps::detect();
        assert!(caps.neon, "aarch64 CPUs always support NEON");
    }

    /// Calling detect twice should return the same value (CPU caps don't
    /// change at runtime).
    ///
    /// Guarded against the byte-identity oracle's forced-scalar mask
    /// (HLD-simd-speed §8.2): when `RVP9_FORCE_SCALAR=1`, `detect()` returns
    /// all-false caps regardless of the host CPU. This test asserts the
    /// *unmasked* path is deterministic, so it skips only when the mask is
    /// actually engaged. The skip predicate reuses `force_scalar_env()` so it
    /// matches `detect()`'s exact-`"1"` trigger — e.g. `RVP9_FORCE_SCALAR=0`
    /// does NOT mask, so it must NOT skip.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn detect_is_stable() {
        if force_scalar_env() {
            return;
        }
        let a = detect();
        let b = detect();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// Non-x86_64 variant of [`detect_is_stable`]: those targets have no
    /// `RVP9_FORCE_SCALAR` mask (the env hook is `#[cfg(target_arch =
    /// "x86_64")]`), so detection is unconditionally deterministic.
    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn detect_is_stable() {
        let a = detect();
        let b = detect();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    /// The forced-scalar mask is two independent pieces, verified here
    /// WITHOUT mutating the process-wide `RVP9_FORCE_SCALAR` env var.
    ///
    /// Mutating it via `set_var` in a parallel test process races with
    /// concurrent `detect()`-based tests — `dispatch.rs` (the
    /// `loopfilter_avx2_default_routing_avx2_cpu` test, ~line 3560)
    /// documents exactly this flake and notes the crate has no
    /// `serial_test` dep to serialise around it. So we prove the two halves
    /// of the contract directly:
    ///
    /// 1. **Mask consequence** — `detect()`'s forced leg returns
    ///    `X86Caps::default()` (all tiers `false`), which `SimdDispatch::build`
    ///    must resolve to scalar reference slots. We build the table from a
    ///    locally-constructed all-false caps (the exact value the forced leg
    ///    returns) and check a representative AVX2-gated slot routes to its
    ///    scalar arm. No env touched.
    /// 2. **Predicate trigger** — `force_scalar_env()` reads the live env; in
    ///    cargo's default test environment the var is unset, so it must read
    ///    `false`. (The exact-`"1"` matching of the predicate is covered by
    ///    its own definition; here we only pin that an unset env does not
    ///    spuriously engage the mask.)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn force_scalar_caps_route_dispatch_to_scalar() {
        use crate::dsp::dispatch::{Convolve8OneAxisFn, SimdDispatch};

        // (1) all-false caps == what detect()'s forced leg returns.
        let masked = X86Caps::default();
        assert!(!masked.avx2, "all-false caps must have avx2 = false");
        assert_eq!(
            masked,
            X86Caps {
                sse2: false,
                ssse3: false,
                sse4_1: false,
                avx2: false,
                bmi2: false,
            },
            "X86Caps::default() must be all-false (the forced-scalar leg value)"
        );

        // Build the dispatch table from those caps and confirm an
        // AVX2-promoted slot resolves to its scalar reference — i.e. the
        // forced-scalar leg yields a scalar table, no SIMD slot survives.
        let d = SimdDispatch::build(masked);
        let scalar_convolve8_horiz: Convolve8OneAxisFn = crate::inter::filter::convolve8_horiz;
        assert_eq!(
            d.convolve8_horiz() as usize,
            scalar_convolve8_horiz as usize,
            "all-false caps must install the scalar convolve8_horiz reference"
        );

        // (2) the predicate does not fire in cargo's default (unset) env.
        assert!(
            !force_scalar_env(),
            "RVP9_FORCE_SCALAR is unset in the default test env; mask must be off"
        );
    }
}
