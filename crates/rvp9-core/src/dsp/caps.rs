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
//! - **aarch64**: scalar, NEON. (NEON-dotprod deferred to v2.)
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
    pub fn detect() -> Self {
        Self {
            sse2: std::is_x86_feature_detected!("sse2"),
            ssse3: std::is_x86_feature_detected!("ssse3"),
            sse4_1: std::is_x86_feature_detected!("sse4.1"),
            avx2: std::is_x86_feature_detected!("avx2"),
            bmi2: std::is_x86_feature_detected!("bmi2"),
        }
    }
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
}

#[cfg(target_arch = "aarch64")]
impl AArch64Caps {
    /// Detect the running CPU's aarch64 ISA support.
    pub fn detect() -> Self {
        Self {
            neon: std::arch::is_aarch64_feature_detected!("neon"),
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
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn aarch64_neon_always_present() {
        let caps = AArch64Caps::detect();
        assert!(caps.neon, "aarch64 CPUs always support NEON");
    }

    /// Calling detect twice should return the same value (CPU caps don't
    /// change at runtime).
    #[test]
    fn detect_is_stable() {
        let a = detect();
        let b = detect();
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}
