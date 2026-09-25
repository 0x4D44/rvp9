# Unit 1423 — integrated encoder performance contradiction

The retained pre-Unit-1421 RVP9 executable (`5b6822c0`, SHA-256
`8d175c1bffba875f8381461a7be447335ad8ec77847ef83bb07b965474b29f6e`)
was rechecked against the same pinned libvpx executable as Unit 1422.
Its paired three-clip runtime was **1.216542× libvpx**, close to the earlier
1.230937× result. The integrated `8759316a` executable built for this check
(SHA-256 `e56466e554d816282b742f0c3805abb4f812968fd539324cf7e507ce1180528a`)
was then directly paired against that retained RVP9 control. Its paired
runtime was **1.124134× control** in a 14-observation-per-clip screen and
**1.138241× control** in a 30-observation-per-clip repeat. The repeat's
preflight load was 9.0 busy logical cores; the six measurement-window
samples were 2.2–5.1 of 32 cores. Every RVP9 output hash matched and was
stable in each arm. Thus the integrated build is reproducibly slower than
the retained control in these screens, contrary to Unit 1421's original
0.983983× candidate/control screen. The cause is not established.

Both RVP9 builds used Rust 1.95.0, native release, `CARGO_PROFILE_RELEASE_DEBUG=2`,
and no declared PGO. Between the retained control's source commit and the
integrated source commit, the only shipped source change was the FDCT16
AVX2 first-pass implementation and runtime dispatch in
`crates/rvp9-core/src/transforms/forward_sse2.rs`; no manifest or Cargo
configuration changed. These source facts do not prove identical effective
build environments or machine-code layout.

To isolate dispatch, a temporary build changed only the FDCT16 runtime
branch to force its existing SSE2 first pass, leaving the AVX2 helper in
the binary. It was **1.010171×** the integrated AVX2 build by paired corpus
ratio (SSE2/AVX2): Ducks 0.986512×, In-to-tree 0.996780×, Old-town-cross
1.048293×. All outputs matched. One of eight host-load samples was 19.6
busy cores, and Ducks' separate medians were heavily skewed, so this screen
does not establish a small speed difference. It does rule out a clear
~12–14% direct benefit from merely forcing SSE2 dispatch in this build.
The diagnostic source edit was removed; no code candidate is retained.

All four screens used the same 60-frame 1920×1080 clips, CPU6, CBR
1260 kb/s, one thread, tile 0, AQ0, noise 0, KF60, one warmup, paired
alternating order, two reversed rotations, and no load ceiling. The
manifests, raw rows, and summary CSVs beside this file give exact commands,
binary and input hashes, per-window load, timings, and output hashes.

**Status:** Do not treat Unit 1421's apparent speedup as retained. The
provisional integrated/libvpx figure remains Unit 1422's 1.400201× paired,
but it should not be used as a precise regression estimate until the build
identity discrepancy is resolved. Next, rebuild the pre-Unit-1421 source
under the current toolchain and path/layout conditions, pair that fresh
binary against both retained old and integrated binaries, and inspect the
effective rustc flags or generated code if the difference persists.
No speculative rollback or new optimization is justified by this evidence.
