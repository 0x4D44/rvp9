# Current PGO Ducks hotspot selection after rejected diagonal bound

This is a selection note, **not a new speed measurement**. The integrated
encoder source is unchanged since the [NSTEP exact-IP capture](../2026-09-26-pgo-nstep-ducks-exact-ip/README.md): PGO EXE SHA-256
`3caee3f4416087ffe13069b4280f48fe92435bc32c9c0a41bebb7c187a73b3ee`,
60,215 direct Ducks samples over three fresh processes. The latest paired
production PGO/libvpx equal-clip CPU ratio is still [1.099305×](../2026-09-26-nstep-duplicate-variance-pgo/README.md), above the 1.05× target.

The fourth-ranked exact IP, RVA `0x153251` (720 samples, 190/276/254), is
`vmovdqu 0x30(%rcx),%xmm1` in `build_inter_prediction_from_tight_plane`.
Disassembly of the hash-pinned PGO EXE on Crucible showed the preceding
`vmovdqu 0x20(%rcx),%xmm0` and following `vmovq 0x68(%rcx),%xmm2`, then
identity-scale checks. The earlier PGO image had the **same sampled load**
at RVA `0x152ed1` (see [older exact-IP profile](../2026-09-26-pgo-ducks-exact-ip/README.md)).
This is a parameter-structure load, not evidence that division dominates.
The prior exact identity-scale specialization saved only 0.41% overall in a
plain-release screen and was rejected (production-parity journal, Unit 1266).
Do not retry that isolated specialization on the strength of this IP.

The current PGO source-line aggregate gives 1,420/60,215 direct samples
(2.36%) to the entire tight-plane prediction symbol. The vertical and
horizontal AVX2 16-tap loop-filter symbols have 698 and 526 (1.16% and
0.87%) respectively. Their samples are diffuse across hundreds of IPs:
the hottest vertical IP has 14 samples, and the hottest horizontal IP has
12. A speculative load rewrite in either filter has no localized cost
evidence. These are self-sample shares, **not** removable-work or speedup
estimates; they do not justify coding to a projected gain.

The larger, measured discrepancy is search volume. The [direct reference
census](../2026-09-26-libvpx-ducks-reference-census/README.md) found 554,931
RVP9 full-tree subpixel entries against 370,507 in pinned libvpx for Ducks.
Of the 184,424 extra entries, 175,844 are GOLDEN-reference entries. This
count does not prove which policy is correct or how much CPU can be saved.
The next bounded diagnostic is to attribute GOLDEN full-tree CPU on the
current exact PGO image and reconcile the upstream admission/rejection path
against pinned libvpx. Only an oracle that preserves the winning search result
and exact output should support an early-exit code change; count reduction
alone is not a correctness or performance result.
