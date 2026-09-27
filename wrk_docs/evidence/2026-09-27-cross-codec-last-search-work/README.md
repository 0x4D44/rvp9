# Same-input LAST full-pel operation census, Ducks 60 frames

This is a diagnostic no-go for the checked-slice SAD4D adapter as the *primary* parity lever, not a new performance result. The last direct integrated production-PGO equal-clip process-CPU ratio remains **1.057638×** pinned libvpx (target ≤1.05×). Counters add work, so their process times are deliberately not compared.

Both codecs encoded the identical 60-frame Ducks input (SHA-256 `6d2d866613f7a8c7b11bd2c27374f7637cd4a0748ff6fea4a671b23cfbcde6d7`) with the CPU6/CBR1260 single-thread command in [libvpx's run script](libvpx/run-libvpx-search-ops.ps1). The pinned libvpx source is submodule commit `12f3a2ac`; its diagnostic binary SHA-256 was `421f91fb690f05291d74f9dfed0bd4c81a837ec346a88f502290c60530f8868c`, and its output SHA-256 was the canonical `b24234d8db46c22e03585f1973b27c2e62315f0464da392a2ecd2769d2124fa8`. RVP9 source was integrated commit `6c1a1f94b95d460e2edadf19d451d808f9444347` plus the diagnostic-only [source patch](rvp9/source.patch.b64) (decoded SHA-256 `b50cb1babb68a5e53ca4cae9b37e7548d31977b725eec2dbf2e2ff2fecad5a95`); its x86 binary SHA-256 was `aac8243cc927581964a57682e5f03449b1ad8e1c36fdf995c4297fbf261a7ba7`, and its output SHA-256 was the canonical `8d0458b89ca00cbfb96961e1c16e6018bf88b18790ea7e9d6eeb8c80bd659089`. RVP9's 60-frame stderr SHA-256 was `6f9ad61fddf41271f8ee0f3d5c27a5f7af25da609503ba8a57e8c27b7a6649f7`; the summarized rows below are its nonzero `origin=final_pass ref=LAST method=NSTEP` rows. The RVP9 patch's counter test failed when its diamond-step increment was deliberately removed, then passed when restored. AArch64 formatting, focused test, and CLI build passed; a three-frame gate-off/on encode had identical IVF hashes.

| LAST/NSTEP, all shapes | RVP9 | libvpx | RVP9/libvpx |
|---|---:|---:|---:|
| Entries | 393,364 | 378,013 | 1.041× |
| Diamond starts | 1,013,558 | 1,099,468 | 0.922× |
| Diamond steps | 3,256,929 | 3,405,827 | 0.956× |
| In-limit steps | 3,256,492 | 3,395,220 | 0.959× |
| Four-way diamond SAD calls | 3,184,277 | 6,790,440 | 0.469× |
| Scalar diamond SAD calls | 216,956 | 57,305 | 3.786× |
| Final refinements | 276,444 | 314,158 | 0.880× |

RVP9 uses one four-way SAD call per full in-limit diamond step; libvpx uses two. This accounts for most of the four-way call difference. Counting lanes, RVP9 performs 12.74 million batched candidate SADs in LAST's diamond loop versus libvpx's 27.16 million. RVP9's 159,651 extra scalar diamond calls do not erase that topology difference. These counts do **not** imply either codec has a faster search: per-call cost, seed work, GOLDEN work, and sub-pel work differ. RVP9 also recorded 236,166 GOLDEN/NSTEP entries, whose libvpx counterpart uses another path; they are excluded from this LAST-only comparison. Variance counts were omitted because the call-boundary scopes differ.

Per-shape RVP9 LAST rows (all from the same stderr):

| Shape | Entries | Starts | Steps | In-limit | Four-way SAD | Scalar SAD | Refinements |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8×8 | 27,311 | 80,321 | 256,222 | 255,807 | 253,254 | 8,820 | 19,370 |
| 16×16 | 365,131 | 931,328 | 2,994,224 | 2,994,202 | 2,924,723 | 207,592 | 256,474 |
| 32×32 | 922 | 1,909 | 6,483 | 6,483 | 6,300 | 544 | 600 |

The [libvpx raw CSV](libvpx/ducks-work.csv) retains its shape split. Both source patches are base64-encoded to keep this evidence docs-only; decode them and check the stated RVP9 hash above and libvpx's decoded SHA-256 `10ef304d57cf76991e5532ff875c4dc1078ac27a5e9dcf2a0b067a54317bf562` before use. Neither diagnostic patch belongs in a production build. The corresponding [design note](../../2026.09.27%20-%20HLD%20-%20same-entry%20motion-search%20work%20census.md) sets the decision guard: a large operation-count mismatch stops the adapter rewrite. A later optimization must still pass exact output differentials and repeatable paired production-PGO CPU screens.
