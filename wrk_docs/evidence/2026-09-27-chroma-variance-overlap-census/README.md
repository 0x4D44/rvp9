# Failed-skip chroma variance overlap: no-go

The integrated large-block `chroma_planes_skippable` probe already reuses chroma prediction bytes when its skip test fails. A diagnostic checked whether the fallback UV model also repeats the probe's variance reduction for those same bytes. The [base64-encoded opt-in diagnostic patch](diagnostic.patch.b64) changes no scoring or bitstream behavior. It counted only completed, ready-mask planes with color sensitivity that reached the UV model after early-termination gates.

Crucible built the diagnostic from local commit `1fc56f77` on integrated `b35a34b9` as equivalent remote commit `415de176d02e051a67b89e7dde3c7459ccdd0bf5`. The x86 release EXE SHA-256 was `6f60f09846b4b308de3821b0f03c9d31eefcced4e0434985efef074a1b2016eb`. It encoded the three pinned 60-frame 1080p inputs at CPU6/CBR1260, one thread, no tiles/AQ/noise, lag zero and KF60. All three IVFs matched the canonical integrated SHA-256 values; the instrumented CPU times are not performance evidence. The focused reconciliation test was deliberately red and then green on AArch64, and passed on x86; the x86 release build and AArch64 CLI check passed.

| Clip | Probe calls | Veto U | Veto V | Sensitive reductions repeated | Pixels repeated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ducks | 5,722 | 0 | 0 | 0 | 0 |
| Tree | 25,544 | 1 | 0 | 1 | 256 |
| Town | 26,232 | 0 | 0 | 0 | 0 |

Every other probe accepted both planes, then skipped the fallback UV model. The single Tree overlap is too little to warrant a production cache or paired PGO screen. This closes only this exact duplicate-reduction hypothesis, not other UV-model work. [Per-block counts](counts.csv), [raw diagnostic lines](tree.stderr.txt), [runner](run-chroma-census.ps1), and [build identity](identity.txt) preserve the observation. The diagnostic branch stays unintegrated.
