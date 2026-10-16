# Final-release candidate qualification evidence

Current candidate audit: [release-readiness report](../../2026.10.15%20-%20STATUS%20-%20final%20release%20readiness.md). The following initial results are historical intake evidence; later source changes require requalification. Candidate tool regressions, Miri diagnostics and Arm scalar-coverage correction are under [candidate-20261016](candidate-20261016/identity.json). Completed intake spatial data and numerical limitations are under [spatial-quality](spatial-quality/audit.md).

Candidate code: `1a84f60a24e8d82e8e6454743fbb80521280df0e` (alpha.4 plus bug claim). Documentation/ledger commits on this task do not change shipped source. Native host: Apple M5 Max, aarch64-apple-darwin, Rust 1.95.0 / LLVM 22.1.2. See `macos-initial-identity.json` for exact binaries and input identities.

Observed terminal results:

| Check | Exit | Result |
| --- | ---: | --- |
| Pinned libvpx initialization/build | 0 | Exact v1.14.1 pin; encoder, decoder and spatial example built |
| Official vector fetch | 0 | 307 vectors and 307 MD5 ledgers verified |
| Canonical corpus setup, twice | 1 | Three matrix clips downloaded; Parkrun URL returns HTTP 404; RVP9-BUG-FLU-00036 |
| Release CLI/harness/bench binary build | 0 | Completed; initial malformed repeated --bins invocation is excluded |
| `cargo fmt --all --check` | 0 | No diagnostic |
| `cargo clippy --workspace --all-targets --locked -- -D warnings` | 0 | Passed |
| `cargo doc --workspace --no-deps --locked` | 0 | Completed with existing warnings; not a warning-free public-doc pass |
| Production matrix self-test | 0 | Plan/count/negative-path checks pass; stub codec observations |
| Production resume self-test | 0 | Receipts/tampering/redecode/partial-timing restrictions pass |
| Production timing self-test | 0 | Missing/malformed evidence and contradictory runner verdicts reject |
| Spatial matrix self-test | 0 | Matrix topology/count/quality-ceiling checks pass |
| Alpha and reverse gates with zero frames | 2 | Both reject before doing work |
| Production invocation without phase | 1 | Rejects; no default real phase |

These are prerequisite/tooling results. Conformance, cross-decoder, quality, speed, latency and safety campaigns are not implied by passing stubbed self-tests. Pending real campaign results are recorded separately when observed.

KIL-00107 singular-fit fixes now have passing candidate regression checks. Separate numerical accuracy defect RVP9-BUG-FLU-00042 blocks trusting affected cubic measurements, and FLU-00041 affects spatial summary CSV. Arm scalar-labelled conformance does not select scalar dispatch (FLU-00043); no actual Arm scalar coverage is established. The user has not authorized integration or release.
