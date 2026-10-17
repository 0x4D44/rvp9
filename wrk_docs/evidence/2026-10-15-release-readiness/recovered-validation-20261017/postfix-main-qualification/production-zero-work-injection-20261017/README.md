# Production-phase zero-work injection

On the exact candidate script, the quality, relative-speed and absolute-latency phase functions were invoked with empty row iterators and no-op host/child boundaries. Each phase exited 1 at its exact-total guard and left `completed/` absent: quality ran 0 of 200 curves, speed 0 of 75 groups, and latency 0 of 200 rows.

A scratch-only negative control removed only those exact-total guard blocks. The same zero-row inputs then exited 0 and published header-only `completed/` directories for all three phases. This confirms the injection would expose a fail-open regression in the guards.

The function bodies came from `scripts/qualify-production-encoder.sh` at product source `fb5c0c917038db6c640b3dbc7c2b84b567233333`; its Git blob matches that candidate. Only the trailing `main "$@"` invocation was omitted to call the functions directly. The test stubs did not run an encoder, decoder, AQ oracle, or host-load check. This verifies phase-count orchestration only; it supplies no codec, quality, throughput, latency, or release-gate result.

The exact command, source and harness identities, per-phase outcomes and SHA-256 hashes are in [qualification.json](qualification.json). The [harness](zero-work-injection.sh), combined result [log](harness.raw.txt), and six phase logs preserve the executed cases.
