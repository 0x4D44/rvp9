# Soak CLI test expectation correction

The earlier exact-candidate workspace run exposed a stale test fixture: it passed `--frames-per-iter 0`, which the current soak CLI rejects before reaching the test's requested missing-`vpxdec` check. The test now uses one frame. The CLI checks the intentionally absent decoder path before entering the soak loop, so the focused regression test exercises that preflight without encoding or decoding.

On revision `daf0e40590dd2578d45ce43c27cf997b72cb278f`, the focused release-mode test passed 1/1 with locked offline dependencies and stdin closed. The change touches only `crates/rvp9-bench/tests/soak_cli.rs`; production code remains the exact candidate source `fb5c0c917038db6c640b3dbc7c2b84b567233333`. The exact raw output is losslessly preserved as `focused-test.raw.txt.gz`; the JSON records both compressed and uncompressed hashes.

This does not replace the failed full-workspace result: that suite has not been rerun on the corrected test tree. Its separate one-time harness stderr timeout-test failure remains unexplained, despite four isolated exact-source retries passing. See `qualification.json` and the linked full-workspace record.
