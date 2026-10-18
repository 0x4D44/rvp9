# Spatial child-process failure probe

On runner revision baa4dd19bc3111f7da8f859b8019744478cf59de, the exact scripts/qualify-spatial-svc.sh blob matches product candidate fb5c0c917038db6c640b3dbc7c2b84b567233333. The harness calls the extracted run_quality_curve and its relevant helpers verbatim for one CBR 720p30, CPU6, two-layer curve. Its bdrate-svc stub writes two passing-looking +0.1% layer rows, then exits 73.

The exact-source caller exited 1 with its child-failure diagnostic before adding summary data or calling the cross-decoder helper, despite the stub's CSV containing both rows. In a mutation control, the child-status conditional was inverted; the same exit-73 output became two PASS rows, reached the cross-check stub, and exited 0. This demonstrates the probe detects loss of the child-status gate.

The probe executes one helper-level path with a child stub. It does not run actual BD-rate calculations, encoders, decoders, the full 96-curve phase, publication, or codec, quality, or performance measurements. The summary format retains the known unquoted bitrate-grid field associated with FLU-00041. FLU-00041 and FLU-00042 remain separate blockers. No new bug was raised because the exact-source guard rejected the failed child.
