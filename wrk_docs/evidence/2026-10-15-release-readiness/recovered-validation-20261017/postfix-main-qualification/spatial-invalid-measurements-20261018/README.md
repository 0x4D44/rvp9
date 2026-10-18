# Spatial invalid-measurement probe

On runner revision afa6d491072aa6a832120c02e1ab6f63242bfc76, the exact scripts/qualify-spatial-svc.sh blob matches product candidate fb5c0c917038db6c640b3dbc7c2b84b567233333. The harness calls the extracted run_quality_curve and its relevant helpers verbatim for one CBR 720p30, CPU6, two-layer curve. A stub bdrate-svc writes the injected per-layer values and exits successfully.

The exact-source caller rejected NaN and inf with its decimal-number diagnostic before adding a summary row. It rejected 0.6% above the +0.5% ceiling, wrote an SL0 FAIL row, and exited 1 before cross-decoder checking. In a mutation control, replacing require_number with a no-op let both NaN rows become PASS, reached the cross-check stub, and exited 0. This demonstrates the probe detects removal of the caller's numeric guard.

The finite over-limit case's summary line retains the script's unquoted comma-separated bitrate grid. That known spatial summary defect is FLU-00041; this probe only exercises the status and exit path and does not validate summary CSV shape. The mutation control is deliberately altered evidence code and does not change repository source.

The child is a stub, and cross-decoder checking is stubbed. This does not run the full 96-curve phase, actual bdrate-svc, encoders, decoders, publication, or any codec, quality, or performance measurement. FLU-00041 and FLU-00042 remain open. No new bug was raised because this probe confirmed an existing guard rather than finding another defect. Raw outputs, generated CSVs, the exact source extract, harnesses, and hashes are in qualification.json.
