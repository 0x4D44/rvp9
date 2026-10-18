# Current-source seeded parser fuzz qualification

At task-worktree revision b29b33112c3a63d6fa45f9af4c9b2477c7d9a067, whose relevant product and fuzz source matches candidate fb5c0c917038db6c640b3dbc7c2b84b567233333, fuzz_parse_packet completed a 300-second libFuzzer campaign from the 307 official conformance vectors (.ivf and .webm, 25,545,475 bytes total). The run exited 0 after 301 seconds and 3,201,081 executions (10,634/s average); libFuzzer reported 2,943 new units, final coverage 1,634 counters / 2,754 features, retained corpus 383 inputs / 176 KiB, and peak RSS 1,541 MiB. No crash or sanitizer artifact was found.

The run used cargo-fuzz 0.13.2 and Rust 1.98.0-nightly on Apple Silicon. Cargo-fuzz's default AddressSanitizer instrumentation was used. This is a bounded parser-target qualification only. It does not qualify fuzz_decode_frame, Windows fuzzing, the approved billion-iteration campaign, or the broader release safety gates. The earlier 31-second parser and full-frame smokes remain separately recorded in ../mac-current-source-bounded-fuzz-smokes-qualification-20261017/qualification.json.

Evidence files:
- command.txt, environment.txt, candidate-identity.txt: invocation and identities.
- runner.raw.txt, libfuzzer.raw.log, exit-code.txt: complete process output and result.
- initial-seed-corpus.tsv: filename, byte length, and SHA-256 for every initial official vector.
- seed-corpus-source.txt: copy provenance and post-run manifest verification.
- artifact-state.txt: crash-artifact and generated-corpus disposition.
- qualification.json: machine-readable result.
- SHA256SUMS.txt: integrity hashes for the retained evidence, checked from the repository root with shasum -a 256 -c.

No project source, manifest, lockfile, fuzz corpus, or toolchain configuration was changed by this run.
