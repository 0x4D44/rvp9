# Current-source seeded full-frame fuzz qualification

On exact product source fb5c0c917038db6c640b3dbc7c2b84b567233333, the fuzz_decode_frame target completed a 300-second Apple Silicon libFuzzer campaign from the first raw VP9 keyframe packet in official vector vp90-2-05-resize.ivf. The packet is 3,964 bytes and its source vector and extracted-packet hashes are recorded in seed-input.txt.

The run exited 0 after 301 seconds and completed 64,129 executions (213/s average). LibFuzzer reported 1,748 new units, 5,118 coverage counters, 21,578 features, retained corpus 1,240 inputs / 3,222 KiB, and peak RSS 1,051 MiB. Its log reached Decoder::decode_inter_frame. No crash or sanitizer artifact was found.

The run used cargo-fuzz 0.13.2 and Rust 1.98.0-nightly with cargo-fuzz's default AddressSanitizer instrumentation. This is bounded full-frame evidence from one official 320x240 keyframe seed. It does not qualify Windows fuzzing, the approved billion-iteration campaign, or the broader release safety gates.

Evidence files:
- command.txt, environment.txt, candidate-identity.txt: invocation, environment, and source/tool identities.
- runner.raw.txt, libfuzzer.raw.log, exit-code.txt: complete raw outputs and result.
- seed-input.txt: official vector provenance and raw packet hash.
- artifact-state.txt: crash-artifact and generated-corpus disposition.
- qualification.json: machine-readable result.
- SHA256SUMS.txt: retained evidence integrity hashes.

No project source, manifest, lockfile, or toolchain configuration changed during the run.
