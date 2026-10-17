# macOS fuzz prerequisites

Installed native nightly is available, but cargo fuzz --version exits 101 because cargo-fuzz is absent. The installed nightly also rejects the former documented -Zsanitizer=undefined recipe. These are prerequisite probes, with zero fuzz executions; no tool was installed. Exact command exits and original output hashes are retained in results.json. The raw archive preserves all four original files; visible text normalizes whitespace, mapped in text-evidence-integrity.json.
