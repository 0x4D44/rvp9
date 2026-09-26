# Realtime CPU6 never reaches pruned-tree second-level checks

Decision: do not prototype the proposed MV-rate lower-bound exit in
`sub_pixel.rs:second_level_checks` for production CPU6. The routine is not on
this workload's path, so its potential whole-encode saving is zero here.

The live realtime NEWMV path in `crates/rvp9-core/src/mode_decision/inter.rs`
sets `RT_NEWMV_SUBPEL_SEARCH_LEVEL = 0`, passes it as `iters_per_step`, and calls
`find_best_sub_pixel_tree` for CPU6. The pruned-more `second_level_checks`
requires `iters_per_step > 1`. Its only encoder caller is the separate
`rvp9-encoder/src/rdo/single_mv_search.rs` path; source reachability therefore
already argues against this candidate. No `RVP9_RT_NEWMV_SUBPEL_ITERS`
override was set on Crucible during the run.

To check runtime reachability, `subpel-rate-census.patch` instrumented every
`second_level_checks` candidate and printed attempt, legal-MV, and rate-reject
counts by block size after encode. A non-PGO Crucible release build passed the
19 focused `sub_pixel` tests. The diagnostic binary SHA-256 was
`430cbfd217f783f21517789d126c818d64417b65119e1bffb220a4d2dbb15dee`.
`run-subpel-rate-census.ps1` ran the pinned 60-frame, CPU6 CBR1260 clips.
All three IVF hashes matched the integrated reference; `run-output.txt`
records **zero diagnostic rows** for Ducks, Tree, and Town. The per-clip stderr
files were all zero bytes. This is a reachability census, not a CPU screen.

The diagnostic patch is rejected and not integrated. This result does not
rule out a different lower bound on the *full-tree* first-level probes, where
candidate costs can also control search direction and must be preserved.
