# Unit 1432 — bounded current x86 source attribution

This is a single diagnostic 60-frame In-to-tree CPU6 encode on Crucible, not
a performance comparison. The exact current encoder is `e588d13f` (EXE SHA-256
`c371d472690baba94dc1374b9ce4b65dddd3f3ac2381feca362efa25e55f0d98`,
PDB SHA-256 `a3995992a6eb0ce343fed14f01830737ce577b708dde8379d6913902c7ae67fc`).
The input Y4M SHA-256 is `ead3aef191242b03cc6019f0527c71bc3d11e856ba4f9f92b926fe804582f791`.
The resulting 337,194-byte IVF matched the prior output SHA-256
`5e3ce26ae1c1394d1e096502add7d87cc38559e7f86265004900580582641b41`.

WPR CPU sampling produced a 91,226,112-byte ETL (SHA-256
`ace4d256823b2e0698e98064950326677b176edcc9501de4539f2e0bd353faee`).
WPR start/stop returned zero and ended idle. A full `xperf -a dumper` pass
produced 30,474 `SampledProfile` rows overall. The target PID 41068 had
1,241 direct `Count=1` rows, of which 1,067 were in `rvp9enc.exe` user code.
The xperf weighted encoder report was 1,067,028, versus 1,067 raw rows at
roughly 1,000 weight per row; this is reconciliation by scale, not exact
integer equality. The profile report completed with exit zero and no loss
warning, but the dumper wrapper did not capture its numeric exit code. No
numeric ETL loss total was collected.

`pid-41068-rvp9enc-samples.offline.csv` retains those 1,067 raw encoder
rows. `pid-41068-rvp9enc-ip-lines.csv` maps their 747 distinct IPs through
the exact PDB. I independently counted 1,067 symbol-mapped rows and 1,065
with a nonzero source line. The mapping file's `line_ok` field incorrectly
marks two `line=0` rows as good; use the numeric `line` field instead.
`reconstruct_via_round_trip` accounts for 164 direct rows, including 142
at `crates/rvp9-core/src/mode_decision/inter.rs:5590`, the SSE2 Hadamard
quant-metrics callsite. That is 13.3% of these direct encoder samples and a
directional *callsite ceiling*, not an expected speedup or TX4 attribution.
The TX4 prototype is supported separately by the Unit 1419 exact-output
runtime diagnostic in `wrk_docs/2026.09.25 - HLD - TX4 batching diagnostic.md`.

The transferred raw files matched these remote SHA-256 values before
line-ending normalization: samples
`f05190809f4647d6189558b837e1bd9e3068b077a6270e0cf21aa2591f8c9a5a`,
IP map `7007a7f094d2bbc3807cf64f2950a9a55d2c66d4606ed6e98395047a9756fc87`.
The checked-in LF-normalized SHA-256 values are
`c038981a44ef9d36fc04d7ab65305c04a540d654e495dc165d40866214c38d5b`
for samples and
`af8396c8e0bed426b76c0d45baa4fc54d7f16fa143901359478aa1b355bf7eae`
for the IP map.
The 238,589,788-byte full dumper output was deliberately not checked in;
its SHA-256 was `7f1d8d7ee3219f3168c7d094e30741d39f5639f055e7e3a4c1806cf474cad73f`.
The first attempt to stream-filter the older 1 GB Unit 1426 trace timed out
with partial rows, so no old-trace line weights were accepted.

## Bounded instruction attribution of `inter.rs:5590`

Using only the saved IP map and the exact EXE/PDB, DbgHelp resolved the
enclosing `reconstruct_via_round_trip` symbol at runtime address
`0x00007ff66d535ad0`, size `0x3fae`. Its 77 line-5590 IP rows collapsed
to 60 distinct instruction-start RVAs with 142 direct counts; all 142
matched exact instruction starts in a bounded `objdump` of that symbol.
I independently summed `line5590-instruction-attribution.csv`: 119 counts
on packed arithmetic, 18 on other vector abs/conversion/max instructions,
and only five on loads, compare, or branch. The dense RVA
`0x87a80`–`0x87c9c` sequence contains `vpunpck*`, `vpsubd`, `vpabsd`,
`vpmuludq`, and `vpaddq`, consistent with the inlined squared-error path
in `MetricAccumulator::add`. This is opcode/source-pattern inference, not
PDB proof of each inline frame; the PDB reports the enclosing source line.
The sampled native binary used EVEX-encoded 128-bit instructions and
extended XMM registers, so these counts do not predict AVX2-only codegen.

The two small CSVs were transferred with SHA-256 values
`56d0eb8e43161d7eb0fc1670751e6e672f28913727c02898a410f8b38375913e`
and `3b60509d2151646d59d0a119636f389c099772d83752eb8072be8bbffc143066`.
The checked-in LF-normalized RVA and instruction CSVs have SHA-256 values
`d06603b512566ff506a6b52bd691b266902dfd29219663fe173e9d20a98c5b8a`
and `7b7b499d78e4c54fdb891e0cb8b2942ff07aad5c3aef8fe45a21aa7d1fc7ef75`.
The unchanged extraction script has SHA-256
`8eff55f09924f46346a189596775f6770414c9f98b62bc6e0e7c54d59c5462fd` and names
its original registered Crucible scratch and must be reparameterized to
rerun after that scratch is removed; the archived tables are self-contained.
