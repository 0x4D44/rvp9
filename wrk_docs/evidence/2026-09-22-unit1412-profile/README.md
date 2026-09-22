# Unit 1412 — x86 picker source-line attribution

This is a fresh native Crucible ETW trace of integrated `cdbf7a6a` using
twenty serial 60-frame 1920x1080@30 CBR Ducks encodes, CPU-used 6, one
thread, tile 0, AQ0, noise 0, lag 0, and KF60. All twenty output IVF hashes
match. The exact executable, PDB, input, ETL, and intermediate hashes are in
`manifest.txt`.

The symbol report assigns 4,796,666 weighted samples (10.9565% of encoder
symbol weight) to the exact `pick_and_simulate_inter_encode_partitions::closure$10`.
The raw IP export contains 4,797 direct samples from that symbol. The
matching PDB maps 4,532 of them (94.48%) to nonzero source lines; 265 map to
line zero. `picker-ip-lines.csv` retains per-IP weights and module-relative
addresses; `picker-line-aggregate.csv` is their source-line grouping.

The largest mapped line is `crates/rvp9-encoder/src/lib.rs:11471`, the call to
`choose_inter_luma_tx_size`: 1,224 samples, 25.52% of the direct callback and
3.09% of all 39,628 raw encoder samples. This reverses the earlier assumption
that the hot callback was dominated by A4 descent setup. Optimized PDB line
tables can charge an inlined region to its callsite, so 3.09% is a directional
selector ceiling, not a measured saving or an internal block breakdown.

Every sample was normalized from its process image base to the exact PDB
image base. All twenty processes happened to load at one relocated base; the
mapper still applied the per-PID normalization. This profile supplies a
specific next target, not a new candidate/control timing result. The last
direct RVP9/libvpx ratio remains 1.227734x (Unit 1410).
