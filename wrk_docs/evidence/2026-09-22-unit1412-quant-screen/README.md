# Unit 1412 — AVX2 all-zero quantizer screen

`screen-b-*` and `screen-c-*` preserve two complete three-clip, three-pair,
two-rotation candidate/control runs on Crucible. All output hashes match in
both runs. Several individual encodes stalled by 3–7× in their second
rotations, despite allowing moderate background load. Thus the apparent
0.895943× and 0.993877× full-run corpus ratios are not reliable speed gains.
The clean first-rotation clip medians were 1.0094×/0.9878×/0.9914× in B and
1.0057×/1.0116×/0.9855× in C (Ducks/Tree/Town). C's geometric mean was
1.000867×: effectively flat. The shortcut was not retained.

The control is exact integrated `cdbf7a6a`; the candidate is the same source
plus the AVX2 all-zero patch. The exact executable hashes, settings, clip
hashes, load samples, raw times, and output hashes are in the manifest and
CSVs. There was no strict idle-load cap; moderate background use is allowed,
but these asymmetric multi-second stalls swamp the sub-percent signal.
`rejected-candidate.patch.gz` preserves the exact uncommitted code experiment
(decompressed SHA-256 `92c90b24298f11a060aa641561b2d91f63e734f659959b0d5575be036ad7ccd9`);
the shipped source files remain unchanged. The direct RVP9/libvpx measurement
therefore remains Unit 1410's 1.227734×, with no fresh libvpx run warranted.
