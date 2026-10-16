# Spatial quality evidence audit

This is historical intake evidence from 2026-10-15/16, not qualification of the merged release candidate. The original manifest reports source `3dd780ccfb04358dcbb1dab0bfe766041e071cd1` and `git_status_clean=no`; the full dirty source state was not captured. All original files here are byte-for-byte copies, including the malformed CSV.

## Completion and observed gates

The campaign finished 96 curves, 240 per-layer gates, 2,016 unique expected full/layer streams and 6,048 decoder invocations. All 240 summary numbers match the individual raw BD-rate CSVs. Each curve contains six anchor and six test measurements per layer. All cross-decoder rows report matching nonempty rvp9dec/vpxdec output hashes, ffmpeg success and PASS. Stream enumeration matches every expected bitrate and full/layer suffix.

| Mode | Curves / layers | Recorded mean BD-rate | Recorded maximum | Ceiling |
| --- | --- | --- | --- | --- |
| Production CBR | 48 / 120 | -4.544458% | +0.488431% | +0.5% per clip/layer |
| Experimental spatial VBR | 48 / 120 | +0.174914% | +22.630645% | +25% per clip/layer |

Worst CBR: `cbr-720p30-old_town_cross-cpu6-s3`, SL0. Worst VBR: `vbr-1080p60-in_to_tree-cpu8-s2`, SL1. Fifty-five VBR layer results exceed the production +0.5% ceiling; its experimental status is essential.

## Separate evidence defects

**RVP9-BUG-FLU-00041: malformed summary CSV.** The header has 13 fields; all 240 data rows have 18 because the six-rate list is unquoted. Raw files remain unchanged. The audit identifies fields by their fixed prefix and suffix instead of accepting the declared schema.

**RVP9-BUG-FLU-00042: numeric evidence.** The harness uses uncentered floating-point normal equations (`crates/rvp9-harness/src/bdrate.rs:294`, raw powers through degree six, then partial-pivot Gauss-Jordan). For `cbr-1080p30-ducks_take_off-cpu8-s2`, SL0, the retained current Rust diagnostic evaluates the same rounded six-point inputs as **-0.714401618768%**. An independent Python standard-library Decimal calculation at 80-digit precision yields **-1.788791640314%**. Both original and centered cubic bases agree; the maximum disagreement across all 240 fits is less than 1.78e-64 percentage points. This is a same-input numeric discrepancy of about 1.07439 percentage points.

The full campaign separately recorded **-3.383557%** for that layer. The CSV rounds the original measured PSNR values, so the campaign result and the rounded-input diagnostic must not be treated as an exact replay of identical full-precision inputs. Numeric conditioning makes rounding material. The diagnostic tests the same published degree-three least-squares semantics using all six points, natural-log bitrate, common PSNR overlap and exact polynomial integration; it does not introduce a different acceptance metric.

The independent rounded-data diagnostic changes no gate outcome: all 120 CBR layers remain below +0.5% (maximum +0.488430125972%); all 120 VBR layers remain below +25% (maximum +22.630639916833%). This diagnostic is not an acceptance substitution or release waiver.

## Provenance and limits

The original manifest records Darwin ARM64, Rust 1.95.0, ffmpeg 8.1.2, libvpx commit `12f3a2ac603e8f10742105519e0cd03c3b8f71dd`, qualification-script and binary SHA256s, three canonical source hashes and eight processed-input hashes. During this audit every recorded file hash matched its currently available original. The PGO encoder hash is `cec75b2ee57b832e08cdf1153774401ab2df8ba7a8aa2c962f02053215c67b80`; this establishes identity of the observed binary, not equivalence with another candidate.

The script deleted generated streams and decoded frames after checking them. Raw stream redecoding is therefore unavailable from this compact archive. All 96 original cross-decoder logs contain ffmpeg nonmonotonic-DTS messages despite success exits; ffmpeg establishes decoding success, while only rvp9dec/vpxdec output hashes establish pixel equality. The cross-decoder logs and large inputs/streams are not copied here. No build or campaign was rerun for this audit.

## Reproducing the diagnostic

Run the retained Python script with this archive directory as its single argument; its output JSON records all 240 layers, raw rounded points, per-input CSV SHA256s and original/centered high-precision results. The archived JSON was originally generated against the scratch campaign and retains that historical absolute path.

The Rust probe source is copied verbatim from `.scratch/candidate-20261016/bdrate-numeric-probe.rs`; its relative `#[path]` assumes that original location. Restore it to that location before compiling against the current harness source. The original `bdrate-numeric-result.log` is retained verbatim. A second output, `bdrate-numeric-probe.log`, was captured by rerunning the already compiled small diagnostic binary during preservation, without rebuilding it. It is a diagnostic observation, not a campaign acceptance artifact.

`integrity.json` hashes every preserved file except itself and records copy origins; it also records the probe binary hash used for the new diagnostic observation.
