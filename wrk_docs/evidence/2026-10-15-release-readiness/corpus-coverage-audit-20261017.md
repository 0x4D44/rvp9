# Release corpus coverage audit — 2026-10-17

Candidate source rechecked at 13:51 UTC: `origin/main` is `fb5c0c917038db6c640b3dbc7c2b84b567233333`. The task-worktree `scripts/qualify-production-encoder.sh` is byte-identical to that source. This audit checks corpus inputs and declared coverage; it is not a quality run.

## Inputs present in the task worktree

`rg --files --no-ignore -g '*.y4m' tests/corpus` found six source Y4M files. The four profile-0 files required by the production qualification script are present and match its declared SHA-256 hashes and headers:

| File | Header | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `in_to_tree_420_720p50.y4m` | `W1280 H720 F50:1 Ip A1:1` | 691,203,035 | `81166517a3054c789923535b5da23fbbb9bbd7fd4c28dac83918ac351e121e74` |
| `ducks_take_off_420_720p50.y4m` | `W1280 H720 F50:1 Ip A1:1` | 691,203,035 | `0e09aa923ce72e2f6d1c978c3c1cc8dd9588d2ede819081f040213e361970efc` |
| `old_town_cross_420_720p50.y4m` | `W1280 H720 F50:1 Ip A1:1` | 691,203,035 | `0fbc2ec0c552ad693528b2e49409c7449880b9230055a7696346473d1aaf4123` |
| `foreman_cif.y4m` | `W352 H288 F30000:1001 Ip A128:117` | 45,621,044 | `b0e3c35eff9fe1da91521a02eb9f0e3ef4bc5aec341ecef1567cd87f1711267b` |

The other two Y4M files are `profile1/foreman_c422.y4m` (`W352 H288 F30000:1001 Ip A128:117 C422`) and `profile1/foreman_c444.y4m` (`W352 H288 F30000:1001 Ip A128:117 C444`). They add chroma formats for the same Foreman scene, not new scene types. The directory also contains encoded IVF/WebM fixtures and MD5 ledgers; these are decoder fixtures, not additional source scenes for the production quality matrix. `tests/corpus/sources/` contains its README but no downloaded source clips.

The production qualification matrix uses In To Tree, Ducks and Old Town for 720p30 and 1080p30; its 1080p60 rows use In To Tree and Old Town. Foreman is the source for a separate AQ gain-parity oracle. Denoiser inputs are generated from these sources at runtime with a fixed seeded noise filter; they do not add scene provenance or content classes. These mappings come from `target_clips`, `required_corpus_inputs` and `prepare_inputs` in the candidate-identical qualification script.

The repository's source-corpus README describes the fetch scripts as a Derf seed, with Blender Open Movies and optional self-recorded material planned separately. Its HLD allows an optional self-recorded WebRTC session when Arthur is both endpoints, while excluding YouTube extracts and captures from Meet/Hangouts. `notes/g8-followups.md` retains that boundary. None of those optional sources is present in the six-file input inventory. The updated release `PLAN.md` in this task branch additionally names an AOM test set, WebRTC-representative corpus and livestream corpus; it does not choose versions, clips, provenance records or a licensed subset. The PLAN file is a task-branch documentation change and differs from `origin/main`.

## Source leads, not selected corpus

- AOM's current AVM/AV2 CTC guide identifies CTC v9.0 as its current default and says the selected version changes the test clips and encoding parameters. AOM's 2024 testing-subgroup update points to AOM CTC test sequences hosted on Xiph. These sources do not establish whether PLAN's phrase “AOM test set” means that AV2 set or a different AOM sequence collection. The Xiph Derf index describes its collection as generally believed freely redistributable while warning that some clips have additional restrictions. A release set therefore needs an explicit set/version, clip list and per-clip rights check. [AOM AVM CTC guide](https://aomedia.googlesource.com/avm/+/c6219d5be733fb295437f0a329a5cef90b7ec301/tools/convexhull_framework/USER_GUIDE.md), [AOM testing-subgroup update](https://aomedia.org/docs/AVM.Test.Result.Update.ICIP.2024.pdf), [Xiph Derf collection](https://media.xiph.org/video/derf/).
- Microsoft's Video Conferencing Dataset is a concrete conferencing-content candidate: its project describes 160 1080p30, 10-second clips across talking-head, opaque-background, blurred-background and mobile scenarios, and publishes the dataset under CDLA-Permissive 2.0. That supports conferencing-content coverage; it does not by itself prove that Arthur considers it “WebRTC-representative” or that the full 160-clip set is the intended size. [Microsoft VCD README](https://github.com/microsoft/VCD/blob/main/README.md).
- Xiph's Derf index has a separate Gaming collection sourced from Twitch and marks its clips Copyright. This is a livestream-content lead, not a license-cleared fixture; no such clip is selected or present in the current inputs. Any use needs a rights review. [Xiph Derf collection](https://media.xiph.org/video/derf/).
- No project-selected livestream clip list or license record is named in the current corpus files or the inspected corpus-source docs. The Xiph Gaming lead does not resolve that gap.

## Acceptance consequence

The present corpus is a verified, narrow input set for the existing matrices. It does not establish the newer AOM, WebRTC-representative and livestream coverage claim. Adding media depends on decisions about which AOM set/version and subset to use, whether self-recorded sessions or a dataset such as VCD satisfy the WebRTC wording, and whether a livestream source such as Xiph's copyright-labeled Gaming collection can be used or another rights-cleared source is needed. No media was downloaded, modified or added for this audit. Until those choices are made and the selected inputs are hash/header-verified, corpus coverage remains incomplete and no compression-quality campaign can claim the full declared coverage.

## Reproduction

From the task worktree, the relevant checks were:

```sh
git ls-remote origin refs/heads/main
git diff --quiet origin/main -- scripts/qualify-production-encoder.sh
rg --files --no-ignore -g '*.y4m' tests/corpus
head -n 1 tests/corpus/<required-file>.y4m
shasum -a 256 tests/corpus/<required-file>.y4m
stat -f '%z' tests/corpus/<required-file>.y4m
```

The four recorded hashes match `corpus_expected_sha256` and the headers match `corpus_expected_header` in the candidate-identical script. The matrix and denoiser derivation can be rechecked with `target_clips`, `required_corpus_inputs` and `prepare_inputs` in that script.
