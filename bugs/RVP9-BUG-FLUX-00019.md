# RVP9-BUG-FLUX-00019 — Official skip-vector ledgers with ordinal filenames fail the conformance harness

- **State:** Closed
- **Priority:** Must
- **Severity:** Low
- **Area:** harness/conformance
- **Raised:** 2026-08-12T14:08:03Z
- **Discovery source:** Agent
- **Owner:** -
- **Owner role:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner fingerprint:** -
- **Owner since:** -
- **Owner until:** -
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T14:13:24Z, deltic:auto role=fix run=fix-20260812T140920Z-p98653-n291313000-c1 branch=task/bug-RVP9-BUG-FLUX-00019-run-fix-20260812T140920Z-p98653-n291313000-c1 code=ecaea8ba gate=manual) -> Closed (2026-10-19T06:51:34Z, deltic:manual role=verify run=verify-20261019T064801Z-f9ea5bb7 verified_fix_run=fix-20260812T140920Z-p98653-n291313000-c1 verdict=close)

## Observation

The pinned libvpx ledgers for vp90-2-19-skip.webm and vp90-2-19-skip-01.webm use 01.yuv-style filenames. crates/rvp9-harness/src/md5file.rs rejects them because it requires WxH-NNNN in every filename, so the strict official-corpus gate fails before decoding. Reproduced in the 307-vector run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

### Verification summary (2026-10-19T06:51:34Z, deltic:auto run=verify-20261019T064801Z-f9ea5bb7 verified_fix_run=fix-20260812T140920Z-p98653-n291313000-c1 verdict=close)

Verifier note: Manual verify: Independent verification: the recorded SHA ecaea8ba is absent; matching landed commit 74ca855b47a32ea0f66909fb53dfe2061de5de45 is an ancestor of origin/main. It adds ordinal-only .yuv parsing alongside resolution-bearing names, preserves sequential-index validation, records missing dimensions as zero, and makes verbose conformance output use decoded dimensions. cargo test -p rvp9-harness md5file::tests::read_md5_file_accepts_official_ordinal_only_filenames -- --exact passed with one selected test. Fails-before removed only the ordinal parse fallback; the selected test failed on ...

## Notes
