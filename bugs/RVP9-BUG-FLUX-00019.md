# RVP9-BUG-FLUX-00019 — Official skip-vector ledgers with ordinal filenames fail the conformance harness

- **State:** Fixed
- **Priority:** Must
- **Severity:** Low
- **Area:** harness/conformance
- **Raised:** 2026-08-12T14:08:03Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** verify
- **Owner run:** verify-20261019T064801Z-f9ea5bb7
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-FLUX-00019-run-verify-20261019T064801Z-f9ea5bb7
- **Owner base:** 97d6f3d31b1e66701d0552962b72239d4772a880
- **Owner fingerprint:** sha256:7f263d069fbe50ffc5e739cdb1000716b0e1da2d2e6f3a0c5f925029f6c90f4d
- **Owner since:** 2026-10-19T06:48:01Z
- **Owner until:** 2026-10-19T10:48:01Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-12T14:08:03Z, raised via `deltic bugs new` model=gpt-5.6-sol@high) -> Fixed (2026-08-12T14:13:24Z, deltic:auto role=fix run=fix-20260812T140920Z-p98653-n291313000-c1 branch=task/bug-RVP9-BUG-FLUX-00019-run-fix-20260812T140920Z-p98653-n291313000-c1 code=ecaea8ba gate=manual)

## Observation

The pinned libvpx ledgers for vp90-2-19-skip.webm and vp90-2-19-skip-01.webm use 01.yuv-style filenames. crates/rvp9-harness/src/md5file.rs rejects them because it requires WxH-NNNN in every filename, so the strict official-corpus gate fails before decoding. Reproduced in the 307-vector run and an isolated 7-vector rerun on 2026-08-12.

## Fix

<unfixed — raised only>

## Notes
