# RVP9-BUG-KIL-00095 — Y4M frame parameters are scanned without a length bound

- **State:** Open
- **Priority:** Should
- **Severity:** Medium
- **Area:** io/y4m parser
- **Raised:** 2026-08-22T14:33:46Z
- **Discovery source:** Agent
- **Owner:** deltic:manual
- **Owner role:** fix
- **Owner run:** fix-20261016T002533Z-f257c74d
- **Owner host:** flux
- **Owner branch:** task/bug-RVP9-BUG-KIL-00095-run-fix-20261016T002533Z-f257c74d
- **Owner base:** 742342e14b13cf7ad2993f31ae63f9253c6574bc
- **Owner fingerprint:** -
- **Owner since:** 2026-10-16T00:25:33Z
- **Owner until:** 2026-10-16T04:25:33Z
- **Verify retry after:** -
- **Held branch:** -
- **Legacy fixed run:** -
- **Attempts:** fix=0, doubt=0, indeterminate=0
- **State history:** Open (2026-08-22T14:33:46Z, raised via `deltic bugs new`)

## Observation

crates/rvp9-io/src/y4m.rs:380-393 handles an extended FRAME header by issuing one-byte File::read_exact calls until newline, with no length cap. A malformed input can force an arbitrary file-length syscall loop, consume frame payload as parameters, and return only at a coincidental newline or EOF. The global Y4M header has a 4 KiB cap, and official libvpx bounds frame-header extensions to 79 bytes, so unbounded scanning is not required for compatibility. Expected: reject an overlong frame header promptly with InvalidData. Fix with a small bounded buffered line read and preserve support for normal short parameters. Add overlong and no-newline regressions that prove rejection at the cap without consuming body bytes. Estimated effort: small. Confirmed independently by performance, security, correctness skeptic, and the lead; no tests or application were run.

## Fix

<unfixed — raised only>

## Notes
