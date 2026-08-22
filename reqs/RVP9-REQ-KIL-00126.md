# RVP9-REQ-KIL-00126 — Prove supported encoder flags across every encode path

- **State:** Draft
- **Priority:** Should
- **Area:** cli/rvp9enc option contract
- **Raised:** 2026-08-22T20:44:05Z
- **Discovery source:** Agent
- **Implemented-by:** —
- **Satisfied-by:** —
- **Violated-by:** —
- **Depends-on:** —
- **Design:** —
- **Flow:** heavy
- **Claimed-by:** —
- **State history:** Draft (2026-08-22T20:44:05Z, raised via `deltic reqs new` model=gpt-5.6-sol@high)

## Statement

Every rvp9enc flag advertised as supported must either reach its intended encoder behavior on every applicable encode path (one-pass, spatial SVC, temporal SVC, and two-pass) or fail loudly for an unsupported combination. The acceptance oracle must enumerate the live flag table and exercise cross-path combinations so a new path cannot silently ignore an existing option.

## Notes
