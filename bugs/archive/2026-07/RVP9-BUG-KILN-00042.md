# RVP9-BUG-KILN-00042 - Y4M header geometry can overflow allocation math

- **State:** Closed
- **Priority:** Should
- **Severity:** High
- **Area:** io/y4m parser
- **Raised:** 2026-07-07
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T172419Z-p68822-n594265000-c1 branch=task/bug-RVP9-BUG-KILN-00042-run-fix-20260730T172419Z-p68822-n594265000-c1 code=795b0b387fef6bbbffad9f009c8a2a559bda0b9f gate=manual) -> Closed (2026-07-31, independently verified and closed by a two-eyes pass (Claude Opus 5))

## Observation

`parse_header` accepts invalid or hostile Y4M numeric fields, then `read_cropped` uses those fields in unchecked size arithmetic and allocations.

`crates/rvp9-io/src/y4m.rs:80`-`87` parse `W`, `H`, and `F` as `u32` values and lines `113`-`119` return them without requiring non-zero dimensions, non-zero framerate parts, a sane maximum, or a chroma-compatible source geometry. `Y4mHeader::framerate` divides by `fps_den` at line `42`.

`read_cropped` casts the header dimensions to `usize` at lines `161`-`164`, multiplies `src_w * src_h` and `(src_w / 2) * (src_h / 2)` at lines `170`-`172`, and allocates `body` from the derived `frame_body` at line `176`. Large dimensions can overflow or trigger huge allocations before the encoder config validation layer gets a chance to reject them.

The source-dimension domain check is also incomplete. Lines `154`-`158` require even crop dimensions, but odd 4:2:0 source dimensions are still accepted and then parsed with floor-divided chroma sizes at line `171`, so the reader consumes the wrong frame body size for such input.

## Fix

Open. Reject zero `W`, `H`, `fps_num`, and `fps_den`; reject or explicitly support odd 4:2:0 source dimensions; cap supported frame byte counts; and use checked multiplication/addition for every derived allocation size.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Related in shape to closed encoder config bounds work, but this bug is at the Y4M parser boundary before encoder validation.

## Verification (2026-07-31, independent two-eyes closure)

Verifier: Claude Opus 5, not the fixer of this bug.

Every clause of the observation is covered: zero W/H/fps_num/fps_den rejected; odd 4:2:0 source
rejected (the wrong-frame-body-size case); dimensions capped at 16384, verified to match
`rate_control::config::MAX_CODED_DIMENSION`; all derived sizes use `checked_mul`/`checked_add`
against a 384 MiB frame cap; every allocation goes through `try_reserve`
(`crates/rvp9-io/src/y4m.rs:47-105`, `:244-250`, `:297-302`, `:416`).
`tests::parse_header_rejects_zero_geometry_and_framerate`,
`::parse_header_rejects_odd_or_oversized_420_geometry` and
`::read_cropped_rejects_unallocatable_frame_count_without_panicking` all observed passing; the
last panicked pre-fix on `Vec::with_capacity(usize::MAX)`.
Over-strict-guard check: no 4:2:0 y4m is checked in (the two present are C422/C444, rejected
earlier by the chroma gate), and the gitignored corpus is 352x288 and 1280x720 -- even and well
inside the envelope.
Two residuals noted, neither blocking. (1) Real shipped-CLI behaviour change: `rvp9enc` now
refuses odd-dimension 4:2:0 y4m that libvpx's `y4minput.c` accepts with ceil chroma. Rejecting
is safer and this record permitted "reject or explicitly support", but CLAUDE.md requires the
departure in `notes/divergence-ledger.md` and no entry exists -- recorded in `scratchpad.md`.
(2) Per-frame bytes are capped but total memory is not: `read_cropped(path, w, h, n)` still
allocates n x frame_bytes.

**Gate state on this tree** (trunk `4abe0d8` + docs commit `3a7aba2`, Windows x86_64
AVX2 host, conformance vectors fetched via `scripts/fetch-vectors.sh`):
`cargo fmt --all -- --check` green; `cargo clippy --workspace --all-targets --locked`
green, zero warnings; `cargo test --workspace --locked --no-fail-fast` 3092 passed,
47 ignored, 3 failed. All three failures are attributed and tracked, and none touches
this bug: two are `w_tpl` panics from the reopened RVP9-BUG-KILN-00020, and one is
`parallel_tile_decode_reuses_isolated_scratch_buffers`, split out as
RVP9-BUG-KILN-00062. The tree is NOT green and is not claimed to be.
