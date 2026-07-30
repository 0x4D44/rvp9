# RVP9-BUG-KILN-00042 - Y4M header geometry can overflow allocation math

- **State:** Fixed
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
- **State history:** Open (2026-07-07, raised by Codex code-review pass over `crates/rvp9-io/` at `6128772`) -> Fixed (2026-07-30, deltic:auto role=fix run=fix-20260730T172419Z-p68822-n594265000-c1 branch=task/bug-RVP9-BUG-KILN-00042-run-fix-20260730T172419Z-p68822-n594265000-c1 code=795b0b387fef6bbbffad9f009c8a2a559bda0b9f gate=manual)

## Observation

`parse_header` accepts invalid or hostile Y4M numeric fields, then `read_cropped` uses those fields in unchecked size arithmetic and allocations.

`crates/rvp9-io/src/y4m.rs:80`-`87` parse `W`, `H`, and `F` as `u32` values and lines `113`-`119` return them without requiring non-zero dimensions, non-zero framerate parts, a sane maximum, or a chroma-compatible source geometry. `Y4mHeader::framerate` divides by `fps_den` at line `42`.

`read_cropped` casts the header dimensions to `usize` at lines `161`-`164`, multiplies `src_w * src_h` and `(src_w / 2) * (src_h / 2)` at lines `170`-`172`, and allocates `body` from the derived `frame_body` at line `176`. Large dimensions can overflow or trigger huge allocations before the encoder config validation layer gets a chance to reject them.

The source-dimension domain check is also incomplete. Lines `154`-`158` require even crop dimensions, but odd 4:2:0 source dimensions are still accepted and then parsed with floor-divided chroma sizes at line `171`, so the reader consumes the wrong frame body size for such input.

## Fix

Open. Reject zero `W`, `H`, `fps_num`, and `fps_den`; reject or explicitly support odd 4:2:0 source dimensions; cap supported frame byte counts; and use checked multiplication/addition for every derived allocation size.

## Notes

Confirmed by all three skeptic passes in the 2026-07-07 IO review. Related in shape to closed encoder config bounds work, but this bug is at the Y4M parser boundary before encoder validation.
