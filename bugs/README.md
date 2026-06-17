# Bug ledger

This `bugs/` directory is the bug ledger - one markdown file per bug,
filename == the bug ID, with the current state a field inside each file.
deltic's Repos->Bugs drill-in assembles the ledger in memory from `bugs/*.md`;
there is no master table by design, and there is no `BUGS.md` - never create one.

## ID grammar

`PREFIX-TYPE-HOST-NNNNN` for newly minted bugs; legacy `PREFIX-NNNNN` IDs stay
valid and are never rewritten.

- **PREFIX** - this repo's acronym: `RVP9`.
- **TYPE** - `BUG` (the only type minted now); `REQ` is reserved for future use.
- **HOST** - the minting machine's normalized hostname (uppercase, `[A-Z0-9]`
  only, not truncated).
- **NNNNN** - a per-host sequence, zero-padded to 5, never reused.

## Allocating an ID (per-host)

Derive HOST at runtime (`echo %COMPUTERNAME%` on Windows, `hostname` on
WSL/Linux), normalize it, then take `max(SEQ) + 1` over the existing
`RVP9-BUG-<HOST>-*` files. No central allocator, no cross-machine coordination.

Two actors sharing a HOST token (worktrees on one box, or a Windows host and its
WSL sharing a hostname) can mint the same SEQ from stale views; that surfaces as
a loud git add/add conflict at integration. Renumber on conflict.

## States and transitions

`Open -> Blocked -> Fixed -> Closed`. Transitions are dated and attributed
(who + when + commit) and append-only. Record a `State history:` line on every
transition; never silently overwrite a prior state.

## Two-eyes closure

The fixer flips a bug to **Fixed**; a different person re-verifies and flips
**Fixed -> Closed**. The fixer never closes their own bug.

## File format

Labelled lines (`- **State:**`, `- **Severity:**`, `- **Area:**`,
`- **Raised:**`, `- **State history:**`) under an `# <ID> - <title>` H1, then
`## Observation` and `## Notes` prose. The state is a field inside the file;
transition it by editing that field, never by renaming the file.
