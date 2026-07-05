# Bugs - `bugs/` directory ledger

This `bugs/` directory is the defect ledger: one markdown file per bug,
filename == the bug ID, with the current state a field inside each file.
deltic assembles the ledger in memory from `bugs/*.md`; there is no master table
by design, and there is no `BUGS.md` - never create one.

Accepted enhancements, design follow-ups, and durable behavioural specs live in
the sibling `reqs/` ledger as `RVP9-REQ-...` files, not in `bugs/`.

## ID grammar

`PREFIX-TYPE-HOST-NNNNN` for newly minted bugs; legacy `PREFIX-NNNNN` IDs stay
valid and are never rewritten.

- **PREFIX** - this repo's acronym: `RVP9`.
- **TYPE** - `BUG` in this ledger. The sibling `reqs/` ledger mints `REQ`.
- **HOST** - the minting machine's normalized hostname: uppercase, `[A-Z0-9]`
  only, not truncated.
- **NNNNN** - a per-host sequence, zero-padded to at least 5 digits, never
  reused.

## Allocating an ID

Derive HOST at runtime (`echo %COMPUTERNAME%` on Windows, `hostname` on
WSL/Linux), normalize it, list existing `RVP9-BUG-<HOST>-*` files, take
`max(NNNNN)+1`, and commit that one new file. No central allocator, no lock.

Two actors sharing a HOST token can mint the same ID from stale views; that
surfaces as a loud git add/add conflict at integration. Renumber on conflict.

## States and transitions

`Open -> Blocked -> Fixed -> Closed`. Transitions are dated and attributed in
the append-only `State history:` line. State is a field inside the file; never
rename a file to change state.

## Two-eyes closure

The fixer flips a bug to **Fixed** after landing the fix with regression
coverage. A different verifier re-checks the original observation and flips
**Fixed -> Closed**. The fixer never closes their own bug.

## Priority vs severity

`Priority` is fix urgency: `Must`, `Should`, or `Could`. `Severity` is defect
impact: `Critical`, `High`, `Medium`, or `Low`. Automation picks by priority
first, then severity.

## Ownership and attempts

The current owner lives in the bug file. `Owner role: human` parks automation
for a named human owner. `Owner role: fix` and `Owner role: verify` are leased
automation claims; automation writes the matching owner fields. The unowned
marker is `-` in every owner field.

`Attempts: fix=N, doubt=N, indeterminate=N` records unattended retry history.
`Verify retry after`, `Held branch`, and `Legacy fixed run` are automation
fields; use `-` when unset.

## File format

Each `bugs/<ID>.md` uses a modeled header above the first `## ` heading:

```markdown
# RVP9-BUG-HOST-00001 - Short title

- **State:** Open
- **Priority:** Should
- **Severity:** High
- **Area:** encoder/foo
- **Raised:** YYYY-MM-DD
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
- **State history:** Open (YYYY-MM-DD, raised by ...)

## Observation
<symptom, repro, expected vs actual>

## Fix
<accepted fix summary and verification notes>

## Notes
<other notes, links, failed attempts>
```

The modeled header fields and `State history` are the machine-readable truth.
Free prose after the first `## ` heading is not parsed as schema.
