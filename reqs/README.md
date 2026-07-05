# Requirements — `reqs/` directory ledger

**Purpose.** A requirement is a **living spec**: a durable, attributed statement
of required behaviour plus a re-runnable **oracle** that proves it. One markdown
file per requirement under `reqs/`, the filename == the requirement ID; deltic
assembles the ledger in memory by globbing `reqs/*.md` and surfaces it as the F2
Repos→Reqs pill. There is **no master table** by design, and **no `REQS.md` —
never create one.** This mirrors the `bugs/` ledger (see `bugs/README.md`); the
difference is that a requirement never auto-closes — once `Satisfied` it can flip
to `Violated` and back, so the ledger answers *"what must this system always do,
and is it still doing it?"*.

**ID grammar.** `RVP9-REQ-HOST-NNNNN`.
- **RVP9** — this repo's acronym.
- **REQ** — the type token (the bugs ledger uses `BUG`; `REQ` is reserved for
  this ledger). deltic never inspects the token — both ledgers share one ID
  parser.
- **HOST** — the minting machine's hostname, uppercased with every non
  `[A-Z0-9]` char stripped, **not truncated**.
- **NNNNN** — a per-host sequence, zero-padded to at least 5 digits.
- Legacy `RVP9-NNNNN` ids (pre-conversion) remain valid, verbatim, forever.

**Per-host allocation.** To raise a requirement: derive HOST at runtime
(`echo %COMPUTERNAME%` on Windows, `hostname` on Linux/WSL), list the existing
`RVP9-REQ-<HOST>-*` files, take `max(NNNNN)+1`, and commit that one new
file. No central allocator, no lock.

**Merge safety (R-SAMEHOST).** The filename **stem is the identity**. Two actors
sharing a HOST token (worktrees on one box, a Win+WSL pair with the same
hostname) can mint the same id from stale views — this surfaces **loudly** as an
add/add conflict at integration, never a silent collision. The resolver
renumbers. If a file's H1 id disagrees with its filename, deltic keys the record
by the **filename** and carries a "differs from filename" warning in the ledger
note.

**States & transitions.**

```
Draft ──► Accepted ──► Integrated ──► Satisfied ⇄ Violated
              │            │              ▲
              │            └─(human REJECT: git revert)─► Accepted
              └──► Implemented ──► Satisfied (legacy/manual/heavy)
any state ──► Retired (terminal)
```

| State | Meaning |
|---|---|
| **Draft** | Proposed, under discussion. |
| **Accepted** | Binding, not yet met (outstanding). |
| **Implemented** | Code is present, but the oracle is missing/incomplete — regression-blind. (The legacy/manual/heavy path; the light auto-land lands on `Integrated`.) |
| **Integrated** | The light `/loop` runner built it oracle-first and deltic ff-pushed the gated-green pair to `origin/main` — code **and** oracle landed, **awaiting human batch acceptance**. |
| **Satisfied** | Code **and** a complete oracle, **accepted by a human** (a second pair of eyes read the source and re-ran the oracle). |
| **Violated** | Built then breached — the alarm. Reachable from `Satisfied` *or* directly from `Implemented`. |
| **Retired** | Withdrawn or superseded (terminal). |

State is a **field inside the file** (`- **State:** …`) — never rename a file to
change its state. Transitions are append-only and dated in the `State history:`
line. deltic only *classifies* the declared state (a case-insensitive
contains-scan, `retired → violated → integrated → satisfied → implemented →
accepted → draft`); it never derives or enforces a transition and never mutates a
file. An unrecognised state word classifies as `Other` so it surfaces in the
counts rather than vanishing.

**Field set.** Modeled fields are the `- **Field:**` bullets **above the first
`## ` heading**; everything from the first level-2 heading on is free prose (it
may quote a `- **State:** …` line without poisoning the record).

| Label | Meaning |
|---|---|
| `State` | One of the seven states above. |
| `Priority` | MoSCoW: `Must` / `Should` / `Could` (an unknown/blank cell → `—`). |
| `Area` | Free text (the module/feature the requirement governs). |
| `Raised` | First-raised date (else the first `State history` date). |
| `Implemented-by` | req→code refs — see traceability. |
| `Satisfied-by` | req→oracle refs — see traceability. |
| `Violated-by` | req→bug refs (ledger IDs) — see traceability. |
| `Flow` | `light` / `heavy` — which build flow the `/loop` runner uses (blank/unknown → `—`). |
| `Claimed-by` | The loop's in-flight marker while a runner holds it, else `—`. A `box:<host> run:<run>` value is a **deltic-written machine claim** (cleared automatically with the owner block); human claims stay free text. |
| `Owner`, `Owner run/host/branch/base/since/until` | deltic's **cross-site auto-land lease** (box exclusion): principal, run id, host, build branch, trunk base SHA, and the lease window — renewed on a ~45-min lease while building. All `-` = unowned; a partially-filled block reads as malformed. deltic-written; humans never edit these. |
| `Auto attempts` | Durable failed-auto-build counter, deltic-written; the auto-land pick parks a req at 3. Absent/garbage reads as 0. |
| `State history` | Dated, append-only transitions. |

**Traceability (three labelled lines).** Multiple refs are comma-separated on the
one line; empty items are dropped; an empty value or `—` / `-` means "none yet".

- `Implemented-by:` — **req→code**. Free-text path/symbol refs (displayed, not
  resolved).
- `Satisfied-by:` — **req→oracle**. Free-text test path / command refs. Its
  **non-emptiness is the honesty check's signal**. When automation is genuinely
  impractical, name a documented re-runnable manual check prefixed `manual:`.
- `Violated-by:` — **req→bug**. Ledger IDs. deltic shape-checks each, then
  resolves it against the sibling `bugs/` directory (dangling / stale-alarm).

**Honesty rule + two-eyes process.** deltic **mechanically flags** a requirement
marked `Satisfied` with an **empty `Satisfied-by`** (no oracle). It does not run
the oracle, so the human gate on truth is the **two-eyes rule**: a requirement
moves `Implemented → Satisfied` only after a **second pair of eyes** confirms the
named oracle genuinely exists and passes.

**RVP9 flow override.** Do not set `Flow: light` in this repo yet. The current
deltic auto-land runner is shaped for a root crate: it stages `src/**`, accepts
one root `tests/<file>.rs` oracle, and runs `cargo test --test <stem>`. `rvp9-2`
is a virtual workspace whose source and integration tests live under
`crates/**`, with codec-specific gates and gitignored vector/corpus data. Until
deltic has an explicit RVP9 workspace policy, Gate 1 should use `Flow: heavy` for
accepted requirements, or leave Drafts as `Flow: —` while they are not ready.
Manual/heavy builders move `Accepted -> Implemented`; the builder never
self-promotes to `Satisfied`.

**File format.** Each `reqs/<ID>.md`:

```markdown
# RVP9-REQ-HOST-00001 — Short imperative requirement title

- **State:** Satisfied
- **Priority:** Must
- **Area:** <module/feature>
- **Raised:** 2026-01-01
- **Implemented-by:** crates/rvp9-encoder/src/foo.rs
- **Satisfied-by:** cargo test -p rvp9-encoder proves_it
- **Violated-by:** —
- **Flow:** heavy
- **Claimed-by:** —
- **Owner:** -
- **Owner run:** -
- **Owner host:** -
- **Owner branch:** -
- **Owner base:** -
- **Owner since:** -
- **Owner until:** -
- **Auto attempts:** 0
- **State history:** Draft (2026-01-01) → Accepted (2026-01-01) → Implemented (2026-01-01, `sha`) → Satisfied (2026-01-02, verified)

## Statement
The system must <durable behavioural spec — not a task>.

## Notes
<design links, the oracle command, history of breaches — free prose; NOT parsed>
```

**The `/loop` runner & the auto-land flow.** The reqs ledger doubles as the
**work queue** for the `/loop` runner. The human gates *batches*, not every diff:

- **Gate 1 (human, batch triage).** Review `Draft` reqs; set `Priority`, `Flow`,
  and — the acceptance oracle — a `Satisfied-by` the build will be proven against,
  then move to `Accepted` (or `Retired` for one-shot tasks). This is the cull.
- **The loop drains *ready* reqs** — `deltic reqs --ready` = `Accepted`, a
  routable `Flow` (not `—`), and **unclaimed**:
  - **`Flow: light`** → currently not enabled for RVP9. In repos where it is
    enabled, the runner builds it **oracle-first** (a cross-family
    agent authors the failing `tests/<oracle>` blind to the impl; a build agent
    makes it green editing `src/` only) and **deltic composes the commit, gates
    it, and fast-forward-pushes the green pair to `origin/main`**, moving the req
    `Accepted → Integrated`. The agents never push — deltic owns git.
  - **`Flow: heavy`** → the runner reports it to the human and skips.
- **Gate 2 (human, batch acceptance — the only writer here).** A human reads the
  landed source + oracle and re-runs it. On confirmation, edit `Integrated →
  Satisfied` and integrate the reqs-only ledger edit through the docs lane; batch
  several accepts in one commit. To reject, `git revert` every commit carrying the
  req's `Req-Id` trailer (`git log --grep='Req-Id: <ID>'` is the audit path), then
  set the req back to `Accepted` with an explaining note.

**The `Accepted → Integrated` ledger edit — deltic-authored.** The build branch
is **forbidden to touch `reqs/`**: the landing fence rejects any branch whose
diff includes the ledger, so an agent can never rewrite the file that carries
the lease. After the code lands (pre-landing rebase + guarded ff-push), **deltic
itself authors the terminal flip** as a deltic-owned commit at origin:

- set `- **State:** Integrated`;
- append `→ Integrated (<YYYY-MM-DD>)` to the existing `- **State history:**`
  line (the dated-transition shape this ledger parses);
- fill `- **Implemented-by:**` from the build report;
- clear the `Owner*` block and the box `Claimed-by` marker.

deltic likewise writes the lease edits at origin: **claim** (before the build —
the `Owner*` block plus the box `Claimed-by` marker), **renew** (`Owner until`
bumps on the ~45-min lease while building), and **release** (on failure or skip;
a genuine build failure also increments `Auto attempts`, and the pick parks a
req at 3 attempts).

A `Claimed-by:` value of the form `box:<host> run:<run>` is that deltic-written
machine claim — cleared automatically together with the owner block. Human
free-text `Claimed-by` values are never touched by automation.

The write split, in short: **deltic owns the machine fields** — the `Owner*`
block, `Auto attempts`, the box `Claimed-by`, and the terminal
`Accepted → Integrated` flip. **Humans own the decisions** — triage to
`Accepted`, batch acceptance to `Satisfied`, `Retired`.

> This ledger was bootstrapped by `deltic reqs init`. Edit it freely — it is your
> repo's own copy, not a fleet-managed cache.
