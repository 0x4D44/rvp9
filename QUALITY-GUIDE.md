<!-- deltic:quality-guide:v1 -->
# Quality programme guide

The authored source is .deltic-quality-program.toml. Deltic owns its schema,
derivation, ordering, JSON contract, and generated projections. Do not hand-edit
QUALITY-STRATEGY-CURRENT.md, QUALITY-MAP.md, BOARD.md, or
READINESS-CURRENT.md.

This programme is a Draft proposal for the RVP9 reviewable final-release candidate.
Its target confidences are Low, activation is not recorded, and the four release-boundary
choices remain open. The generated quality-bar readiness view does not replace the
final-release checklist, include the bug ledger, or authorize a release.

## First investigation method: evidence-audit

Use this method to assess one Area against the current candidate. Start with the exact
product-source commit and the runner revision; verify that the evidence identifies both.
For each cited record, check its platform, toolchain, inputs, command, exit status, and
raw-output hashes. Recompute hashes where the artifact is available. Separate real product
runs from self-tests, stubs, injections, and historical results. Mark missing platforms,
configurations, clips, and ignored tests as gaps; never infer a pass from a successful
harness self-test. If a concrete defect appears, preserve the failing state and route it
through bug-tracking; do not take over another agent's assigned fix. If the question is
scope or a target bar, leave it as stakeholder work.

Mission records belong under wrk_docs/quality/missions/<MSN-ID>/; use the fleet quality
Mission, Charter, and sentinel-bearing Session templates. The Session is the canonical
assessment, and each Area update must cite the exact actor, channel, method, intent, and
probe. Do not retrofit older ad-hoc results as Sessions.

## Local testing mechanics

Run from an isolated task worktree at a recorded candidate. Prefer the existing commands
in the local release qualification runbook; use deltic timeout for bounded runs and
close stdin. Keep native macOS Apple Silicon and Windows x86-64 results separate.
Preserve commands, source identities, host/toolchain details, input hashes, exit status,
and raw-output hashes in the evidence directory. Do not add hosted CI or dependencies,
and do not claim full-corpus, production quality, throughput, latency, or long-duration
assurance from synthetic or bounded smoke results.
