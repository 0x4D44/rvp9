<!-- deltic:quality-projection:v1 -->
<!-- schema: 1 -->
<!-- projection-digest: sha256:f01fd6ec29648ca2b51a4eef4e1b7ec38e9330f6408fbb7b0fb411a570f67829 -->
<!-- view: strategy -->

# Quality strategy — RVP9 reviewable final-release candidate

- **Programme:** `next-release`
- **Lifecycle:** `Draft`
- **Target date:** —
- **Activated by/on:** — / —

## Stakeholders

- Agent evidence consumer/reviewer: delight—machine-readable commands, inputs, and hashes; discussed—scope and provenance gaps; good enough—reproducible local checks; dealbreaker—stale or ambiguous evidence.
- Human repository partner and release authority (identity pending): delight—defensible exact-source evidence; discussed—the four boundary choices; good enough—all confirmed gates reviewed; dealbreaker—unsupported pass claims.

## Valued outcomes

- Make release claims reproducible, traceable to inputs and tools, and limited to what the evidence proves.
- Produce a reviewable final-release candidate with exact-source, platform-specific evidence for every confirmed acceptance gate.

## Dealbreakers

- A required correctness, compression, real-time, or safety result fails or is represented as passing without applicable evidence.
- The candidate violates the approved feature boundary or a supported native platform has an unresolved release-blocking failure.
## Band anchors

- **Weak:** At least one in-scope release outcome has a confirmed failure, an unresolved material defect, or missing applicable evidence.
- **Sound:** Every confirmed release outcome has applicable exact-candidate evidence on each required native platform; all exceptions are explicitly approved and claims match the evidence.
- **Strong:** Sound is met, independent repeats corroborate the key results, and representative inputs cover the confirmed production surface without a material evidence gap.

## Stakeholder-owned Area targets

| Area | Kind | Importance | Required | Conf(R) | Confirmed | Impact | Milestone blocking | Rationale |
|---|---|---|---|---|---|---|---:|---|
| `api-and-local-gates` — Public API documentation and local regression gates | Ility | Medium | Sound | Low | — / — | Medium | true | The objective requires accurate public API documentation and repeatable local regression commands without hosted CI. |
| `bounded-reliability` — Bounded encode/decode reliability | Ility | High | Sound | Low | — / — | High | true | The current release objective requires exact-source bounded smoke runs; the previously approved billion-iteration fuzz and 72-hour Intel soak remain deferred. |
| `codec-correctness` — VP9 bitstream correctness | Functional | High | Strong | Low | — / — | High | true | Provisional from the final-candidate objective and PLAN §2; target confidence remains Low because native scalar coverage and some secondary criteria are unresolved. |
| `compression-quality` — Production compression quality | Ility | High | Strong | Low | — / — | High | true | The objective requires the production CBR surface to meet the documented aggregate and important-clip limits; target confidence remains Low pending the exact release boundary. |
| `corpus-representativeness` — Production corpus coverage | Ility | High | Strong | Low | — / — | High | false | PLAN §2 names AOM, WebRTC-representative, and livestream content; accept a narrower set only through an explicit scope decision. |
| `decoder-throughput` — Decoder throughput | Ility | Medium | Strong | Low | — / — | Medium | false | PLAN §3 proposes decoder throughput within 2× libvpx, while Q.1.A calls its head-to-head ledger informational; the release-gate choice is unresolved. |
| `encoder-throughput` — Production encoder throughput | Ility | High | Strong | Low | — / — | High | true | The objective requires the full current-source relative-speed matrix; targets remain provisional until the final modes and corpus are confirmed. |
| `feature-boundary` — Production feature boundary | Functional | High | Strong | Low | — / — | High | false | PLAN §2 lists frame dropping and EXT_RATECTRL; the proposed candidate boundary omits them. Keep the target provisional until the release owner resolves this scope choice. |
| `frame-latency` — Production frame latency | Ility | High | Strong | Low | — / — | High | true | PLAN §2 specifies 33/16 ms median and 50/25 ms p99 limits, while scripts use 33.333/16.667 ms medians; do not relax until the release owner confirms the bar. |
| `memory-safety` — Memory safety and unsafe contracts | Ility | High | Strong | Low | — / — | High | true | The goal requires bounded safety assurance across parser, decoder, threading, and SIMD boundaries; whether the full shared-infrastructure unsafe audit is a release gate remains open. |
| `package-consumption` — Package build and consumption | Functional | High | Strong | Low | — / — | High | true | The objective requires package/build verification on both native platforms, including tests of the extracted archives. |
