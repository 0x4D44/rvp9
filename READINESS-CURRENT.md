<!-- deltic:quality-projection:v1 -->
<!-- schema: 1 -->
<!-- projection-digest: sha256:f01fd6ec29648ca2b51a4eef4e1b7ec38e9330f6408fbb7b0fb411a570f67829 -->
<!-- view: readiness -->

# Quality-bar readiness

**0 of 11 above-bar-and-known — NOT READY**

- **Programme lifecycle:** `Draft`
- **Readiness reasons:** `programme_not_active`, `areas_not_above_bar_and_known`
- **Confirmed parked Areas:** 0
- **Pending scope decisions:** 0
- **Unavailable methods:** none

## Blocked routes

- none

| Area | Scope | Actual vs bar | Confidence / evidence | Action | Readiness |
|---|---|---|---|---|---|
| `api-and-local-gates` — Public API documentation and local regression gates | targeted Medium / Sound | Weak vs Sound (gap 1) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `bounded-reliability` — Bounded encode/decode reliability | targeted High / Sound | Sound vs Sound (gap 0) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `codec-correctness` — VP9 bitstream correctness | targeted High / Strong | Sound vs Strong (gap 1) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `compression-quality` — Production compression quality | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `corpus-representativeness` — Production corpus coverage | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `decoder-throughput` — Decoder throughput | targeted Medium / Strong | Weak vs Strong (gap 2) | target=Low, actual=Low, channels= | Stakeholder | not ready |
| `encoder-throughput` — Production encoder throughput | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `feature-boundary` — Production feature boundary | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `frame-latency` — Production frame latency | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `memory-safety` — Memory safety and unsafe contracts | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |
| `package-consumption` — Package build and consumption | targeted High / Strong | Weak vs Strong (gap 2) | target=Low, actual=Medium, channels= | Stakeholder | not ready |

This is **quality-bar readiness only**. It does not compute open-bug readiness or authorize a release.
