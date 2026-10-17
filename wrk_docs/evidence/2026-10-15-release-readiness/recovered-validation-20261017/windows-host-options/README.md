# Alternative Windows host metadata — 2026-10-17

Bounded read-only investigation after Crucible canonical host-load guard refusals (lead supplied1.86 and0.802 busy-core results, zero metrics). Used fleet-comms skill and `deltic hosts` inventory; no board posts, peer messages, process control, transfers, worktree creation, builds, installs or remote file writes. SSH used existing trust with BatchMode and StrictHostKeyChecking=yes. Only RVP9-root path metadata, git identity/status count, installed toolchain names, ffmpeg location and one CPU metadata sample were queried. No credentials or unrelated repository content read.

## Outcome

No currently ready safe alternative was verified in this bounded search. KILN has the strongest existing RVP9 setup evidence, but it is incomplete for the canonical performance campaign. ANVIL has a potentially quiet instantaneous sample but lacks a verified RVP9 setup and the pinned toolchain. This is not a claim that no other host can be prepared or that the whole fleet was inspected.

| Host | Verified prerequisites | Missing or unverified | Capacity observation |
| --- | --- | --- | --- |
| KILN | Native Windows x86-64/Intel Ultra9 285HX,24 logical CPUs; SSH works; Rust1.95.0 MSVC installed/default; ffmpeg found; existing D:\language\rvp9-2 clean at75c9cc8c035c652a3354c150af42803d813257cf; vpxenc.exe5671968 bytes and vpxdec.exe5542677 bytes; Foreman45621044 bytes | Tree/Ducks/Town canonical720p files absent at inspected repo; target/release rvp9enc.exe and encode-headtohead.exe absent. Reference pin/config/hashes, Foreman hash, current candidate equivalence, production PGO artifact/manifest, compiler/linker and other required tooling not qualified. D:\language\rvp9 also exists at a different old HEAD with9 status lines, no inspected references/corpus/binaries. | One instantaneous5:55:32UTC sample9% of24 logical processors (~2.16 busy cores); not canonical idle evidence and not encouraging for0.75 guard. |
| ANVIL | Native Windows x86-64/i7-14700KF,28 logical CPUs; SSH works; ffmpeg found | No Cargo.toml at D:\language\rvp9-2, D:\language\rvp9 or C:\language\rvp9. Other possible repo locations were not searched. Rust1.95 absent from installed toolchain list (default1.98); candidate/reference/corpus/PGO unknown. | One instantaneous5:55:36UTC sample0%; this does not establish sustained idleness or any canonical60-second guard pass. |

Inventory lists additional native Windows hosts (ANNEAL, NMI, QUENCH, TEMPER, NOMAD, LIGHTNINGCLAUDE, THUNDERCLAUDE), but their RVP9 prerequisite availability and current capacity were not verified here. Cards are peer metadata and13–16h old for most of these hosts; NOMAD card is5days old. No broad host/path sweep performed.

## Decision boundary

Do not move the current timing claim to either inspected host. Any later host choice still requires candidate/reference/corpus/toolchain/production-PGO identity, its own native qualification, and the unchanged canonical60-second <=0.75busy-logical-core guard. Missing heavyweight processes, hardware size or a zero instantaneous sample is not an idle pass. Results from Crucible do not transfer to Intel hosts. Preparing/transfer/build work would be a separate authorized action, not performed here.

## Evidence

inventory.raw.txt records the bounded fleet inventory. Sanitized host cards omit SSH keys/trust details. metadata-probe.ps1 is the exact read-only script; checks.json records bounded commands/exits/raw hashes. Both SSH checks exited0. Original PowerShell CLIXML/CRLF output is preserved in *-metadata.raw.txt; *-metadata.json parses only the JSON object without changing the originals. No runtime benchmark was run. integrity.json hashes every other compact file. Files are final.
