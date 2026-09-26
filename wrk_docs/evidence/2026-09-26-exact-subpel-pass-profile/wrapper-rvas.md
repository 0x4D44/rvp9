# Inlined wrapper RVA classification

RVA counts from the exact integrated EXE. All values are sampled IPs, not
function timings.

| Clip | Stage | RVA=count |
| --- | --- | --- |
| Ducks | Full horizontal | `0x3c550=1`, `0x3c56d=1`, `0x3c577=1`, `0x3c581=2`, `0x3c59d=1`, `0x3c5a5=1`, `0x3c5b3=2`, `0x3c5b8=1`, `0x3c635=1`, `0x3c638=1`, `0x3c652=2`, `0x3c675=1`, `0x3c6a8=2`, `0x3c6e0=1`, `0x3c6eb=1` |
| Ducks | Full vertical | `0x3c780=1`, `0x3c7b1=1`, `0x3c805=1`, `0x3c831=1`, `0x3c958=2` |
| Ducks | Vertical-only | `0x3ca43=1`, `0x3ca54=1`, `0x3ca94=2`, `0x3caa2=1`, `0x3caaa=1`, `0x3cacc=1`, `0x3cada=2`, `0x3cae9=1`, `0x3caed=1`, `0x3cb01=2`, `0x3cb14=1`, `0x3cb25=1`, `0x3cb2b=1`, `0x3cb2f=1`, `0x3cb35=1`, `0x3f969=1` |
| Ducks | Horizontal-only | `0x3cb8a=1`, `0x3cbbf=1`, `0x3cbec=1`, `0x3cbf4=1`, `0x3cc0a=1`, `0x3cc22=1`, `0x3cc42=2`, `0x3cc4d=3`, `0x3cc64=1`, `0x3cc6a=2`, `0x3cc6e=1` |
| Ducks | Variance | `0x3cca3=3`, `0x3ccb6=1`, `0x401b6=1`, `0x401cf=1` |
| Ducks | Ambiguous setup | `0x3c460=1`, `0x3c48e=1`, `0x3c4a9=2` |
| Tree | Full horizontal | `0x3c5a9=1`, `0x3c6e5=1` |
| Tree | Full vertical | `0x3c776=1`, `0x3c9c2=1`, `0x3f600=1` |
| Tree | Vertical-only | `0x3ca87=1` |
| Tree | Horizontal-only | `0x3cbd4=2`, `0x3cbf0=1` |
| Tree | Variance | `0x3cca3=1`, `0x3ffc1=1` |
| Tree | Ambiguous setup | `0x3c456=1`, `0x5e780=1`, `0x5e784=1` |
| Town | Full horizontal | `0x3c671=1`, `0x3c6b1=1` |
| Town | Full vertical | `0x3c6fc=1`, `0x3c7ce=1`, `0x3c94a=1` |
| Town | Horizontal-only | `0x3cbf4=1`, `0x3cc0a=1`, `0x3cc64=2` |
| Town | Variance | `0x3cca3=1`, `0x3f711=1` |

The remaining ambiguous Tree samples are in generic entry/TLS setup at
`0x6a353=1` and `0x6a39b=1`. The `0x3c5a9` bounds guard is within the
first-pass loop. The `0x3c*` RVAs above are the 16×16 wrapper, `0x3f*` the
8×8 wrapper, and `0x5e*` the 32×32 wrapper.

For Ducks, the exact 16×16 disassembly boundaries were full horizontal
`0x3c4d6–0x3c6f3`, full vertical `0x3c6f4–0x3ca0c`, vertical-only
`0x3ca11–0x3cb52`, horizontal-only `0x3cb53–0x3cc82`, and variance
`0x3cc83–0x3ccbc`. The three 8×8 wrapper RVAs above were inspected
individually. Entry/setup samples below `0x3c4d6` stay ambiguous.
