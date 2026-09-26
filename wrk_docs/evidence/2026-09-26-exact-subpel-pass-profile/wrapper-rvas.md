# Inlined wrapper RVA classification

RVA counts from the exact integrated EXE. All values are sampled IPs, not
function timings.

| Clip | Stage | RVA=count |
| --- | --- | --- |
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
