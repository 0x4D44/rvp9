# Unit 1407 pinned-libvpx comparison

This screen compares the Unit 1407 candidate (`857c80af`) with the pinned
`vpxenc.exe` shipped by the acceptance workspace. It uses the same 1080p30
clips, 60-frame limit, CPU-used 6 CBR 1260 settings, five? No: three paired
three paired samples, two rotations, and the relaxed Crucible load policy as screen B.

The candidate remains byte-identical to its RVP9 control, while libvpx emits
its own expected bitstream. Candidate time divided by pinned libvpx time:

| clip | paired candidate/libvpx |
| --- | ---: |
| ducks | 1.360468x |
| intree | 1.232105x |
| oldtown | 1.154496x |
| corpus | **1.246167x** |

The pinned-libvpx comparison therefore confirms a substantial remaining parity
gap; it is not a release-quality parity claim. The manifest records the pinned
binary and all input hashes.
