# E1 — rvp9 vs libvpx realtime encoder gap analysis (foreman_cif)

Source: foreman_cif centre-crop 256x256, 64 frames @ 29.97002997002997 fps. rvp9/libvpx run as one-pass VBR (`vpxenc --rt --cpu-used=7 --lag-in-frames=0 --passes=1 --aq-mode=0`). BOTH bitstreams decoded through the SAME rvp9 decoder (oracle); all per-block stats use identical definitions.

Area-weighted % = fraction of coded 8x8 cells (so a 64x64 leaf counts 64x more than an 8x8 leaf). 'count' = number of distinct partition leaves. Mode/skip/ref stats are over INTER frames only.


## Operating point: 25 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 56.9 | 42.8 |
| keyframe bits | 19912 | 22824 |
| avg inter-frame bits | 1611 | 1090 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 254.8 | 250.2 |
| min / max | 240 / 255 | 201 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 440 | 0.7 | 1828 | 2.8 |
| 8x16 | 184 | 0.6 | 0 | 0.0 |
| 16x8 | 454 | 1.4 | 0 | 0.0 |
| 16x16 | 15199 | 94.2 | 13495 | 83.7 |
| 16x32 | 6 | 0.1 | 0 | 0.0 |
| 32x16 | 54 | 0.7 | 0 | 0.0 |
| 32x32 | 95 | 2.4 | 544 | 13.5 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.1 | 0.7 |
| NEARESTMV % | 63.8 | 81.4 |
| NEARMV % | 1.8 | 3.9 |
| ZEROMV % | 25.7 | 5.1 |
| NEWMV % | 8.5 | 8.8 |
| skip=1 % | 82.8 | 94.3 |
| ref LAST % | 89.4 | 92.7 |
| ref GOLDEN % | 10.6 | 6.6 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 50 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 66.8 | 69.1 |
| keyframe bits | 39560 | 41168 |
| avg inter-frame bits | 1638 | 1687 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 253.9 | 238.5 |
| min / max | 218 / 255 | 171 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 520 | 0.8 | 1764 | 2.7 |
| 8x16 | 218 | 0.7 | 0 | 0.0 |
| 16x8 | 482 | 1.5 | 0 | 0.0 |
| 16x16 | 14920 | 92.5 | 12619 | 78.2 |
| 16x32 | 10 | 0.1 | 0 | 0.0 |
| 32x16 | 68 | 0.8 | 0 | 0.0 |
| 32x32 | 143 | 3.5 | 763 | 18.9 |
| 64x64 | 0 | 0.0 | 1 | 0.1 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.1 | 0.7 |
| NEARESTMV % | 63.1 | 74.5 |
| NEARMV % | 2.0 | 4.6 |
| ZEROMV % | 26.0 | 4.3 |
| NEWMV % | 8.8 | 15.8 |
| skip=1 % | 82.5 | 91.9 |
| ref LAST % | 89.9 | 91.2 |
| ref GOLDEN % | 10.0 | 8.1 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 100 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 99.7 | 134.5 |
| keyframe bits | 68376 | 74536 |
| avg inter-frame bits | 2296 | 3376 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 239.4 | 204.9 |
| min / max | 130 / 252 | 114 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 1332 | 2.1 | 5804 | 9.0 |
| 8x16 | 462 | 1.4 | 0 | 0.0 |
| 16x8 | 872 | 2.7 | 0 | 0.0 |
| 16x16 | 14604 | 90.6 | 11509 | 71.4 |
| 16x32 | 8 | 0.1 | 0 | 0.0 |
| 32x16 | 68 | 0.8 | 0 | 0.0 |
| 32x32 | 93 | 2.3 | 792 | 19.6 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.4 | 1.0 |
| NEARESTMV % | 58.3 | 63.2 |
| NEARMV % | 3.2 | 6.2 |
| ZEROMV % | 25.7 | 3.8 |
| NEWMV % | 12.3 | 25.9 |
| skip=1 % | 75.8 | 86.5 |
| ref LAST % | 90.3 | 91.6 |
| ref GOLDEN % | 9.3 | 7.4 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 250 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 191.6 | 285.1 |
| keyframe bits | 68376 | 109608 |
| avg inter-frame bits | 5408 | 7924 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 179.2 | 149.9 |
| min / max | 130 / 194 | 60 / 185 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 7936 | 12.3 | 23912 | 37.1 |
| 8x16 | 2676 | 8.3 | 0 | 0.0 |
| 16x8 | 3354 | 10.4 | 0 | 0.0 |
| 16x16 | 10617 | 65.8 | 8430 | 52.3 |
| 16x32 | 12 | 0.1 | 0 | 0.0 |
| 32x16 | 40 | 0.5 | 0 | 0.0 |
| 32x32 | 102 | 2.5 | 430 | 10.7 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 2.8 | 2.0 |
| NEARESTMV % | 41.2 | 58.0 |
| NEARMV % | 4.8 | 7.7 |
| ZEROMV % | 22.8 | 3.2 |
| NEWMV % | 28.4 | 29.1 |
| skip=1 % | 72.2 | 77.5 |
| ref LAST % | 89.2 | 88.9 |
| ref GOLDEN % | 8.0 | 9.1 |
| ref ALTREF % | 0.0 | 0.0 |

### Per-frame bits & base_qindex (100 kbps)

| frame | kind | rvp9 bits | rvp9 q | rvp9 intra% | libvpx bits | libvpx q | libvpx intra% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | KEY | 68376 | 65 | 100 | 74536 | 57 | 100 |
| 1 | inter | 9760 | 130 | 2 | 8752 | 114 | 3 |
| 2 | inter | 4024 | 193 | 3 | 2816 | 186 | 2 |
| 3 | inter | 3152 | 217 | 1 | 4296 | 210 | 1 |
| 4 | inter | 2656 | 227 | 1 | 2992 | 237 | 0 |
| 5 | inter | 2392 | 231 | 1 | 2864 | 251 | 0 |
| 6 | inter | 2024 | 233 | 1 | 2808 | 255 | 0 |
| 7 | inter | 2192 | 233 | 1 | 1936 | 255 | 0 |
| 8 | inter | 2840 | 234 | 0 | 1768 | 255 | 0 |
| 9 | inter | 2048 | 239 | 0 | 688 | 255 | 0 |
| 10 | inter | 2592 | 239 | 0 | 15224 | 139 | 12 |
| 11 | inter | 2120 | 243 | 0 | 1048 | 239 | 0 |
| 12 | inter | 2648 | 243 | 0 | 1712 | 229 | 0 |
| 13 | inter | 2608 | 247 | 0 | 2152 | 228 | 0 |
| 14 | inter | 2280 | 251 | 0 | 2104 | 230 | 1 |
| 15 | inter | 1992 | 252 | 0 | 1736 | 233 | 1 |
| 16 | inter | 2016 | 251 | 0 | 1808 | 232 | 1 |
| 17 | inter | 1744 | 251 | 0 | 1528 | 232 | 0 |
| 18 | inter | 2144 | 249 | 0 | 1560 | 229 | 0 |
| 19 | inter | 1696 | 249 | 0 | 1416 | 226 | 0 |
| 20 | inter | 1936 | 246 | 0 | 19536 | 134 | 7 |
| 21 | inter | 2024 | 245 | 0 | 1016 | 222 | 0 |
| 22 | inter | 2464 | 245 | 0 | 1760 | 211 | 0 |
| 23 | inter | 2256 | 248 | 0 | 1888 | 210 | 0 |
| 24 | inter | 2296 | 249 | 0 | 1728 | 210 | 0 |
| 25 | inter | 2008 | 250 | 0 | 1440 | 209 | 0 |
| 26 | inter | 1816 | 250 | 0 | 1536 | 205 | 0 |
| 27 | inter | 1520 | 248 | 0 | 1256 | 202 | 0 |
| 28 | inter | 1712 | 243 | 0 | 1336 | 196 | 0 |
| 29 | inter | 1720 | 240 | 0 | 1768 | 191 | 0 |
| 30 | inter | 2112 | 237 | 0 | 15000 | 135 | 3 |
| 31 | inter | 1880 | 237 | 0 | 1288 | 190 | 0 |
| 32 | inter | 2552 | 236 | 0 | 2048 | 184 | 1 |
| 33 | inter | 2416 | 239 | 1 | 2368 | 186 | 2 |
| 34 | inter | 2464 | 241 | 0 | 2040 | 191 | 1 |
| 35 | inter | 2168 | 244 | 0 | 2520 | 192 | 1 |
| 36 | inter | 2104 | 244 | 1 | 2320 | 199 | 1 |
| 37 | inter | 2136 | 244 | 0 | 2104 | 203 | 1 |
| 38 | inter | 1904 | 244 | 0 | 1888 | 205 | 1 |
| 39 | inter | 1864 | 243 | 0 | 1592 | 206 | 0 |
| 40 | inter | 2152 | 241 | 0 | 16928 | 126 | 7 |
| 41 | inter | 2184 | 241 | 1 | 1384 | 209 | 1 |
| 42 | inter | 2400 | 241 | 0 | 1640 | 206 | 0 |
| 43 | inter | 1464 | 243 | 0 | 1512 | 206 | 0 |
| 44 | inter | 2080 | 238 | 0 | 1784 | 204 | 1 |
| 45 | inter | 1960 | 238 | 0 | 1672 | 205 | 0 |
| 46 | inter | 1840 | 237 | 0 | 1664 | 205 | 0 |
| 47 | inter | 2416 | 236 | 1 | 2408 | 205 | 1 |
| 48 | inter | 2808 | 238 | 1 | 2344 | 213 | 3 |
| 49 | inter | 1944 | 244 | 0 | 1992 | 220 | 1 |
| 50 | inter | 2280 | 243 | 0 | 15696 | 133 | 1 |
| 51 | inter | 1712 | 244 | 0 | 1024 | 218 | 0 |
| 52 | inter | 2120 | 241 | 0 | 1624 | 207 | 0 |
| 53 | inter | 2040 | 241 | 1 | 2288 | 205 | 2 |
| 54 | inter | 2440 | 241 | 1 | 2264 | 209 | 0 |
| 55 | inter | 2080 | 244 | 1 | 2240 | 212 | 2 |
| 56 | inter | 1968 | 244 | 1 | 1720 | 216 | 0 |
| 57 | inter | 1920 | 243 | 0 | 1704 | 215 | 0 |
| 58 | inter | 2616 | 242 | 1 | 1744 | 213 | 1 |
| 59 | inter | 1856 | 246 | 0 | 1448 | 212 | 0 |
| 60 | inter | 2216 | 244 | 0 | 17464 | 130 | 1 |
| 61 | inter | 2080 | 244 | 0 | 1304 | 211 | 0 |
| 62 | inter | 1800 | 244 | 0 | 1720 | 207 | 0 |
| 63 | inter | 1976 | 242 | 0 | 1456 | 207 | 0 |
