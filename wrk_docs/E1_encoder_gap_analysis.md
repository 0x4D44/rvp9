# E1 — rvp9 vs libvpx realtime encoder gap analysis (foreman_cif)

Source: foreman_cif centre-crop 256x256, 64 frames @ 29.97002997002997 fps. rvp9/libvpx run as one-pass VBR (`vpxenc --rt --cpu-used=7 --lag-in-frames=0 --passes=1 --aq-mode=0`). BOTH bitstreams decoded through the SAME rvp9 decoder (oracle); all per-block stats use identical definitions.

Area-weighted % = fraction of coded 8x8 cells (so a 64x64 leaf counts 64x more than an 8x8 leaf). 'count' = number of distinct partition leaves. Mode/skip/ref stats are over INTER frames only.


## Operating point: 25 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 51.6 | 42.8 |
| keyframe bits | 10256 | 22824 |
| avg inter-frame bits | 1587 | 1090 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 255.0 | 250.2 |
| min / max | 255 / 255 | 201 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 412 | 0.6 | 1828 | 2.8 |
| 8x16 | 178 | 0.6 | 0 | 0.0 |
| 16x8 | 430 | 1.3 | 0 | 0.0 |
| 16x16 | 14869 | 92.2 | 13495 | 83.7 |
| 16x32 | 12 | 0.1 | 0 | 0.0 |
| 32x16 | 72 | 0.9 | 0 | 0.0 |
| 32x32 | 171 | 4.2 | 544 | 13.5 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.1 | 0.7 |
| NEARESTMV % | 63.0 | 81.4 |
| NEARMV % | 1.9 | 3.9 |
| ZEROMV % | 26.4 | 5.1 |
| NEWMV % | 8.6 | 8.8 |
| skip=1 % | 82.8 | 94.3 |
| ref LAST % | 89.7 | 92.7 |
| ref GOLDEN % | 10.3 | 6.6 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 50 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 51.6 | 69.1 |
| keyframe bits | 10256 | 41168 |
| avg inter-frame bits | 1587 | 1687 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 255.0 | 238.5 |
| min / max | 255 / 255 | 171 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 412 | 0.6 | 1764 | 2.7 |
| 8x16 | 178 | 0.6 | 0 | 0.0 |
| 16x8 | 430 | 1.3 | 0 | 0.0 |
| 16x16 | 14869 | 92.2 | 12619 | 78.2 |
| 16x32 | 12 | 0.1 | 0 | 0.0 |
| 32x16 | 72 | 0.9 | 0 | 0.0 |
| 32x32 | 171 | 4.2 | 763 | 18.9 |
| 64x64 | 0 | 0.0 | 1 | 0.1 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.1 | 0.7 |
| NEARESTMV % | 63.0 | 74.5 |
| NEARMV % | 1.9 | 4.6 |
| ZEROMV % | 26.4 | 4.3 |
| NEWMV % | 8.6 | 15.8 |
| skip=1 % | 82.8 | 91.9 |
| ref LAST % | 89.7 | 91.2 |
| ref GOLDEN % | 10.3 | 8.1 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 100 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 66.7 | 134.5 |
| keyframe bits | 10256 | 74536 |
| avg inter-frame bits | 2098 | 3376 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 243.4 | 204.9 |
| min / max | 234 / 255 | 114 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 1012 | 1.6 | 5804 | 9.0 |
| 8x16 | 352 | 1.1 | 0 | 0.0 |
| 16x8 | 734 | 2.3 | 0 | 0.0 |
| 16x16 | 14556 | 90.3 | 11509 | 71.4 |
| 16x32 | 16 | 0.2 | 0 | 0.0 |
| 32x16 | 70 | 0.9 | 0 | 0.0 |
| 32x32 | 151 | 3.7 | 792 | 19.6 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.3 | 1.0 |
| NEARESTMV % | 58.5 | 63.2 |
| NEARMV % | 3.0 | 6.2 |
| ZEROMV % | 26.3 | 3.8 |
| NEWMV % | 11.9 | 25.9 |
| skip=1 % | 76.5 | 86.5 |
| ref LAST % | 90.3 | 91.6 |
| ref GOLDEN % | 9.4 | 7.4 |
| ref ALTREF % | 0.0 | 0.0 |

## Operating point: 250 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 154.7 | 285.1 |
| keyframe bits | 10256 | 109608 |
| avg inter-frame bits | 5082 | 7924 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 186.7 | 149.9 |
| min / max | 174 / 224 | 60 / 185 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 8x8 | 7192 | 11.1 | 23912 | 37.1 |
| 8x16 | 2544 | 7.9 | 0 | 0.0 |
| 16x8 | 3022 | 9.4 | 0 | 0.0 |
| 16x16 | 11043 | 68.5 | 8430 | 52.3 |
| 16x32 | 12 | 0.1 | 0 | 0.0 |
| 32x16 | 38 | 0.5 | 0 | 0.0 |
| 32x32 | 101 | 2.5 | 430 | 10.7 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 2.5 | 2.0 |
| NEARESTMV % | 41.9 | 58.0 |
| NEARMV % | 4.6 | 7.7 |
| ZEROMV % | 23.5 | 3.2 |
| NEWMV % | 27.5 | 29.1 |
| skip=1 % | 71.9 | 77.5 |
| ref LAST % | 88.8 | 88.9 |
| ref GOLDEN % | 8.7 | 9.1 |
| ref ALTREF % | 0.0 | 0.0 |

### Per-frame bits & base_qindex (100 kbps)

| frame | kind | rvp9 bits | rvp9 q | rvp9 intra% | libvpx bits | libvpx q | libvpx intra% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | KEY | 10256 | 195 | 100 | 74536 | 57 | 100 |
| 1 | inter | 1672 | 255 | 0 | 8752 | 114 | 3 |
| 2 | inter | 2592 | 242 | 0 | 2816 | 186 | 2 |
| 3 | inter | 1848 | 246 | 0 | 4296 | 210 | 1 |
| 4 | inter | 2288 | 244 | 1 | 2992 | 237 | 0 |
| 5 | inter | 1904 | 245 | 0 | 2864 | 251 | 0 |
| 6 | inter | 1960 | 244 | 0 | 2808 | 255 | 0 |
| 7 | inter | 1664 | 244 | 0 | 1936 | 255 | 0 |
| 8 | inter | 2512 | 241 | 0 | 1768 | 255 | 0 |
| 9 | inter | 2136 | 244 | 0 | 688 | 255 | 0 |
| 10 | inter | 2296 | 244 | 0 | 15224 | 139 | 12 |
| 11 | inter | 2152 | 245 | 0 | 1048 | 239 | 0 |
| 12 | inter | 2384 | 245 | 0 | 1712 | 229 | 0 |
| 13 | inter | 2672 | 247 | 0 | 2152 | 228 | 0 |
| 14 | inter | 2088 | 251 | 0 | 2104 | 230 | 1 |
| 15 | inter | 2232 | 251 | 0 | 1736 | 233 | 1 |
| 16 | inter | 1736 | 252 | 0 | 1808 | 232 | 1 |
| 17 | inter | 2112 | 249 | 0 | 1528 | 232 | 0 |
| 18 | inter | 1880 | 249 | 0 | 1560 | 229 | 0 |
| 19 | inter | 1744 | 248 | 0 | 1416 | 226 | 0 |
| 20 | inter | 1792 | 245 | 0 | 19536 | 134 | 7 |
| 21 | inter | 2240 | 243 | 0 | 1016 | 222 | 0 |
| 22 | inter | 2464 | 244 | 0 | 1760 | 211 | 0 |
| 23 | inter | 2240 | 246 | 0 | 1888 | 210 | 0 |
| 24 | inter | 2384 | 247 | 0 | 1728 | 210 | 0 |
| 25 | inter | 1952 | 249 | 0 | 1440 | 209 | 0 |
| 26 | inter | 1824 | 248 | 0 | 1536 | 205 | 0 |
| 27 | inter | 1432 | 246 | 0 | 1256 | 202 | 0 |
| 28 | inter | 2232 | 240 | 0 | 1336 | 196 | 0 |
| 29 | inter | 1400 | 241 | 0 | 1768 | 191 | 0 |
| 30 | inter | 2280 | 234 | 1 | 15000 | 135 | 3 |
| 31 | inter | 2112 | 236 | 1 | 1288 | 190 | 0 |
| 32 | inter | 2304 | 236 | 0 | 2048 | 184 | 1 |
| 33 | inter | 2560 | 237 | 1 | 2368 | 186 | 2 |
| 34 | inter | 2504 | 240 | 0 | 2040 | 191 | 1 |
| 35 | inter | 1936 | 243 | 0 | 2520 | 192 | 1 |
| 36 | inter | 2328 | 242 | 0 | 2320 | 199 | 1 |
| 37 | inter | 1944 | 243 | 0 | 2104 | 203 | 1 |
| 38 | inter | 2096 | 242 | 0 | 1888 | 205 | 1 |
| 39 | inter | 1816 | 242 | 0 | 1592 | 206 | 0 |
| 40 | inter | 2216 | 240 | 0 | 16928 | 126 | 7 |
| 41 | inter | 2176 | 241 | 1 | 1384 | 209 | 1 |
| 42 | inter | 2472 | 241 | 0 | 1640 | 206 | 0 |
| 43 | inter | 1448 | 244 | 0 | 1512 | 206 | 0 |
| 44 | inter | 2392 | 238 | 0 | 1784 | 204 | 1 |
| 45 | inter | 1736 | 240 | 0 | 1672 | 205 | 0 |
| 46 | inter | 2056 | 238 | 0 | 1664 | 205 | 0 |
| 47 | inter | 2496 | 238 | 0 | 2408 | 205 | 1 |
| 48 | inter | 2560 | 240 | 0 | 2344 | 213 | 3 |
| 49 | inter | 1992 | 244 | 1 | 1992 | 220 | 1 |
| 50 | inter | 2432 | 243 | 0 | 15696 | 133 | 1 |
| 51 | inter | 1616 | 245 | 0 | 1024 | 218 | 0 |
| 52 | inter | 1816 | 241 | 0 | 1624 | 207 | 0 |
| 53 | inter | 2296 | 240 | 1 | 2288 | 205 | 2 |
| 54 | inter | 2344 | 241 | 1 | 2264 | 209 | 0 |
| 55 | inter | 2240 | 242 | 1 | 2240 | 212 | 2 |
| 56 | inter | 2016 | 243 | 1 | 1720 | 216 | 0 |
| 57 | inter | 2104 | 242 | 0 | 1704 | 215 | 0 |
| 58 | inter | 2632 | 242 | 1 | 1744 | 213 | 1 |
| 59 | inter | 1888 | 246 | 0 | 1448 | 212 | 0 |
| 60 | inter | 2032 | 245 | 0 | 17464 | 130 | 1 |
| 61 | inter | 2096 | 245 | 0 | 1304 | 211 | 0 |
| 62 | inter | 1760 | 245 | 0 | 1720 | 207 | 0 |
| 63 | inter | 1672 | 243 | 0 | 1456 | 207 | 0 |
