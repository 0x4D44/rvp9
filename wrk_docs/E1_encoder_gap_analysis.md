# E1 — rvp9 vs libvpx realtime encoder gap analysis (foreman_cif)

Source: foreman_cif centre-crop 256x256, 64 frames @ 29.97002997002997 fps. libvpx run as `vpxenc --rt --cpu-used=7 --passes=1 --aq-mode=0`. BOTH bitstreams decoded through the SAME rvp9 decoder (oracle); all per-block stats use identical definitions.

Area-weighted % = fraction of coded 8x8 cells (so a 64x64 leaf counts 64x more than an 8x8 leaf). 'count' = number of distinct partition leaves. Mode/skip/ref stats are over INTER frames only.


## Operating point: 25 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 127.0 | 42.7 |
| keyframe bits | 3896 | 22824 |
| avg inter-frame bits | 4243 | 1086 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 247.7 | 252.7 |
| min / max | 210 / 255 | 199 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 4x4 | 0 | 0.0 | 11 | 0.0 |
| 4x8 | 0 | 0.0 | 7 | 0.0 |
| 8x4 | 0 | 0.0 | 19 | 0.0 |
| 8x8 | 53212 | 82.5 | 371 | 0.6 |
| 8x16 | 0 | 0.0 | 36 | 0.1 |
| 16x8 | 0 | 0.0 | 44 | 0.1 |
| 16x16 | 957 | 5.9 | 442 | 2.7 |
| 16x32 | 0 | 0.0 | 22 | 0.3 |
| 32x16 | 0 | 0.0 | 42 | 0.5 |
| 32x32 | 55 | 1.4 | 174 | 4.3 |
| 32x64 | 0 | 0.0 | 2 | 0.1 |
| 64x64 | 103 | 10.2 | 919 | 91.2 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.0 | 95.5 |
| NEARESTMV % | 78.9 | 3.8 |
| NEARMV % | 0.0 | 0.1 |
| ZEROMV % | 21.1 | 0.1 |
| NEWMV % | 0.0 | 0.5 |
| skip=1 % | 74.0 | 2.1 |
| ref LAST % | 100.0 | 2.8 |
| ref GOLDEN % | 0.0 | 1.4 |
| ref ALTREF % | 0.0 | 0.4 |

## Operating point: 50 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 159.7 | 71.0 |
| keyframe bits | 7272 | 41168 |
| avg inter-frame bits | 5297 | 1752 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 237.8 | 238.3 |
| min / max | 190 / 255 | 164 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 4x4 | 0 | 0.0 | 39 | 0.1 |
| 4x8 | 0 | 0.0 | 38 | 0.1 |
| 8x4 | 0 | 0.0 | 115 | 0.2 |
| 8x8 | 54232 | 84.1 | 1112 | 1.7 |
| 8x16 | 0 | 0.0 | 144 | 0.4 |
| 16x8 | 0 | 0.0 | 90 | 0.3 |
| 16x16 | 718 | 4.5 | 469 | 2.9 |
| 16x32 | 0 | 0.0 | 30 | 0.4 |
| 32x16 | 0 | 0.0 | 38 | 0.5 |
| 32x32 | 75 | 1.9 | 174 | 4.3 |
| 32x64 | 0 | 0.0 | 16 | 0.8 |
| 64x32 | 0 | 0.0 | 10 | 0.5 |
| 64x64 | 97 | 9.6 | 886 | 87.9 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.0 | 95.3 |
| NEARESTMV % | 79.9 | 2.6 |
| NEARMV % | 0.0 | 0.5 |
| ZEROMV % | 20.1 | 0.1 |
| NEWMV % | 0.0 | 1.5 |
| skip=1 % | 69.9 | 15.9 |
| ref LAST % | 100.0 | 3.7 |
| ref GOLDEN % | 0.0 | 0.6 |
| ref ALTREF % | 0.0 | 0.4 |

## Operating point: 100 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 249.0 | 140.1 |
| keyframe bits | 13680 | 74536 |
| avg inter-frame bits | 8222 | 3565 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 211.4 | 203.9 |
| min / max | 171 / 243 | 114 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 4x4 | 0 | 0.0 | 22 | 0.0 |
| 4x8 | 0 | 0.0 | 39 | 0.1 |
| 8x4 | 0 | 0.0 | 73 | 0.1 |
| 8x8 | 55028 | 85.3 | 1614 | 2.5 |
| 8x16 | 0 | 0.0 | 204 | 0.6 |
| 16x8 | 0 | 0.0 | 328 | 1.0 |
| 16x16 | 427 | 2.6 | 633 | 3.9 |
| 16x32 | 0 | 0.0 | 38 | 0.5 |
| 32x16 | 0 | 0.0 | 138 | 1.7 |
| 32x32 | 86 | 2.1 | 310 | 7.7 |
| 32x64 | 0 | 0.0 | 34 | 1.7 |
| 64x32 | 0 | 0.0 | 16 | 0.8 |
| 64x64 | 100 | 9.9 | 800 | 79.4 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.0 | 89.1 |
| NEARESTMV % | 79.4 | 5.9 |
| NEARMV % | 0.0 | 1.0 |
| ZEROMV % | 20.6 | 0.4 |
| NEWMV % | 0.0 | 3.5 |
| skip=1 % | 55.5 | 14.8 |
| ref LAST % | 100.0 | 7.1 |
| ref GOLDEN % | 0.0 | 1.7 |
| ref ALTREF % | 0.0 | 2.1 |

## Operating point: 250 kbps target

### 1. Bit allocation

| metric | rvp9 | libvpx |
|---|---:|---:|
| achieved bitrate (kbps) | 514.6 | 304.9 |
| keyframe bits | 32896 | 109608 |
| avg inter-frame bits | 16921 | 8596 |
| frames rvp9-decoder FAILED to decode | 0 | 0 |

### 5. base_qindex (inter frames)

| metric | rvp9 | libvpx |
|---|---:|---:|
| avg base_qindex | 168.9 | 149.7 |
| min / max | 143 / 195 | 57 / 255 |

### 2. Partition size distribution (inter frames)

Distinct coded leaf blocks (count) and area coverage (% of coded 8x8 cells).

| block | rvp9 count | rvp9 area% | libvpx count | libvpx area% |
|---|---:|---:|---:|---:|
| 4x4 | 0 | 0.0 | 57 | 0.1 |
| 4x8 | 0 | 0.0 | 70 | 0.1 |
| 8x4 | 0 | 0.0 | 133 | 0.2 |
| 8x8 | 60960 | 94.5 | 1444 | 2.2 |
| 8x16 | 0 | 0.0 | 298 | 0.9 |
| 16x8 | 0 | 0.0 | 316 | 1.0 |
| 16x16 | 252 | 1.6 | 1155 | 7.2 |
| 16x32 | 0 | 0.0 | 118 | 1.5 |
| 32x16 | 0 | 0.0 | 124 | 1.5 |
| 32x32 | 79 | 2.0 | 427 | 10.6 |
| 32x64 | 0 | 0.0 | 16 | 0.8 |
| 64x32 | 0 | 0.0 | 88 | 4.4 |
| 64x64 | 20 | 2.0 | 701 | 69.5 |

### 3-4-6. Mode / skip / ref distribution (inter frames, area-weighted % of coded 8x8 cells)

| metric | rvp9 | libvpx |
|---|---:|---:|
| intra % | 0.0 | 82.3 |
| NEARESTMV % | 86.1 | 4.9 |
| NEARMV % | 0.0 | 2.0 |
| ZEROMV % | 13.9 | 1.6 |
| NEWMV % | 0.0 | 9.2 |
| skip=1 % | 44.0 | 16.1 |
| ref LAST % | 100.0 | 8.3 |
| ref GOLDEN % | 0.0 | 5.7 |
| ref ALTREF % | 0.0 | 3.8 |

### Per-frame bits & base_qindex (100 kbps)

| frame | kind | rvp9 bits | rvp9 q | rvp9 intra% | libvpx bits | libvpx q | libvpx intra% |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | KEY | 13680 | 177 | 100 | 74536 | 57 | 100 |
| 1 | inter | 3192 | 238 | 0 | 8752 | 114 | 31 |
| 2 | inter | 3768 | 242 | 0 | 2816 | 186 | 56 |
| 3 | inter | 5016 | 231 | 0 | 4296 | 210 | 12 |
| 4 | inter | 5416 | 221 | 0 | 2992 | 237 | 60 |
| 5 | inter | 6352 | 211 | 0 | 2864 | 251 | 28 |
| 6 | inter | 7512 | 202 | 0 | 2808 | 255 | 39 |
| 7 | inter | 8800 | 194 | 0 | 1936 | 255 | 87 |
| 8 | inter | 10832 | 187 | 0 | 1768 | 255 | 91 |
| 9 | inter | 12528 | 182 | 0 | 688 | 255 | 98 |
| 10 | inter | 13440 | 177 | 0 | 15680 | 139 | 38 |
| 11 | inter | 14528 | 174 | 0 | 24272 | 239 | 96 |
| 12 | inter | 16688 | 172 | 0 | 1752 | 229 | 93 |
| 13 | inter | 18320 | 171 | 0 | 2216 | 228 | 91 |
| 14 | inter | 17504 | 171 | 0 | 1952 | 231 | 97 |
| 15 | inter | 17984 | 172 | 0 | 1912 | 232 | 95 |
| 16 | inter | 16528 | 174 | 0 | 1760 | 232 | 89 |
| 17 | inter | 14752 | 176 | 0 | 1384 | 231 | 96 |
| 18 | inter | 13128 | 178 | 0 | 1112 | 226 | 97 |
| 19 | inter | 11696 | 180 | 0 | 968 | 218 | 100 |
| 20 | inter | 11520 | 182 | 0 | 144 | 255 | 92 |
| 21 | inter | 12792 | 183 | 0 | 17632 | 206 | 99 |
| 22 | inter | 15312 | 185 | 0 | 1784 | 199 | 98 |
| 23 | inter | 14800 | 189 | 0 | 2000 | 199 | 98 |
| 24 | inter | 13464 | 193 | 0 | 1872 | 200 | 100 |
| 25 | inter | 11552 | 197 | 0 | 1472 | 200 | 98 |
| 26 | inter | 8376 | 200 | 0 | 1568 | 196 | 100 |
| 27 | inter | 6352 | 202 | 0 | 1328 | 194 | 94 |
| 28 | inter | 5720 | 203 | 0 | 1288 | 189 | 100 |
| 29 | inter | 5696 | 203 | 0 | 1152 | 183 | 97 |
| 30 | inter | 6760 | 204 | 0 | 136 | 255 | 94 |
| 31 | inter | 7480 | 205 | 0 | 18864 | 175 | 100 |
| 32 | inter | 8280 | 206 | 0 | 2560 | 175 | 99 |
| 33 | inter | 8544 | 209 | 0 | 2368 | 181 | 100 |
| 34 | inter | 8760 | 211 | 0 | 2344 | 186 | 98 |
| 35 | inter | 8160 | 213 | 0 | 2304 | 190 | 100 |
| 36 | inter | 7696 | 216 | 0 | 2344 | 194 | 100 |
| 37 | inter | 7032 | 218 | 0 | 2192 | 198 | 99 |
| 38 | inter | 6536 | 219 | 0 | 1680 | 201 | 99 |
| 39 | inter | 5920 | 221 | 0 | 1360 | 199 | 99 |
| 40 | inter | 5936 | 222 | 0 | 144 | 234 | 94 |
| 41 | inter | 6088 | 224 | 0 | 2232 | 180 | 100 |
| 42 | inter | 5632 | 225 | 0 | 2600 | 179 | 100 |
| 43 | inter | 4648 | 226 | 0 | 2472 | 180 | 100 |
| 44 | inter | 4736 | 227 | 0 | 2816 | 180 | 100 |
| 45 | inter | 5000 | 227 | 0 | 2344 | 183 | 93 |
| 46 | inter | 5048 | 228 | 0 | 2544 | 183 | 99 |
| 47 | inter | 6024 | 229 | 0 | 3200 | 184 | 96 |
| 48 | inter | 6112 | 231 | 0 | 3064 | 190 | 98 |
| 49 | inter | 5840 | 232 | 0 | 2936 | 195 | 98 |
| 50 | inter | 4816 | 233 | 0 | 2552 | 199 | 92 |
| 51 | inter | 4248 | 234 | 0 | 2184 | 200 | 100 |
| 52 | inter | 4264 | 235 | 0 | 2448 | 198 | 95 |
| 53 | inter | 4712 | 235 | 0 | 3064 | 198 | 93 |
| 54 | inter | 4776 | 236 | 0 | 2888 | 203 | 93 |
| 55 | inter | 4640 | 237 | 0 | 23840 | 124 | 95 |
| 56 | inter | 4536 | 238 | 0 | 1656 | 213 | 93 |
| 57 | inter | 4264 | 238 | 0 | 1776 | 210 | 96 |
| 58 | inter | 4864 | 239 | 0 | 2048 | 207 | 95 |
| 59 | inter | 5096 | 240 | 0 | 1600 | 207 | 93 |
| 60 | inter | 5064 | 241 | 0 | 1880 | 202 | 84 |
| 61 | inter | 4504 | 242 | 0 | 1920 | 201 | 84 |
| 62 | inter | 4536 | 242 | 0 | 2072 | 199 | 80 |
| 63 | inter | 3872 | 243 | 0 | 2024 | 199 | 90 |
