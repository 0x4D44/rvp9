# Stage-0 raw timings

Nine alternating current/fused runs on Crucible, nanoseconds per whole grid; sorted within each arm. Compile and run [probe.rs](probe.rs) with `rustc --edition=2024 -C opt-level=3 -C target-cpu=native -C debuginfo=1 probe.rs -o probe.exe` and `probe.exe`.

```text
32 hot current [69.361, 69.42, 69.6195, 69.6245, 69.707, 69.75, 69.783, 69.7865, 71.265]
32 hot fused   [64.7645, 65.028, 65.0295, 65.0845, 65.0885, 65.215, 65.216, 65.2255, 65.5455]
32 cold current [509.487, 509.550, 514.315, 523.185, 524.272, 528.398, 539.215, 542.672, 564.305]
32 cold fused   [499.180, 501.072, 502.187, 504.712, 505.660, 509.617, 515.491, 518.122, 528.615]
64 hot current [256.6635, 256.683, 257.191, 257.3465, 257.4005, 257.449, 257.452, 257.8955, 259.532]
64 hot fused   [237.67, 237.752, 237.774, 237.8625, 238.012, 238.1015, 238.215, 238.241, 238.8865]
64 cold current [836.548, 839.856, 847.253, 850.110, 850.895, 851.619, 856.181, 856.673, 864.303]
64 cold fused   [775.594, 783.773, 797.534, 812.337, 814.351, 819.910, 825.700, 832.218, 858.671]
```
