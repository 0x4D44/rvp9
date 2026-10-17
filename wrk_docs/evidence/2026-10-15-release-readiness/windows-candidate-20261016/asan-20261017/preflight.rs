fn main() {
    let source: Vec<u8> = (0..4096).map(|i| (i & 255) as u8).collect();
    let mut data = std::hint::black_box(source);
    for (index, value) in data.iter_mut().enumerate() {
        *value = value.wrapping_add(index as u8);
    }
    let sum: u64 = data.iter().map(|&value| u64::from(value)).sum();
    assert_eq!(data.len(), 4096);
    assert_eq!(sum, 520192);
    println!("native_asan_preflight_ok arch={} pointer_bits={} avx2={} bytes={} checksum={}", std::env::consts::ARCH, usize::BITS, std::is_x86_feature_detected!("avx2"), data.len(), sum);
}
