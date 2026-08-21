//! Exercise the complete VP9 packet parser and frame reconstruction path.
//!
//! Raw bytes work well here when the corpus is seeded with valid encoder or
//! conformance packets: libFuzzer preserves those deep paths while mutating
//! compressed headers, tile payloads, modes, coefficients, and motion data.

#![no_main]

use libfuzzer_sys::fuzz_target;
use rvp9_decoder::Decoder;

const MAX_PACKET_BYTES: usize = 16 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_PACKET_BYTES {
        return;
    }

    let mut decoder = Decoder::with_threads(std::num::NonZeroUsize::MIN);
    let Ok(packet) = decoder.parse_packet(data) else {
        return;
    };
    for frame in packet.frames() {
        if decoder.decode_frame(frame).is_err() {
            break;
        }
    }
});
