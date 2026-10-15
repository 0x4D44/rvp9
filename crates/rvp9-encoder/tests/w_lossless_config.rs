//! Unsupported lossless requests fail before stream construction.

use rvp9_decoder::Decoder;
use rvp9_encoder::{
    EncodeError, EncoderConfig, EncoderParallelism, RcMode, StreamEncoder, YuvFrameRef,
};

fn rejects_lossless(config: EncoderConfig) -> bool {
    let rejected = |result| matches!(result, Err(EncodeError::InvalidConfig(message)) if message.contains("lossless"));
    let default = rejected(StreamEncoder::new(config));
    let parallel = rejected(StreamEncoder::new_with_parallelism(
        config,
        EncoderParallelism::default(),
    ));
    default && parallel
}

#[test]
fn q_mode_rejects_lossless_inter_and_boosted_keyframe_requests() {
    // Q mode ignores min_qindex and boosts keyframes by eight qindices.
    let accepted: Vec<_> = (0..=8)
        .filter(|&cq_level| {
            let mut config = EncoderConfig::new(64, 64);
            config.rc_mode = RcMode::Q;
            config.cq_level = cq_level;
            !rejects_lossless(config)
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "lossless Q requests must fail construction, including boosted keyframes: {accepted:?}"
    );
}

#[test]
fn regulated_modes_reject_bounds_that_allow_lossless() {
    let accepted: Vec<_> = [RcMode::Cbr, RcMode::Vbr, RcMode::VbrTwoPass, RcMode::Cq]
        .into_iter()
        .filter(|&mode| {
            let mut config = EncoderConfig::new(64, 64);
            config.rc_mode = mode;
            config.min_qindex = 0;
            config.cq_level = 0;
            !rejects_lossless(config)
        })
        .collect();
    assert!(
        accepted.is_empty(),
        "modes permitting lossless picks must fail construction: {accepted:?}"
    );
}

#[test]
fn lossy_boundaries_remain_supported_and_match_wire_quantizers() {
    // cq_level=0 remains inert for the default CBR configuration.
    StreamEncoder::new(EncoderConfig::new(64, 64)).expect("default CBR remains supported");
    for mode in [RcMode::Cbr, RcMode::Vbr, RcMode::VbrTwoPass] {
        let mut config = EncoderConfig::new(64, 64);
        config.rc_mode = mode;
        config.min_qindex = 1;
        StreamEncoder::new(config).expect("positive minimum remains supported");
    }

    // Q=9 gives the smallest supported keyframe Q (1), then inter Q=9.
    // CQ's positive floor permits min_qindex=0 safely; force its upper
    // bound to one so both emitted frame kinds exercise that boundary.
    for (mode, cq_level, min_qindex, max_qindex, expected) in [
        (RcMode::Q, 9, 0, 255, [1, 9]),
        (RcMode::Cq, 1, 0, 1, [1, 1]),
        (RcMode::Cq, 0, 1, 1, [1, 1]),
        (RcMode::Cbr, 0, 1, 1, [1, 1]),
    ] {
        let mut config = EncoderConfig::new(64, 64);
        config.rc_mode = mode;
        config.cq_level = cq_level;
        config.min_qindex = min_qindex;
        config.max_qindex = max_qindex;
        let mut encoder = StreamEncoder::new(config).expect("lossy boundary is supported");
        let mut decoder = Decoder::new();
        let y: Vec<_> = (0..64 * 64).map(|i| (96 + i % 64) as u8).collect();
        let uv = vec![128; 32 * 32];
        for q in expected {
            let encoded = encoder
                .encode_frame(&YuvFrameRef {
                    y: &y,
                    u: &uv,
                    v: &uv,
                    y_stride: 64,
                    u_stride: 32,
                    v_stride: 32,
                })
                .expect("encode supported lossy boundary");
            assert_eq!(encoded.q_index, q, "reported {mode:?} quantizer");
            let parsed: Vec<_> = decoder
                .parse_packet(&encoded.data)
                .expect("parse packet")
                .into_iter()
                .collect();
            assert_eq!(parsed.len(), 1);
            for frame in &parsed {
                assert_eq!(
                    frame
                        .header
                        .tail
                        .as_ref()
                        .expect("coded frame tail")
                        .quantization
                        .base_qindex,
                    q,
                    "wire {mode:?} quantizer"
                );
                decoder.decode_frame(frame).expect("decode boundary frame");
            }
        }
    }
}
