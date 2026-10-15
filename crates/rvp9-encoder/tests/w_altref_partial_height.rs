//! Regression coverage for RVP9-BUG-KILN-00069.

use rvp9_decoder::Decoder;
use rvp9_encoder::{AqMode, EncoderConfig, StreamEncoder, YuvFrameRef};

fn partial_height_altref(width: u32, height: u32, emit_during_flush: bool) {
    let result = std::panic::catch_unwind(|| {
        let mut config = EncoderConfig::new(width, height);
        config.gop_length = 8;
        config.lag_in_frames = if emit_during_flush { 10 } else { 3 };
        config.enable_alt_ref = true;
        config.aq_mode = AqMode::Off;
        config.speed = 8;
        let mut encoder = StreamEncoder::new(config).expect("valid even dimensions");
        // ARF is scheduled at display offset 5. Nine inputs leave forward
        // neighbours available whether that offset is emitted here or in flush.
        let y = vec![96; (width * height) as usize];
        let u = vec![112; (width * height / 4) as usize];
        let v = vec![144; (width * height / 4) as usize];
        let source = YuvFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride: width as usize,
            u_stride: width as usize / 2,
            v_stride: width as usize / 2,
        };
        let mut packets = Vec::new();
        for _ in 0..9 {
            if let Some(packet) = encoder
                .encode_frame_delayed(&source)
                .expect("delayed encode")
            {
                packets.push((false, packet.data));
            }
        }
        for packet in encoder.flush() {
            packets.push((true, packet.expect("flush encode").data));
        }
        let mut decoder = Decoder::new();
        let mut shown = 0;
        let mut arfs = 0;
        for (from_flush, packet) in packets {
            let parsed = decoder.parse_packet(&packet).expect("parse superframe");
            let contains_arf = parsed.frames().iter().any(|f| !f.header.show_frame);
            if contains_arf {
                assert!(parsed.frames().len() > 1, "ARF must be in a superframe");
                assert_eq!(from_flush, emit_during_flush, "ARF emission phase");
            }
            for frame in parsed.frames() {
                decoder
                    .decode_frame(frame)
                    .expect("decode every ARF and show frame");
                if frame.header.show_frame {
                    shown += 1;
                } else {
                    arfs += 1;
                }
            }
        }
        assert!(
            arfs > 0,
            "partial-height regression must reach ARF emission"
        );
        assert_eq!(shown, 9, "one decoded show frame per input");
    });
    assert!(
        result.is_ok(),
        "RVP9-BUG-KILN-00069: ALTREF must filter, reconstruct and decode {width}x{height} (flush={emit_during_flush}) without panicking"
    );
}

#[test]
fn altref_partial_height_720p_delayed() {
    partial_height_altref(1280, 720, false);
}
#[test]
fn altref_partial_height_720p_flush() {
    partial_height_altref(1280, 720, true);
}
#[test]
fn altref_partial_height_1080p_delayed() {
    partial_height_altref(1920, 1080, false);
}
#[test]
fn altref_partial_height_1080p_flush() {
    partial_height_altref(1920, 1080, true);
}
#[test]
fn altref_partial_height_32_not_64_delayed() {
    partial_height_altref(64, 96, false);
}
#[test]
fn altref_partial_height_32_not_64_flush() {
    partial_height_altref(64, 96, true);
}

/// Exercise both edges, including sub-macroblock dimensions and row padding.
#[test]
fn altref_partial_blocks_preserve_visible_pixels_and_strides() {
    use rvp9_encoder::temporal_filter::{
        ArnrFilterData, ArnrFrameRef, temporal_filter_iterate_row, vp9_temporal_filter,
    };
    let fixed_divide = std::array::from_fn(|i| if i == 0 { 0 } else { 0x80000 / i as u32 });
    for (width, height, y_stride, uv_stride) in [(34, 18, 34, 17), (34, 38, 40, 24), (2, 2, 4, 3)] {
        let mut y = vec![255; y_stride * height];
        let mut u = vec![255; uv_stride * height / 2];
        let mut v = u.clone();
        for row in 0..height {
            for col in 0..width {
                y[row * y_stride + col] = ((row * 3 + col * 7) % 200) as u8;
            }
        }
        for row in 0..height / 2 {
            for col in 0..width / 2 {
                u[row * uv_stride + col] = (row * 2 + col + 20) as u8;
                v[row * uv_stride + col] = (row + col * 2 + 80) as u8;
            }
        }
        let frame = ArnrFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride,
            uv_stride,
            width: width as u32,
            height: height as u32,
        };
        let data = ArnrFilterData::new(vec![frame; 3], 1, 3);
        let output = vp9_temporal_filter(&data);
        for (source, result, stride, row_width, rows) in [
            (&y, &output.y, y_stride, width, height),
            (&u, &output.u, uv_stride, width / 2, height / 2),
            (&v, &output.v, uv_stride, width / 2, height / 2),
        ] {
            assert_eq!(result.len(), row_width * rows);
            for row in 0..rows {
                assert_eq!(
                    &result[row * row_width..(row + 1) * row_width],
                    &source[row * stride..row * stride + row_width],
                    "RVP9-BUG-KILN-00069: partial edge pixels must survive filtering"
                );
            }
        }
        // The public row entry point retains the source stride; only
        // visible pixels are written, leaving row padding untouched.
        let mut row_y = vec![251; y.len()];
        let mut row_u = vec![251; u.len()];
        let mut row_v = vec![251; v.len()];
        for mb_row in 0..height.div_ceil(32) {
            temporal_filter_iterate_row(
                &data,
                mb_row,
                0,
                width.div_ceil(32),
                &fixed_divide,
                &mut row_y,
                &mut row_u,
                &mut row_v,
            );
        }
        for (source, result, stride, row_width, rows) in [
            (&y, &row_y, y_stride, width, height),
            (&u, &row_u, uv_stride, width / 2, height / 2),
            (&v, &row_v, uv_stride, width / 2, height / 2),
        ] {
            for row in 0..rows {
                assert_eq!(
                    &result[row * stride..row * stride + row_width],
                    &source[row * stride..row * stride + row_width]
                );
                assert!(
                    result[row * stride + row_width..(row + 1) * stride]
                        .iter()
                        .all(|&p| p == 251)
                );
            }
        }
    }
}
