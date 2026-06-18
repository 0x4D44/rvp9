//! G-C: differential decode oracle for the per-segment SEG_LVL_SKIP /
//! SEG_LVL_REF_FRAME features.
//!
//! ## Why this test exists
//!
//! No official VP9 conformance vector and no `vpxenc` CLI flag activates
//! `SEG_LVL_SKIP` / `SEG_LVL_REF_FRAME` (see the G-C HLD §4.1-4.2). The
//! only way to mint a stream that exercises them is the libvpx encoder's
//! `VP9E_SET_ROI_MAP` C API (the `vp9_roi_test.cc` RoiMaskBackgroundSkip
//! pattern). `tests/corpus/segmentation/roi_segfeatures.ivf` was minted
//! by `roi_segfeatures_gen.c` (committed alongside it) with realtime +
//! cpu-used=7 + aq-mode=3, `roi.skip[3] = 1`, `roi.ref_frame[4] = 1`,
//! over 60 frames of part-static content.
//!
//! ## What it provably exercises
//!
//! `SEG_LVL_SKIP` fires on **351 real inter blocks** in this fixture
//! (segment 3, the right static third): each forces `skip = true` +
//! `mode = ZEROMV`, suppressing the skip bit AND the inter-mode bits.
//! This was confirmed live during bring-up: decoding with the forcing
//! DISABLED yields a *different* MD5 (the bool reader desyncs), proving
//! the match is non-vacuous. The frame headers additionally activate
//! `SEG_LVL_REF_FRAME` on segment 4 (verified in the resolved
//! per-frame feature table), so the segmentation-header parse must stay
//! in sync with libvpx too — but libvpx's realtime nonrd encoder does
//! not place inter blocks on the REF_FRAME segment in this fixture, so
//! the per-block REF_FRAME *force* path is covered by the unit oracle
//! in `rvp9-core` (`read_ref_frames_seg_force_consumes_zero_bits`), not
//! here. See the HLD §4.5 honest verdict.
//!
//! ## The oracle
//!
//! `roi_segfeatures.ivf.md5` holds the single MD5 that
//! `vpxdec --i420 --md5` computes over the whole decoded I420 stream
//! (all 60 shown frames, Y‖U‖V at native dims). We decode the same
//! fixture with rvp9, concatenate the visible planes in the identical
//! `vpxdec --md5` order, and assert the MD5 matches byte-for-byte.

use rvp9_core::bitstream::FrameKind;
use rvp9_core::frame::{PlaneRef, Yv12Buffer};
use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use rvp9_harness::{hex16, md5};
use std::num::NonZeroUsize;
use std::path::PathBuf;

/// Resolve `tests/corpus/segmentation/<name>` from the workspace root.
fn corpus(name: &str) -> PathBuf {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_root
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("tests")
        .join("corpus")
        .join("segmentation")
        .join(name)
}

/// Append the visible (cropped) bytes of a plane in raster order,
/// excluding stride/border padding — exactly what `vpxdec --md5` hashes.
fn push_plane(dst: &mut Vec<u8>, plane: PlaneRef<'_>, w: usize, h: usize) {
    let stride = plane.stride();
    let data = plane.data();
    for r in 0..h {
        let start = r * stride;
        dst.extend_from_slice(&data[start..start + w]);
    }
}

/// Concatenate one decoded frame's visible Y, U, V planes in
/// `vpxdec --md5` order/dims.
fn append_visible_i420(out: &mut Vec<u8>, buf: &Yv12Buffer) {
    let crop_w = buf.width() as usize;
    let crop_h = buf.height() as usize;
    let uv_w = buf.chroma_width() as usize;
    let uv_h = buf.chroma_height() as usize;
    push_plane(out, buf.y_plane(), crop_w, crop_h);
    push_plane(out, buf.u_plane(), uv_w, uv_h);
    push_plane(out, buf.v_plane(), uv_w, uv_h);
}

#[test]
fn roi_segfeatures_decodes_bit_exact_vs_vpxdec() {
    let fixture = corpus("roi_segfeatures.ivf");
    let md5_path = corpus("roi_segfeatures.ivf.md5");
    assert!(fixture.exists(), "G-C fixture missing: {}", fixture.display());
    assert!(
        md5_path.exists(),
        "G-C gold MD5 missing: {}",
        md5_path.display(),
    );

    let gold = std::fs::read_to_string(&md5_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", md5_path.display()))
        .trim()
        .to_ascii_lowercase();

    let (data, packets) =
        read_packets(&fixture).unwrap_or_else(|e| panic!("container parse {}: {e}", fixture.display()));
    // Single worker — deterministic, matches the vpxdec gold which is
    // single-threaded for this small frame size.
    let mut decoder = Decoder::with_threads(NonZeroUsize::new(1).expect("1 is non-zero"));

    let mut all_bytes: Vec<u8> = Vec::new();
    let mut shown = 0usize;
    for range in &packets {
        let parsed = decoder
            .parse_packet(range.slice(&data))
            .unwrap_or_else(|e| panic!("parse_packet {}: {e}", fixture.display()));
        for frame in parsed {
            let is_shown = frame.header.show_frame
                || matches!(frame.header.kind, FrameKind::ShowExistingFrame { .. });
            let buf = decoder.decode_frame(&frame).unwrap_or_else(|e| {
                panic!(
                    "{}: decode error on shown-frame {shown} (kind={:?}): {e}",
                    fixture.display(),
                    frame.header.kind,
                )
            });
            if is_shown {
                append_visible_i420(&mut all_bytes, &buf);
                shown += 1;
            }
        }
    }

    // A truncated decode must not pass vacuously.
    assert_eq!(shown, 60, "expected 60 shown frames, decoded {shown}");

    let got = hex16(md5(&all_bytes));
    assert_eq!(
        got, gold,
        "G-C SEG_LVL_SKIP/REF_FRAME fixture: rvp9 whole-stream I420 MD5 \
         ({got}) does not match vpxdec gold ({gold}) — segmentation \
         feature forcing desynced the bool reader",
    );
}
