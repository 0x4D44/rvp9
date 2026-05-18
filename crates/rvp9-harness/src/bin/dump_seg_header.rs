//! D.10 — quick diagnostic for segmentation-header state per frame.
//! Run e.g. `cargo run --bin dump_seg_header -- tests/conformance/vectors/vp90-2-09-aq2.webm`.

use rvp9_decoder::Decoder;
use rvp9_harness::container::read_packets;
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let vec_path = PathBuf::from(args.next().expect("vector path"));
    let max_frames: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    let (data, packets) = read_packets(&vec_path).expect("container parse");
    let mut decoder = Decoder::new();
    let mut frame_idx: usize = 0;
    for range in &packets {
        let parsed = match decoder.parse_packet(range.slice(&data)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("parse err: {e:?}");
                return;
            }
        };
        for frame in parsed {
            if let Some(tail) = frame.header.tail.as_ref() {
                let s = &tail.segmentation;
                println!(
                    "frame {frame_idx}: show={} kind={:?} refresh_fc={} frame_par={} fc_idx={} en={} upd_map={} t_upd={} upd_data={} abs={} feat[0]={:?} feat[1]={:?} feat[2]={:?} tree_probs={:?} pred_probs={:?}",
                    frame.header.show_frame,
                    frame.header.kind,
                    tail.refresh_frame_context,
                    tail.frame_parallel_decoding_mode,
                    tail.frame_context_idx,
                    s.enabled,
                    s.update_map,
                    s.temporal_update,
                    s.update_data,
                    s.abs_delta,
                    s.feature_data[0],
                    s.feature_data[1],
                    s.feature_data[2],
                    s.tree_probs,
                    s.pred_probs,
                );
            } else {
                println!("frame {frame_idx}: show_existing");
            }
            // Now actually decode.
            match decoder.decode_frame(&frame) {
                Ok(_) => eprintln!("frame {frame_idx}: OK"),
                Err(e) => eprintln!("frame {frame_idx}: ERR {e:?}"),
            }
            frame_idx += 1;
            if frame_idx >= max_frames {
                return;
            }
        }
    }
}
