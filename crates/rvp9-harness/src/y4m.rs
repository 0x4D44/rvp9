//! Harness Y4M helpers.
//!
//! The Y4M reader/writer itself lives in the shared [`rvp9_io::y4m`]
//! crate (so the CLIs can use it without depending on this test
//! harness). This module re-exports that surface and adds the
//! harness-specific test-corpus path probes.

pub use rvp9_io::y4m::*;

/// Probe whether a known testsrc y4m corpus file is available at one of
/// the canonical paths. Returns `Some(path)` for the first hit.
///
/// Used by harness tests to opt into real-content runs without making
/// the path-finding logic the test's concern.
#[must_use]
pub fn probe_testsrc_720p() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from("tests/corpus/testsrc_720p30.y4m"),
        std::path::PathBuf::from(r"D:\language\rvp9-2\tests\corpus\testsrc_720p30.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

/// Probe whether the classic Foreman CIF (352×288, ~30 fps, 300 frames)
/// y4m corpus file is available at one of the canonical paths. Returns
/// `Some(path)` for the first hit.
///
/// Foreman is a real-content clip with substantial motion (head turn,
/// camera pan to construction site) and texture that stresses every
/// component (DCT, MV search, MC). It complements the saturating
/// testsrc 720p clip — see `tests/corpus/sources/README.md` for the
/// acquisition steps.
#[must_use]
pub fn probe_foreman_cif() -> Option<std::path::PathBuf> {
    let candidates = [
        std::path::PathBuf::from("tests/corpus/foreman_cif.y4m"),
        std::path::PathBuf::from(r"D:\language\rvp9-2\tests\corpus\foreman_cif.y4m"),
        // Sibling-tree cache populated by the legacy rvp9 repo's fixtures.
        std::path::PathBuf::from(r"D:\language\rvp9\crates\rvp9\tests\data\cache\foreman_cif.y4m"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_some_when_testsrc_exists() {
        // Skip if testsrc isn't present (CI without corpus).
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let mut f = std::fs::File::open(&p).expect("open testsrc");
        let hdr = parse_header(&mut f).expect("parse header");
        assert_eq!(hdr.w, 1280);
        assert_eq!(hdr.h, 720);
        assert!(hdr.framerate() > 0.0);
    }

    #[test]
    fn read_cropped_512x512_8_frames_from_testsrc() {
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let (frames, framerate) = read_cropped(&p, 512, 512, 8).expect("read 8 frames at 512x512");
        assert_eq!(frames.len(), 8);
        assert!(framerate > 0.0);
        // Y plane is 512*512 = 262144 bytes per frame.
        assert_eq!(frames[0].0.len(), 512 * 512);
        // Chroma at half-rate: 256*256 = 65536.
        assert_eq!(frames[0].1.len(), 256 * 256);
        assert_eq!(frames[0].2.len(), 256 * 256);
        // Frames should differ across the sequence (testsrc has motion).
        assert_ne!(frames[0].0, frames[4].0, "testsrc should have motion");
    }

    #[test]
    fn read_cropped_rejects_odd_crop() {
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let err = read_cropped(&p, 511, 511, 1).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
