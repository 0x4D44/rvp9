//! Harness Y4M helpers.
//!
//! The Y4M reader/writer itself lives in the shared [`rvp9_io::y4m`]
//! crate (so the CLIs can use it without depending on this test
//! harness). This module re-exports that surface and adds the
//! harness-specific test-corpus path probes.

pub use rvp9_io::y4m::*;

fn workspace_corpus_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("corpus")
}

/// Return a corpus path anchored to the workspace that built this harness.
#[must_use]
pub fn workspace_corpus_file(name: &str) -> std::path::PathBuf {
    workspace_corpus_dir().join(name)
}

fn corpus_candidates(name: &str) -> [std::path::PathBuf; 1] {
    [workspace_corpus_file(name)]
}

/// Probe whether a known testsrc y4m corpus file is available in this
/// workspace. Returns `Some(path)` when present.
///
/// Used by harness tests to opt into real-content runs without making
/// the path-finding logic the test's concern.
#[must_use]
pub fn probe_testsrc_720p() -> Option<std::path::PathBuf> {
    corpus_candidates("testsrc_720p30.y4m")
        .into_iter()
        .find(|path| path.is_file())
}

/// Probe whether the classic Foreman CIF (352×288, ~30 fps, 300 frames)
/// y4m corpus file is available in this workspace. Returns `Some(path)`
/// when present.
///
/// Foreman is a real-content clip with substantial motion (head turn,
/// camera pan to construction site) and texture that stresses every
/// component (DCT, MV search, MC). It complements the saturating
/// testsrc 720p clip — see `tests/corpus/sources/README.md` for the
/// acquisition steps.
#[must_use]
pub fn probe_foreman_cif() -> Option<std::path::PathBuf> {
    corpus_candidates("foreman_cif.y4m")
        .into_iter()
        .find(|path| path.is_file())
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
        // Some local fallback fixtures are static color bars; still make
        // sure the crop read real non-flat luma content.
        assert!(
            frames[0].0.windows(2).any(|w| w[0] != w[1]),
            "testsrc crop should not be flat"
        );
    }

    #[test]
    fn read_cropped_rejects_odd_crop() {
        let Some(p) = probe_testsrc_720p() else {
            return;
        };
        let err = read_cropped(&p, 511, 511, 1).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn corpus_candidates_stay_inside_this_workspace() {
        let workspace = workspace_corpus_dir();
        let candidates = corpus_candidates("foreman_cif.y4m");

        assert_eq!(candidates.len(), 1);
        assert!(candidates.iter().all(|path| path.starts_with(&workspace)));
    }
}
