//! Named, fail-closed selections for the official VP9 conformance corpus.

use std::path::PathBuf;

/// Number of 8-bit VP9 vectors selected from pinned libvpx v1.14.1.
pub const OFFICIAL_VECTOR_COUNT: usize = 307;
/// Expected displayed frames across the complete official corpus.
pub const OFFICIAL_FRAME_COUNT: usize = 4_816;
/// Official vector outside the v0.2.0-alpha.1 decoder surface.
pub const ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR: &str = "vp90-2-22-svc_1280x720_3.ivf";
/// Number of official vectors in the v0.2.0-alpha.1 supported surface.
pub const ALPHA_SUPPORTED_VECTOR_COUNT: usize = 306;
/// Expected displayed frames across the alpha-supported official vectors.
pub const ALPHA_SUPPORTED_FRAME_COUNT: usize = 4_796;

/// Select the v0.2.0-alpha.1 surface from a complete official corpus.
///
/// The selection fails if the corpus is incomplete, has grown, or does not
/// contain exactly the one known spatial-SVC vector. That makes the deliberate
/// alpha boundary visible without weakening the default full-corpus run.
pub fn alpha_supported_vectors(mut vectors: Vec<PathBuf>) -> Result<Vec<PathBuf>, String> {
    if vectors.len() != OFFICIAL_VECTOR_COUNT {
        return Err(format!(
            "alpha-supported mode requires {OFFICIAL_VECTOR_COUNT} official vectors, found {}",
            vectors.len()
        ));
    }

    let unsupported_count = vectors
        .iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR)
        })
        .count();
    if unsupported_count != 1 {
        return Err(format!(
            "alpha-supported mode requires exactly one {ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR}, found {unsupported_count}"
        ));
    }

    vectors.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_none_or(|name| name != ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR)
    });
    debug_assert_eq!(vectors.len(), ALPHA_SUPPORTED_VECTOR_COUNT);
    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_corpus() -> Vec<PathBuf> {
        let mut vectors: Vec<PathBuf> = (0..ALPHA_SUPPORTED_VECTOR_COUNT)
            .map(|index| PathBuf::from(format!("vector-{index:03}.webm")))
            .collect();
        vectors.push(PathBuf::from(ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR));
        vectors
    }

    #[test]
    fn alpha_scope_excludes_exactly_the_spatial_svc_vector() {
        let selected = alpha_supported_vectors(complete_corpus()).expect("complete corpus");

        assert_eq!(selected.len(), ALPHA_SUPPORTED_VECTOR_COUNT);
        assert!(selected.iter().all(|path| {
            path.file_name().and_then(|name| name.to_str())
                != Some(ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR)
        }));
    }

    #[test]
    fn alpha_scope_rejects_incomplete_corpus() {
        let mut vectors = complete_corpus();
        vectors.pop();
        let error = alpha_supported_vectors(vectors).expect_err("incomplete corpus must fail");
        assert!(error.contains("requires 307"), "unexpected error: {error}");
    }

    #[test]
    fn alpha_scope_rejects_missing_spatial_svc_vector() {
        let mut vectors = complete_corpus();
        let last = vectors
            .last_mut()
            .expect("complete corpus has unsupported vector");
        *last = PathBuf::from("replacement.ivf");
        let error = alpha_supported_vectors(vectors).expect_err("missing boundary must fail");
        assert!(
            error.contains(ALPHA_UNSUPPORTED_SPATIAL_SVC_VECTOR),
            "unexpected error: {error}"
        );
    }
}
