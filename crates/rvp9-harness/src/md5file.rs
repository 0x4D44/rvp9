//! Per-frame MD5 ledger reader.
//!
//! libvpx's conformance vectors ship with a sibling `<vector>.md5`
//! file holding one line per *displayed* frame. Format (one frame per
//! line, two-space separator):
//!
//! ```text
//! c3fbb7abbdb5bd4ed4a7e34768c17df1  vp90-2-00-quantizer-00-352x288-0001.i420
//! 08203c2595bdb2d58ead6f921345d699  vp90-2-00-quantizer-00-352x288-0002.i420
//! ```
//!
//! The filename column encodes the frame's resolution (`352x288`) and
//! 1-based index (`0001`). Some vectors switch resolution mid-stream
//! (e.g. `vp90-2-05-resize.ivf.md5`), so the width/height vary across
//! lines for the same vector.
//!
//! Counts only **displayed** frames — `show_frame == 0` frames have
//! no entry. `ShowExistingFrame` headers DO have entries (they emit
//! a displayed buffer).

use std::path::Path;

/// One parsed line of a `.md5` ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Md5Entry {
    /// 32-char lowercase hex MD5 string.
    pub md5_hex: String,
    /// Visible/displayed frame width, parsed from the filename.
    pub width: u32,
    /// Visible/displayed frame height, parsed from the filename.
    pub height: u32,
    /// 1-based frame index within the displayed stream.
    pub index: u32,
}

/// Errors parsing the `.md5` ledger.
#[derive(Debug)]
pub enum Md5FileError {
    /// File I/O failed.
    Io(std::io::Error),
    /// A line did not match the expected two-column format.
    BadLine {
        /// 1-based line number.
        lineno: usize,
        /// The original line text (trimmed).
        line: String,
    },
    /// A line's filename column did not embed a `<W>x<H>-<NNNN>.i420`
    /// resolution / index suffix.
    BadFilename {
        /// 1-based line number.
        lineno: usize,
        /// The filename column.
        filename: String,
    },
}

impl std::fmt::Display for Md5FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::BadLine { lineno, line } => write!(f, "line {lineno}: bad format: {line:?}"),
            Self::BadFilename { lineno, filename } => write!(
                f,
                "line {lineno}: filename does not encode WxH-NNNN: {filename:?}"
            ),
        }
    }
}

impl std::error::Error for Md5FileError {}

impl From<std::io::Error> for Md5FileError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Read a `.md5` ledger file and return one [`Md5Entry`] per line.
///
/// Blank lines are ignored.
///
/// # Errors
///
/// - [`Md5FileError::Io`] if the file can't be read.
/// - [`Md5FileError::BadLine`] if a non-blank line doesn't match the
///   `<md5_hex>  <filename>` shape (two spaces).
/// - [`Md5FileError::BadFilename`] if the filename column doesn't
///   embed a `<W>x<H>-<NNNN>.i420` resolution / index suffix.
pub fn read_md5_file(path: &Path) -> Result<Vec<Md5Entry>, Md5FileError> {
    let text = std::fs::read_to_string(path)?;
    let mut entries = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        let (md5_hex, fname) =
            line.split_once("  ").ok_or_else(|| Md5FileError::BadLine {
                lineno: i + 1,
                line: line.to_string(),
            })?;
        if md5_hex.len() != 32 || !md5_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(Md5FileError::BadLine {
                lineno: i + 1,
                line: line.to_string(),
            });
        }
        let (w, h, idx) =
            parse_resolution_index(fname).ok_or_else(|| Md5FileError::BadFilename {
                lineno: i + 1,
                filename: fname.to_string(),
            })?;
        entries.push(Md5Entry {
            md5_hex: md5_hex.to_ascii_lowercase(),
            width: w,
            height: h,
            index: idx,
        });
    }
    Ok(entries)
}

/// Parse a libvpx conformance-vector frame filename and extract
/// `(width, height, index)`. The shape is always:
///
/// ```text
///   <stem>-<W>x<H>-<NNNN>.i420
/// ```
///
/// Returns `None` for any other shape.
fn parse_resolution_index(filename: &str) -> Option<(u32, u32, u32)> {
    let stem = filename.strip_suffix(".i420")?;
    // Last '-' splits off the NNNN; the one before that splits off WxH.
    let (rest, idx_str) = stem.rsplit_once('-')?;
    let (_, wxh) = rest.rsplit_once('-')?;
    let (w_str, h_str) = wxh.split_once('x')?;
    let w: u32 = w_str.parse().ok()?;
    let h: u32 = h_str.parse().ok()?;
    let idx: u32 = idx_str.parse().ok()?;
    Some((w, h, idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_resolution_index_basic() {
        assert_eq!(
            parse_resolution_index("vp90-2-00-quantizer-00-352x288-0001.i420"),
            Some((352, 288, 1))
        );
        assert_eq!(
            parse_resolution_index("vp90-2-05-resize.ivf-282x173-0004.i420"),
            Some((282, 173, 4))
        );
    }

    #[test]
    fn parse_resolution_index_rejects_garbage() {
        assert!(parse_resolution_index("foo").is_none());
        assert!(parse_resolution_index("no-extension").is_none());
        assert!(parse_resolution_index("missing-dim.i420").is_none());
        assert!(parse_resolution_index("nox-extension-0001.i420").is_none());
    }

    #[test]
    fn read_md5_file_smoke() {
        let dir = std::env::temp_dir();
        let p = dir.join("rvp9_test_md5_smoke.md5");
        std::fs::write(
            &p,
            "c3fbb7abbdb5bd4ed4a7e34768c17df1  foo-352x288-0001.i420\n\
             08203c2595bdb2d58ead6f921345d699  foo-352x288-0002.i420\n\
             \n",
        )
        .unwrap();
        let entries = read_md5_file(&p).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].md5_hex, "c3fbb7abbdb5bd4ed4a7e34768c17df1");
        assert_eq!((entries[0].width, entries[0].height, entries[0].index), (352, 288, 1));
        assert_eq!((entries[1].width, entries[1].height, entries[1].index), (352, 288, 2));
    }

    #[test]
    fn read_md5_file_bad_format() {
        let dir = std::env::temp_dir();
        let p = dir.join("rvp9_test_md5_bad.md5");
        std::fs::write(&p, "not a valid line\n").unwrap();
        assert!(matches!(read_md5_file(&p), Err(Md5FileError::BadLine { .. })));
    }
}
