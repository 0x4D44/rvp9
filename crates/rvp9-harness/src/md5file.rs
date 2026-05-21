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
//!
//! ## In-memory shape (G8 — Wave B harness widening)
//!
//! Per HLD V6 §G8 the in-memory [`Md5Entry`] carries enough context to
//! drive the per-packet sub-frame-count sanity oracle on top of the
//! libvpx visible-MD5 reference. Concretely each row pairs a
//! `(packet_index, sub_frame_index)` tuple with a `show_frame` bit and
//! an optional 16-byte digest:
//!
//! - `packet_index` — 0-based IVF/WebM packet position in the stream.
//! - `sub_frame_index` — sub-frame within a superframe-packed packet
//!   (0 for single-frame packets).
//! - `show_frame` — `true` whenever the entry carries a displayed
//!   frame; `false` is reserved for future hidden-frame entries when
//!   we widen beyond the libvpx `.md5` format.
//! - `md5_visible` — `Some([u8; 16])` for displayed frames; `None`
//!   reserved for future hidden-frame rows (libvpx's `.md5` format
//!   only carries displayed frames, so this is always `Some` when
//!   parsed from a `.md5` file).
//!
//! The `.md5` on-disk format is **not** widened — only the in-memory
//! representation is — so parsing remains backward-compatible with the
//! AOM-shipped fixtures. Each parsed row gets `packet_index =
//! row_index`, `sub_frame_index = 0`, `show_frame = true`,
//! `md5_visible = Some(_)`. Stream packets that carry multiple
//! displayed sub-frames (atypical for the AOM corpus) are not
//! representable from a stock `.md5` alone — the in-memory ledger
//! becomes the source of truth in that scenario and the per-packet
//! sanity oracle (in `tests/conformance.rs`) pins the count from the
//! superframe-index byte tail.

use std::path::Path;

/// One parsed line of a `.md5` ledger.
///
/// See module docs for the in-memory shape rationale.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Md5Entry {
    /// 0-based packet index within the stream (IVF/WebM packet position).
    pub packet_index: usize,
    /// Sub-frame index within a superframe-packed packet (0 for
    /// non-superframe packets).
    pub sub_frame_index: usize,
    /// Whether this row represents a displayed frame.
    ///
    /// Always `true` for entries parsed from the libvpx `.md5` format
    /// (which only records displayed frames). Reserved for future use
    /// when the in-memory ledger grows beyond the on-disk format.
    pub show_frame: bool,
    /// 16-byte MD5 digest of the visible YV12 buffer.
    ///
    /// `Some(_)` whenever `show_frame == true` — matches the libvpx
    /// `.md5` format which only ledgers displayed frames. `None` is
    /// reserved for hypothetical hidden-frame rows.
    pub md5_visible: Option<[u8; 16]>,
    /// Visible/displayed frame width, parsed from the filename.
    pub width: u32,
    /// Visible/displayed frame height, parsed from the filename.
    pub height: u32,
}

impl Md5Entry {
    /// Return the visible MD5 as a lowercase 32-char hex string, or
    /// `None` if the entry has no visible digest (hidden-frame rows).
    ///
    /// Convenience for the harness call sites that compare against a
    /// hex string produced by [`crate::hex16`].
    ///
    /// # Transitional shim (G8.0 follow-up)
    ///
    /// G8.0 review (2026-05-21, DA "Concerns"): every call site
    /// currently writes `entry.md5_hex_visible().as_deref() ==
    /// Some(got)`, which collapses two semantically distinct states
    /// into a single "mismatch" bucket:
    ///
    ///   - `show_frame == true && md5_visible == None` — the ledger
    ///     row claims a displayed frame but carries no digest. This
    ///     should be a ledger-parse error, not a frame mismatch.
    ///   - `md5_visible == Some(_)` but the digest doesn't match what
    ///     the decoder produced. This is a real frame mismatch.
    ///
    ///  Today the on-disk libvpx `.md5` format never lets the first
    ///  state occur (every row carries a digest), so the conflation is
    ///  harmless. Once the in-memory schema grows hidden-frame rows or
    ///  partial-ledger fixtures, the call sites must distinguish them.
    ///
    /// Tracked for G8.1 removal once consumers learn to handle the
    /// `Option` directly.
    #[must_use]
    pub fn md5_hex_visible(&self) -> Option<String> {
        self.md5_visible.map(crate::hex16)
    }
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
/// Each parsed row is assigned `packet_index = row_index`,
/// `sub_frame_index = 0`, `show_frame = true`, `md5_visible =
/// Some(parsed)` — matching the libvpx on-disk `.md5` format which
/// only carries displayed frames. See module docs for the rationale.
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
    // `row_index` tracks the displayed-frame ordinal (incremented only
    // for non-blank lines that successfully parse). It is therefore the
    // `packet_index` we want to write into `Md5Entry`. It is NOT the
    // same as `i` (which is `.enumerate()` over EVERY line including
    // blanks): packet_index = displayed-frame ordinal (skipping blank
    // rows) ≠ enumerate index.
    let mut row_index: usize = 0;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end();
        if line.is_empty() {
            continue;
        }
        let (md5_hex, fname) = line.split_once("  ").ok_or_else(|| Md5FileError::BadLine {
            lineno: i + 1,
            line: line.to_string(),
        })?;
        if md5_hex.len() != 32 || !md5_hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(Md5FileError::BadLine {
                lineno: i + 1,
                line: line.to_string(),
            });
        }
        let (w, h, _idx) =
            parse_resolution_index(fname).ok_or_else(|| Md5FileError::BadFilename {
                lineno: i + 1,
                filename: fname.to_string(),
            })?;
        let digest = parse_md5_hex(md5_hex).ok_or_else(|| Md5FileError::BadLine {
            lineno: i + 1,
            line: line.to_string(),
        })?;
        entries.push(Md5Entry {
            packet_index: row_index,
            sub_frame_index: 0,
            show_frame: true,
            md5_visible: Some(digest),
            width: w,
            height: h,
        });
        row_index += 1;
    }
    Ok(entries)
}

/// Parse a 32-char lowercase hex string into a 16-byte digest. Returns
/// `None` if the string is the wrong length or contains a non-hex
/// character.
fn parse_md5_hex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 {
        return None;
    }
    let mut out = [0u8; 16];
    let bytes = s.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_val(bytes[2 * i])?;
        let lo = hex_val(bytes[2 * i + 1])?;
        *slot = (hi << 4) | lo;
    }
    Some(out)
}

#[inline]
fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
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
    fn parse_md5_hex_round_trip() {
        let hex = "c3fbb7abbdb5bd4ed4a7e34768c17df1";
        let bytes = parse_md5_hex(hex).unwrap();
        assert_eq!(crate::hex16(bytes), hex);
    }

    #[test]
    fn parse_md5_hex_rejects_garbage() {
        assert!(parse_md5_hex("").is_none());
        assert!(parse_md5_hex("nothex").is_none());
        // 31 chars — wrong length.
        assert!(parse_md5_hex("c3fbb7abbdb5bd4ed4a7e34768c17df").is_none());
        // 33 chars.
        assert!(parse_md5_hex("c3fbb7abbdb5bd4ed4a7e34768c17df12").is_none());
        // Non-hex char.
        assert!(parse_md5_hex("c3fbb7abbdb5bd4ed4a7e34768c17dfZ").is_none());
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
        // G8 widened shape: row 0 carries packet_index=0, sub_frame=0,
        // show_frame=true, md5_visible=Some(...).
        assert_eq!(entries[0].packet_index, 0);
        assert_eq!(entries[0].sub_frame_index, 0);
        assert!(entries[0].show_frame);
        assert_eq!(
            entries[0].md5_hex_visible().as_deref(),
            Some("c3fbb7abbdb5bd4ed4a7e34768c17df1")
        );
        assert_eq!((entries[0].width, entries[0].height), (352, 288));
        assert_eq!(entries[1].packet_index, 1);
        assert_eq!(
            entries[1].md5_hex_visible().as_deref(),
            Some("08203c2595bdb2d58ead6f921345d699")
        );
        assert_eq!((entries[1].width, entries[1].height), (352, 288));
    }

    #[test]
    fn read_md5_file_bad_format() {
        let dir = std::env::temp_dir();
        let p = dir.join("rvp9_test_md5_bad.md5");
        std::fs::write(&p, "not a valid line\n").unwrap();
        assert!(matches!(
            read_md5_file(&p),
            Err(Md5FileError::BadLine { .. })
        ));
    }

    /// Round-trip: parse a `.md5` row, format the digest back to hex,
    /// confirm it matches the original 32-char string. Pins the
    /// G8 in-memory representation against the pre-G8 `md5_hex: String`
    /// behaviour.
    #[test]
    fn md5_hex_visible_round_trip() {
        let dir = std::env::temp_dir();
        let p = dir.join("rvp9_test_md5_round_trip.md5");
        let hex0 = "c3fbb7abbdb5bd4ed4a7e34768c17df1";
        let hex1 = "08203c2595bdb2d58ead6f921345d699";
        std::fs::write(
            &p,
            format!(
                "{hex0}  foo-352x288-0001.i420\n\
                 {hex1}  foo-352x288-0002.i420\n",
            ),
        )
        .unwrap();
        let entries = read_md5_file(&p).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].md5_hex_visible().as_deref(), Some(hex0));
        assert_eq!(entries[1].md5_hex_visible().as_deref(), Some(hex1));
        // Hidden-frame row representation: build by hand and confirm
        // `md5_hex_visible` returns `None`.
        let hidden = Md5Entry {
            packet_index: 7,
            sub_frame_index: 1,
            show_frame: false,
            md5_visible: None,
            width: 352,
            height: 288,
        };
        assert!(hidden.md5_hex_visible().is_none());
    }
}
