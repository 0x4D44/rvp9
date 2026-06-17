//! MD5 (RFC 1321) and the `vpxdec --md5` visible-plane digest.
//!
//! Computed in-process so the toolchain doesn't depend on an external
//! `md5sum`. Used by the conformance harness and by `rvp9dec --md5` /
//! `--framemd5`.

use rvp9_core::frame::Yv12Buffer;
use std::path::Path;

/// MD5 of a file. Computed in-process so we don't depend on an external
/// `md5sum`. Implementation uses the standard RFC 1321 spec; ~80 LOC.
///
/// # Errors
///
/// Forwards any [`std::io::Error`] from reading `path`.
pub fn md5_of_file(path: &Path) -> std::io::Result<[u8; 16]> {
    let data = std::fs::read(path)?;
    Ok(md5(&data))
}

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

const K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// RFC 1321 MD5 digest of `data`.
#[must_use]
pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    // Pre-processing: pad with 0x80, then zeros, then 64-bit LE length.
    let bit_len: u64 = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            m[i] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g): (u32, usize) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&a0.to_le_bytes());
    out[4..8].copy_from_slice(&b0.to_le_bytes());
    out[8..12].copy_from_slice(&c0.to_le_bytes());
    out[12..16].copy_from_slice(&d0.to_le_bytes());
    out
}

/// Lower-case hex encoding of a 16-byte digest.
#[must_use]
pub fn hex16(bytes: [u8; 16]) -> String {
    let mut s = String::with_capacity(32);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// MD5 of a decoded YV12 buffer in the order/dimensions libvpx's
/// `vpxdec --md5` uses: Y, then U, then V, each row hashed in raster
/// order over the **visible** (cropped) plane width only — border and
/// stride padding are excluded.
///
/// The U/V plane dimensions are derived from the buffer's chroma
/// format via [`Yv12Buffer::chroma_width`] / [`chroma_height`]:
/// - `Yuv420` → `((crop_w + 1) / 2, (crop_h + 1) / 2)`
/// - `Yuv422` → `((crop_w + 1) / 2, crop_h)`
/// - `Yuv440` → `(crop_w, (crop_h + 1) / 2)`
/// - `Yuv444` → `(crop_w, crop_h)`
///
/// Without this widening (the v0 spelling was `crop_w.div_ceil(2)` /
/// `crop_h.div_ceil(2)` hardcoded for 4:2:0) the harness would MD5
/// phantom bytes — or stop short of valid bytes — for Profile 1
/// streams.
///
/// Reference: `vpxdec.c:update_image_md5` (loops `Y, U, V` planes,
/// hashing `vpx_img_plane_width(img, plane)` bytes per row for
/// `vpx_img_plane_height(img, plane)` rows) + `tools_common.c:
/// vpx_img_plane_width/height` (the `(crop + ss) >> ss` rounding for
/// chroma planes).
///
/// [`Yv12Buffer::chroma_width`]: rvp9_core::frame::Yv12Buffer::chroma_width
/// [`chroma_height`]: rvp9_core::frame::Yv12Buffer::chroma_height
#[must_use]
pub fn md5_of_yv12_visible(buf: &Yv12Buffer) -> [u8; 16] {
    use rvp9_core::frame::PlaneRef;

    let crop_w = buf.width() as usize;
    let crop_h = buf.height() as usize;
    let uv_w = buf.chroma_width() as usize;
    let uv_h = buf.chroma_height() as usize;

    // Pre-size: Y + 2 × chroma.
    let mut bytes: Vec<u8> = Vec::with_capacity(crop_w * crop_h + 2 * uv_w * uv_h);
    let push_plane = |dst: &mut Vec<u8>, plane: PlaneRef<'_>, w: usize, h: usize| {
        let stride = plane.stride();
        let data = plane.data();
        for r in 0..h {
            let start = r * stride;
            dst.extend_from_slice(&data[start..start + w]);
        }
    };
    push_plane(&mut bytes, buf.y_plane(), crop_w, crop_h);
    push_plane(&mut bytes, buf.u_plane(), uv_w, uv_h);
    push_plane(&mut bytes, buf.v_plane(), uv_w, uv_h);
    md5(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvp9_core::bitstream::ChromaFormat;
    use rvp9_core::frame::{BorderUsage, ByteAlignment, Yv12Buffer};

    #[test]
    fn md5_empty() {
        assert_eq!(hex16(md5(b"")), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn md5_abc() {
        assert_eq!(hex16(md5(b"abc")), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn md5_quick_brown_fox() {
        assert_eq!(
            hex16(md5(b"The quick brown fox jumps over the lazy dog")),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }

    /// G7 oracle. `md5_of_yv12_visible` must hash exactly the visible
    /// plane bytes for every chroma sub-mode — for each format the
    /// total hashed byte count is `Y_visible + 2 × chroma_visible`,
    /// and the chroma extent follows libvpx's `(crop + ss) >> ss`. The
    /// pre-G7 spelling hardcoded `crop.div_ceil(2)` and would silently
    /// either hash phantom bytes (4:4:4 → reads past the visible
    /// chroma extent) or stop short (4:2:2 / 4:4:0).
    ///
    /// Verifies by:
    /// 1. Filling each plane with a known constant.
    /// 2. Recomputing the expected MD5 over the same byte stream the
    ///    function should walk (`Y, U, V`, each `chroma_w × chroma_h`
    ///    for U/V), and asserting equality.
    /// 3. Also checking the digest *differs* across formats — same Y
    ///    dims + same fills + different chroma extents → different hash.
    #[test]
    fn md5_of_yv12_visible_handles_all_chroma_formats() {
        let formats = [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv440,
            ChromaFormat::Yuv444,
        ];
        // 17×11 — odd width AND height, so the `(crop+ss)>>ss` rounding
        // bites on both axes for the subsampled sub-modes.
        let (crop_w, crop_h) = (17u32, 11u32);
        let mut digests = Vec::with_capacity(formats.len());

        for cf in formats {
            let mut buf = Yv12Buffer::alloc(
                crop_w,
                crop_h,
                cf,
                BorderUsage::Decoder,
                ByteAlignment::Legacy,
            )
            .expect("alloc");
            buf.y_plane_mut().fill(0x11);
            buf.u_plane_mut().fill(0x22);
            buf.v_plane_mut().fill(0x33);

            // Compute the expected stream the function should walk.
            let cw = buf.chroma_width() as usize;
            let ch = buf.chroma_height() as usize;
            let mut expected = Vec::with_capacity((crop_w * crop_h) as usize + 2 * cw * ch);
            expected.extend(std::iter::repeat_n(0x11u8, (crop_w * crop_h) as usize));
            expected.extend(std::iter::repeat_n(0x22u8, cw * ch));
            expected.extend(std::iter::repeat_n(0x33u8, cw * ch));
            let expected_md5 = md5(&expected);

            let got = md5_of_yv12_visible(&buf);
            assert_eq!(
                got, expected_md5,
                "md5_of_yv12_visible mismatch for {cf:?} (chroma {cw}×{ch})"
            );
            digests.push((cf, got));
        }

        // Every chroma format produces a distinct digest (different
        // total byte count under the same fill).
        for i in 0..digests.len() {
            for j in (i + 1)..digests.len() {
                assert_ne!(
                    digests[i].1, digests[j].1,
                    "{:?} and {:?} produced the same digest",
                    digests[i].0, digests[j].0
                );
            }
        }
    }
}
