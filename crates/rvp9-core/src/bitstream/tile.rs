//! VP9 per-tile byte-buffer parser.
//!
//! Mirrors libvpx `get_tile_buffer` / `get_tile_buffers`
//! (`vp9/decoder/vp9_decodeframe.c:1649-1697`). Given the compressed
//! tile data (everything after the compressed header) and the
//! `(tile_cols, tile_rows)` configuration, this module splits the byte
//! stream into one `&[u8]` per tile.
//!
//! ## Wire format
//!
//! For all tiles except the last:
//!
//! - 4 bytes big-endian `size` field.
//! - `size` bytes of tile payload.
//!
//! For the last tile: no size field — the payload runs to the end of
//! the input slice.
//!
//! Tiles are stored in **row-major** order: row 0 cols 0..N, row 1
//! cols 0..N, etc.
//!
//! ## Maximum tile count
//!
//! libvpx caps `log2_tile_cols` at 6 (≤ 64 columns) and
//! `log2_tile_rows` at 2 (≤ 4 rows). VP9 spec: 64 × 4 = 256 tiles max.
//! `TileBuffers` here uses a `Vec<TileBuffer<'a>>` rather than a
//! fixed-size matrix; the caller knows `(tile_cols, tile_rows)` from
//! the uncompressed header.

use super::error::BitstreamError;

/// One tile's byte payload + position. Mirrors libvpx `TileBuffer`
/// (`vp9_decoder.h` — `data`, `size`, `col`; we add `row` since our
/// `tile_buffers()` returns a flat `Vec`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TileBuffer<'a> {
    /// Byte payload of this tile.
    pub data: &'a [u8],
    /// Tile column index (0..tile_cols).
    pub col: u32,
    /// Tile row index (0..tile_rows).
    pub row: u32,
}

/// Parse the per-tile byte buffers from a compressed-tile-data slice.
/// `tile_cols × tile_rows` tiles are extracted in row-major order.
///
/// Mirrors libvpx `get_tile_buffers` (`vp9_decodeframe.c:1682-1697`).
///
/// # Errors
///
/// - [`BitstreamError::Truncated`] if the slice is shorter than the
///   4-byte size prefix demanded by a non-last tile.
/// - [`BitstreamError::InvalidTileSize`] if a tile's declared `size`
///   exceeds the remaining bytes (libvpx
///   `vp9_decodeframe.c:1669-1671`).
///
/// # Panics
///
/// None.
pub fn parse_tile_buffers<'a>(
    data: &'a [u8],
    tile_cols: u32,
    tile_rows: u32,
) -> Result<Vec<TileBuffer<'a>>, BitstreamError> {
    if tile_cols == 0 || tile_rows == 0 {
        return Ok(Vec::new());
    }

    let total = (tile_cols as usize) * (tile_rows as usize);
    let mut out: Vec<TileBuffer<'a>> = Vec::with_capacity(total);

    let mut cursor: usize = 0;
    let total_minus_one = total - 1;
    let mut tile_index = 0usize;

    for r in 0..tile_rows {
        for c in 0..tile_cols {
            let is_last = tile_index == total_minus_one;
            let size = if is_last {
                data.len() - cursor
            } else {
                // 4-byte big-endian size prefix.
                if data.len() < cursor + 4 {
                    return Err(BitstreamError::Truncated {
                        needed: cursor + 4,
                        available: data.len(),
                    });
                }
                let bytes = &data[cursor..cursor + 4];
                let size = (u32::from(bytes[0]) << 24)
                    | (u32::from(bytes[1]) << 16)
                    | (u32::from(bytes[2]) << 8)
                    | u32::from(bytes[3]);
                cursor += 4;
                size as usize
            };

            if cursor + size > data.len() {
                return Err(BitstreamError::InvalidTileSize {
                    tile_row: r,
                    tile_col: c,
                    declared: size,
                    available: data.len() - cursor,
                });
            }

            out.push(TileBuffer {
                data: &data[cursor..cursor + size],
                col: c,
                row: r,
            });
            cursor += size;
            tile_index += 1;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a tile-buffer payload from per-tile byte slices,
    /// in row-major order. Emits a 4-byte BE size prefix before each
    /// tile except the last.
    fn build_tiles(tiles: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, t) in tiles.iter().enumerate() {
            if i + 1 < tiles.len() {
                let sz = t.len() as u32;
                out.extend_from_slice(&sz.to_be_bytes());
            }
            out.extend_from_slice(t);
        }
        out
    }

    #[test]
    fn single_tile_takes_whole_buffer() {
        let payload = [0x01u8, 0x02, 0x03, 0x04, 0x05];
        let tiles = parse_tile_buffers(&payload, 1, 1).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].data, &payload);
        assert_eq!(tiles[0].col, 0);
        assert_eq!(tiles[0].row, 0);
    }

    #[test]
    fn two_tiles_row_major() {
        let a = [0xaau8; 8];
        let b = [0xbbu8; 12];
        let buf = build_tiles(&[&a, &b]);
        let tiles = parse_tile_buffers(&buf, 2, 1).unwrap();
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].data, &a);
        assert_eq!(tiles[0].col, 0);
        assert_eq!(tiles[0].row, 0);
        assert_eq!(tiles[1].data, &b);
        assert_eq!(tiles[1].col, 1);
        assert_eq!(tiles[1].row, 0);
    }

    #[test]
    fn two_rows_one_col() {
        let a = [0xa1u8; 7];
        let b = [0xb1u8; 9];
        let buf = build_tiles(&[&a, &b]);
        let tiles = parse_tile_buffers(&buf, 1, 2).unwrap();
        assert_eq!(tiles[0].row, 0);
        assert_eq!(tiles[0].col, 0);
        assert_eq!(tiles[1].row, 1);
        assert_eq!(tiles[1].col, 0);
    }

    #[test]
    fn four_tile_grid_row_major_indexing() {
        let payloads: Vec<Vec<u8>> = (0..4).map(|i| vec![i as u8; 4 + i]).collect();
        let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
        let buf = build_tiles(&refs);
        let tiles = parse_tile_buffers(&buf, 2, 2).unwrap();
        assert_eq!(tiles.len(), 4);
        // Row-major: (0,0), (0,1), (1,0), (1,1).
        assert_eq!((tiles[0].row, tiles[0].col), (0, 0));
        assert_eq!((tiles[1].row, tiles[1].col), (0, 1));
        assert_eq!((tiles[2].row, tiles[2].col), (1, 0));
        assert_eq!((tiles[3].row, tiles[3].col), (1, 1));
        for (i, t) in tiles.iter().enumerate() {
            assert_eq!(t.data.len(), 4 + i);
        }
    }

    #[test]
    fn truncated_size_prefix_rejected() {
        // Claim 2 tiles, but only 3 bytes available (need 4 for size).
        let buf = [0x00u8, 0x00, 0x00];
        let err = parse_tile_buffers(&buf, 2, 1).unwrap_err();
        assert!(matches!(err, BitstreamError::Truncated { .. }));
    }

    #[test]
    fn declared_tile_size_exceeds_buffer_rejected() {
        // 4-byte BE size = 100, but only 4 trailing bytes available.
        let mut buf = vec![0u8, 0, 0, 100];
        buf.extend_from_slice(&[0xff; 4]); // 4 trailing payload bytes
        let err = parse_tile_buffers(&buf, 2, 1).unwrap_err();
        assert!(matches!(err, BitstreamError::InvalidTileSize { declared: 100, available: 4, .. }));
    }

    #[test]
    fn empty_tile_payload_ok() {
        // Single tile with empty payload — valid.
        let buf: [u8; 0] = [];
        let tiles = parse_tile_buffers(&buf, 1, 1).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].data.len(), 0);
    }

    #[test]
    fn zero_tile_dims_yields_empty() {
        let buf = [0u8; 10];
        let tiles = parse_tile_buffers(&buf, 0, 5).unwrap();
        assert!(tiles.is_empty());
        let tiles = parse_tile_buffers(&buf, 3, 0).unwrap();
        assert!(tiles.is_empty());
    }
}
