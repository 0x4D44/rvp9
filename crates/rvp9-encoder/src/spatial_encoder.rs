//! Packet-level plumbing for spatially layered VP9 access units.
//!
//! This first slice is intentionally keyframe-only. It proves the source
//! pyramid, per-layer encoder ownership, low-to-high component order, and VP9
//! superframe assembly without pretending that independent inter streams are
//! spatial SVC. Inter pictures need a shared decoder reference-map schedule
//! and cross-layer reconstruction injection; those land in the next slice.

use rvp9_core::bitstream::superframe::SuperframeBuilder;

use crate::EncodeError;
use crate::frame_loop::{
    EncodedFrame, EncoderConfig, EncoderParallelism, FrameKind, StreamEncoder, YuvFrameRef,
    strided_plane_covers,
};
use crate::scale_source::{ScaleSourceError, ScaledI420, scale_i420_2x};
use crate::svc::{SpatialLayerPlan, SpatialLayerSpec, SpatialSvcConfig, SpatialSvcConfigError};

/// Error returned by packet-level spatial encoding.
#[derive(Debug)]
#[non_exhaustive]
pub enum SpatialEncodeError {
    /// The canonical layer plan is invalid.
    Config(SpatialSvcConfigError),
    /// A canonical layer cannot yet be represented by the single-layer
    /// encoder's current superblock geometry.
    UnsupportedLayerGeometry {
        /// Layer width in pixels.
        width: u32,
        /// Layer height in pixels.
        height: u32,
    },
    /// The keyframe-only wrapper requires the synchronous single-temporal-
    /// layer encoder path.
    UnsupportedMode(&'static str),
    /// Input planes or strides do not cover the configured source frame.
    InvalidInput(&'static str),
    /// Source-pyramid construction failed.
    Scale(ScaleSourceError),
    /// A component encoder failed. The wrapper is poisoned after this error
    /// because some lower layers may already have advanced.
    Encoder(EncodeError),
    /// A previous component failure left per-layer state out of step.
    Poisoned,
}

impl core::fmt::Display for SpatialEncodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Config(error) => write!(f, "invalid spatial configuration: {error}"),
            Self::UnsupportedLayerGeometry { width, height } => write!(
                f,
                "spatial layer {width}x{height} is outside the encoder's current multiple-of-64x8 geometry"
            ),
            Self::UnsupportedMode(reason) => write!(f, "unsupported spatial encode mode: {reason}"),
            Self::InvalidInput(reason) => write!(f, "invalid spatial source frame: {reason}"),
            Self::Scale(error) => write!(f, "spatial source scaling failed: {error}"),
            Self::Encoder(error) => write!(f, "spatial component encode failed: {error}"),
            Self::Poisoned => f.write_str(
                "spatial encoder state is out of step after an earlier component failure",
            ),
        }
    }
}

impl std::error::Error for SpatialEncodeError {}

impl From<SpatialSvcConfigError> for SpatialEncodeError {
    fn from(error: SpatialSvcConfigError) -> Self {
        Self::Config(error)
    }
}

impl From<ScaleSourceError> for SpatialEncodeError {
    fn from(error: ScaleSourceError) -> Self {
        Self::Scale(error)
    }
}

/// Metadata for one low-to-high component in a spatial access unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpatialEncodedLayer {
    /// Zero-based spatial layer id.
    pub spatial_layer_id: u8,
    /// Coded luma width.
    pub width: u32,
    /// Coded luma height.
    pub height: u32,
    /// Component byte length, excluding the shared superframe index.
    pub byte_len: usize,
    /// Q-index selected by this layer's rate controller.
    pub q_index: u8,
    /// Component bits seen by that layer's rate controller.
    pub actual_bits: u32,
}

/// One packet containing spatial components in low-to-high decode order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpatialAccessUnit {
    /// VP9 packet bytes. Multiple layers carry a trailing superframe index.
    pub data: Vec<u8>,
    /// Per-component metadata in the same low-to-high order as `data`.
    pub layers: Vec<SpatialEncodedLayer>,
}

/// Keyframe-only spatial access-unit encoder.
///
/// Each layer owns an ordinary [`StreamEncoder`] forced to a one-frame GOP.
/// This makes every component independently intra-coded, so packing them into
/// one low-to-high superframe is valid without shared reference-map state.
/// The type deliberately has no general `encode_frame` method: persistent
/// inter-frame spatial SVC belongs to the forthcoming shared-map encoder.
pub struct SpatialKeyframeEncoder {
    plan: SpatialLayerPlan,
    encoders: Vec<StreamEncoder>,
    poisoned: bool,
}

impl SpatialKeyframeEncoder {
    /// Construct a keyframe-only spatial encoder using default parallelism.
    ///
    /// The template supplies all ordinary encoding choices. Width, height,
    /// target bitrate, and GOP length are replaced per layer; the GOP is fixed
    /// at one so every call emits only keyframes.
    ///
    /// # Errors
    ///
    /// Rejects delayed/ALTREF/temporal-SVC templates, invalid layer plans, and
    /// canonical dimensions the current single-layer encoder cannot encode.
    pub fn new(
        template: EncoderConfig,
        spatial: SpatialSvcConfig,
    ) -> Result<Self, SpatialEncodeError> {
        Self::new_with_parallelism(template, spatial, EncoderParallelism::default())
    }

    /// Construct with explicit per-layer worker and tile limits.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn new_with_parallelism(
        template: EncoderConfig,
        spatial: SpatialSvcConfig,
        parallelism: EncoderParallelism,
    ) -> Result<Self, SpatialEncodeError> {
        if template.lag_in_frames != 0 {
            return Err(SpatialEncodeError::UnsupportedMode(
                "keyframe access units require lag_in_frames == 0",
            ));
        }
        if template.enable_alt_ref {
            return Err(SpatialEncodeError::UnsupportedMode(
                "keyframe access units are mutually exclusive with ALTREF",
            ));
        }
        if template.temporal_layers != 1 {
            return Err(SpatialEncodeError::UnsupportedMode(
                "keyframe access units currently require temporal_layers == 1",
            ));
        }

        let plan = SpatialLayerPlan::canonical(
            spatial,
            template.width,
            template.height,
            template.target_bitrate_bps,
        )?;
        let mut encoders = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            if !layer.width.is_multiple_of(64) || !layer.height.is_multiple_of(8) {
                return Err(SpatialEncodeError::UnsupportedLayerGeometry {
                    width: layer.width,
                    height: layer.height,
                });
            }
            let mut layer_config = template;
            layer_config.width = layer.width;
            layer_config.height = layer.height;
            layer_config.target_bitrate_bps = layer.target_bitrate_bps;
            layer_config.gop_length = 1;
            encoders.push(
                StreamEncoder::new_with_parallelism(layer_config, parallelism)
                    .map_err(SpatialEncodeError::Encoder)?,
            );
        }
        Ok(Self {
            plan,
            encoders,
            poisoned: false,
        })
    }

    /// Validated canonical layer plan owned by this encoder.
    #[must_use]
    pub fn plan(&self) -> &SpatialLayerPlan {
        &self.plan
    }

    /// Encode one source into a low-to-high, keyframe-only access unit.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialEncodeError::InvalidInput`] for invalid source plane
    /// coverage, [`SpatialEncodeError::Scale`] for a failed canonical scale,
    /// or [`SpatialEncodeError::Encoder`] for a component failure. A component
    /// failure poisons the wrapper so partially advanced layer state cannot be
    /// reused accidentally.
    pub fn encode_keyframe_access_unit(
        &mut self,
        source: &YuvFrameRef<'_>,
    ) -> Result<SpatialAccessUnit, SpatialEncodeError> {
        if self.poisoned {
            return Err(SpatialEncodeError::Poisoned);
        }
        let top = self
            .plan
            .layers
            .last()
            .expect("a validated spatial plan has at least one layer");
        let full = copy_tight_i420(source, top.width as usize, top.height as usize)?;
        let sources = self
            .plan
            .layers
            .iter()
            .map(|layer| build_layer_source(&full, layer))
            .collect::<Result<Vec<_>, _>>()?;

        let mut builder = SuperframeBuilder::new();
        let mut layers = Vec::with_capacity(self.encoders.len());
        for ((encoder, layer), input) in self
            .encoders
            .iter_mut()
            .zip(&self.plan.layers)
            .zip(&sources)
        {
            let view = YuvFrameRef {
                y: &input.y,
                u: &input.u,
                v: &input.v,
                y_stride: input.width,
                u_stride: input.width / 2,
                v_stride: input.width / 2,
            };
            let encoded = match encoder.encode_frame(&view) {
                Ok(encoded) => encoded,
                Err(error) => {
                    self.poisoned = true;
                    return Err(SpatialEncodeError::Encoder(error));
                }
            };
            debug_assert_eq!(encoded.frame_kind, FrameKind::Key);
            layers.push(layer_metadata(layer, &encoded));
            if let Err(error) = builder.push_frame(encoded.data) {
                self.poisoned = true;
                return Err(SpatialEncodeError::Encoder(EncodeError::Bitstream(error)));
            }
        }

        Ok(SpatialAccessUnit {
            data: builder.finish(),
            layers,
        })
    }
}

fn layer_metadata(layer: &SpatialLayerSpec, encoded: &EncodedFrame) -> SpatialEncodedLayer {
    SpatialEncodedLayer {
        spatial_layer_id: layer.spatial_layer_id,
        width: layer.width,
        height: layer.height,
        byte_len: encoded.data.len(),
        q_index: encoded.q_index,
        actual_bits: encoded.actual_bits,
    }
}

fn copy_tight_i420(
    source: &YuvFrameRef<'_>,
    width: usize,
    height: usize,
) -> Result<ScaledI420, SpatialEncodeError> {
    let uv_width = width / 2;
    let uv_height = height / 2;
    if !strided_plane_covers(source.y.len(), width, height, source.y_stride)
        || !strided_plane_covers(source.u.len(), uv_width, uv_height, source.u_stride)
        || !strided_plane_covers(source.v.len(), uv_width, uv_height, source.v_stride)
    {
        return Err(SpatialEncodeError::InvalidInput(
            "planes must cover every configured strided row",
        ));
    }
    let mut y = vec![0; width * height];
    let mut u = vec![0; uv_width * uv_height];
    let mut v = vec![0; uv_width * uv_height];
    for row in 0..height {
        y[row * width..(row + 1) * width]
            .copy_from_slice(&source.y[row * source.y_stride..row * source.y_stride + width]);
    }
    for row in 0..uv_height {
        u[row * uv_width..(row + 1) * uv_width]
            .copy_from_slice(&source.u[row * source.u_stride..row * source.u_stride + uv_width]);
        v[row * uv_width..(row + 1) * uv_width]
            .copy_from_slice(&source.v[row * source.v_stride..row * source.v_stride + uv_width]);
    }
    Ok(ScaledI420 {
        width,
        height,
        y,
        u,
        v,
    })
}

fn build_layer_source(
    full: &ScaledI420,
    layer: &SpatialLayerSpec,
) -> Result<ScaledI420, ScaleSourceError> {
    let Some(filter) = layer.source_filter else {
        return Ok(full.clone());
    };
    let mut scaled = scale_i420_2x(&full.y, &full.u, &full.v, full.width, full.height, filter)?;
    while scaled.width > layer.width as usize || scaled.height > layer.height as usize {
        scaled = scale_i420_2x(
            &scaled.y,
            &scaled.u,
            &scaled.v,
            scaled.width,
            scaled.height,
            filter,
        )?;
    }
    if scaled.width != layer.width as usize || scaled.height != layer.height as usize {
        return Err(ScaleSourceError::InvalidDimensions);
    }
    Ok(scaled)
}

#[cfg(test)]
mod tests {
    use rvp9_core::bitstream::{FrameKind as WireFrameKind, Superframe, parse_superframe_index};
    use rvp9_decoder::Decoder;

    use super::*;
    use crate::svc::InterLayerPred;

    fn source(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let y = (0..width * height)
            .map(|index| ((index * 29 + index / width * 17) & 0xff) as u8)
            .collect();
        let uv_len = width * height / 4;
        let u = (0..uv_len)
            .map(|index| ((index * 11 + 43) & 0xff) as u8)
            .collect();
        let v = (0..uv_len)
            .map(|index| ((index * 7 + 101) & 0xff) as u8)
            .collect();
        (y, u, v)
    }

    #[test]
    fn two_layer_keyframe_access_units_parse_and_decode_low_to_high() {
        let width = 128usize;
        let height = 64usize;
        let mut template = EncoderConfig::new(width as u32, height as u32);
        template.target_bitrate_bps = 100_000;
        template.speed = 8;
        let mut encoder = SpatialKeyframeEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 2,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .unwrap();
        let (y, u, v) = source(width, height);
        let view = YuvFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride: width,
            u_stride: width / 2,
            v_stride: width / 2,
        };

        for _ in 0..2 {
            let access_unit = encoder.encode_keyframe_access_unit(&view).unwrap();
            assert_eq!(
                access_unit
                    .layers
                    .iter()
                    .map(|layer| (layer.spatial_layer_id, layer.width, layer.height))
                    .collect::<Vec<_>>(),
                [(0, 64, 32), (1, 128, 64)]
            );
            match parse_superframe_index(&access_unit.data).unwrap() {
                Superframe::Multi { sizes, count, .. } => {
                    assert_eq!(count, 2);
                    assert_eq!(
                        &sizes[..count],
                        &access_unit
                            .layers
                            .iter()
                            .map(|layer| layer.byte_len as u32)
                            .collect::<Vec<_>>()
                    );
                }
                Superframe::Single => panic!("two layers must carry a superframe index"),
            }

            let mut decoder = Decoder::new();
            let parsed = decoder.parse_packet(&access_unit.data).unwrap();
            assert_eq!(parsed.count(), 2);
            assert_eq!(parsed.output_frame_index(), Some(1));
            for (component, &(want_width, want_height)) in
                parsed.frames().iter().zip(&[(64, 32), (128, 64)])
            {
                assert_eq!(component.header.kind, WireFrameKind::KeyFrame);
                assert!(component.header.show_frame);
                assert_eq!(component.header.width, want_width);
                assert_eq!(component.header.height, want_height);
                let decoded = decoder.decode_frame(component).unwrap();
                assert_eq!(decoded.width(), want_width);
                assert_eq!(decoded.height(), want_height);
            }
        }
    }

    #[test]
    fn canonical_720p_three_layer_geometry_is_explicitly_deferred() {
        let template = EncoderConfig::new(1280, 720);
        let error = SpatialKeyframeEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .err()
        .expect("320x180 is not yet encodable");
        assert!(matches!(
            error,
            SpatialEncodeError::UnsupportedLayerGeometry {
                width: 320,
                height: 180
            }
        ));
    }

    #[test]
    fn keyframe_wrapper_rejects_stateful_modes() {
        let spatial = SpatialSvcConfig {
            number_spatial_layers: 2,
            inter_layer_pred: InterLayerPred::Off,
        };
        let mut template = EncoderConfig::new(128, 64);
        template.lag_in_frames = 1;
        assert!(matches!(
            SpatialKeyframeEncoder::new(template, spatial),
            Err(SpatialEncodeError::UnsupportedMode(_))
        ));
        template.lag_in_frames = 0;
        template.enable_alt_ref = true;
        assert!(matches!(
            SpatialKeyframeEncoder::new(template, spatial),
            Err(SpatialEncodeError::UnsupportedMode(_))
        ));
        template.enable_alt_ref = false;
        template.temporal_layers = 2;
        assert!(matches!(
            SpatialKeyframeEncoder::new(template, spatial),
            Err(SpatialEncodeError::UnsupportedMode(_))
        ));
    }

    #[test]
    fn tight_copy_accepts_padding_and_rejects_short_planes() {
        let width = 16usize;
        let height = 8usize;
        let (want_y, want_u, want_v) = source(width, height);
        let y_stride = width + 3;
        let uv_stride = width / 2 + 2;
        let mut y = vec![0xee; y_stride * height];
        let mut u = vec![0xdd; uv_stride * (height / 2)];
        let mut v = vec![0xcc; uv_stride * (height / 2)];
        for row in 0..height {
            y[row * y_stride..row * y_stride + width]
                .copy_from_slice(&want_y[row * width..(row + 1) * width]);
        }
        for row in 0..height / 2 {
            u[row * uv_stride..row * uv_stride + width / 2]
                .copy_from_slice(&want_u[row * width / 2..(row + 1) * width / 2]);
            v[row * uv_stride..row * uv_stride + width / 2]
                .copy_from_slice(&want_v[row * width / 2..(row + 1) * width / 2]);
        }
        let view = YuvFrameRef {
            y: &y,
            u: &u,
            v: &v,
            y_stride,
            u_stride: uv_stride,
            v_stride: uv_stride,
        };
        let copied = copy_tight_i420(&view, width, height).unwrap();
        assert_eq!(copied.y, want_y);
        assert_eq!(copied.u, want_u);
        assert_eq!(copied.v, want_v);

        let short = YuvFrameRef {
            y: &y[..y.len() - y_stride],
            ..view
        };
        assert!(matches!(
            copy_tight_i420(&short, width, height),
            Err(SpatialEncodeError::InvalidInput(_))
        ));
    }
}
