//! Packet-level plumbing for spatially layered VP9 access units.
//!
//! The stateful encoder shares decoder-visible reference and entropy state
//! across otherwise independent per-layer encoders. Components are emitted
//! low-to-high so enhancement pictures can use the freshly reconstructed base
//! layer. A keyframe-only wrapper remains available for narrow callers.

use rvp9_core::bitstream::superframe::SuperframeBuilder;

use crate::EncodeError;
use crate::frame_loop::{
    EncodedFrame, EncoderConfig, EncoderParallelism, FrameKind, SpatialFrameControl,
    SpatialSharedState, StreamEncoder, YuvFrameRef, strided_plane_covers,
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
                "spatial layer {width}x{height} is outside the encoder's current supported geometry"
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

/// Stateful two-layer spatial-SVC encoder.
///
/// Components are emitted low-to-high in one VP9 superframe. The base layer
/// refreshes map entry 0. The enhancement layer refreshes entry 1 and uses
/// the freshly coded base reconstruction as its spatial reference. Later
/// enhancement pictures also retain their prior same-resolution picture as
/// LAST, matching pinned libvpx's fixed no-temporal-layer topology.
pub struct SpatialEncoder {
    plan: SpatialLayerPlan,
    encoders: Vec<StreamEncoder>,
    shared: SpatialSharedState,
    keyframe_interval: u32,
    access_unit_index: u32,
    poisoned: bool,
}

impl SpatialEncoder {
    /// Construct a two-layer spatial encoder using default parallelism.
    ///
    /// # Errors
    ///
    /// Rejects delayed, ALTREF, temporal-SVC, denoised, non-two-layer, and
    /// non-inter-layer-prediction configurations in this first production
    /// slice, plus invalid geometry or rate plans.
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
                "spatial access units require lag_in_frames == 0",
            ));
        }
        if template.enable_alt_ref {
            return Err(SpatialEncodeError::UnsupportedMode(
                "spatial SVC is mutually exclusive with ALTREF",
            ));
        }
        if template.temporal_layers != 1 {
            return Err(SpatialEncodeError::UnsupportedMode(
                "combined temporal and spatial SVC is not wired yet",
            ));
        }
        if template.noise_sensitivity != 0 {
            return Err(SpatialEncodeError::UnsupportedMode(
                "spatial SVC denoiser state is not wired yet",
            ));
        }
        if spatial.number_spatial_layers != 2 {
            return Err(SpatialEncodeError::UnsupportedMode(
                "the stateful spatial encoder currently requires exactly two layers",
            ));
        }
        if !matches!(
            spatial.inter_layer_pred,
            crate::svc::InterLayerPred::On | crate::svc::InterLayerPred::OnConstrained
        ) {
            return Err(SpatialEncodeError::UnsupportedMode(
                "the stateful spatial encoder currently requires inter-layer prediction",
            ));
        }

        let keyframe_interval = template.gop_length;
        let plan = SpatialLayerPlan::canonical(
            spatial,
            template.width,
            template.height,
            template.target_bitrate_bps,
        )?;
        let mut encoders = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            if !layer.width.is_multiple_of(64)
                || !layer.height.is_multiple_of(8)
                || layer.height < 64
            {
                return Err(SpatialEncodeError::UnsupportedLayerGeometry {
                    width: layer.width,
                    height: layer.height,
                });
            }
            let mut layer_config = template;
            layer_config.width = layer.width;
            layer_config.height = layer.height;
            layer_config.target_bitrate_bps = layer.target_bitrate_bps;
            encoders.push(
                StreamEncoder::new_with_parallelism(layer_config, parallelism)
                    .map_err(SpatialEncodeError::Encoder)?,
            );
        }
        let base = plan
            .layers
            .first()
            .expect("a validated two-layer plan has a base layer");
        let base_dimensions = (base.width as usize, base.height as usize);
        Ok(Self {
            plan,
            encoders,
            shared: SpatialSharedState::new(base_dimensions.0, base_dimensions.1),
            keyframe_interval,
            access_unit_index: 0,
            poisoned: false,
        })
    }

    /// Validated canonical layer plan owned by this encoder.
    #[must_use]
    pub fn plan(&self) -> &SpatialLayerPlan {
        &self.plan
    }

    /// Number of access units successfully encoded so far.
    #[must_use]
    pub const fn access_unit_index(&self) -> u32 {
        self.access_unit_index
    }

    /// Encode one top-resolution source into a two-component spatial access
    /// unit in decoder order.
    ///
    /// # Errors
    ///
    /// Returns an input, scale, bitstream, or component error. Any component
    /// failure poisons the coordinator because lower layers may have already
    /// advanced their private rate-control state.
    pub fn encode_access_unit(
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
            .expect("a validated spatial plan has a top layer");
        let full = copy_tight_i420(source, top.width as usize, top.height as usize)?;
        let sources = self
            .plan
            .layers
            .iter()
            .map(|layer| build_layer_source(&full, layer))
            .collect::<Result<Vec<_>, _>>()?;
        let key_access_unit = self
            .access_unit_index
            .is_multiple_of(self.keyframe_interval);

        let mut builder = SuperframeBuilder::new();
        let mut layers = Vec::with_capacity(self.encoders.len());
        for (layer_index, ((encoder, layer), input)) in self
            .encoders
            .iter_mut()
            .zip(&self.plan.layers)
            .zip(&sources)
            .enumerate()
        {
            let control = spatial_frame_control(layer_index, key_access_unit);
            let view = YuvFrameRef {
                y: &input.y,
                u: &input.u,
                v: &input.v,
                y_stride: input.width,
                u_stride: input.width / 2,
                v_stride: input.width / 2,
            };
            encoder.swap_spatial_shared_state(&mut self.shared);
            let encoded = encoder.encode_spatial_component(&view, control);
            encoder.swap_spatial_shared_state(&mut self.shared);
            let encoded = match encoded {
                Ok(encoded) => encoded,
                Err(error) => {
                    self.poisoned = true;
                    return Err(SpatialEncodeError::Encoder(error));
                }
            };
            layers.push(layer_metadata(layer, &encoded));
            if let Err(error) = builder.push_frame(encoded.data) {
                self.poisoned = true;
                return Err(SpatialEncodeError::Encoder(EncodeError::Bitstream(error)));
            }
        }
        self.access_unit_index = self.access_unit_index.saturating_add(1);

        Ok(SpatialAccessUnit {
            data: builder.finish(),
            layers,
        })
    }
}

fn spatial_frame_control(layer_index: usize, key_access_unit: bool) -> SpatialFrameControl {
    match (layer_index, key_access_unit) {
        (0, true) => SpatialFrameControl::Key,
        (0, false) => SpatialFrameControl::Inter {
            refresh_frame_flags: 0x01,
            ref_frame_indices: [0, 0, 2],
            allow_golden_ref: false,
        },
        (1, true) => SpatialFrameControl::Inter {
            refresh_frame_flags: 0x02,
            ref_frame_indices: [0, 1, 2],
            allow_golden_ref: false,
        },
        (1, false) => SpatialFrameControl::Inter {
            refresh_frame_flags: 0x02,
            ref_frame_indices: [1, 0, 2],
            allow_golden_ref: true,
        },
        _ => unreachable!("the stateful spatial encoder validates exactly two layers"),
    }
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

    fn decoded_i420(frame: &rvp9_core::frame::FrameBuf) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        fn tight(data: &[u8], stride: usize, row_width: usize, rows: usize) -> Vec<u8> {
            let mut result = Vec::with_capacity(row_width * rows);
            for row in 0..rows {
                let offset = row * stride;
                result.extend_from_slice(&data[offset..offset + row_width]);
            }
            result
        }

        let width = frame.width() as usize;
        let height = frame.height() as usize;
        (
            tight(
                frame.y_plane().data(),
                frame.y_plane().stride(),
                width,
                height,
            ),
            tight(
                frame.u_plane().data(),
                frame.u_plane().stride(),
                width / 2,
                height / 2,
            ),
            tight(
                frame.v_plane().data(),
                frame.v_plane().stride(),
                width / 2,
                height / 2,
            ),
        )
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
    fn two_layer_inter_access_units_share_reference_map_and_decode_in_order() {
        let width = 256usize;
        let height = 128usize;
        let mut template = EncoderConfig::new(width as u32, height as u32);
        template.target_bitrate_bps = 100_000;
        template.speed = 8;
        template.gop_length = 3;
        let mut encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 2,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .unwrap();
        let mut decoder = Decoder::new();

        for access_unit_index in 0..4usize {
            let (mut y, mut u, mut v) = source(width, height);
            for (index, value) in y.iter_mut().enumerate() {
                *value =
                    value.wrapping_add((access_unit_index * 19 + index / width + index % 7) as u8);
            }
            for value in &mut u {
                *value = value.wrapping_add((access_unit_index * 11) as u8);
            }
            for value in &mut v {
                *value = value.wrapping_add((access_unit_index * 17) as u8);
            }
            let access_unit = encoder
                .encode_access_unit(&YuvFrameRef {
                    y: &y,
                    u: &u,
                    v: &v,
                    y_stride: width,
                    u_stride: width / 2,
                    v_stride: width / 2,
                })
                .unwrap();
            assert_eq!(access_unit.layers.len(), 2);
            assert_eq!(encoder.access_unit_index(), access_unit_index as u32 + 1);

            let parsed = decoder.parse_packet(&access_unit.data).unwrap();
            assert_eq!(parsed.count(), 2);
            assert_eq!(parsed.output_frame_index(), Some(1));
            let key_access_unit = access_unit_index.is_multiple_of(3);
            let expected = if key_access_unit {
                [(0xff, [0, 0, 0]), (0x02, [0, 1, 2])]
            } else {
                [(0x01, [0, 0, 2]), (0x02, [1, 0, 2])]
            };

            for (component_index, (component, &(want_width, want_height))) in parsed
                .frames()
                .iter()
                .zip(&[(128, 64), (256, 128)])
                .enumerate()
            {
                assert!(!component.header.error_resilient);
                assert!(
                    component
                        .header
                        .tail
                        .as_ref()
                        .expect("coded component has a header tail")
                        .frame_parallel_decoding_mode
                );
                match component.header.kind {
                    WireFrameKind::KeyFrame => {
                        assert!(key_access_unit && component_index == 0);
                    }
                    WireFrameKind::InterFrame {
                        refresh_frame_flags,
                        ref_frame_indices,
                        ..
                    } => {
                        assert_eq!(
                            (refresh_frame_flags, ref_frame_indices),
                            expected[component_index]
                        );
                    }
                    kind => panic!("unexpected spatial component kind: {kind:?}"),
                }
                assert_eq!(component.header.width, want_width);
                assert_eq!(component.header.height, want_height);
                let decoded = decoder.decode_frame(component).unwrap();
                assert_eq!(decoded.width(), want_width);
                assert_eq!(decoded.height(), want_height);
                let decoded = decoded_i420(&decoded);
                let encoded = encoder
                    .shared
                    .named_reference_planes_for_test(component_index);
                assert_eq!(decoded.0, encoded.0);
                assert_eq!(decoded.1, encoded.1);
                assert_eq!(decoded.2, encoded.2);
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
