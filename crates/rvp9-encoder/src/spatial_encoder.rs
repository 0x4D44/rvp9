//! Packet-level plumbing for spatially layered VP9 access units.
//!
//! The stateful encoder shares decoder-visible reference and entropy state
//! across otherwise independent per-layer encoders. Components are emitted
//! low-to-high so enhancement pictures can use the freshly reconstructed base
//! layer. A keyframe-only wrapper remains available for narrow callers.

use rvp9_core::bitstream::superframe::SuperframeBuilder;

use crate::frame_loop::{
    EncodedFrame, EncoderConfig, EncoderParallelism, FrameKind, SpatialFrameControl,
    SpatialSharedState, StreamEncoder, YuvFrameRef, strided_plane_covers,
};
use crate::scale_source::{
    ScaleSourceError, ScaledI420, scale_i420_2x, scale_i420_nonnormative_4x,
};
use crate::svc::{SpatialLayerPlan, SpatialLayerSpec, SpatialSvcConfig, SpatialSvcConfigError};
use crate::{EncodeError, f02_oracle};

/// Error returned by packet-level spatial encoding.
#[derive(Debug)]
#[non_exhaustive]
pub enum SpatialEncodeError {
    /// The canonical layer plan is invalid.
    Config(SpatialSvcConfigError),
    /// A canonical layer cannot be represented by the spatial encoder's
    /// 4:2:0 MI geometry.
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
                "spatial layer {width}x{height} requires a positive 8-pixel-aligned width and positive even height"
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

/// Stateful two- or three-layer spatial-SVC encoder.
///
/// Components are emitted low-to-high in one VP9 superframe. The base layer
/// refreshes map entry 0. Each enhancement layer refreshes its matching map
/// entry and uses the freshly coded lower-layer reconstruction as its spatial
/// reference. Later enhancement pictures also retain their prior
/// same-resolution picture as LAST, matching pinned libvpx's fixed
/// no-temporal-layer topology.
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
    /// Rejects delayed, ALTREF, temporal-SVC, denoised, unsupported-layer,
    /// and non-inter-layer-prediction configurations in this production
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
        if !matches!(spatial.number_spatial_layers, 2 | 3) {
            return Err(SpatialEncodeError::UnsupportedMode(
                "the stateful spatial encoder currently requires two or three layers",
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
            if layer.width == 0
                || !layer.width.is_multiple_of(8)
                || layer.height == 0
                || !layer.height.is_multiple_of(2)
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
            let layer_parallelism = if layer.width.is_multiple_of(64) {
                parallelism
            } else {
                // Row/tile-MT currently owns whole SB columns. Keep a partial
                // right-edge component on the proven serial walker.
                EncoderParallelism::SINGLE_THREADED
            };
            encoders.push(
                StreamEncoder::new_spatial_with_parallelism(layer_config, layer_parallelism)
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
        let sources = build_layer_sources(&full, &self.plan.layers)?;
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
            let base_key_q = key_access_unit
                .then(|| {
                    layers
                        .first()
                        .map(|layer: &SpatialEncodedLayer| layer.q_index)
                })
                .flatten();
            let control = spatial_frame_control(
                layer_index,
                key_access_unit,
                base_key_q,
                self.plan.layers.len() as u8,
            );
            let view = YuvFrameRef {
                y: &input.y,
                u: &input.u,
                v: &input.v,
                y_stride: input.width,
                u_stride: input.width / 2,
                v_stride: input.width / 2,
            };
            encoder.swap_spatial_shared_state(&mut self.shared);
            let encoded = f02_oracle::with_spatial_id(layer_index as u8, || {
                encoder.encode_spatial_component(&view, control)
            });
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

fn spatial_frame_control(
    layer_index: usize,
    key_access_unit: bool,
    base_key_q: Option<u8>,
    number_spatial_layers: u8,
) -> SpatialFrameControl {
    match (layer_index, key_access_unit) {
        (0, true) => SpatialFrameControl::Key,
        (0, false) => SpatialFrameControl::Inter {
            number_spatial_layers,
            refresh_frame_flags: 0x01,
            ref_frame_indices: [0, 0, 0],
            allow_golden_ref: false,
            base_key_q: None,
        },
        (_, true) => {
            let map_index = layer_index as u8;
            SpatialFrameControl::Inter {
                number_spatial_layers,
                refresh_frame_flags: 1 << map_index,
                ref_frame_indices: [map_index - 1, map_index, map_index - 1],
                allow_golden_ref: false,
                base_key_q,
            }
        }
        (_, false) => {
            let map_index = layer_index as u8;
            SpatialFrameControl::Inter {
                number_spatial_layers,
                refresh_frame_flags: 1 << map_index,
                ref_frame_indices: [map_index, map_index - 1, map_index],
                allow_golden_ref: true,
                base_key_q: None,
            }
        }
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
            if layer.width == 0
                || !layer.width.is_multiple_of(8)
                || layer.height == 0
                || !layer.height.is_multiple_of(2)
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
            layer_config.gop_length = 1;
            let layer_parallelism = if layer.width.is_multiple_of(64) {
                parallelism
            } else {
                EncoderParallelism::SINGLE_THREADED
            };
            encoders.push(
                StreamEncoder::new_spatial_keyframe_with_parallelism(
                    layer_config,
                    layer_parallelism,
                )
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
        let sources = build_layer_sources(&full, &self.plan.layers)?;

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

fn build_layer_sources(
    full: &ScaledI420,
    layers: &[SpatialLayerSpec],
) -> Result<Vec<ScaledI420>, ScaleSourceError> {
    let top = layers
        .len()
        .checked_sub(1)
        .expect("a validated spatial plan has a top layer");
    let mut sources = vec![None; layers.len()];
    sources[top] = Some(full.clone());
    // Build top-down and retain each adjacent 2x result. Pinned libvpx's 4x
    // path likewise produces layer 1 with layer 1's filter, then produces
    // layer 0 from that saved image with layer 0's filter
    // (`vp9_encoder.c:3990-4003`, `:4976-4998`).
    for layer_index in (0..top).rev() {
        let higher = sources[layer_index + 1]
            .as_ref()
            .expect("higher spatial source was built first");
        let filter = layers[layer_index]
            .source_filter
            .expect("non-top spatial layer has a source filter");
        let scaled = if layer_index == 0
            && top == 2
            && (!higher.width.is_multiple_of(8) || !higher.height.is_multiple_of(8))
        {
            // libvpx allocates its retained half-resolution SVC source with
            // dimensions rounded up to eight pixels. If either visible
            // dimension needs padding, `vp9_svc_twostage_scale` rejects that
            // intermediate and the 4x base takes `vp9_resize_plane` directly
            // from the full source (`vp9_encoder.c:3990-4015`). Canonical
            // 1080p reaches this branch because 540 rounds up to 544.
            scale_i420_nonnormative_4x(&full.y, &full.u, &full.v, full.width, full.height)?
        } else {
            scale_i420_2x(
                &higher.y,
                &higher.u,
                &higher.v,
                higher.width,
                higher.height,
                filter,
            )?
        };
        if (scaled.width, scaled.height)
            != (
                layers[layer_index].width as usize,
                layers[layer_index].height as usize,
            )
        {
            return Err(ScaleSourceError::InvalidDimensions);
        }
        sources[layer_index] = Some(scaled);
    }
    Ok(sources
        .into_iter()
        .map(|source| source.expect("every spatial source was built"))
        .collect())
}

#[cfg(test)]
mod tests {
    use rvp9_core::bitstream::{FrameKind as WireFrameKind, Superframe, parse_superframe_index};
    use rvp9_decoder::Decoder;
    use sha2::{Digest, Sha256};

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

    fn random_source(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut state = 0x5ca1_ab1eu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        };
        let y = (0..width * height).map(|_| next()).collect();
        let uv_len = width * height / 4;
        let u = (0..uv_len).map(|_| next()).collect();
        let v = (0..uv_len).map(|_| next()).collect();
        (y, u, v)
    }

    #[test]
    fn spatial_cbr_layers_use_pinned_realtime_buffer_geometry() {
        let mut template = EncoderConfig::new(1280, 720);
        template.target_bitrate_bps = 2_100_000;
        let encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .unwrap();

        assert_eq!(
            encoder
                .encoders
                .iter()
                .map(StreamEncoder::rate_control_buffer_levels_for_test)
                .collect::<Vec<_>>(),
            [
                (100_000, 50_000, 60_000),
                (400_000, 200_000, 240_000),
                (1_600_000, 800_000, 960_000),
            ]
        );
    }

    #[test]
    fn first_spatial_enhancement_components_use_base_key_q_ceiling() {
        // Retain serialization for shared diagnostic counters.
        let _guard = crate::env_test_mutex();
        let width = 1280usize;
        let height = 720usize;
        let mut template = EncoderConfig::new(width as u32, height as u32);
        template.target_bitrate_bps = 2_100_000;
        template.speed = 8;
        template.min_qindex = 8;
        template.max_qindex = 224;
        let mut encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .unwrap();
        let (y, u, v) = source(width, height);
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

        assert_eq!(access_unit.layers.len(), 3);
        assert_eq!(access_unit.layers[0].q_index, 138);
        assert_eq!(access_unit.layers[1].q_index, 148);
        assert_eq!(access_unit.layers[2].q_index, 148);
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

    fn assert_plane_exact(
        decoded: &[u8],
        encoded: &[u8],
        access_unit_index: usize,
        component_index: usize,
        plane: &str,
    ) {
        assert_eq!(
            decoded.len(),
            encoded.len(),
            "AU {access_unit_index} component {component_index} {plane} length"
        );
        if let Some(index) = decoded
            .iter()
            .zip(encoded)
            .position(|(decoded, encoded)| decoded != encoded)
        {
            panic!(
                "AU {access_unit_index} component {component_index} {plane} differs at {index}: decoder={} encoder={}",
                decoded[index], encoded[index]
            );
        }
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

    fn exercise_stateful_spatial_stream(
        width: usize,
        height: usize,
        layer_count: u8,
        layer_dimensions: &[(u32, u32)],
    ) {
        let mut template = EncoderConfig::new(width as u32, height as u32);
        template.target_bitrate_bps = 100_000;
        template.speed = 8;
        template.gop_length = 3;
        let mut encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: layer_count,
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
            assert_eq!(access_unit.layers.len(), usize::from(layer_count));
            assert_eq!(encoder.access_unit_index(), access_unit_index as u32 + 1);

            let parsed = decoder.parse_packet(&access_unit.data).unwrap();
            assert_eq!(parsed.count(), usize::from(layer_count));
            assert_eq!(
                parsed.output_frame_index(),
                Some(usize::from(layer_count - 1))
            );
            let key_access_unit = access_unit_index.is_multiple_of(3);

            for (component_index, (component, &(want_width, want_height))) in
                parsed.frames().iter().zip(layer_dimensions).enumerate()
            {
                if key_access_unit && component_index == 0 {
                    assert!(!component.header.error_resilient);
                } else {
                    assert!(component.header.error_resilient);
                }
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
                        let map_index = component_index as u8;
                        let expected = if key_access_unit {
                            (1 << map_index, [map_index - 1, map_index, map_index - 1])
                        } else if component_index == 0 {
                            (0x01, [0, 0, 0])
                        } else {
                            (1 << map_index, [map_index, map_index - 1, map_index])
                        };
                        assert_eq!((refresh_frame_flags, ref_frame_indices), expected);
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
                assert_plane_exact(
                    &decoded.0,
                    encoded.0,
                    access_unit_index,
                    component_index,
                    "Y",
                );
                assert_plane_exact(
                    &decoded.1,
                    encoded.1,
                    access_unit_index,
                    component_index,
                    "U",
                );
                assert_plane_exact(
                    &decoded.2,
                    encoded.2,
                    access_unit_index,
                    component_index,
                    "V",
                );
            }
        }
    }

    #[test]
    fn two_layer_inter_access_units_share_reference_map_and_decode_in_order() {
        exercise_stateful_spatial_stream(128, 64, 2, &[(64, 32), (128, 64)]);
    }

    #[test]
    fn three_layer_inter_access_units_share_reference_map_and_decode_in_order() {
        exercise_stateful_spatial_stream(256, 128, 3, &[(64, 32), (128, 64), (256, 128)]);
    }

    #[test]
    fn three_layer_partial_right_edge_stream_decodes_in_order() {
        let _guard = crate::env_test_mutex();
        // Prove the opt-in compact-row path is rejected by geometry even
        // when this test explicitly enables the diagnostic override.
        crate::diagnostic_env::set_var("RVP9_COMPACT_RECON_ROWS", "1");
        exercise_stateful_spatial_stream(384, 192, 3, &[(96, 48), (192, 96), (384, 192)]);
        crate::diagnostic_env::remove_var("RVP9_COMPACT_RECON_ROWS");
    }

    #[test]
    fn base_spatial_operating_point_decodes_without_enhancement_components() {
        let width = 256usize;
        let height = 128usize;
        let mut template = EncoderConfig::new(width as u32, height as u32);
        template.target_bitrate_bps = 630_000;
        template.speed = 8;
        template.gop_length = 60;
        let mut encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .unwrap();
        let mut base_decoder = Decoder::new();

        for access_unit_index in 0..4usize {
            let (mut y, mut u, mut v) = source(width, height);
            for value in &mut y {
                *value = value.wrapping_add((access_unit_index * 19) as u8);
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
            let parsed = base_decoder.parse_packet(&access_unit.data).unwrap();
            let decoded = base_decoder
                .decode_frame(&parsed.frames()[0])
                .unwrap_or_else(|error| {
                    panic!("base component {access_unit_index} must decode alone: {error}")
                });
            assert_eq!((decoded.width(), decoded.height()), (64, 32));
        }
    }

    #[test]
    fn canonical_720p_three_layer_geometry_constructs_every_layer() {
        let template = EncoderConfig::new(1280, 720);
        let encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .expect("canonical 720p spatial pyramid");
        let dimensions: Vec<_> = encoder
            .plan()
            .layers
            .iter()
            .map(|layer| (layer.width, layer.height))
            .collect();
        assert_eq!(dimensions, [(320, 180), (640, 360), (1280, 720)]);
    }

    #[test]
    fn canonical_1080p_three_layer_geometry_constructs_partial_base_layer() {
        let template = EncoderConfig::new(1920, 1080);
        let encoder = SpatialEncoder::new(
            template,
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
        )
        .expect("canonical 1080p spatial pyramid");
        let dimensions: Vec<_> = encoder
            .plan()
            .layers
            .iter()
            .map(|layer| (layer.width, layer.height))
            .collect();
        assert_eq!(dimensions, [(480, 270), (960, 540), (1920, 1080)]);
    }

    #[test]
    fn canonical_three_layer_base_source_uses_intermediate_layer_filter_first() {
        let plan = SpatialLayerPlan::canonical(
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
            1280,
            720,
            1_500_000,
        )
        .unwrap();
        let (y, u, v) = source(1280, 720);
        let full = ScaledI420 {
            width: 1280,
            height: 720,
            y,
            u,
            v,
        };

        let actual = build_layer_sources(&full, &plan.layers).unwrap().remove(0);
        let intermediate = scale_i420_2x(
            &full.y,
            &full.u,
            &full.v,
            full.width,
            full.height,
            plan.layers[1].source_filter.unwrap(),
        )
        .unwrap();
        let expected = scale_i420_2x(
            &intermediate.y,
            &intermediate.u,
            &intermediate.v,
            intermediate.width,
            intermediate.height,
            plan.layers[0].source_filter.unwrap(),
        )
        .unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn unaligned_three_layer_intermediate_uses_pinned_nonnormative_base_scale() {
        let plan = SpatialLayerPlan::canonical(
            SpatialSvcConfig {
                number_spatial_layers: 3,
                inter_layer_pred: InterLayerPred::On,
            },
            64,
            40,
            210_000,
        )
        .unwrap();
        let (y, u, v) = random_source(64, 40);
        let full = ScaledI420 {
            width: 64,
            height: 40,
            y,
            u,
            v,
        };

        let base = build_layer_sources(&full, &plan.layers).unwrap().remove(0);
        let mut hasher = Sha256::new();
        hasher.update(&base.y);
        hasher.update(&base.u);
        hasher.update(&base.v);

        assert_eq!(
            format!("{:x}", hasher.finalize()),
            "81d022192e6444b2f150de4ea57e7ecf7b2d3e84dc08a06877b7d8b7bee57a6b"
        );
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
