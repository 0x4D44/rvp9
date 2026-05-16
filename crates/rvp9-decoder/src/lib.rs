//! rvp9-decoder: VP9 video decoder.
//!
//! Implements the VP9 Bitstream & Decoding Process Specification. Validated
//! end-to-end against the official VP9 conformance test vectors.
//!
//! Built before the encoder (Stage 3) because:
//!
//! 1. The decoder is spec-defined — unambiguous success criterion.
//! 2. Conformance failure is a fast oracle for bugs in the shared kernels
//!    (transforms, entropy, intra pred, loop filter, frame management).
//! 3. Having a working decoder lets us round-trip our encoder output during
//!    Stage 4 without depending solely on libvpx for verification.
//!
//! Status: empty scaffold (Stage 0).
