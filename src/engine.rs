//! The pluggable OCR engine seam: one trait, one recognized-word type.
//!
//! There is no formal OCR API standard, so `ocrspine` exposes a small engine
//! trait ([`OcrEngine`]) that any backend can implement, with
//! [`crate::PaddleOcr`] as the bundled pure-Rust adapter. A different engine
//! (cloud API, alternate ONNX models) can be dropped in later by implementing
//! this one trait — nothing else in the crate depends on Paddle directly.

use crate::error::Result;
use crate::geom::BBox;
use crate::input::OcrImage;

/// One recognized word, in **image pixel coordinates** (origin top-left, y down,
/// matching the input [`OcrImage`]).
#[derive(Clone, Debug, PartialEq)]
pub struct OcrWord {
    /// The recognized text (a single whitespace-delimited token).
    pub text: String,
    /// The word's axis-aligned bounding box in image pixel coordinates.
    pub bbox: BBox,
    /// The engine's confidence on a `[0.0, 100.0]` scale.
    pub confidence: f32,
    /// The four corners of the detected (possibly rotated) text quad in image
    /// pixel coordinates, ordered top-left, top-right, bottom-right, bottom-left
    /// along the text's own axes. For upright text the quad coincides with the
    /// corners of [`bbox`](OcrWord::bbox).
    pub quad: [(f32, f32); 4],
}

/// A pluggable OCR backend. Implementors recognize text in an image and return
/// per-word boxes in pixel space.
///
/// The trait is intentionally tiny — one method — so a non-Paddle engine can be
/// wired in without touching the rest of the crate.
pub trait OcrEngine {
    /// Recognizes the words in `image`.
    ///
    /// # Errors
    ///
    /// A typed [`crate::OcrError`] when the engine is unavailable
    /// (`kind == "unsupported"`) or fails; implementors must never panic.
    fn recognize(&self, image: &OcrImage) -> Result<Vec<OcrWord>>;
}
