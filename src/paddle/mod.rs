//! Pure-Rust PaddleOCR (PP-OCRv5) engine.
//!
//! [`PaddleOcr`] is the bundled [`OcrEngine`](crate::OcrEngine) implementation.
//! It runs the shipped PP-OCRv5 detection/recognition and PP-LCNet
//! text-line-orientation ONNX models on CPU via [`tract`], with no Python and no
//! C/C++ runtime.
//!
//! Pipeline per image: **detect** text boxes (minimum-area rotated rects) → for
//! each box **crop** (axis-aligned for upright text, or de-rotated to horizontal
//! via the rotated quad for skewed text) → **classify** orientation (rotate 180°
//! if needed) → **recognize** (CRNN+CTC) → emit one [`OcrWord`] per box with the
//! box in image pixel coordinates, the detected quad, and confidence on the
//! `[0,100]` scale.
//!
//! Rotated/skewed text is detected as a rotated rectangle and de-rotated before
//! recognition; the public [`OcrWord::bbox`](crate::OcrWord::bbox) remains the
//! axis-aligned bounding box of that rotated quad, while
//! [`OcrWord::quad`](crate::OcrWord::quad) carries the four rotated corners.

mod classify;
mod detect;
mod model;
mod preprocess;
mod recognize;

use rayon::prelude::*;

use crate::engine::{OcrEngine, OcrWord};
use crate::error::Result;
use crate::geom::BBox;
use crate::input::OcrImage;

use self::detect::DetectParams;
use self::model::Models;

/// Boxes within this angle of horizontal (radians, ≈2.9°) are treated as upright
/// and use the axis-aligned crop path — keeping upright text byte-for-byte as it
/// was before rotated-rect detection.
const UPRIGHT_ANGLE_RAD: f32 = 0.05;

/// A pure-Rust PaddleOCR engine running PP-OCRv5 ONNX models via `tract`.
///
/// Construct once with [`PaddleOcr::new`] and reuse: optimized model runnables
/// are cached per input-shape bucket across [`recognize`](OcrEngine::recognize)
/// calls, so the expensive optimization cost is paid at most once per shape.
pub struct PaddleOcr {
    models: Models,
}

impl PaddleOcr {
    /// Builds the engine. The ONNX model files are loaded lazily from disk on
    /// first use (not at construction), and no optimization runs yet (that also
    /// happens lazily on first use of each input-shape bucket), so this is cheap.
    ///
    /// # Errors
    /// Model loading/parsing/optimization is deferred to the first
    /// [`recognize`](OcrEngine::recognize) call — a missing model directory
    /// surfaces there as [`OcrError::Unsupported`](crate::OcrError::Unsupported),
    /// pointing at the `OCRSPINE_MODELS` override.
    pub fn new() -> Result<Self> {
        Ok(PaddleOcr {
            models: Models::new()?,
        })
    }
}

impl OcrEngine for PaddleOcr {
    /// Recognizes the words in `image`. Empty / low-confidence results are
    /// skipped. Boxes are emitted in image pixel coordinates.
    fn recognize(&self, image: &OcrImage) -> Result<Vec<OcrWord>> {
        let rgb = image.rgb();
        // 检测/识别后处理参数（env 覆盖缝）：不设任何 env → 历史默认，逐位不变。
        let params = DetectParams::from_env();
        let boxes = detect::detect(&self.models, rgb, &params)?;

        // Recognize each detected box in parallel. The work is CPU-bound (crop →
        // classify → CRNN+CTC) and an image yields dozens–hundreds of boxes, so
        // `par_iter` near-linearly cuts wall time across cores. The shared
        // runnable cache (`self.models`, `&self` + `Mutex`/`OnceLock`) is
        // thread-safe to share, so no per-box state is duplicated.
        //
        // DETERMINISM: `par_iter().map(..).collect::<Vec<_>>()` is an *indexed*
        // collect — output position equals input box index — so the per-box
        // `Option<OcrWord>` vector is byte-identical to the sequential version
        // regardless of completion order. We then drop the skipped (`None`) boxes
        // in that same order. Any per-box error short-circuits the whole call.
        let per_box: Vec<Option<OcrWord>> = boxes
            .par_iter()
            .map(|b| -> Result<Option<OcrWord>> {
                // Upright (~0°) boxes use the exact axis-aligned crop path as before;
                // skewed boxes are de-rotated to horizontal via the rotated quad so
                // the recognizer (which needs horizontal text) sees an upright line.
                let crop = if b.angle.abs() <= UPRIGHT_ANGLE_RAD {
                    preprocess::crop(rgb, b.x0, b.y0, b.x1, b.y1)
                } else {
                    preprocess::crop_rotated(rgb, &b.quad)
                };
                let oriented = classify::classify_and_orient(&self.models, crop)?;
                let rec = recognize::recognize(&self.models, &oriented)?;
                let text = rec.text.trim().to_string();
                if text.is_empty() || rec.confidence < params.text_score {
                    return Ok(None);
                }
                Ok(Some(OcrWord {
                    text,
                    bbox: BBox::new(b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64),
                    // Combine detection + recognition confidence onto the [0,100] scale.
                    confidence: (rec.confidence * b.score * 100.0).clamp(0.0, 100.0),
                    quad: b.quad,
                }))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(per_box.into_iter().flatten().collect())
    }
}
