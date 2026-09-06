//! CRNN+CTC text recognition: a (possibly rotated) crop → recognized string +
//! confidence.
//!
//! Matches RapidOCR's PP-OCRv5 rec config: resize the crop to height 48, width
//! `round(48 * w/h)`; bucket the width UP to a multiple of 64 (min 16); resize
//! to `(48, w_prop)` and right-pad with mid-gray (127, which normalizes to
//! ~0.0 — the CTC blank) to the bucket width; normalize
//! `(px/255-0.5)/0.5`; run → probs `[1,T,18385]`; CTC greedy decode (per
//! timestep argmax, skip blank index 0, collapse consecutive-equal indices);
//! confidence = mean of the max-softmax-prob over the kept (non-blank)
//! timesteps. Results below `text_score=0.5` are dropped by the caller.

use image::{Rgb, RgbImage};
use tract_onnx::prelude::*;

use crate::error::{OcrError as Error, Result};
use crate::paddle::model::CharTable;
use crate::paddle::model::Models;
use crate::paddle::preprocess::{resize_exact, to_tensor};

/// Recognition input height.
const REC_H: u32 = 48;
/// Width bucket granularity (must match the rec runnable cache key). Coarse
/// (64px) buckets keep the number of distinct rec input shapes small, so the
/// expensive per-shape `into_optimized()` runs few times and the runnable cache
/// hits far more often across boxes/pages. The extra right-pad (≤63px of
/// mid-gray, which normalizes to ~0 — the CTC blank) does not change
/// recognition output.
const WIDTH_BUCKET: u32 = 64;
/// Minimum bucketed width.
const MIN_WIDTH: u32 = 16;
/// Symmetric `(px/255-0.5)/0.5` normalization.
const REC_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const REC_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// Right-pad fill value. Mirrors PaddleOCR's `resize_norm_img`, which zero-pads
/// the *normalized* image: mid-gray 127 normalizes to ~0.0, which the model
/// reads as a CTC blank. Black (0) would normalize to -1.0, read as ink, and the
/// BiLSTM would smear that phantom stroke across the whole line.
const PAD_GRAY: u8 = 127;

/// A recognized text line: the decoded string and its mean per-step confidence
/// in `[0,1]`.
pub(crate) struct RecResult {
    pub text: String,
    pub confidence: f32,
}

/// Buckets the proportional width up to a multiple of `WIDTH_BUCKET`.
fn bucket_width(crop: &RgbImage) -> (u32, u32) {
    let (cw, ch) = (crop.width().max(1), crop.height().max(1));
    let prop = (((REC_H as f32) * cw as f32) / ch as f32).round() as u32;
    let prop = prop.max(1);
    let bucket = prop.div_ceil(WIDTH_BUCKET) * WIDTH_BUCKET;
    let bucket = bucket.max(MIN_WIDTH);
    (prop.min(bucket), bucket)
}

/// Pastes `resized` onto a `(bucket_w, REC_H)` canvas pre-filled with mid-gray,
/// leaving the right `bucket_w - resized.width()` columns as `PAD_GRAY`.
fn pad_to_bucket(resized: &RgbImage, bucket_w: u32) -> RgbImage {
    let mut canvas = RgbImage::from_pixel(bucket_w, REC_H, Rgb([PAD_GRAY; 3]));
    let w = resized.width().min(bucket_w);
    for y in 0..REC_H {
        for x in 0..w {
            canvas.put_pixel(x, y, *resized.get_pixel(x, y));
        }
    }
    canvas
}

/// Recognizes the text in a single crop.
pub(crate) fn recognize(models: &Models, crop: &RgbImage) -> Result<RecResult> {
    let (prop_w, bucket_w) = bucket_width(crop);

    // Resize to (48, prop_w) then right-pad with mid-gray to (48, bucket_w).
    let resized = resize_exact(crop, prop_w, REC_H);
    let canvas = pad_to_bucket(&resized, bucket_w);

    let tensor = to_tensor(&canvas, REC_MEAN, REC_STD);
    let runnable = models.rec(bucket_w as usize)?;
    let out = runnable
        .run(tvec!(tensor.into()))
        .map_err(|e| Error::Unsupported(format!("paddle: rec inference failed: {e}")))?;

    let view = out[0]
        .to_array_view::<f32>()
        .map_err(|e| Error::Unsupported(format!("paddle: bad rec output: {e}")))?;
    let shape = view.shape();
    // Expect [1, T, C]; tolerate [T, C].
    let (t, c) = match shape.len() {
        3 => (shape[1], shape[2]),
        2 => (shape[0], shape[1]),
        _ => {
            return Err(Error::Unsupported(format!(
                "paddle: unexpected rec output rank {}",
                shape.len()
            )))
        }
    };
    let flat: Vec<f32> = view.iter().copied().collect();
    Ok(ctc_greedy_decode(&flat, t, c, &models.chars))
}

/// CTC greedy decode over the rec output laid out as `[t][c]` (row-major,
/// length `t*c`).
///
/// The PP-OCRv5 rec model already applies softmax in-graph, so each row is a
/// probability distribution over the `c` classes. Per timestep we take the
/// argmax class; we skip the blank (index 0) and collapse runs of the same
/// index, mapping kept indices through the dictionary. Confidence is the mean
/// over kept timesteps of the winning class's probability (the row max). A tiny
/// `softmax` guard handles the (unexpected) case of raw logits: if a row's
/// max value exceeds 1, we softmax that row to recover a probability.
fn ctc_greedy_decode(probs: &[f32], t: usize, c: usize, chars: &CharTable) -> RecResult {
    let mut text = String::new();
    let mut conf_sum = 0.0f32;
    let mut conf_n = 0u32;
    let mut prev: isize = -1;

    for ti in 0..t {
        let row = &probs[ti * c..ti * c + c];
        // argmax + value.
        let mut best_idx = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        // Collapse repeats and skip blank (index 0).
        if best_idx != 0 && best_idx as isize != prev {
            // Bound the dictionary lookup to the model's class count.
            if best_idx < chars.len() {
                text.push_str(chars.get(best_idx));
            }
            // Probability of the winning class. The model outputs softmax probs
            // already (row max in [0,1]); if it ever emits logits (max > 1),
            // recover the probability via a stable softmax of this row.
            let p = if best_val > 1.0 {
                let denom: f32 = row.iter().map(|&v| (v - best_val).exp()).sum();
                if denom > 0.0 {
                    1.0 / denom
                } else {
                    best_val
                }
            } else {
                best_val
            };
            conf_sum += p;
            conf_n += 1;
        }
        prev = best_idx as isize;
    }

    let confidence = if conf_n > 0 {
        conf_sum / conf_n as f32
    } else {
        0.0
    };
    RecResult { text, confidence }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbImage};

    use super::{bucket_width, pad_to_bucket, PAD_GRAY, REC_H, REC_MEAN, REC_STD};
    use crate::paddle::preprocess::{resize_exact, to_tensor};

    /// A narrow crop pads on the right with mid-gray, and that pad normalizes to
    /// ~0.0 (CTC blank) rather than the -1.0 that black padding would produce.
    #[test]
    fn pad_to_bucket_fills_right_with_mid_gray() {
        // 10x48 all-white crop -> prop_w = 10, bucket_w = 64.
        let crop = RgbImage::from_pixel(10, REC_H, Rgb([255, 255, 255]));
        let (prop_w, bucket_w) = bucket_width(&crop);
        assert_eq!((prop_w, bucket_w), (10, 64));

        let resized = resize_exact(&crop, prop_w, REC_H);
        let canvas = pad_to_bucket(&resized, bucket_w);
        assert_eq!(canvas.dimensions(), (64, 48));

        // Left columns keep the resized pixels; right columns are PAD_GRAY.
        for y in 0..REC_H {
            for x in 0..prop_w {
                assert_eq!(canvas.get_pixel(x, y), resized.get_pixel(x, y));
            }
            for x in prop_w..bucket_w {
                assert_eq!(*canvas.get_pixel(x, y), Rgb([PAD_GRAY, PAD_GRAY, PAD_GRAY]));
            }
        }

        // The pad region normalizes to ~0.0, not -1.0.
        let tensor = to_tensor(&canvas, REC_MEAN, REC_STD);
        let view = tensor.to_array_view::<f32>().expect("rec tensor is f32");
        for c in 0..3 {
            for y in 0..REC_H as usize {
                for x in prop_w as usize..bucket_w as usize {
                    let v = view[[0, c, y, x]];
                    assert!(
                        v.abs() < 0.01,
                        "pad value {v} at (c={c},y={y},x={x}) not ~0"
                    );
                }
            }
        }
    }
}
