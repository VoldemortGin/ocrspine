//! `ocrspine` — domain-neutral, pure-Rust OCR.
//!
//! Input an image (RGB / grayscale pixels, or encoded PNG/JPEG/TIFF/BMP bytes),
//! get back a [`Vec<OcrWord>`]: each word carries its text, an axis-aligned
//! [`BBox`] in image pixel coordinates, a `[0,100]` confidence, and the detected
//! (possibly rotated) text quad. Recognition runs PP-OCRv5 ONNX models (DBNet
//! detection + 180° text-line-orientation classifier + CRNN/CTC recognition) on
//! CPU via [`tract`](tract_onnx), with no Python and no C/C++ runtime — fully
//! offline and deterministic.
//!
//! This crate has **zero domain coupling**: it knows nothing about PDFs, slides,
//! documents, or any host concept. It operates purely on pixels.
//!
//! ```no_run
//! use ocrspine::{OcrEngine, OcrImage, PaddleOcr};
//!
//! let bytes = std::fs::read("page.png")?;
//! let image = OcrImage::from_encoded(&bytes)?;
//! let engine = PaddleOcr::new()?;
//! for word in engine.recognize(&image)? {
//!     println!("{:?} @ {:?} ({:.1})", word.text, word.bbox, word.confidence);
//! }
//! # Ok::<(), ocrspine::OcrError>(())
//! ```
//!
//! ## Models
//!
//! The three ONNX model files are loaded from disk at runtime (offline, no
//! network): the `OCRSPINE_MODELS` environment variable when set, else the
//! in-crate `models/` directory (via `CARGO_MANIFEST_DIR`) so `cargo test` works
//! in a checkout with no setup. See `models/PROVENANCE.md` for provenance and
//! licensing of the bundled PP-OCRv5 weights (Apache-2.0, PaddlePaddle Authors).

#![forbid(unsafe_code)]

mod engine;
mod error;
mod geom;
mod input;
mod paddle;

pub use engine::{OcrEngine, OcrWord};
pub use error::{OcrError, Result};
pub use geom::BBox;
pub use input::OcrImage;
pub use paddle::PaddleOcr;
