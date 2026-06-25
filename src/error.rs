//! `ocrspine` error type.
//!
//! Arbitrary input, a missing/unusable model, or a failed recognition yields a
//! typed [`OcrError`], **never** a panic. The stable [`OcrError::kind`]
//! discriminant is a short, machine-greppable string for callers that map errors
//! across an FFI / language boundary.

/// The `ocrspine` error type.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum OcrError {
    /// The OCR engine is unavailable or unusable: a model file could not be
    /// located, an ONNX graph failed to parse/optimize, or inference reported a
    /// fatal error. The field is a stable English description.
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A caller-supplied argument violates a documented contract (e.g. an RGB
    /// buffer whose length does not match `width * height * 3`).
    #[error("invalid argument: {0}")]
    InvalidArgument(&'static str),

    /// An I/O failure reading a model file from disk.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Failed to decode an encoded image (PNG / JPEG / TIFF / BMP). The field is
    /// a description of the underlying decode error.
    #[error("decode error: {0}")]
    Decode(String),
}

impl OcrError {
    /// A short, stable discriminant string (machine-greppable, never localized).
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            OcrError::Unsupported(_) => "unsupported",
            OcrError::InvalidArgument(_) => "invalid-argument",
            OcrError::Io(_) => "io",
            OcrError::Decode(_) => "decode",
        }
    }
}

/// Convenience alias used throughout `ocrspine`.
pub type Result<T> = std::result::Result<T, OcrError>;
