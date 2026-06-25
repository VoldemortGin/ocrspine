//! Domain-neutral image input for the OCR engine.
//!
//! [`OcrImage`] is the single input type every [`OcrEngine`](crate::OcrEngine)
//! accepts. It wraps an `image::RgbImage` internally and offers three
//! constructors: from raw RGB bytes, from raw grayscale bytes, and from encoded
//! image bytes (PNG / JPEG / TIFF / BMP) decoded via the `image` crate. This
//! replaces any host-specific raster (e.g. a PDF `Pixmap`) — callers convert
//! their own pixels into one of these constructors.

use image::RgbImage;

use crate::error::{OcrError, Result};

/// A decoded RGB image, the input to OCR recognition.
pub struct OcrImage {
    rgb: RgbImage,
}

impl OcrImage {
    /// Builds an image from a tightly-packed RGB byte buffer (`R,G,B` per pixel,
    /// row-major, no padding). `rgb.len()` must equal `width * height * 3`.
    ///
    /// # Errors
    /// [`OcrError::InvalidArgument`] if the buffer length does not match the
    /// declared dimensions (or the dimensions are zero).
    pub fn from_rgb(width: u32, height: u32, rgb: Vec<u8>) -> Result<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|p| p.checked_mul(3));
        if width == 0 || height == 0 || expected != Some(rgb.len()) {
            return Err(OcrError::InvalidArgument(
                "OcrImage::from_rgb: buffer length must equal width * height * 3",
            ));
        }
        let img = RgbImage::from_raw(width, height, rgb).ok_or(OcrError::InvalidArgument(
            "OcrImage::from_rgb: buffer length must equal width * height * 3",
        ))?;
        Ok(OcrImage { rgb: img })
    }

    /// Builds an image from a tightly-packed grayscale byte buffer (one byte per
    /// pixel, row-major). Each gray value is expanded to `[g, g, g]` RGB.
    /// `gray.len()` must equal `width * height`.
    ///
    /// # Errors
    /// [`OcrError::InvalidArgument`] if the buffer length does not match the
    /// declared dimensions (or the dimensions are zero).
    pub fn from_gray(width: u32, height: u32, gray: Vec<u8>) -> Result<Self> {
        let expected = (width as usize).checked_mul(height as usize);
        if width == 0 || height == 0 || expected != Some(gray.len()) {
            return Err(OcrError::InvalidArgument(
                "OcrImage::from_gray: buffer length must equal width * height",
            ));
        }
        let mut rgb = Vec::with_capacity(gray.len() * 3);
        for g in gray {
            rgb.push(g);
            rgb.push(g);
            rgb.push(g);
        }
        let img = RgbImage::from_raw(width, height, rgb).ok_or(OcrError::InvalidArgument(
            "OcrImage::from_gray: buffer length must equal width * height",
        ))?;
        Ok(OcrImage { rgb: img })
    }

    /// Decodes encoded image bytes (PNG / JPEG / TIFF / BMP) into an RGB image.
    ///
    /// # Errors
    /// [`OcrError::Decode`] if the bytes are not a supported, well-formed image.
    pub fn from_encoded(bytes: &[u8]) -> Result<Self> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| OcrError::Decode(e.to_string()))?
            .to_rgb8();
        Ok(OcrImage { rgb: img })
    }

    /// The image width in pixels.
    #[inline]
    #[must_use]
    pub fn width(&self) -> u32 {
        self.rgb.width()
    }

    /// The image height in pixels.
    #[inline]
    #[must_use]
    pub fn height(&self) -> u32 {
        self.rgb.height()
    }

    /// The backing RGB buffer (crate-internal: the engine reads it directly).
    #[inline]
    pub(crate) fn rgb(&self) -> &RgbImage {
        &self.rgb
    }
}
