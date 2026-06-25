//! Domain-neutral axis-aligned bounding box in image pixel coordinates.

/// An axis-aligned bounding box, `(x0, y0)` top-left and `(x1, y1)`
/// bottom-right, in image pixel coordinates (origin top-left, y down).
///
/// Fields are `f64` so a box derived from sub-pixel detection geometry round-trips
/// without precision loss. This type carries no domain semantics — it is just a
/// rectangle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    /// Left edge (smallest x).
    pub x0: f64,
    /// Top edge (smallest y).
    pub y0: f64,
    /// Right edge (largest x).
    pub x1: f64,
    /// Bottom edge (largest y).
    pub y1: f64,
}

impl BBox {
    /// Constructs a box from its four edges.
    #[inline]
    #[must_use]
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        BBox { x0, y0, x1, y1 }
    }

    /// The box width (`x1 - x0`), clamped to be non-negative.
    #[inline]
    #[must_use]
    pub fn width(&self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }

    /// The box height (`y1 - y0`), clamped to be non-negative.
    #[inline]
    #[must_use]
    pub fn height(&self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }

    /// The box center `(cx, cy)`.
    #[inline]
    #[must_use]
    pub fn center(&self) -> (f64, f64) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }
}
