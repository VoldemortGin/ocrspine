# ocrspine

Domain-neutral, pure-Rust OCR. Input an image, get back words with bounding
boxes and confidences. No Python, no C/C++, no cloud, no network — fully offline
and deterministic.

## Spine 家族 / Spine family

本仓库是 Spine 家族的成员之一（角色：L0 底座）。家族全部成员、分层、依赖方向、依赖形式与当前差距见 [`docs/spine-family.md`](docs/spine-family.md)；该文件在每个家族仓库中的副本内容相同，真源在家族根目录 `~/startup/spine/docs/spine-family.md`，用根目录 `make family-doc-sync` 同步。

## What it is

A generic OCR engine: feed it pixels (RGB or grayscale) or encoded image bytes
(PNG / JPEG / TIFF / BMP), and it returns `Vec<OcrWord>`, where each word has its
recognized `text`, an axis-aligned `BBox` in image pixel coordinates, a `[0,100]`
`confidence`, and the detected (possibly rotated) text `quad`.

Under the hood it runs **PP-OCRv5** ONNX models on CPU via
[`tract-onnx`](https://crates.io/crates/tract-onnx):

1. **DBNet** detection → minimum-area rotated rectangles,
2. **PP-LCNet** 180° text-line-orientation classification,
3. **CRNN + CTC** recognition.

The engine knows nothing about PDFs, slides, or any document concept. It is just
`pixels → words`.

## Usage

```rust
use ocrspine::{OcrEngine, OcrImage, PaddleOcr};

// From encoded bytes (PNG / JPEG / TIFF / BMP):
let bytes = std::fs::read("page.png")?;
let image = OcrImage::from_encoded(&bytes)?;

// Or from raw pixels:
// let image = OcrImage::from_rgb(width, height, rgb_bytes)?;
// let image = OcrImage::from_gray(width, height, gray_bytes)?;

let engine = PaddleOcr::new()?;
for word in engine.recognize(&image)? {
    println!("{:?} @ {:?} ({:.1})", word.text, word.bbox, word.confidence);
}
# Ok::<(), ocrspine::OcrError>(())
```

## Models

The default three ONNX model files (~28 MB) live in `models/` and are loaded from disk at
runtime (offline, no network). Resolution order:

1. the `OCRSPINE_MODELS` environment variable, if set, points at the directory
   holding the `*.onnx` files;
2. otherwise the in-crate `models/` directory (via `CARGO_MANIFEST_DIR`), so
   `cargo test` and a source checkout work with no setup.

The tiny recognition dictionary is embedded into the binary at compile time.

The separately versioned `ocrspine-models` data package is currently **0.0.3**.
It ships the default zh/en/ja set: `ppocrv5_det.onnx`, `ppocrv5_rec.onnx`,
`ppocrv5_cls.onnx` and `ppocr_keys_v5.txt`. Thai evaluation files in this
repository are excluded from that package. See the
[data-package scope and version policy](packages/ocrspine-models/README.md).

See [`models/PROVENANCE.md`](models/PROVENANCE.md) for the full provenance,
licensing (Apache-2.0, PaddlePaddle Authors), and conversion record of the
bundled PP-OCRv5 weights, and [`NOTICE`](NOTICE) for the required attribution.

## Build & test

```bash
# From the crate root:
OCRSPINE_MODELS=$(pwd)/models cargo build --release
cargo test --release      # runs the real ONNX models against the fixtures
```

The acceptance test (`tests/ocr.rs`) loads a fixture PNG, runs the full pipeline
against the real models, and asserts the reference lines are recognized.

## Publishing

This crate remains `publish = false` (not published on crates.io). Consumers
use the public git repository with a complete, fixed commit, for example:

```toml
[dependencies]
ocrspine = { git = "https://github.com/VoldemortGin/ocrspine", rev = "732975f0233cd6500edfbbb82bc06c2332369871" }
```

This is the commit used by docspine and pptspine in their verified 2026-09-10
consumer builds. It is a reproducible consumption example, not a promise that
it contains the latest OCR changes. Consumers select and validate any later
revision explicitly; this documentation does not change their dependencies.
Cargo may fetch git/crate sources during preparation. The git checkout includes
`models/`; inference loads local files and does not download models at runtime.

The model weights, separately, **are** published — as a standalone pure-data
Python package, [`ocrspine-models`](https://pypi.org/project/ocrspine-models/)
(`pip install ocrspine-models`), which the spine document hosts (pdfspine /
pptspine / docspine) take as a runtime dependency so they can ship OCR weights
without embedding them in their wheels. See
[`packages/ocrspine-models/`](packages/ocrspine-models/). This does not change the
fact that the `ocrspine` crate itself stays `publish = false` and off crates.io.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
