# `ocrspine-models` — bundled OCR model provenance

This data distribution redistributes the *permissively-licensed* PaddleOCR
**PP-OCRv5** default model weights (converted to ONNX, then made `tract`-parseable)
that the spine family's pure-Rust PaddleOCR engine (`ocrspine`) loads at runtime.
It is the shared data companion that `pdfspine` / `pptspine` / `docspine` depend
on; the models are the same files tracked once in the `ocrspine` repo at
`models/` (the build force-includes them from there — it does not keep a second
copy).

The project thesis is **license cleanliness**: every redistributed byte has a
recorded, affirmatively-permissive license and a recorded upstream source.

All four files originate from the **PaddleOCR** project
(<https://github.com/PaddlePaddle/PaddleOCR>), which — including its published
PP-OCR model weights — is distributed under the **Apache License, Version 2.0**
(SPDX: `Apache-2.0`), compatible with this project's Apache-2.0 license. The
required attribution that must accompany binary distributions is carried in the
[`NOTICE`](./NOTICE) shipped alongside this file.

> **Scope.** Only the DEFAULT zh/en/ja model set ships here. Language-evaluation
> artifacts that live in the `ocrspine` repo (e.g. the Thai recognition model
> `ppocrv5_rec_th.onnx` and its dictionary) are **not** part of this package, and
> neither are any test-fixture fonts.

> **Conversion + strip note.** The upstream PaddleOCR weights are published in
> PaddlePaddle's native inference format. The bundled `*.onnx` files were first
> converted to ONNX with [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)
> (a mechanical format transcode — it does not change the weights or licensing),
> then post-processed by a deterministic strip step that renames illegal dynamic
> dimension names (`DynamicDimension.N`, containing a `.`) to legal identifiers,
> clears `value_info`, and clears output shape hints, so the pure-Rust `tract`
> runtime can parse them. **It changes no weights** — only graph metadata.

## `ppocrv5_det.onnx` — PP-OCRv5 text detection (DBNet)

| field | value |
|---|---|
| **What** | DBNet text-detection model. Input `[1,3,H,W]`, output probability map `[1,1,H,W]`. |
| **Upstream model** | PP-OCRv5 mobile detection (`PP-OCRv5_mobile_det`). |
| **Upstream** | <https://github.com/PaddlePaddle/PaddleOCR> |
| **Pre-converted source** | <https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx> |
| **License** | **Apache-2.0** (PaddlePaddle Authors). SPDX: `Apache-2.0`. |
| **Bundled sha256** | `62f0e79763dfd9bdadae166dd61078e703d9149df891e11a58c48d18595fad6f` |
| **Conversion** | PaddlePaddle inference model → ONNX via Paddle2ONNX; then deterministic dim-name/shape-hint cleanup (no weight modification). |

## `ppocrv5_rec.onnx` — PP-OCRv5 text recognition (CRNN + CTC)

| field | value |
|---|---|
| **What** | CRNN + CTC recognition model. Input `[1,3,48,W]`, output softmax probs `[1,T,18385]`. |
| **Upstream model** | PP-OCRv5 mobile recognition (`PP-OCRv5_mobile_rec`), index-aligned to `ppocr_keys_v5.txt`. |
| **Upstream** | <https://github.com/PaddlePaddle/PaddleOCR> |
| **Pre-converted source** | <https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx> |
| **License** | **Apache-2.0** (PaddlePaddle Authors). SPDX: `Apache-2.0`. |
| **Bundled sha256** | `22edcdb81193286bd1f2c34069d4dc4296d3ac268fc736b4a83bba3eaae26dbb` |
| **Conversion** | PaddlePaddle inference model → ONNX via Paddle2ONNX; then deterministic dim-name/shape-hint cleanup (no weight modification). |

## `ppocrv5_cls.onnx` — PP-OCRv5 text-line orientation classifier (180°)

| field | value |
|---|---|
| **What** | Text-line orientation classifier (PP-LCNet). Input concrete `[1,3,80,160]`, output `[1,2]` (0° / 180°). |
| **Upstream model** | `PP-LCNet_x1_0_textline_ori` (PP-OCRv5 default text-line-orientation model). |
| **Upstream** | <https://github.com/PaddlePaddle/PaddleOCR> |
| **Pre-converted source** | <https://huggingface.co/monkt/paddleocr-onnx> (`preprocessing/textline-orientation/`) |
| **License** | **Apache-2.0** (PaddlePaddle Authors). SPDX: `Apache-2.0`. |
| **Bundled sha256** | `ebe902aa38f936ba1d38ef6876727f29ab3f3b41a3b3ba8fbec754030b1100c9` |
| **Conversion** | PaddlePaddle inference model → ONNX via Paddle2ONNX; then deterministic dim-name/shape-hint cleanup (no weight modification). |

## `ppocr_keys_v5.txt` — PP-OCRv5 recognition character dictionary

| field | value |
|---|---|
| **What** | Recognition character dictionary, **index-aligned** to the `ppocrv5_rec.onnx` output class axis (line 0 = CTC blank, lines 1..18383 = characters, last line = a single space). Total 18385 lines = the rec model's output width. |
| **Upstream** | <https://huggingface.co/monkt/paddleocr-onnx> (`languages/chinese/dict.txt`, 18383 characters — PP-OCRv5 character set covering Simplified/Traditional Chinese, Japanese, Latin, digits, and symbols). |
| **License** | **Apache-2.0** (PaddlePaddle Authors). SPDX: `Apache-2.0`. |
| **Bundled sha256** | `4941d35625bf15f7ef89fcbff443893e2fef40e25a7ecde885203994d4a9be25` |
| **Baking** | The raw 18383-line dict is baked to the index-aligned form (prepend `blank`, append a space line → 18385 lines), matching the engine's `CharTable::load()` expectation. Shipped here for self-containment; the engine embeds its own copy via `include_str!`. |

See the [`NOTICE`](./NOTICE) for the attribution that must accompany binary
distributions of these bundled models.
