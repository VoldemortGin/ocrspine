# ocrspine-models

[![PyPI](https://img.shields.io/pypi/v/ocrspine-models.svg)](https://pypi.org/project/ocrspine-models/)

The default PP-OCRv5 ONNX model weights for [ocrspine](https://github.com/VoldemortGin/ocrspine)'s
pure-Rust PaddleOCR engine — the **shared data companion** for the spine document
family (`pdfspine` / `pptspine` / `docspine`).

This is a **pure-data distribution**. The host wheels already contain the OCR
*code* (compiled in) but ship **no models**; this one package supplies the ~28 MB
of weights, so the family keeps exactly one git-tracked copy of the models (in the
`ocrspine` repo) and one published copy of the data wheel.

It is **published on PyPI** — `pip install ocrspine-models` — though you normally
do not install it directly: it is a hard dependency of the hosts, so
`pip install pdfspine` (or `pptspine` / `docspine`) pulls it in automatically. The
host then resolves the models at runtime by reading
`ocrspine_models.models_dir()` and exporting it as the engine's `OCRSPINE_MODELS`
environment variable. Everything is offline — no model download at runtime.

```python
import ocrspine_models

print(ocrspine_models.models_dir())  # dir holding the 3 ONNX files
print(ocrspine_models.det_path())    # .../ppocrv5_det.onnx
print(ocrspine_models.rec_path())    # .../ppocrv5_rec.onnx
print(ocrspine_models.cls_path())    # .../ppocrv5_cls.onnx
print(ocrspine_models.keys_path())   # .../ppocr_keys_v5.txt
```

## What ships here

| file | what |
|---|---|
| `ppocrv5_det.onnx` | PP-OCRv5 text detection (DBNet) |
| `ppocrv5_rec.onnx` | PP-OCRv5 text recognition (CRNN + CTC), zh/en/ja |
| `ppocrv5_cls.onnx` | PP-LCNet text-line orientation classifier (180°) |
| `ppocr_keys_v5.txt` | recognition character dictionary (carried for self-containment) |

Only the default zh/en/ja model set ships here. Language-evaluation artifacts that
live in the `ocrspine` repo (e.g. the Thai recognition model) are out of scope.

## License

Apache-2.0. The redistributed PP-OCR model weights are Copyright (c) PaddlePaddle
Authors (Apache-2.0), converted to ONNX via Paddle2ONNX. See
[`ocrspine_models/NOTICE`](./ocrspine_models/NOTICE) and
[`ocrspine_models/PROVENANCE.md`](./ocrspine_models/PROVENANCE.md).
