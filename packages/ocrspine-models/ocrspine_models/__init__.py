"""``ocrspine-models`` — the default PP-OCRv5 ONNX weights for ocrspine's PaddleOCR.

This is a pure-data companion distribution shared by the spine family
(pdfspine / pptspine / docspine). The host wheels have the OCR *code* compiled in
but ship no models; this package supplies the three ONNX weights (and the zh/en/ja
recognition dictionary) and exposes :func:`models_dir`, which the hosts read at
runtime to set the engine's ``OCRSPINE_MODELS`` environment variable. Everything
is offline — no model download at runtime.

Models: PP-OCRv5 detection / recognition + PP-LCNet text-line-orientation
classifier, redistributed from PaddleOCR under Apache-2.0. See ``PROVENANCE.md`` /
``NOTICE`` in this package.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = [
    "models_dir",
    "det_path",
    "rec_path",
    "cls_path",
    "keys_path",
    "__version__",
]

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("ocrspine-models")
except Exception:  # pragma: no cover - source tree without dist metadata
    __version__ = "0.0.1"

# Filenames as they are laid out inside the installed package directory.
_DET_FILE = "ppocrv5_det.onnx"
_REC_FILE = "ppocrv5_rec.onnx"
_CLS_FILE = "ppocrv5_cls.onnx"
_KEYS_FILE = "ppocr_keys_v5.txt"

# The three ONNX weights the Rust PaddleOCR engine loads at runtime; a directory
# only counts as a usable model dir when all three are present.
_REQUIRED_FILES = (_DET_FILE, _REC_FILE, _CLS_FILE)


def _package_dir() -> Path:
    """The installed package directory (where the data files live)."""
    with resources.as_file(resources.files(__package__)) as path:
        return Path(path)


def models_dir() -> Path:
    """Return the directory holding the three default PP-OCRv5 ONNX models.

    The directory is guaranteed to contain ``ppocrv5_det.onnx``,
    ``ppocrv5_rec.onnx`` and ``ppocrv5_cls.onnx`` (placed there by the build
    hook). The spine hosts pass it to the Rust engine via the ``OCRSPINE_MODELS``
    environment variable.

    Raises:
        FileNotFoundError: on a corrupt / partial install missing an ONNX weight.
    """
    directory = _package_dir()
    missing = [f for f in _REQUIRED_FILES if not (directory / f).is_file()]
    if missing:  # pragma: no cover - corrupt/partial install
        raise FileNotFoundError(
            f"ocrspine-models: ONNX model(s) missing from {directory}: "
            f"{', '.join(missing)}. Reinstall with "
            "`pip install --force-reinstall ocrspine-models`."
        )
    return directory


def det_path() -> Path:
    """Absolute path of the PP-OCRv5 text-detection ONNX model."""
    return models_dir() / _DET_FILE


def rec_path() -> Path:
    """Absolute path of the PP-OCRv5 text-recognition ONNX model."""
    return models_dir() / _REC_FILE


def cls_path() -> Path:
    """Absolute path of the PP-LCNet text-line-orientation ONNX model."""
    return models_dir() / _CLS_FILE


def keys_path() -> Path:
    """Absolute path of the zh/en/ja recognition character dictionary.

    Raises:
        FileNotFoundError: if the dictionary was not carried into this build (it
            is optional — the engine embeds its own copy via ``include_str!``).
    """
    path = _package_dir() / _KEYS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"ocrspine-models: recognition dictionary {_KEYS_FILE!r} is not "
            f"present in {path.parent} (it ships optionally)."
        )
    return path
