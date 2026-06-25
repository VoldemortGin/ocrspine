"""Hatchling build hook for the ``ocrspine-models`` data distribution.

The default PP-OCRv5 model files are git-tracked exactly once, in the canonical
``ocrspine/models/`` directory (the same files the ``ocrspine`` Rust crate loads
at runtime). To avoid duplicating ~28 MB in git, this hook ``force_include``s
them into both the sdist and the wheel at build time instead of vendoring a
second copy.

Only the four DEFAULT (zh/en/ja) files ship here. The three ``*.onnx`` weights
are REQUIRED; the recognition dictionary ``ppocr_keys_v5.txt`` is carried as a
convenience (it is otherwise embedded into the Rust binary via ``include_str!``).
The Thai evaluation files (``ppocrv5_rec_th.onnx`` / ``ppocr_keys_th.txt``) are
NEVER included — they are evaluation-only and out of scope for this package.

Resolution of the models source dir (first that exists wins), so it works both
for an in-repo build AND for building a wheel from an unpacked sdist:

  1. ``<package>/ocrspine_models/`` — already-injected files (e.g. an sdist that
     carried them into the package dir);
  2. ``<package>/../../models/`` — the canonical ``ocrspine/models`` source.
"""

from __future__ import annotations

import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# The three ONNX weights the engine loads at runtime — REQUIRED in every build.
_REQUIRED_FILES = ("ppocrv5_det.onnx", "ppocrv5_rec.onnx", "ppocrv5_cls.onnx")
# The recognition dictionary — carried if present (optional; embedded in the
# Rust binary by default, shipped here so the data dir is self-describing).
_OPTIONAL_FILES = ("ppocr_keys_v5.txt",)


class CustomBuildHook(BuildHookInterface):
    def _models_src_dir(self) -> str:
        candidates = (
            os.path.join(self.root, "ocrspine_models"),
            os.path.join(self.root, os.pardir, os.pardir, "models"),
        )
        for cand in candidates:
            if all(os.path.isfile(os.path.join(cand, f)) for f in _REQUIRED_FILES):
                return os.path.abspath(cand)
        searched = "\n  ".join(os.path.abspath(c) for c in candidates)
        raise RuntimeError(
            "ocrspine-models: could not locate the default PP-OCRv5 ONNX models "
            f"({', '.join(_REQUIRED_FILES)}). Searched:\n  {searched}\n"
            "Build this distribution from a full ocrspine checkout (the models are "
            "git-tracked at ocrspine/models), or from its sdist."
        )

    def initialize(self, version: str, build_data: dict) -> None:
        src = self._models_src_dir()
        force_include = build_data.setdefault("force_include", {})
        for fname in _REQUIRED_FILES + _OPTIONAL_FILES:
            path = os.path.join(src, fname)
            if fname in _OPTIONAL_FILES and not os.path.isfile(path):
                continue
            # Wheel: land inside the import package. Sdist: land inside the package
            # dir too, so a later wheel build from the sdist finds them via
            # candidate (1) above. Same destination works for both targets.
            force_include[path] = os.path.join("ocrspine_models", fname)
