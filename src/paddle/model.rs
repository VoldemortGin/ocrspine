//! Lazily-built, shape-bucketed [`tract`] runnables for the three PP-OCR models.
//!
//! `into_optimized()` is expensive (~1–2s per distinct concrete input shape), so
//! each model caches one runnable per shape bucket in a `HashMap`. The detection
//! and recognition models have dynamic input dims (symbolic `H`/`W` in the
//! shipped ONNX), so we pin a concrete fact at first use of each bucket; the
//! classifier takes a fixed 3×80×160 crop and uses a single runnable.
//!
//! The ~16 MB ONNX model files are NOT embedded in the binary. Instead they are
//! loaded from a resolvable directory at runtime (offline, no network): the
//! `OCRSPINE_MODELS` environment variable when set, else the in-crate `models`
//! directory (via `CARGO_MANIFEST_DIR`) so `cargo test` works in a checkout with
//! no setup. The tiny (~26 KB) recognition dictionary stays embedded with
//! [`include_str!`].

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;

use tract_onnx::prelude::*;

use crate::error::{OcrError as Error, Result};

/// The optimized + runnable typed model produced by `into_optimized().into_runnable()`.
pub(crate) type Runnable = TypedRunnableModel<TypedModel>;

// --- Model files: loaded from disk at runtime (NOT embedded). ---

/// Environment variable pointing at the directory that holds the three
/// `*.onnx` model files. Overrides the in-crate default.
const ENV_MODELS_DIR: &str = "OCRSPINE_MODELS";

/// 识别模型路径覆盖（语言切换缝，镜像 [`ENV_MODELS_DIR`] 的覆盖语义）：设置后
/// rec ONNX 从该**文件**加载；未设置时仍用 bundled `ppocrv5_rec.onnx`。用于换上
/// 其他语言的 rec 模型（如泰文），检测/方向分类与语言无关、原样复用。
const ENV_REC_MODEL: &str = "OCRSPINE_REC_MODEL";

/// 识别字典路径覆盖：设置后字典从该**磁盘文件**加载；未设置时用编译期内嵌的
/// `ppocr_keys_v5.txt`。字典须与所用 rec 模型 index 对齐（行数 == rec 输出宽度）。
/// 与 [`ENV_REC_MODEL`] 配套使用。
const ENV_REC_DICT: &str = "OCRSPINE_REC_DICT";

/// PP-OCRv5 DBNet text-detection model. Input `[1,3,H,W]`, output prob map
/// `[1,1,H,W]`.
const DET_FILE: &str = "ppocrv5_det.onnx";
/// PP-OCRv5 CRNN+CTC recognition model. Input `[1,3,48,W]`, output softmax probs
/// `[1,T,18385]`.
const REC_FILE: &str = "ppocrv5_rec.onnx";
/// PP-OCRv5 text-line orientation classifier (PP-LCNet_x1_0_textline_ori). Input
/// concrete `[1,3,80,160]`, output `[1,2]` (0° / 180°).
const CLS_FILE: &str = "ppocrv5_cls.onnx";

/// The recognition dictionary, INDEX-ALIGNED to the rec output's class axis:
/// line 0 = the CTC blank, lines 1.. = characters, last line = a single space.
/// This is tiny (~26 KB) and stays embedded; only the multi-MB ONNX weights are
/// loaded from disk. We must preserve the trailing space line, so we split on
/// `'\n'` (not `lines()`, which would also be fine, but we keep this explicit)
/// and do NOT trim.
const KEYS: &str = include_str!("../../models/ppocr_keys_v5.txt");

/// Resolves the directory holding the ONNX model files. Prefers the
/// `OCRSPINE_MODELS` environment variable (a user override); falls back to the
/// in-crate `models` directory so a source checkout works with no setup. Never
/// touches the network.
fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(ENV_MODELS_DIR) {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

/// Reads a model file from the resolved [`models_dir`], mapping a missing file
/// or read error to a clear `Unsupported` error that points the user at the
/// `OCRSPINE_MODELS` override.
fn read_model(file: &str) -> Result<Vec<u8>> {
    let path = models_dir().join(file);
    std::fs::read(&path).map_err(|e| {
        Error::Unsupported(format!(
            "paddle: OCR model {file:?} not found at {} ({e}). Set \
             `{ENV_MODELS_DIR}` to the dir holding the PP-OCR `*.onnx` files.",
            path.display(),
        ))
    })
}

/// 读取识别模型字节：优先 [`ENV_REC_MODEL`] 指向的文件（语言切换缝），否则回退到
/// [`models_dir`] 下的 bundled rec 文件——**未设置该 env 时行为与之前字节一致**。
fn read_rec_model() -> Result<Vec<u8>> {
    if let Some(path) = std::env::var_os(ENV_REC_MODEL) {
        let path = PathBuf::from(path);
        return std::fs::read(&path).map_err(|e| {
            Error::Unsupported(format!(
                "paddle: rec model not found at {} ({e}). Unset `{ENV_REC_MODEL}` \
                 to use the bundled rec model.",
                path.display(),
            ))
        });
    }
    read_model(REC_FILE)
}

/// Builds an `InferenceModel` from ONNX bytes, mapping any tract error into our
/// typed `Unsupported` error (a failure here is a build/environment problem,
/// surfaced — never a panic).
fn proto(bytes: &[u8]) -> Result<InferenceModel> {
    tract_onnx::onnx()
        .model_for_read(&mut Cursor::new(bytes))
        .map_err(|e| Error::Unsupported(format!("paddle: failed to parse ONNX model: {e}")))
}

/// Pins a concrete `[1,3,h,w]` f32 input fact, optimizes, and makes the model
/// runnable. This is the per-bucket cost we cache.
fn build_runnable(model: InferenceModel, h: usize, w: usize) -> Result<Runnable> {
    model
        .with_input_fact(0, f32::fact([1, 3, h, w]).into())
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| Error::Unsupported(format!("paddle: failed to optimize model: {e}")))
}

/// The recognition character table (decoded once at construction).
///
/// `table[i]` is the string emitted for class index `i`. Index 0 (the CTC
/// blank) is stored as an empty string so the decoder can index uniformly; it is
/// also skipped explicitly during decode.
pub(crate) struct CharTable {
    table: Vec<String>,
}

impl CharTable {
    /// 默认字典：从编译期内嵌的 `ppocr_keys_v5.txt` 构建。**默认路径行为不变。**
    fn load() -> Self {
        Self::from_text(KEYS)
    }

    /// 从磁盘字典文件构建（[`ENV_REC_DICT`] 覆盖用）。字典须为 index-aligned 形式：
    /// 第 0 行为 CTC blank、其后为字符、可含末行空格，**行数 == rec 输出宽度**。
    fn from_path(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            Error::Unsupported(format!(
                "paddle: rec dictionary not found at {} ({e}). Unset `{ENV_REC_DICT}` \
                 to use the embedded dictionary.",
                path.display(),
            ))
        })?;
        Ok(Self::from_text(&text))
    }

    /// 解析 index-aligned 字典文本（任意长度，**不硬编码宽度**）。
    fn from_text(text: &str) -> Self {
        // Split on '\n' preserving every line, including a trailing space line.
        // A final empty element from a trailing newline is dropped (the file
        // ends with the space line + '\n'); the space line itself is kept.
        let mut table: Vec<String> = text.split('\n').map(|s| s.to_string()).collect();
        if table.last().map(|s| s.is_empty()).unwrap_or(false) {
            table.pop();
        }
        // Index 0 is the blank: blank out its label so it never contributes text.
        if let Some(first) = table.first_mut() {
            first.clear();
        }
        CharTable { table }
    }

    /// The label for class index `i` (`""` for the blank or out-of-range).
    #[inline]
    pub(crate) fn get(&self, i: usize) -> &str {
        self.table.get(i).map(String::as_str).unwrap_or("")
    }

    /// The number of classes (equals the rec model's output width — 18385 for the
    /// bundled zh/en/ja model, or e.g. 526 for the Thai rec model via the dict
    /// override). Used to bound decode lookups, so it must NOT be hard-coded.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.table.len()
    }
}

/// Holds the three models' lazily-built runnables and the recognition dict.
///
/// Detection caches per `(h, w)` and recognition per padded width (`(48, w)`),
/// both keyed by the same `(h, w)` map. The classifier is concrete (80×160), so
/// it builds exactly one runnable. Caches are behind a `Mutex` so `recognize(&self, ..)`
/// stays `&self` (the `OcrEngine` contract) while still memoizing across calls.
pub(crate) struct Models {
    det: Mutex<HashMap<(usize, usize), std::sync::Arc<Runnable>>>,
    rec: Mutex<HashMap<(usize, usize), std::sync::Arc<Runnable>>>,
    cls: std::sync::OnceLock<std::sync::Arc<Runnable>>,
    pub(crate) chars: CharTable,
}

impl Models {
    /// Constructs the model holder. This does NOT optimize any model yet (the
    /// expensive `into_optimized()` happens lazily per shape bucket), so it is
    /// cheap; only the dictionary is decoded eagerly.
    pub(crate) fn new() -> Result<Self> {
        // 字典来源：设置了 `OCRSPINE_REC_DICT` 则从磁盘加载（语言切换），
        // 否则用编译期内嵌的中文字典（默认行为不变）。
        let chars = match std::env::var_os(ENV_REC_DICT) {
            Some(path) => CharTable::from_path(std::path::Path::new(&path))?,
            None => CharTable::load(),
        };
        Ok(Models {
            det: Mutex::new(HashMap::new()),
            rec: Mutex::new(HashMap::new()),
            cls: std::sync::OnceLock::new(),
            chars,
        })
    }

    /// The detection runnable for input height `h`, width `w` (cached).
    pub(crate) fn det(&self, h: usize, w: usize) -> Result<std::sync::Arc<Runnable>> {
        if let Some(r) = self.det.lock().unwrap().get(&(h, w)) {
            return Ok(r.clone());
        }
        let runnable = std::sync::Arc::new(build_runnable(proto(&read_model(DET_FILE)?)?, h, w)?);
        self.det.lock().unwrap().insert((h, w), runnable.clone());
        Ok(runnable)
    }

    /// The recognition runnable for a padded crop of height 48 and width `w`
    /// (cached per width).
    pub(crate) fn rec(&self, w: usize) -> Result<std::sync::Arc<Runnable>> {
        let key = (48usize, w);
        if let Some(r) = self.rec.lock().unwrap().get(&key) {
            return Ok(r.clone());
        }
        let runnable = std::sync::Arc::new(build_runnable(proto(&read_rec_model()?)?, 48, w)?);
        self.rec.lock().unwrap().insert(key, runnable.clone());
        Ok(runnable)
    }

    /// The (single, concrete) classifier runnable, built on first use.
    pub(crate) fn cls(&self) -> Result<std::sync::Arc<Runnable>> {
        if let Some(r) = self.cls.get() {
            return Ok(r.clone());
        }
        let runnable =
            std::sync::Arc::new(build_runnable(proto(&read_model(CLS_FILE)?)?, 80, 160)?);
        // OnceLock: if a concurrent caller won the race, use theirs.
        let _ = self.cls.set(runnable);
        Ok(self.cls.get().expect("just set").clone())
    }
}
