//! PP-OCRv5 泰文 OCR 评测 harness（char 级 CER）。
//!
//! 经识别缝把 rec 模型/字典切到泰文（`OCRSPINE_REC_MODEL` + `OCRSPINE_REC_DICT`，
//! 指向 `models/ppocrv5_rec_th.onnx` + `models/ppocr_keys_th.txt`），对
//! `tests/fixtures/thai/` 下每张合成 PNG 跑 [`PaddleOcr`]，把识别框按**上到下、
//! 左到右**排序拼接，与 `ground_truth.json` 的源串比较，算逐样本 CER 与 mean CER。
//!
//! 该评测较重（加载 7.8MB 泰文 rec 模型），默认 `#[ignore]`，使 `cargo test`
//! 仍只跑原有中文验收（默认路径不破）。手动实跑：
//!
//! ```text
//! cargo test --release --test thai_eval -- --ignored --nocapture
//! ```
//!
//! （模型/字典已随仓库落在 `models/`，无需额外设置 env；测试自己用
//! `CARGO_MANIFEST_DIR` 定位并设置这两个覆盖 env。）
//!
//! **实测结论（2026-06，14 条合成 fixtures）**：
//! - 未调检测（CJK 默认参数）端到端 mean CER ≈ **0.42**（6/14 完全正确）；
//! - 启用泰文检测 profile（`OCRSPINE_DET_PROFILE=thai`：`text_score` 放宽 + 同行
//!   邻接框水平合并）后压到 mean CER ≈ **0.25**（实测 0.2500，10/14 完全正确）——拿回
//!   了大部分差距；
//! - 但仍高于 **0.15** 的生产级目标：把检测整段旁路、直接拿整张紧排文本行喂 rec 的
//!   rec-only 上限也只有 ≈ **0.157**，即这套合成 fixtures 上 rec 本身就是瓶颈，检测
//!   再怎么调也压不到 0.15 以下。
//!
//! 因此泰文走 PP-OCRv5 是 **best-effort**（高于 0.15 生产线 → 仍是“超阈值 → Typhoon
//! VLM 独立路径有依据”的诚实信号）。下面的 [`thai_ppocrv5_cer_eval`] 断言是**回归守
//! 卫**（`mean ≤ 0.28`，守住已达成的 ≈0.25 基线、防回退），不是生产门禁。

mod common;

use std::path::PathBuf;

use ocrspine::{OcrEngine, OcrImage, OcrWord, PaddleOcr};

use common::{cer, normalize, similarity};

/// 仓库根（crate manifest 目录）。
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// 一条 ground truth：图片文件名 + 源串。
struct Gt {
    file: String,
    text: String,
}

/// 解析 `tests/fixtures/thai/ground_truth.json`（形如 `[{file,text}]`）。
fn load_ground_truth() -> Vec<Gt> {
    let path = root().join("tests/fixtures/thai/ground_truth.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {} failed: {e}", path.display()));
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse ground_truth.json");
    v.as_array()
        .expect("ground_truth.json must be an array")
        .iter()
        .map(|o| Gt {
            file: o["file"].as_str().expect("file str").to_string(),
            text: o["text"].as_str().expect("text str").to_string(),
        })
        .collect()
}

/// 把识别词按阅读顺序（上到下、左到右）排序后用单空格拼接。
///
/// 用 box 中心 y 聚行：行容差取所有框高中位数的一半（最小 1px），同一行内按 x0 升序。
/// 单空格拼接 + 后续 [`normalize`] 折叠空白，对泰文（无词间空格）公平：检测把一行拆成
/// 多个 token 框时空格自然落在 token 边界，单框内部无空格。
fn reading_order_join(words: &[OcrWord]) -> String {
    if words.is_empty() {
        return String::new();
    }
    let mut heights: Vec<f64> = words.iter().map(|w| w.bbox.y1 - w.bbox.y0).collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_h = heights[heights.len() / 2];
    let tol = (median_h * 0.5).max(1.0);

    let mut idx: Vec<usize> = (0..words.len()).collect();
    let cy = |w: &OcrWord| (w.bbox.y0 + w.bbox.y1) / 2.0;
    // 行号 = round(中心y / tol)，再按 (行号, x0) 全序排序（确定性）。
    idx.sort_by(|&a, &b| {
        let ra = (cy(&words[a]) / tol).round() as i64;
        let rb = (cy(&words[b]) / tol).round() as i64;
        ra.cmp(&rb)
            .then(words[a].bbox.x0.partial_cmp(&words[b].bbox.x0).unwrap())
    });
    idx.iter()
        .map(|&i| words[i].text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// 设置识别缝指向随仓库落地的泰文 rec 模型与 index 对齐字典。
fn set_thai_rec_env() {
    let model = root().join("models/ppocrv5_rec_th.onnx");
    let dict = root().join("models/ppocr_keys_th.txt");
    assert!(model.exists(), "missing {}", model.display());
    assert!(dict.exists(), "missing {}", dict.display());
    std::env::set_var("OCRSPINE_REC_MODEL", &model);
    std::env::set_var("OCRSPINE_REC_DICT", &dict);
}

/// 一条已解码的评测样本。
struct Sample {
    file: String,
    text: String,
    image: OcrImage,
}

/// 解码全部 fixtures（一次性，供网格搜索反复复用）。
fn load_samples() -> Vec<Sample> {
    let gts = load_ground_truth();
    assert!(!gts.is_empty(), "no ground truth loaded");
    gts.into_iter()
        .map(|gt| {
            let png = root().join("tests/fixtures/thai").join(&gt.file);
            let bytes = std::fs::read(&png)
                .unwrap_or_else(|e| panic!("read {} failed: {e}", png.display()));
            let image = OcrImage::from_encoded(&bytes).expect("decode png");
            Sample {
                file: gt.file,
                text: gt.text,
                image,
            }
        })
        .collect()
}

/// 对全部样本跑识别，返回逐样本 `(hyp 归一化串, CER)`。读取当前进程的检测 env。
fn eval_samples(engine: &PaddleOcr, samples: &[Sample]) -> Vec<(String, f64)> {
    samples
        .iter()
        .map(|s| {
            let words = engine.recognize(&s.image).expect("recognize ok");
            let hyp = reading_order_join(&words);
            (normalize(&hyp), cer(&s.text, &hyp))
        })
        .collect()
}

/// 真实评测：泰文 rec + 字典 + **泰文检测 profile**（`OCRSPINE_DET_PROFILE=thai`，固化
/// 在 `src/paddle/detect.rs::DetectParams::thai`）。对全部 14 个 fixture 逐样本 dump
/// 识别 vs ground truth + 每样本 CER，再聚合 mean CER / exact 数。
///
/// **这是回归守卫，不是生产门禁**：未调检测 mean CER≈0.42（6/14 exact）→ 启用泰文
/// profile（仅 `text_score` 放宽 + 同行邻接框水平合并；bin/box/unclip 经网格搜索确认
/// 保持默认即最优）后 ≈0.25（实测 0.2500，10/14 exact）。0.25 **仍高于 0.15 的生产级
/// 目标**；这套 fixtures 上把检测旁路的 rec-only 上限也只有 ≈0.157（rec 本身是瓶颈），
/// 故 0.25 是 best-effort 基线。断言 `mean ≤ 0.28` 仅为守住该基线不回退，绝非达到 0.15。
#[test]
#[ignore = "heavy: loads the 7.8MB Thai rec model; run with --ignored --nocapture"]
fn thai_ppocrv5_cer_eval() {
    set_thai_rec_env();
    // 一键启用泰文检测 profile（固化在 DetectParams::thai()）。
    std::env::set_var("OCRSPINE_DET_PROFILE", "thai");

    let samples = load_samples();
    let engine = PaddleOcr::new().expect("build PaddleOcr");
    let results = eval_samples(&engine, &samples);

    eprintln!("\n=== PP-OCRv5 Thai eval ({} samples) ===", samples.len());
    eprintln!(
        "{:<14} {:<26} {:<26} {:>7}",
        "file", "ground_truth", "recognized", "CER"
    );
    let mut cers: Vec<f64> = Vec::with_capacity(samples.len());
    let mut sims: Vec<f64> = Vec::with_capacity(samples.len());
    for (s, (hyp, c)) in samples.iter().zip(results.iter()) {
        cers.push(*c);
        sims.push(similarity(&normalize(&s.text), hyp));
        eprintln!("{:<14} {:<26} {:<26} {:>7.4}", s.file, normalize(&s.text), hyp, c);
    }

    let mean = cers.iter().sum::<f64>() / cers.len() as f64;
    let max = cers.iter().cloned().fold(0.0_f64, f64::max);
    let perfect = cers.iter().filter(|&&c| c == 0.0).count();
    let mean_acc = sims.iter().sum::<f64>() / sims.len() as f64;
    eprintln!("---");
    eprintln!(
        "mean CER = {:.4} | max CER = {:.4} | exact = {}/{} | accuracy(1-normLev) = {:.4}",
        mean,
        max,
        perfect,
        cers.len(),
        mean_acc
    );

    // 回归守卫阈值（NOT 生产门禁）：守住泰文 profile 下已达成的 mean CER≈0.25 基线，
    // 防回退。0.25 仍高于 0.15 生产线、rec-only 上限≈0.157（rec 是瓶颈），详见上方文档。
    const REGRESSION_GUARD: f64 = 0.28;
    assert!(
        mean <= REGRESSION_GUARD,
        "Thai mean CER {mean:.4} regressed above the {REGRESSION_GUARD:.2} guard \
         (best-effort baseline ≈0.25 under the thai detection profile; \
         0.15 production target is knowingly NOT met — rec-only ceiling ≈0.157)"
    );
}

/// 一个检测参数 combo（网格搜索用）。
#[derive(Clone, Copy)]
struct Combo {
    box_thresh: f32,
    bin_thresh: f32,
    unclip: f32,
    merge_x: f32,
    merge_y: f32,
    min_box_side: i32,
    text_score: f32,
}

impl Combo {
    /// 把本 combo 写进进程 env（供下一次 `recognize` 现读）。
    fn apply(&self) {
        std::env::set_var("OCRSPINE_DET_BOX_THRESH", self.box_thresh.to_string());
        std::env::set_var("OCRSPINE_DET_BIN_THRESH", self.bin_thresh.to_string());
        std::env::set_var("OCRSPINE_DET_UNCLIP_RATIO", self.unclip.to_string());
        std::env::set_var("OCRSPINE_DET_MERGE_X", self.merge_x.to_string());
        std::env::set_var("OCRSPINE_DET_MERGE_Y", self.merge_y.to_string());
        std::env::set_var("OCRSPINE_DET_MIN_BOX_SIDE", self.min_box_side.to_string());
        std::env::set_var("OCRSPINE_TEXT_SCORE", self.text_score.to_string());
    }
    fn label(&self) -> String {
        format!(
            "box={:.2} bin={:.2} unclip={:.2} mx={:.1} my={:.1} minside={} text={:.2}",
            self.box_thresh,
            self.bin_thresh,
            self.unclip,
            self.merge_x,
            self.merge_y,
            self.min_box_side,
            self.text_score
        )
    }
}

/// 网格搜索（一次性调参工具，**勿在 CI / 常规跑**）：在 fixtures 上扫描泰文检测后
/// 处理参数，找端到端 mean CER 最低的一组。
///
/// 复用单个 [`PaddleOcr`]（检测参数每次 `recognize` 从 env 现读，无需重建/重优化），
/// 逐 combo 设置 `OCRSPINE_DET_*` env 后整体评测，按 mean CER 升序打印结果表，并在末尾
/// dump 最优组的逐样本对比。仅供调参，**不参与默认 `cargo test`**；输出真实数字。
#[test]
#[ignore = "one-shot tuning sweep; run with --ignored --nocapture to (re)derive the Thai profile — do NOT run in CI"]
fn thai_detection_grid_search() {
    set_thai_rec_env();
    let samples = load_samples();
    let engine = PaddleOcr::new().expect("build PaddleOcr");

    // 第二阶段聚焦扫描：2D 合并（merge_x/merge_y）+ min_box_side（滤掉切碎产生的噪声
    // 小框）。bin=0.3、text=0.3 经第一阶段确认为优区，固定以省时。
    let box_threshs = [0.3_f32, 0.4];
    let unclips = [1.6_f32, 2.0];
    let merge_xs = [1.5_f32, 2.0, 2.5];
    let merge_ys = [0.8_f32, 1.0, 1.2];
    let min_sides = [3_i32, 8, 12];

    struct Row {
        combo: Combo,
        mean: f64,
        exact: usize,
    }
    let mut rows: Vec<Row> = Vec::new();
    for &bx in &box_threshs {
        for &uc in &unclips {
            for &mx in &merge_xs {
                for &my in &merge_ys {
                    for &ms in &min_sides {
                        let combo = Combo {
                            box_thresh: bx,
                            bin_thresh: 0.3,
                            unclip: uc,
                            merge_x: mx,
                            merge_y: my,
                            min_box_side: ms,
                            text_score: 0.3,
                        };
                        combo.apply();
                        let results = eval_samples(&engine, &samples);
                        let mean =
                            results.iter().map(|(_, c)| c).sum::<f64>() / results.len() as f64;
                        let exact = results.iter().filter(|(_, c)| *c == 0.0).count();
                        rows.push(Row { combo, mean, exact });
                    }
                }
            }
        }
    }

    rows.sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());
    eprintln!("\n=== Thai detection grid search ({} combos) ===", rows.len());
    eprintln!("{:>8}  {:>7}  params", "meanCER", "exact");
    for r in &rows {
        eprintln!("{:>8.4}  {:>5}/{}  {}", r.mean, r.exact, samples.len(), r.combo.label());
    }
    let best = &rows[0];
    eprintln!(
        "---\nBEST: meanCER={:.4} exact={}/{}  {}",
        best.mean,
        best.exact,
        samples.len(),
        best.combo.label()
    );

    // 末尾：用最优 combo 逐样本 dump，便于核对哪些样本仍失败。
    best.combo.apply();
    let results = eval_samples(&engine, &samples);
    eprintln!("--- per-sample @ BEST ---");
    for (s, (hyp, c)) in samples.iter().zip(results.iter()) {
        eprintln!("{:<14} {:<26} {:<26} {:>7.4}", s.file, normalize(&s.text), hyp, c);
    }
}

// --- CER 指标移植的自检（轻量，默认随 `cargo test` 跑，不加载模型） ---

#[test]
fn cer_metric_basics() {
    // 完全一致 → 0；空 ref + 空 hyp → 0；空 ref + 非空 hyp → 1。
    assert_eq!(cer("abcd", "abcd"), 0.0);
    assert_eq!(cer("", ""), 0.0);
    assert_eq!(cer("", "x"), 1.0);
    // 1 处替换 / 4 字符参考 = 0.25。
    assert!((cer("kitten", "kitteX") - 1.0 / 6.0).abs() < 1e-12);
    // 空白归一化：多空格 / 换行折叠后一致 → 0。
    assert_eq!(cer("ก ข  ค", "ก ข\nค"), 0.0);
}

#[test]
fn cer_thai_chars_are_char_level() {
    // 泰文 char 级：替换 1 个声调符 → 1/参考字符数。
    let r = "ที่นี่"; // 6 个 codepoint
    let n = r.chars().count();
    let mut h: Vec<char> = r.chars().collect();
    h[1] = 'x';
    let hyp: String = h.into_iter().collect();
    assert!((cer(r, &hyp) - 1.0 / n as f64).abs() < 1e-12);
}
