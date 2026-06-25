//! DBNet text detection: image → text boxes in original pixel coordinates.
//!
//! Pipeline (matching RapidOCR's PP-OCRv5 det config):
//!   1. resize so `min(h,w) ≈ 736` (limit_type=min), cap max side ≤ 2000,
//!      min side ≥ 30, both rounded to a multiple of 32 for the model;
//!   2. normalize per channel and run DBNet → prob map `[1,1,H,W]`;
//!   3. binarize at `thresh=0.3`, dilate the mask 2×2 (`use_dilation`);
//!   4. for each connected component compute the **minimum-area rotated
//!      rectangle** (rotating calipers over the component's convex hull);
//!   5. drop boxes whose mean prob (score_mode=fast) `< box_thresh=0.5`;
//!   6. unclip (inflate) the rotated rect by `area*unclip_ratio/perimeter`
//!      (`unclip_ratio=1.6`) outward along both axes;
//!   7. scale the rotated quad back to ORIGINAL image pixels;
//!   8. sort top-to-bottom then left-to-right.
//!
//! Each [`DetBox`] carries both the axis-aligned bounding box (the public
//! `OcrWord.bbox`) AND the four rotated-quad corners + angle, so the recognizer
//! can de-rotate skewed crops to horizontal. A ~0° min-area rect collapses to
//! the axis-aligned box, so upright text behaves exactly as before.

use image::RgbImage;
use tract_onnx::prelude::*;

use crate::error::{OcrError as Error, Result};
use crate::paddle::model::Models;
use crate::paddle::preprocess::{resize_exact, to_tensor};

/// `limit_side_len`: resize so the SHORT side lands near this (limit_type=min).
const LIMIT_SIDE_LEN: u32 = 736;
/// Hard cap on the long side after resize.
const MAX_SIDE: u32 = 2000;
/// Hard floor on the short side after resize.
const MIN_SIDE: u32 = 30;
/// Model spatial dims must be multiples of this.
const STRIDE: u32 = 32;
/// Probability-map binarization threshold.
const BIN_THRESH: f32 = 0.3;
/// Minimum mean-prob (fast mode) for a kept box.
const BOX_THRESH: f32 = 0.5;
/// Box inflation ratio (Vatti-clip approximation).
const UNCLIP_RATIO: f32 = 1.6;
/// Minimum box side (px, in MODEL space) to keep a component.
const MIN_BOX_SIDE: i32 = 3;

/// 识别置信度丢弃阈值的默认值（rec `text_score`，与历史 `mod.rs` 中的常量一致）。
const TEXT_SCORE: f32 = 0.5;

/// ImageNet normalization (the variant RapidOCR's det model is trained with).
const DET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// --- 检测后处理可调参数（env 覆盖缝，镜像 `OCRSPINE_*` 模式） ---

/// 命名检测 profile 选择：`thai` 选用 [`DetectParams::thai`] 作为基线，其它/未设
/// 时用 [`DetectParams::default`]（=历史硬编码常量）。个别 `OCRSPINE_DET_*` 仍可
/// 在所选基线之上逐项覆盖。
const ENV_DET_PROFILE: &str = "OCRSPINE_DET_PROFILE";
/// 短边目标长度 `limit_side_len`（limit_type=min）。
const ENV_DET_LIMIT_SIDE: &str = "OCRSPINE_DET_LIMIT_SIDE";
/// 概率图二值化阈值 `db_thresh`。
const ENV_DET_BIN_THRESH: &str = "OCRSPINE_DET_BIN_THRESH";
/// 保留框的最小 fast-score `db_box_thresh`（<0.5 丢框；泰文短词需调低）。
const ENV_DET_BOX_THRESH: &str = "OCRSPINE_DET_BOX_THRESH";
/// unclip 膨胀比例 `db_unclip_ratio`（泰文上叠/下挂声调需更大膨胀含进整簇）。
const ENV_DET_UNCLIP_RATIO: &str = "OCRSPINE_DET_UNCLIP_RATIO";
/// 最小框边（model 空间像素）；避免把声调小连通域当独立框或被滤掉。
const ENV_DET_MIN_BOX_SIDE: &str = "OCRSPINE_DET_MIN_BOX_SIDE";
/// 同行邻接框水平合并阈值（gap ≤ ratio×行高 则合并）；负值/未设=不合并。
const ENV_DET_MERGE_X: &str = "OCRSPINE_DET_MERGE_X";
/// 合并的垂直可达比例（× 行高）：把上叠/下挂声调簇等垂直偏移的框并入同一行。
const ENV_DET_MERGE_Y: &str = "OCRSPINE_DET_MERGE_Y";
/// 识别置信度丢弃阈值 `text_score`（rec 置信度低于此值的框被丢，致“为空”）。
const ENV_TEXT_SCORE: &str = "OCRSPINE_TEXT_SCORE";

/// DBNet 后处理 + 识别丢弃的可调参数集合。
///
/// **默认即历史常量**：[`DetectParams::default`] 的每个字段都等于上面的硬编码常量，
/// 且 `merge_x_ratio = None`（不做水平合并），故不设任何 env 时 zh/en/ja 的检测行为
/// 与改动前**逐位一致**。[`from_env`](DetectParams::from_env) 先按 `OCRSPINE_DET_PROFILE`
/// 选基线（`thai` → [`thai`](DetectParams::thai)，否则 default），再用个别
/// `OCRSPINE_DET_*` env 逐项覆盖。
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetectParams {
    /// 短边目标长度（resize 到此，limit_type=min）。
    pub limit_side_len: u32,
    /// 概率图二值化阈值。
    pub bin_thresh: f32,
    /// 保留框的最小 fast-score。
    pub box_thresh: f32,
    /// unclip 膨胀比例。
    pub unclip_ratio: f32,
    /// 最小框边（model 空间像素）。
    pub min_box_side: i32,
    /// 同行邻接框水平合并阈值（gap ≤ ratio×行高 合并）；`None`=不合并（整个合并步关闭）。
    pub merge_x_ratio: Option<f32>,
    /// 合并的垂直可达比例（× 行高）：连通上叠/下挂声调簇等垂直偏移的框。仅在
    /// `merge_x_ratio` 开启时生效。
    pub merge_y_ratio: f32,
    /// 识别置信度丢弃阈值（rec `text_score`）。
    pub text_score: f32,
}

impl Default for DetectParams {
    /// 历史硬编码常量（默认路径），与改动前逐位一致。
    fn default() -> Self {
        Self {
            limit_side_len: LIMIT_SIDE_LEN,
            bin_thresh: BIN_THRESH,
            box_thresh: BOX_THRESH,
            unclip_ratio: UNCLIP_RATIO,
            min_box_side: MIN_BOX_SIDE,
            merge_x_ratio: None,
            merge_y_ratio: 0.0,
            text_score: TEXT_SCORE,
        }
    }
}

impl DetectParams {
    /// 泰文检测 profile：`tests/thai_eval.rs::thai_detection_grid_search` 网格搜索固化
    /// 的端到端最优组（14 条合成 fixtures 上实测 mean CER≈0.25，10/14 exact）。相对默认
    /// 检测**只动两处**：(1) `text_score` 0.5→0.3，避免泰文短词/声调簇因 rec 置信度偏低
    /// 被整框丢空；(2) 启用同行邻接框水平合并 `merge_x_ratio=2.0`，把被按 CJK 调参的
    /// DBNet 切碎的一行重新拼回单个轴对齐框喂给 rec（接近“整行直喂 rec”的上限）。二值
    /// 化/框 score/unclip 三个阈值经网格搜索确认**保持默认即最优**（bin=0.3、box=0.5、
    /// unclip=1.6），故此处直接复用默认常量。通过 `OCRSPINE_DET_PROFILE=thai` 一键启用
    /// （rec 切到泰文模型时）。
    pub(crate) fn thai() -> Self {
        Self {
            // bin/box/unclip：网格搜索确认默认值即最优，复用默认常量（值同 default）。
            limit_side_len: LIMIT_SIDE_LEN,
            bin_thresh: BIN_THRESH,
            box_thresh: BOX_THRESH,
            unclip_ratio: UNCLIP_RATIO,
            min_box_side: MIN_BOX_SIDE,
            // 关键改动一：同行邻接框水平合并（x_ratio=2.0），把切碎的一行拼回单框喂 rec。
            // merge_y 保持 default=0（即 grid-search BEST 的基线值）；这套 fixtures 上
            // merge_y 取 0 / 0.8 结果一致，故取与 BEST 一致的 0。
            merge_x_ratio: Some(2.0),
            merge_y_ratio: 0.0,
            // 关键改动二：放宽 rec 置信度丢弃阈值，不丢泰文短词。
            text_score: 0.3,
        }
    }

    /// 按 env 构建：先选基线 profile，再逐项覆盖。未设任何 env → [`default`](Self::default)。
    pub(crate) fn from_env() -> Self {
        let mut p = match std::env::var(ENV_DET_PROFILE).ok().as_deref() {
            Some("thai") => Self::thai(),
            _ => Self::default(),
        };
        if let Some(v) = env_parse::<u32>(ENV_DET_LIMIT_SIDE) {
            p.limit_side_len = v.max(STRIDE);
        }
        if let Some(v) = env_parse::<f32>(ENV_DET_BIN_THRESH) {
            p.bin_thresh = v;
        }
        if let Some(v) = env_parse::<f32>(ENV_DET_BOX_THRESH) {
            p.box_thresh = v;
        }
        if let Some(v) = env_parse::<f32>(ENV_DET_UNCLIP_RATIO) {
            p.unclip_ratio = v;
        }
        if let Some(v) = env_parse::<i32>(ENV_DET_MIN_BOX_SIDE) {
            p.min_box_side = v;
        }
        if let Some(v) = env_parse::<f32>(ENV_DET_MERGE_X) {
            // 负值显式关闭合并；非负值启用并设阈值。
            p.merge_x_ratio = if v < 0.0 { None } else { Some(v) };
        }
        if let Some(v) = env_parse::<f32>(ENV_DET_MERGE_Y) {
            p.merge_y_ratio = v.max(0.0);
        }
        if let Some(v) = env_parse::<f32>(ENV_TEXT_SCORE) {
            p.text_score = v;
        }
        p
    }
}

/// 读取并解析一个 env 标量；未设置或解析失败时返回 `None`（保持基线值）。
fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.trim().parse::<T>().ok()
}

/// A detected text box in ORIGINAL image pixel coordinates.
///
/// `x0,y0,x1,y1` is the axis-aligned bounding box of the rotated quad (this is
/// the box surfaced as `OcrWord.bbox`). `quad` holds the four corners of the
/// minimum-area rotated rectangle, ordered top-left, top-right, bottom-right,
/// bottom-left along the rect's own axes; `angle` is the rect's rotation in
/// radians (the angle of its long/text axis from horizontal, in `(-π/2, π/2]`).
/// For upright text `angle ≈ 0` and the quad coincides with the AABB corners.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DetBox {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub score: f32,
    pub quad: [(f32, f32); 4],
    pub angle: f32,
}

/// Computes the model input size for `(w, h)`: scale the short side to
/// `LIMIT_SIDE_LEN`, clamp the long side to `MAX_SIDE` and short to `MIN_SIDE`,
/// then round both to a multiple of `STRIDE`. Returns `(model_w, model_h)`.
fn det_input_size(w: u32, h: u32, limit_side_len: u32) -> (u32, u32) {
    let short = w.min(h).max(1) as f32;
    let scale = limit_side_len as f32 / short;
    let mut mw = (w as f32 * scale).round() as u32;
    let mut mh = (h as f32 * scale).round() as u32;
    // Clamp long/short sides.
    let long = mw.max(mh);
    if long > MAX_SIDE {
        let s = MAX_SIDE as f32 / long as f32;
        mw = (mw as f32 * s).round() as u32;
        mh = (mh as f32 * s).round() as u32;
    }
    mw = mw.max(MIN_SIDE);
    mh = mh.max(MIN_SIDE);
    // Round up to multiple of STRIDE.
    mw = mw.div_ceil(STRIDE) * STRIDE;
    mh = mh.div_ceil(STRIDE) * STRIDE;
    (mw.max(STRIDE), mh.max(STRIDE))
}

/// Runs detection on `img`, returning boxes in original-image pixel coords.
///
/// 后处理阈值/膨胀/合并由 `params` 决定（[`DetectParams::from_env`] 提供 env 覆盖）；
/// 传 [`DetectParams::default`] 即历史行为。
pub(crate) fn detect(models: &Models, img: &RgbImage, params: &DetectParams) -> Result<Vec<DetBox>> {
    let (ow, oh) = (img.width(), img.height());
    let (mw, mh) = det_input_size(ow, oh, params.limit_side_len);

    let resized = resize_exact(img, mw, mh);
    let tensor = to_tensor(&resized, DET_MEAN, DET_STD);

    let runnable = models.det(mh as usize, mw as usize)?;
    let out = runnable
        .run(tvec!(tensor.into()))
        .map_err(|e| Error::Unsupported(format!("paddle: detection inference failed: {e}")))?;

    // Output prob map [1,1,H,W] (or [1,H,W]); read as a flat H*W f32 view.
    let view = out[0]
        .to_array_view::<f32>()
        .map_err(|e| Error::Unsupported(format!("paddle: bad detection output: {e}")))?;
    let shape = view.shape();
    let (ph, pw) = match shape.len() {
        4 => (shape[2], shape[3]),
        3 => (shape[1], shape[2]),
        2 => (shape[0], shape[1]),
        _ => {
            return Err(Error::Unsupported(format!(
                "paddle: unexpected detection output rank {}",
                shape.len()
            )))
        }
    };
    let prob: Vec<f32> = view.iter().copied().collect();
    debug_assert_eq!(prob.len(), ph * pw);

    // 1) Binarize.
    let mut mask = vec![false; ph * pw];
    for (m, &p) in mask.iter_mut().zip(prob.iter()) {
        *m = p >= params.bin_thresh;
    }
    // 2) Dilate 2×2 (structuring element anchored top-left, like cv2 with a
    //    2×2 kernel: a pixel turns on if itself or its right/below/diagonal
    //    neighbour was on). This thickens strokes so adjacent glyphs merge.
    let dilated = dilate_2x2(&mask, pw, ph);

    // 3) Connected components (8-connectivity), keeping each component's
    //    foreground pixels (in model/prob-map space, which equals model-input
    //    space — DBNet output is full resolution) so we can fit a rotated rect.
    let comps = connected_components(&dilated, pw, ph);

    // Scale from model space back to original image pixels.
    let sx = ow as f32 / mw as f32;
    let sy = oh as f32 / mh as f32;

    let mut boxes = Vec::new();
    for c in comps {
        // Skip tiny components (use the AABB extent as a cheap pre-filter).
        if (c.x1 - c.x0) < params.min_box_side || (c.y1 - c.y0) < params.min_box_side {
            continue;
        }
        // 4) Minimum-area rotated rectangle over the component's convex hull.
        let mar = min_area_rect(&c.pixels);
        // Skip degenerate rects (a thin line of pixels).
        if mar.w < params.min_box_side as f32 || mar.h < params.min_box_side as f32 {
            continue;
        }
        // 5) Fast score: mean prob over the rotated rect's polygon (RapidOCR's
        //    box_score_fast masks the box, not its AABB — essential for skewed
        //    boxes, whose AABB is mostly background).
        let score = mean_prob_quad(&prob, pw, ph, &rect_corners(&mar));
        if score < params.box_thresh {
            continue;
        }
        // 6) Unclip: inflate the rect outward along both axes.
        let quad_model = unclip_rect(&mar, params.unclip_ratio);

        // 7) Scale the quad to original pixels.
        let mut quad = [(0.0f32, 0.0f32); 4];
        for (i, &(px, py)) in quad_model.iter().enumerate() {
            quad[i] = (px * sx, py * sy);
        }
        // Axis-aligned bbox of the (scaled) rotated quad → OcrWord.bbox.
        let (mut bx0, mut by0, mut bx1, mut by1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for &(px, py) in &quad {
            bx0 = bx0.min(px);
            by0 = by0.min(py);
            bx1 = bx1.max(px);
            by1 = by1.max(py);
        }
        // Rect angle measured in original pixel space (sx/sy may differ, but for
        // OCR pages they are equal, so this is the true text-axis angle).
        let angle = (quad[1].1 - quad[0].1).atan2(quad[1].0 - quad[0].0);
        boxes.push(DetBox {
            x0: (bx0.floor() as i32).clamp(0, ow as i32),
            y0: (by0.floor() as i32).clamp(0, oh as i32),
            x1: (bx1.ceil() as i32).clamp(0, ow as i32),
            y1: (by1.ceil() as i32).clamp(0, oh as i32),
            score,
            quad,
            angle,
        });
    }

    // 6b) 可选：把同一文本行被切碎的框（横向相邻 + 上叠/下挂声调垂直偏移）聚成单个
    //     轴对齐框，使 rec 看到完整文本行 —— 对无词间空格、含声调簇的泰文短词尤其关键。
    //     仅在 profile 显式开启时运行；默认 `None` 不动（zh/en/ja 逐位不变）。
    if let Some(x_ratio) = params.merge_x_ratio {
        boxes = merge_row_boxes(boxes, x_ratio, params.merge_y_ratio, ow, oh);
    }

    // 7) Sort top-to-bottom, then left-to-right. Group rows by a y-tolerance so
    //    boxes on the same visual line read left-to-right.
    boxes.sort_by(|a, b| {
        let ay = (a.y0 + a.y1) / 2;
        let by = (b.y0 + b.y1) / 2;
        // Same line if vertical centers are within half the smaller box height.
        let tol = ((a.y1 - a.y0).min(b.y1 - b.y0) / 2).max(1);
        if (ay - by).abs() <= tol {
            a.x0.cmp(&b.x0)
        } else {
            ay.cmp(&by)
        }
    });

    Ok(boxes)
}

/// 文本行框聚类合并（仅泰文 profile 等显式开启时调用）。
///
/// 用并查集把“应属同一文本行”的框聚成一组，每组取并集 AABB 输出**一个轴对齐框**
/// （`quad` 退化为 AABB 四角、`angle≈0`，走 upright 裁剪路径；`score` 取组内最大）。
/// 连通判据：两框各按 `参考高度 = max(两框高)` 把 x 方向外扩 `x_ratio×h/2`、y 方向
/// 外扩 `y_ratio×h/2` 后若仍相交即连通。这样**既**桥接横向相邻的同行碎框（x 方向），
/// **又**把上叠声调/下挂元音这类垂直偏移、但 x 范围重叠的框并进同一行（y 方向）——
/// 后者正是泰文短词（น้ำ / ที่นี่ / ผู้ใหญ่）此前被切碎/漏识的根因。
/// `x_ratio` 适中可只桥接被切碎的同词碎框、保留真实词间空格对应的较大间隙；
/// `y_ratio` 取行高的一个零头即可连通声调簇而不致跨行误并（多行间距通常更大）。
fn merge_row_boxes(boxes: Vec<DetBox>, x_ratio: f32, y_ratio: f32, ow: u32, oh: u32) -> Vec<DetBox> {
    let n = boxes.len();
    if n < 2 {
        return boxes;
    }
    // 并查集。
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while parent[r] != r {
            r = parent[r];
        }
        // 路径压缩。
        let mut c = i;
        while parent[c] != r {
            let next = parent[c];
            parent[c] = r;
            c = next;
        }
        r
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if boxes_connected(&boxes[i], &boxes[j], x_ratio, y_ratio) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    // 按根聚组，组内取并集 AABB；用 HashMap 把根映射到 out 索引，保持稳定。
    let mut root_to_out: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut out: Vec<DetBox> = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        let r = find(&mut parent, i);
        match root_to_out.get(&r) {
            Some(&oi) => {
                let acc = &mut out[oi];
                acc.x0 = acc.x0.min(b.x0);
                acc.y0 = acc.y0.min(b.y0);
                acc.x1 = acc.x1.max(b.x1);
                acc.y1 = acc.y1.max(b.y1);
                acc.score = acc.score.max(b.score);
            }
            None => {
                root_to_out.insert(r, out.len());
                out.push(*b);
            }
        }
    }

    // 收尾：夹回图像范围，并把 quad/angle 规整为合并后的轴对齐框。
    for b in &mut out {
        b.x0 = b.x0.clamp(0, ow as i32);
        b.y0 = b.y0.clamp(0, oh as i32);
        b.x1 = b.x1.clamp(0, ow as i32);
        b.y1 = b.y1.clamp(0, oh as i32);
        b.quad = aabb_quad(b.x0, b.y0, b.x1, b.y1);
        b.angle = 0.0;
    }
    out
}

/// 两框是否“属同一文本行”：以 `max(两框高)` 为参考高度，x 方向外扩 `x_ratio×h/2`、
/// y 方向外扩 `y_ratio×h/2` 后若 AABB 仍相交即连通（对称、与遍历顺序无关）。
fn boxes_connected(a: &DetBox, b: &DetBox, x_ratio: f32, y_ratio: f32) -> bool {
    let ah = (a.y1 - a.y0).max(1) as f32;
    let bh = (b.y1 - b.y0).max(1) as f32;
    let h = ah.max(bh);
    let mx = x_ratio * h * 0.5;
    let my = y_ratio * h * 0.5;
    let (ax0, ay0, ax1, ay1) = (a.x0 as f32, a.y0 as f32, a.x1 as f32, a.y1 as f32);
    let (bx0, by0, bx1, by1) = (b.x0 as f32, b.y0 as f32, b.x1 as f32, b.y1 as f32);
    let x_overlap = (ax0 - mx) <= (bx1 + mx) && (bx0 - mx) <= (ax1 + mx);
    let y_overlap = (ay0 - my) <= (by1 + my) && (by0 - my) <= (ay1 + my);
    x_overlap && y_overlap
}

/// 轴对齐 bbox 的四角（top-left, top-right, bottom-right, bottom-left）。
fn aabb_quad(x0: i32, y0: i32, x1: i32, y1: i32) -> [(f32, f32); 4] {
    let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
    [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
}

/// 2×2 dilation: output pixel `(x,y)` is on if any of `(x,y)`, `(x+1,y)`,
/// `(x,y+1)`, `(x+1,y+1)` was on in the input. Matches cv2.dilate with a 2×2
/// all-ones kernel (anchored at the top-left).
fn dilate_2x2(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut on = mask[y * w + x];
            if !on && x + 1 < w {
                on = mask[y * w + x + 1];
            }
            if !on && y + 1 < h {
                on = mask[(y + 1) * w + x];
            }
            if !on && x + 1 < w && y + 1 < h {
                on = mask[(y + 1) * w + x + 1];
            }
            out[y * w + x] = on;
        }
    }
    out
}

/// One connected component: its axis-aligned bbox (`x1`/`y1` exclusive, spanning
/// `[x0,x1) × [y0,y1)`) plus every foreground pixel `(x,y)` it contains (mask
/// coords), which feeds the rotated-rect fit.
struct Comp {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    pixels: Vec<(i32, i32)>,
}

/// 8-connected component extraction via iterative flood fill, collecting each
/// component's pixels and bounding box.
fn connected_components(mask: &[bool], w: usize, h: usize) -> Vec<Comp> {
    let mut visited = vec![false; w * h];
    let mut comps = Vec::new();
    let mut stack: Vec<(i32, i32)> = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            let idx = sy * w + sx;
            if !mask[idx] || visited[idx] {
                continue;
            }
            // New component: flood fill, tracking the bbox and pixels.
            let (mut x0, mut y0) = (sx as i32, sy as i32);
            let (mut x1, mut y1) = (sx as i32, sy as i32);
            let mut pixels = Vec::new();
            stack.clear();
            stack.push((sx as i32, sy as i32));
            visited[idx] = true;
            while let Some((cx, cy)) = stack.pop() {
                x0 = x0.min(cx);
                y0 = y0.min(cy);
                x1 = x1.max(cx);
                y1 = y1.max(cy);
                pixels.push((cx, cy));
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = cx + dx;
                        let ny = cy + dy;
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let nidx = ny as usize * w + nx as usize;
                        if mask[nidx] && !visited[nidx] {
                            visited[nidx] = true;
                            stack.push((nx, ny));
                        }
                    }
                }
            }
            // Make x1/y1 exclusive.
            comps.push(Comp {
                x0,
                y0,
                x1: x1 + 1,
                y1: y1 + 1,
                pixels,
            });
        }
    }
    comps
}

/// A rotated rectangle: center, side lengths (`w` is the long/text axis, `h` the
/// short axis), and the text axis as a unit vector `(ux, uy)`. Lives in
/// mask/model space.
pub(crate) struct RotatedRect {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub ux: f32,
    pub uy: f32,
}

/// Computes the minimum-area enclosing rectangle of a point set via rotating
/// calipers over its convex hull. The returned rect's `w` is the longer side
/// (taken as the text axis) and `(ux,uy)` points along it.
///
/// For an axis-aligned point cloud this yields an axis-aligned rect (`uy≈0`), so
/// upright text collapses to today's behavior.
pub(crate) fn min_area_rect(points: &[(i32, i32)]) -> RotatedRect {
    let hull = convex_hull(points);
    // Degenerate hulls: fall back to the axis-aligned bbox.
    if hull.len() < 3 {
        let (mut x0, mut y0, mut x1, mut y1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for &(px, py) in points {
            let (px, py) = (px as f32, py as f32);
            x0 = x0.min(px);
            y0 = y0.min(py);
            x1 = x1.max(px);
            y1 = y1.max(py);
        }
        return RotatedRect {
            cx: (x0 + x1) / 2.0,
            cy: (y0 + y1) / 2.0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
            ux: 1.0,
            uy: 0.0,
        };
    }

    let mut best_area = f32::INFINITY;
    let mut best = RotatedRect {
        cx: 0.0,
        cy: 0.0,
        w: 0.0,
        h: 0.0,
        ux: 1.0,
        uy: 0.0,
    };
    let n = hull.len();
    // Each hull edge is a candidate rect orientation (the optimal rect always
    // has one side flush with an edge of the hull).
    for i in 0..n {
        let (ax, ay) = hull[i];
        let (bx, by) = hull[(i + 1) % n];
        let (ex, ey) = (bx - ax, by - ay);
        let len = (ex * ex + ey * ey).sqrt();
        if len < f32::EPSILON {
            continue;
        }
        // Edge direction (u) and its perpendicular (v).
        let (ux, uy) = (ex / len, ey / len);
        let (vx, vy) = (-uy, ux);
        // Project every hull vertex onto (u, v).
        let (mut min_u, mut max_u) = (f32::INFINITY, f32::NEG_INFINITY);
        let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
        for &(hx, hy) in &hull {
            let pu = hx * ux + hy * uy;
            let pv = hx * vx + hy * vy;
            min_u = min_u.min(pu);
            max_u = max_u.max(pu);
            min_v = min_v.min(pv);
            max_v = max_v.max(pv);
        }
        let su = max_u - min_u;
        let sv = max_v - min_v;
        let area = su * sv;
        if area < best_area {
            best_area = area;
            // Center in (u,v) → back to xy.
            let mu = (min_u + max_u) / 2.0;
            let mv = (min_v + max_v) / 2.0;
            let cx = mu * ux + mv * vx;
            let cy = mu * uy + mv * vy;
            // Orient so w is the longer side (the text axis).
            if su >= sv {
                best = RotatedRect {
                    cx,
                    cy,
                    w: su,
                    h: sv,
                    ux,
                    uy,
                };
            } else {
                best = RotatedRect {
                    cx,
                    cy,
                    w: sv,
                    h: su,
                    ux: vx,
                    uy: vy,
                };
            }
        }
    }

    // Normalize the axis so the angle stays in (-π/2, π/2] (point rightward, and
    // for the vertical edge case point downward). Keeps `angle` ~0 for upright.
    if best.ux < 0.0 || (best.ux == 0.0 && best.uy < 0.0) {
        best.ux = -best.ux;
        best.uy = -best.uy;
    }
    best
}

/// Andrew's monotone-chain convex hull. Input mask pixels (`i32`), output hull
/// vertices in `f32`, counter-clockwise, without the duplicated endpoint.
fn convex_hull(points: &[(i32, i32)]) -> Vec<(f32, f32)> {
    if points.len() < 3 {
        return points.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    }
    let mut pts: Vec<(i32, i32)> = points.to_vec();
    pts.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    pts.dedup();
    if pts.len() < 3 {
        return pts.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    }

    // 2D cross product of OA × OB (i64 to avoid overflow).
    let cross = |o: (i32, i32), a: (i32, i32), b: (i32, i32)| -> i64 {
        (a.0 - o.0) as i64 * (b.1 - o.1) as i64 - (a.1 - o.1) as i64 * (b.0 - o.0) as i64
    };

    let mut lower: Vec<(i32, i32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(i32, i32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower.iter().map(|&(x, y)| (x as f32, y as f32)).collect()
}

/// The four corners of a rotated rect (no unclip), in mask space, ordered
/// top-left, top-right, bottom-right, bottom-left along the rect's own axes.
pub(crate) fn rect_corners(r: &RotatedRect) -> [(f32, f32); 4] {
    let hw = r.w / 2.0;
    let hh = r.h / 2.0;
    let (ux, uy) = (r.ux, r.uy);
    let (vx, vy) = (-uy, ux);
    let corner = |su: f32, sv: f32| -> (f32, f32) {
        (
            r.cx + su * hw * ux + sv * hh * vx,
            r.cy + su * hw * uy + sv * hh * vy,
        )
    };
    [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ]
}

/// Mean probability over the (convex) quad polygon — the fast score for a
/// rotated box. Scans the quad's AABB and includes only pixels inside the
/// polygon (point-in-convex-polygon via consistent edge sign).
fn mean_prob_quad(prob: &[f32], w: usize, h: usize, quad: &[(f32, f32); 4]) -> f32 {
    let (mut x0, mut y0, mut x1, mut y1) = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for &(px, py) in quad {
        x0 = x0.min(px);
        y0 = y0.min(py);
        x1 = x1.max(px);
        y1 = y1.max(py);
    }
    let ix0 = (x0.floor() as i32).clamp(0, w as i32);
    let iy0 = (y0.floor() as i32).clamp(0, h as i32);
    let ix1 = (x1.ceil() as i32).clamp(0, w as i32);
    let iy1 = (y1.ceil() as i32).clamp(0, h as i32);
    if ix1 <= ix0 || iy1 <= iy0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in iy0..iy1 {
        for x in ix0..ix1 {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            if point_in_quad(p, quad) {
                sum += prob[y as usize * w + x as usize];
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

/// Point-in-convex-quad test: the point is inside iff it is on the same side of
/// every directed edge (all cross products share one sign).
fn point_in_quad(p: (f32, f32), quad: &[(f32, f32); 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let cross = (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
        if cross.abs() > f32::EPSILON {
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
    }
    true
}

/// Unclips (inflates) a rotated rect outward along both of its axes by
/// `distance = area * unclip_ratio / perimeter` (Vatti-clip approximation) and
/// returns the four corners ordered along the rect's own axes: top-left,
/// top-right, bottom-right, bottom-left (in mask space).
pub(crate) fn unclip_rect(r: &RotatedRect, unclip_ratio: f32) -> [(f32, f32); 4] {
    let area = r.w * r.h;
    let perimeter = 2.0 * (r.w + r.h);
    let dist = if perimeter > 0.0 {
        area * unclip_ratio / perimeter
    } else {
        0.0
    };
    let hw = r.w / 2.0 + dist;
    let hh = r.h / 2.0 + dist;
    let (ux, uy) = (r.ux, r.uy);
    let (vx, vy) = (-uy, ux); // perpendicular (short axis)
                              // Corners: -u-v, +u-v, +u+v, -u+v.
    let corner = |su: f32, sv: f32| -> (f32, f32) {
        (
            r.cx + su * hw * ux + sv * hh * vx,
            r.cy + su * hw * uy + sv * hh * vy,
        )
    };
    [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ]
}
