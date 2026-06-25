//! 评测公用文本指标：char 级 Levenshtein 编辑距离、相似度、空白归一化、CER。
//!
//! 从 `pdfspine/python/tests/test_text.py` 的 `_levenshtein/_similarity/_normalize`
//! 移植为 Rust（char 级，对泰文/CJK 这类无词间空格的文字才有意义）。仅供 `tests/`
//! 下的评测 harness 复用（`mod common;`），不进入 crate 公共 API。

/// 标准编辑距离 DP（char 级；输入已切成 `&[char]`）。
pub fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a == b {
        return 0;
    }
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            let v = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
            cur.push(v);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// 空白归一化：把任意空白串（含换行/制表）折叠成单个空格并去首尾——避免布局分隔
/// 不公平地拉低相似度，同时不丢任何真实字形。等价于 Python `" ".join(s.split())`。
pub fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// 归一化相似度 = `1 − dist/max(len)`（char 级）；两者皆空 → 1.0。
pub fn similarity(a: &str, b: &str) -> f64 {
    let ca: Vec<char> = a.chars().collect();
    let cb: Vec<char> = b.chars().collect();
    if ca.is_empty() && cb.is_empty() {
        return 1.0;
    }
    let d = levenshtein(&ca, &cb);
    1.0 - d as f64 / ca.len().max(cb.len()) as f64
}

/// 字符错误率 CER = `levenshtein(ref, hyp) / len(ref)`（char 级，**两边先做空白归一化**）。
///
/// 参考串为空时：假设串也空 → 0.0；否则 → 1.0。CER 可超过 1.0（插入过多时）。
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let r: Vec<char> = normalize(reference).chars().collect();
    let h: Vec<char> = normalize(hypothesis).chars().collect();
    if r.is_empty() {
        return if h.is_empty() { 0.0 } else { 1.0 };
    }
    levenshtein(&r, &h) as f64 / r.len() as f64
}
