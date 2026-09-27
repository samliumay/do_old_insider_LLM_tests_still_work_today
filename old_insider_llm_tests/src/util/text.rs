//! Text helpers for log lines.

/// The first `n` characters of `s` (for log lines).
pub fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}
