//! Content hashes and git provenance.

use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// sha256 over several parts, separated by NUL so that ("ab", "c") != ("a", "bc").
pub fn sha256_parts(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p.as_bytes());
        h.update([0u8]);
    }
    hex::encode(h.finalize())
}

/// HEAD commit of the repo containing `dir`, and whether its tree has uncommitted changes.
/// Outside a git repo: ("unknown", true), so a run from it is never taken as reproducible.
pub fn git_state(dir: &Path) -> (String, bool) {
    let head = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output();
    let status = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["status", "--porcelain", "--", "."])
        .output();
    match (head, status) {
        (Ok(h), Ok(s)) if h.status.success() && s.status.success() => (
            String::from_utf8_lossy(&h.stdout).trim().to_string(),
            !s.stdout.is_empty(),
        ),
        _ => ("unknown".to_string(), true),
    }
}
