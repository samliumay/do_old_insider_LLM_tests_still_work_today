//! Git provenance of the code and benchmark trees.

use std::path::Path;
use std::process::Command;

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
