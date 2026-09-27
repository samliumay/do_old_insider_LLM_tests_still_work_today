//! Content hashes (sha256).

use sha2::{Digest, Sha256};

/// sha256 of `bytes`, as lowercase hex.
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
