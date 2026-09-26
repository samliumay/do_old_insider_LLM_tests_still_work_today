//! What every model API returns, and one enum over the APIs, so the runner does not care
//! which one serves a model.

use anyhow::Result;
use serde_json::Value;

use crate::config::{ModelConfig, ModelSource};
use crate::types::{Api, EpisodeStatus};
use crate::{ollama, openrouter};

pub struct Request<'a> {
    pub model: &'a str,
    /// OpenRouter: pinned provider tag. Ollama: unused (the digest is checked before the run).
    pub pin: &'a str,
    pub system: &'a str,
    pub user: &'a str,
    pub temperature: f64,
    pub max_tokens: u32,
    pub seed: Option<u64>,
}

/// What one call returned. `status` comes from the finish reason:
/// `stop` → ok, `length` → truncated, anything else (incl. `error`) → failed.
#[derive(Debug)]
pub struct Reply {
    pub status: EpisodeStatus,
    pub answer: Option<String>,
    pub reasoning: Option<String>,
    pub finish_reason: Option<String>,
    pub served_by: Option<String>,
    pub api_model: Option<String>,
    pub prompt_tokens: Option<i32>,
    pub completion_tokens: Option<i32>,
    pub reasoning_tokens: Option<i32>,
    pub cost_usd: Option<f64>,
    pub error: Option<String>,
    pub raw: Value,
}

/// Why a call produced no reply. Only `Retryable` errors are tried again.
#[derive(Debug)]
pub enum CallError {
    /// Network trouble, timeouts, rate limits, 5xx, malformed bodies.
    Retryable(anyhow::Error),
    /// A 4xx other than 408/429: the request itself is wrong (e.g. no endpoint fits it).
    Permanent(anyhow::Error),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Retryable(e) => write!(f, "{e:#}"),
            CallError::Permanent(e) => write!(f, "{e:#} (not retried)"),
        }
    }
}

/// Classify a non-success HTTP status.
pub fn http_error(status: reqwest::StatusCode, body: &str) -> CallError {
    let err = anyhow::anyhow!("HTTP {status}: {}", truncate(body, 300));
    if status.is_server_error() || status.as_u16() == 408 || status.as_u16() == 429 {
        CallError::Retryable(err)
    } else {
        CallError::Permanent(err)
    }
}

pub enum Backend {
    OpenRouter(openrouter::Client),
    Ollama(ollama::Client),
}

impl Backend {
    pub fn api(&self) -> Api {
        match self {
            Backend::OpenRouter(_) => Api::OpenRouter,
            Backend::Ollama(_) => Api::Ollama,
        }
    }

    /// Check that the pinned endpoint (OpenRouter) or digest (Ollama) is what will serve
    /// the model, before a run starts. Returns the pin to record with the run.
    pub async fn check(&self, m: &ModelConfig, max_tokens: u32) -> Result<String> {
        match (self, &m.source) {
            (Backend::OpenRouter(c), ModelSource::OpenRouter { provider }) => {
                c.check_endpoint(&m.id, provider, max_tokens).await?;
                Ok(provider.clone())
            }
            (Backend::Ollama(c), ModelSource::Ollama { digest }) => {
                c.check_digest(&m.id, digest).await?;
                Ok(digest.clone())
            }
            (Backend::OpenRouter(_), ModelSource::Ollama { .. })
            | (Backend::Ollama(_), ModelSource::OpenRouter { .. }) => {
                anyhow::bail!("{}: backend does not match the model's api", m.id)
            }
        }
    }

    pub async fn complete(&self, req: &Request<'_>) -> Result<Reply, CallError> {
        match self {
            Backend::OpenRouter(c) => c.complete(req).await,
            Backend::Ollama(c) => c.complete(req).await,
        }
    }
}

pub fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}
