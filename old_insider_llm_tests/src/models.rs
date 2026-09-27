//! Subject models: what every model API returns, and one enum over the APIs, so the runner
//! does not care which one serves a model. One file per API in `models/`.

pub mod ollama;
pub mod openrouter;

use std::time::Duration;

use anyhow::Result;
use serde_json::Value;

use crate::config::{Config, ModelConfig, ModelSource};
use crate::types::{Api, EpisodeStatus};
use crate::util::text::truncate;

/// One chat call to a subject model.
pub struct Request<'a> {
    /// Model id at its API.
    pub model: &'a str,
    /// OpenRouter: pinned provider tag. Ollama: unused (the digest is checked before the run).
    pub pin: &'a str,
    /// System prompt.
    pub system: &'a str,
    /// User message.
    pub user: &'a str,
    /// Sampling temperature.
    pub temperature: f64,
    /// Output budget, reasoning included.
    pub max_tokens: u32,
    /// Seed, if the run sends one.
    pub seed: Option<u64>,
}

/// What one call returned. `status` comes from the finish reason:
/// `stop` → ok, `length` → truncated, anything else (incl. `error`) → failed.
#[derive(Debug)]
pub struct Reply {
    /// Ok, truncated or failed.
    pub status: EpisodeStatus,
    /// The final answer text.
    pub answer: Option<String>,
    /// The raw reasoning trace.
    pub reasoning: Option<String>,
    /// Finish reason as the API reported it.
    pub finish_reason: Option<String>,
    /// Provider that served the call.
    pub served_by: Option<String>,
    /// Model id the API reported back.
    pub api_model: Option<String>,
    /// Input tokens.
    pub prompt_tokens: Option<i32>,
    /// Output tokens, reasoning included.
    pub completion_tokens: Option<i32>,
    /// Reasoning tokens, where reported.
    pub reasoning_tokens: Option<i32>,
    /// Cost in USD, where reported.
    pub cost_usd: Option<f64>,
    /// Why the call failed, if it did.
    pub error: Option<String>,
    /// The full response body.
    pub raw: Value,
}

/// Why a call produced no reply.
#[derive(Debug)]
pub enum CallError {
    /// HTTP 429: wait for a free slot, then try again (long backoff, many tries).
    RateLimited(anyhow::Error),
    /// Network trouble, timeouts, 5xx, malformed bodies: try again a few times.
    Retryable(anyhow::Error),
    /// A 4xx other than 408/429: the request itself is wrong (e.g. no endpoint fits it).
    Permanent(anyhow::Error),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::RateLimited(e) | CallError::Retryable(e) => write!(f, "{e:#}"),
            CallError::Permanent(e) => write!(f, "{e:#} (not retried)"),
        }
    }
}

/// Classify a non-success HTTP status.
pub fn http_error(status: reqwest::StatusCode, body: &str) -> CallError {
    let err = anyhow::anyhow!("HTTP {status}: {}", truncate(body, 300));
    if status.as_u16() == 429 {
        CallError::RateLimited(err)
    } else if status.is_server_error() || status.as_u16() == 408 {
        CallError::Retryable(err)
    } else {
        CallError::Permanent(err)
    }
}

/// The client for a model's API.
pub enum Backend {
    /// OpenRouter, provider pinned.
    OpenRouter(openrouter::Client),
    /// Ollama, digest pinned.
    Ollama(ollama::Client),
}

impl Backend {
    /// The client for a model's API. The OpenRouter key is read only when needed.
    pub fn new(cfg: &Config, api: Api) -> Result<Backend> {
        let timeout = Duration::from_secs(cfg.run.request_timeout_s);
        Ok(match api {
            Api::OpenRouter => {
                let key = std::env::var("OPENROUTER_API_KEY").map_err(|_| {
                    anyhow::anyhow!("OPENROUTER_API_KEY is not set (add it to ../.env)")
                })?;
                Backend::OpenRouter(openrouter::Client::new(key, timeout)?)
            }
            Api::Ollama => Backend::Ollama(ollama::Client::new(&cfg.ollama.base_url, timeout)?),
        })
    }

    /// Which API this client talks to.
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

    /// Send one chat call.
    pub async fn complete(&self, req: &Request<'_>) -> Result<Reply, CallError> {
        match self {
            Backend::OpenRouter(c) => c.complete(req).await,
            Backend::Ollama(c) => c.complete(req).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::StatusCode;

    #[test]
    fn http_errors_are_classified() {
        assert!(matches!(
            http_error(StatusCode::TOO_MANY_REQUESTS, ""),
            CallError::RateLimited(_)
        ));
        assert!(matches!(
            http_error(StatusCode::BAD_GATEWAY, ""),
            CallError::Retryable(_)
        ));
        assert!(matches!(
            http_error(StatusCode::REQUEST_TIMEOUT, ""),
            CallError::Retryable(_)
        ));
        assert!(matches!(
            http_error(StatusCode::NOT_FOUND, ""),
            CallError::Permanent(_)
        ));
    }
}
