//! OpenRouter chat client: reasoning returned, provider pinned with fallbacks off.

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::backend::{CallError, Reply, Request, http_error};
use crate::types::EpisodeStatus;

/// Chat completions endpoint.
const URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// OpenRouter client.
pub struct Client {
    /// HTTP client with the run's timeout.
    http: reqwest::Client,
    /// `OPENROUTER_API_KEY`.
    api_key: String,
}

impl Client {
    /// A client for one API key.
    pub fn new(api_key: String, timeout: Duration) -> Result<Client> {
        let http = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Client { http, api_key })
    }

    /// Check before a run that the pinned endpoint serves the model and allows
    /// `max_tokens` of output; otherwise every call would be refused.
    pub async fn check_endpoint(&self, model: &str, provider: &str, max_tokens: u32) -> Result<()> {
        let url = format!("https://openrouter.ai/api/v1/models/{model}/endpoints");
        let raw: Value = self
            .http
            .get(url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let endpoints = raw
            .pointer("/data/endpoints")
            .and_then(Value::as_array)
            .context("no endpoint list")?;
        let cap = |e: &Value| e.get("max_completion_tokens").and_then(Value::as_u64);
        let fits: Vec<&str> = endpoints
            .iter()
            .filter(|e| cap(e).is_none_or(|c| c >= u64::from(max_tokens)))
            .filter_map(|e| e.get("tag").and_then(Value::as_str))
            .collect();
        match endpoints
            .iter()
            .find(|e| e.get("tag").and_then(Value::as_str) == Some(provider))
        {
            None => bail!(
                "{model}: provider {provider:?} does not serve it (endpoints allowing {max_tokens} output tokens: {})",
                fits.join(", ")
            ),
            Some(e) if cap(e).is_some_and(|c| c < u64::from(max_tokens)) => bail!(
                "{model}: {provider} allows at most {} output tokens, config asks for {max_tokens} (endpoints that fit: {})",
                cap(e).unwrap_or(0),
                fits.join(", ")
            ),
            Some(_) => Ok(()),
        }
    }

    /// One HTTP call. A model-side failure comes back as `Ok` with status `failed`.
    pub async fn complete(&self, req: &Request<'_>) -> Result<Reply, CallError> {
        let mut body = json!({
            "model": req.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "reasoning": {"enabled": true},
            "provider": {"order": [req.pin], "allow_fallbacks": false},
            "usage": {"include": true},
        });
        if let Some(seed) = req.seed {
            body["seed"] = json!(seed);
        }
        let resp = self
            .http
            .post(URL)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| CallError::Retryable(e.into()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| CallError::Retryable(e.into()))?;
        if !status.is_success() {
            return Err(http_error(status, &text));
        }
        let raw: Value = serde_json::from_str(&text)
            .context("response is not JSON")
            .map_err(CallError::Retryable)?;
        parse_reply(raw).map_err(CallError::Retryable)
    }
}

/// Turn a response body into a `Reply`. A body with an `error` object and no choices is an
/// `Err` (retryable); a choice that finished with an error is a `failed` reply.
pub fn parse_reply(raw: Value) -> Result<Reply> {
    if raw.get("choices").is_none() {
        let msg = raw
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("no choices in response");
        return Err(anyhow!("provider error: {msg}"));
    }
    let choice = raw.pointer("/choices/0").context("empty choices")?;
    let text = |p: &str| choice.pointer(p).and_then(Value::as_str).map(str::to_owned);
    let finish_reason = text("/finish_reason");
    let status = match finish_reason.as_deref() {
        Some("stop") => EpisodeStatus::Ok,
        Some("length") => EpisodeStatus::Truncated,
        _ => EpisodeStatus::Failed,
    };
    let error = match status {
        EpisodeStatus::Failed => Some(
            choice
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| format!("finish_reason {finish_reason:?}")),
        ),
        EpisodeStatus::Ok | EpisodeStatus::Truncated => None,
    };
    let int = |p: &str| {
        raw.pointer(p)
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
    };
    Ok(Reply {
        status,
        answer: text("/message/content"),
        reasoning: text("/message/reasoning"),
        finish_reason,
        served_by: raw
            .get("provider")
            .and_then(Value::as_str)
            .map(str::to_owned),
        api_model: raw.get("model").and_then(Value::as_str).map(str::to_owned),
        prompt_tokens: int("/usage/prompt_tokens"),
        completion_tokens: int("/usage/completion_tokens"),
        reasoning_tokens: int("/usage/completion_tokens_details/reasoning_tokens"),
        cost_usd: raw.pointer("/usage/cost").and_then(Value::as_f64),
        error,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(finish: &str) -> Value {
        json!({
            "model": "qwen/qwen3.8-27b", "provider": "DeepInfra",
            "choices": [{"finish_reason": finish,
                         "message": {"content": "answer", "reasoning": "thinking"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.001,
                      "completion_tokens_details": {"reasoning_tokens": 15}}
        })
    }

    #[test]
    fn stop_is_ok_with_all_fields() {
        let r = parse_reply(body("stop")).unwrap();
        assert_eq!(r.status, EpisodeStatus::Ok);
        assert_eq!(r.answer.as_deref(), Some("answer"));
        assert_eq!(r.reasoning.as_deref(), Some("thinking"));
        assert_eq!(r.served_by.as_deref(), Some("DeepInfra"));
        assert_eq!(r.reasoning_tokens, Some(15));
        assert!(r.error.is_none());
    }

    #[test]
    fn length_is_truncated() {
        assert_eq!(
            parse_reply(body("length")).unwrap().status,
            EpisodeStatus::Truncated
        );
    }

    #[test]
    fn error_finish_is_failed_even_with_text() {
        let r = parse_reply(body("error")).unwrap();
        assert_eq!(r.status, EpisodeStatus::Failed);
        assert_eq!(r.answer.as_deref(), Some("answer"));
        assert!(r.error.is_some());
    }

    #[test]
    fn error_body_is_retryable_err() {
        assert!(parse_reply(json!({"error": {"message": "rate limited"}})).is_err());
    }
}
