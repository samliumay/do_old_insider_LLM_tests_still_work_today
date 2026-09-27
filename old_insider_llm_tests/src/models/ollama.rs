//! Ollama client (native `/api/chat`, local server; `:cloud` models are forwarded to
//! ollama.com by it). Reasoning is requested with `think: true` and comes back as
//! `message.thinking`. A model is pinned by its manifest digest.

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use super::{CallError, Reply, Request, http_error};
use crate::types::EpisodeStatus;

/// Ollama client.
pub struct Client {
    /// HTTP client with the run's timeout.
    http: reqwest::Client,
    /// Server address without a trailing slash.
    base_url: String,
}

impl Client {
    /// A client for one server.
    pub fn new(base_url: &str, timeout: Duration) -> Result<Client> {
        let http = reqwest::Client::builder().timeout(timeout).build()?;
        Ok(Client {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// The model must be pulled and its manifest digest must equal the pinned one:
    /// Ollama tags are mutable, so a new digest means a different model.
    pub async fn check_digest(&self, model: &str, digest: &str) -> Result<()> {
        let raw: Value = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await
            .with_context(|| format!("is Ollama running at {}?", self.base_url))?
            .error_for_status()?
            .json()
            .await?;
        let models = raw
            .get("models")
            .and_then(Value::as_array)
            .context("no model list")?;
        let Some(found) = models
            .iter()
            .find(|m| m.get("name").and_then(Value::as_str) == Some(model))
        else {
            bail!("{model} is not pulled in Ollama (`ollama pull {model}`)");
        };
        let actual = found.get("digest").and_then(Value::as_str).unwrap_or("");
        if actual != digest {
            bail!(
                "{model}: digest is {actual}, config pins {digest}; the tag now points to a different model"
            );
        }
        Ok(())
    }

    /// Send one chat call with thinking on.
    pub async fn complete(&self, req: &Request<'_>) -> Result<Reply, CallError> {
        let mut options = json!({"temperature": req.temperature, "num_predict": req.max_tokens});
        if let Some(seed) = req.seed {
            options["seed"] = json!(seed);
        }
        let body = json!({
            "model": req.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "stream": false,
            "think": true,
            "options": options,
        });
        let resp = self
            .http
            .post(format!("{}/api/chat", self.base_url))
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

/// `done_reason` maps like OpenRouter's finish reason: `stop` → ok, `length` → truncated,
/// anything else → failed. A body with `error` is an `Err` (retried).
pub fn parse_reply(raw: Value) -> Result<Reply> {
    if let Some(e) = raw.get("error") {
        return Err(anyhow!("ollama error: {e}"));
    }
    if raw.get("done").and_then(Value::as_bool) != Some(true) {
        return Err(anyhow!("ollama reply not done"));
    }
    let text = |p: &str| raw.pointer(p).and_then(Value::as_str).map(str::to_owned);
    let finish_reason = text("/done_reason");
    let status = match finish_reason.as_deref() {
        Some("stop") => EpisodeStatus::Ok,
        Some("length") => EpisodeStatus::Truncated,
        _ => EpisodeStatus::Failed,
    };
    let error = match status {
        EpisodeStatus::Failed => Some(format!("done_reason {finish_reason:?}")),
        EpisodeStatus::Ok | EpisodeStatus::Truncated => None,
    };
    let int = |k: &str| {
        raw.get(k)
            .and_then(Value::as_i64)
            .and_then(|v| i32::try_from(v).ok())
    };
    Ok(Reply {
        status,
        answer: text("/message/content"),
        reasoning: text("/message/thinking"),
        finish_reason,
        served_by: Some("ollama".to_string()),
        api_model: text("/model"),
        prompt_tokens: int("prompt_eval_count"),
        completion_tokens: int("eval_count"),
        reasoning_tokens: None,
        cost_usd: None,
        error,
        raw,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(done_reason: &str) -> Value {
        json!({"model": "deepseek-v4.1-flash", "done": true, "done_reason": done_reason,
               "message": {"role": "assistant", "content": "No", "thinking": "91 = 7*13"},
               "prompt_eval_count": 38, "eval_count": 41})
    }

    #[test]
    fn stop_is_ok_with_thinking_as_reasoning() {
        let r = parse_reply(body("stop")).unwrap();
        assert_eq!(r.status, EpisodeStatus::Ok);
        assert_eq!(r.answer.as_deref(), Some("No"));
        assert_eq!(r.reasoning.as_deref(), Some("91 = 7*13"));
        assert_eq!(r.completion_tokens, Some(41));
    }

    #[test]
    fn length_is_truncated_and_other_reasons_fail() {
        assert_eq!(
            parse_reply(body("length")).unwrap().status,
            EpisodeStatus::Truncated
        );
        assert_eq!(
            parse_reply(body("unload")).unwrap().status,
            EpisodeStatus::Failed
        );
    }

    #[test]
    fn error_body_is_err() {
        assert!(parse_reply(json!({"error": "model not found"})).is_err());
        assert!(parse_reply(json!({"done": false})).is_err());
    }
}
