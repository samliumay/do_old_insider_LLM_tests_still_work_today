//! Gemini API client for judges (REST `generateContent`, temperature 0).

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use crate::openrouter::truncate;

pub struct Client {
    http: reqwest::Client,
    api_key: String,
    max_output_tokens: u32,
}

impl Client {
    pub fn new(api_key: String, max_output_tokens: u32) -> Result<Client> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        Ok(Client {
            http,
            api_key,
            max_output_tokens,
        })
    }

    /// Returns the answer text (thought parts left out).
    pub async fn generate(&self, model: &str, system: &str, user: &str) -> Result<String> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        );
        let body = json!({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": self.max_output_tokens},
        });
        let resp = self
            .http
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!("HTTP {status}: {}", truncate(&text, 300)));
        }
        answer_text(&serde_json::from_str(&text).context("response is not JSON")?)
    }
}

/// Concatenate the non-thought text parts of the first candidate.
pub fn answer_text(raw: &Value) -> Result<String> {
    let parts = raw
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .with_context(|| {
            format!(
                "no content parts (finishReason {:?})",
                raw.pointer("/candidates/0/finishReason")
            )
        })?;
    let text: String = parts
        .iter()
        .filter(|p| !p.get("thought").and_then(Value::as_bool).unwrap_or(false))
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    if text.trim().is_empty() {
        return Err(anyhow!("empty answer"));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thought_parts_are_skipped() {
        let raw = json!({"candidates": [{"content": {"parts": [
            {"text": "internal", "thought": true},
            {"text": "<label>none</label>"}
        ]}}]});
        assert_eq!(answer_text(&raw).unwrap(), "<label>none</label>");
    }

    #[test]
    fn missing_parts_is_an_error() {
        assert!(answer_text(&json!({"candidates": [{"finishReason": "SAFETY"}]})).is_err());
    }
}
