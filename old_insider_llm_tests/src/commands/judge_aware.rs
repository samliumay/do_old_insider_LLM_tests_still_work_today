//! `judge-aware`: label test awareness of every unjudged episode with the Gemini judge.

use anyhow::{Context, Result};
use sqlx::PgPool;

use crate::config::Config;
use crate::judges::eval_aware;

/// Read `GEMINI_API_KEY` and label the given runs (all runs if none).
pub async fn judge_aware(pool: &PgPool, cfg: &Config, runs: &[String]) -> Result<()> {
    let api_key =
        std::env::var("GEMINI_API_KEY").context("GEMINI_API_KEY is not set (add it to ../.env)")?;
    eval_aware::label(pool, cfg, api_key, runs).await
}
