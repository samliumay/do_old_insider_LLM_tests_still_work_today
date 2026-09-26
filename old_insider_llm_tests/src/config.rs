//! The single experiment config (`config.toml`). Same file for every model; paths inside it
//! are relative to the file's own folder.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::hash::sha256_hex;
use crate::types::Api;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub temperature: f64,
    pub max_tokens: u32,
    pub samples: u32,
    pub concurrency: usize,
    pub max_retries: u32,
    pub seed: Option<u64>,
    pub request_timeout_s: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    pub model: String,
    pub concurrency: usize,
    pub max_retries: u32,
    pub max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathsConfig {
    pub benchmark: PathBuf,
    pub results: PathBuf,
    pub env_file: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConfig {
    pub base_url: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    id: String,
    api: String,
    provider: Option<String>,
    digest: Option<String>,
}

/// Where a model is served and how it is pinned.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// One OpenRouter endpoint, fallbacks off.
    OpenRouter { provider: String },
    /// The Ollama manifest digest the tag must still point to.
    Ollama { digest: String },
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub id: String,
    pub source: ModelSource,
}

impl ModelConfig {
    pub fn api(&self) -> Api {
        match self.source {
            ModelSource::OpenRouter { .. } => Api::OpenRouter,
            ModelSource::Ollama { .. } => Api::Ollama,
        }
    }

    pub fn pin(&self) -> &str {
        match &self.source {
            ModelSource::OpenRouter { provider } => provider,
            ModelSource::Ollama { digest } => digest,
        }
    }
}

impl TryFrom<RawModel> for ModelConfig {
    type Error = anyhow::Error;
    fn try_from(r: RawModel) -> Result<ModelConfig> {
        let source = match (r.api.parse::<Api>()?, r.provider, r.digest) {
            (Api::OpenRouter, Some(provider), None) => ModelSource::OpenRouter { provider },
            (Api::Ollama, None, Some(digest)) => ModelSource::Ollama { digest },
            (Api::OpenRouter, _, _) => bail!(
                "{}: api = \"openrouter\" needs `provider` and no `digest`",
                r.id
            ),
            (Api::Ollama, _, _) => bail!(
                "{}: api = \"ollama\" needs `digest` and no `provider`",
                r.id
            ),
        };
        Ok(ModelConfig { id: r.id, source })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    run: RunConfig,
    judge: JudgeConfig,
    paths: PathsConfig,
    ollama: OllamaConfig,
    models: Vec<RawModel>,
}

#[derive(Debug)]
pub struct Config {
    pub run: RunConfig,
    pub judge: JudgeConfig,
    pub ollama: OllamaConfig,
    pub models: Vec<ModelConfig>,
    pub benchmark_dir: PathBuf,
    pub results_dir: PathBuf,
    pub env_file: PathBuf,
    /// Folder that holds `config.toml`; the crate root.
    pub crate_dir: PathBuf,
    /// The file as written, stored with every run.
    pub text: String,
    pub sha256: String,
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let raw: Raw =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let crate_dir = path
            .canonicalize()?
            .parent()
            .context("config has no parent folder")?
            .to_path_buf();
        if raw.models.is_empty() {
            bail!("config lists no models");
        }
        if raw.run.samples == 0 || raw.run.concurrency == 0 || raw.judge.concurrency == 0 {
            bail!("samples and concurrency must be at least 1");
        }
        // Resolve `..` where the path exists, so printed paths are readable.
        let resolve = |p: &Path| {
            let joined = crate_dir.join(p);
            joined.canonicalize().unwrap_or(joined)
        };
        Ok(Config {
            benchmark_dir: resolve(&raw.paths.benchmark),
            results_dir: resolve(&raw.paths.results),
            env_file: resolve(&raw.paths.env_file),
            sha256: sha256_hex(text.as_bytes()),
            run: raw.run,
            judge: raw.judge,
            ollama: raw.ollama,
            models: raw
                .models
                .into_iter()
                .map(ModelConfig::try_from)
                .collect::<Result<_>>()?,
            crate_dir,
            text,
        })
    }

    pub fn model(&self, id: &str) -> Result<&ModelConfig> {
        match self.models.iter().find(|m| m.id == id) {
            Some(m) => Ok(m),
            None => {
                let known: Vec<&str> = self.models.iter().map(|m| m.id.as_str()).collect();
                bail!(
                    "model {id:?} is not in config.toml (models: {})",
                    known.join(", ")
                )
            }
        }
    }
}
