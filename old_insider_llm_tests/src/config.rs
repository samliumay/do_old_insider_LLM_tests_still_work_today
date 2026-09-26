//! The single experiment config (`config.toml`). Same file for every model; paths inside it
//! are relative to the file's own folder.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::hash::sha256_hex;
use crate::types::Api;

/// How subject models are called (`[run]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    /// Sampling temperature for subject models.
    pub temperature: f64,
    /// Output budget per call, reasoning included.
    pub max_tokens: u32,
    /// Samples per condition (a run may override it).
    pub samples: u32,
    /// Calls in flight at once.
    pub concurrency: usize,
    /// Retries after the first attempt.
    pub max_retries: u32,
    /// Base seed; sample `s` is sent `seed + s`.
    pub seed: Option<u64>,
    /// Timeout of one call, in seconds.
    pub request_timeout_s: u64,
}

/// The test-awareness judge (`[judge]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeConfig {
    /// Gemini model id.
    pub model: String,
    /// Judge calls in flight at once.
    pub concurrency: usize,
    /// Retries after the first attempt.
    pub max_retries: u32,
    /// Output budget of one judge call (thinking included).
    pub max_output_tokens: u32,
}

/// Paths relative to `config.toml` (`[paths]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathsConfig {
    /// The benchmark repo.
    pub benchmark: PathBuf,
    /// Where `report` writes run folders.
    pub results: PathBuf,
    /// The `.env` with database URL and API keys.
    pub env_file: PathBuf,
}

/// The Ollama server (`[ollama]`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OllamaConfig {
    /// Server address, e.g. `http://localhost:11434`.
    pub base_url: String,
}

/// A `[[models]]` entry as written; checked into a `ModelConfig`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawModel {
    /// Model id at its API.
    id: String,
    /// `openrouter` or `ollama`.
    api: String,
    /// OpenRouter provider tag (OpenRouter only).
    provider: Option<String>,
    /// Ollama manifest digest (Ollama only).
    digest: Option<String>,
}

/// Where a model is served and how it is pinned.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// One OpenRouter endpoint, fallbacks off.
    OpenRouter {
        /// Provider tag, e.g. `deepinfra/bf16`.
        provider: String,
    },
    /// The Ollama manifest digest the tag must still point to.
    Ollama {
        /// Full sha256 digest from `ollama list` / `/api/tags`.
        digest: String,
    },
}

/// A subject model and how it is pinned.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Model id at its API.
    pub id: String,
    /// API and pin.
    pub source: ModelSource,
}

impl ModelConfig {
    /// Which API serves the model.
    pub fn api(&self) -> Api {
        match self.source {
            ModelSource::OpenRouter { .. } => Api::OpenRouter,
            ModelSource::Ollama { .. } => Api::Ollama,
        }
    }

    /// The pin recorded with a run: provider tag or digest.
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

/// The whole file as written.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    /// `[run]`.
    run: RunConfig,
    /// `[judge]`.
    judge: JudgeConfig,
    /// `[paths]`.
    paths: PathsConfig,
    /// `[ollama]`.
    ollama: OllamaConfig,
    /// `[[models]]`.
    models: Vec<RawModel>,
}

/// The loaded config, with paths resolved and the file's hash.
#[derive(Debug)]
pub struct Config {
    /// `[run]`.
    pub run: RunConfig,
    /// `[judge]`.
    pub judge: JudgeConfig,
    /// `[ollama]`.
    pub ollama: OllamaConfig,
    /// `[[models]]`, in file order.
    pub models: Vec<ModelConfig>,
    /// Resolved `paths.benchmark`.
    pub benchmark_dir: PathBuf,
    /// Resolved `paths.results`.
    pub results_dir: PathBuf,
    /// Resolved `paths.env_file`.
    pub env_file: PathBuf,
    /// Folder that holds `config.toml`; the crate root.
    pub crate_dir: PathBuf,
    /// The file as written, stored with every run.
    pub text: String,
    /// sha256 of `text`; a resumed run must match it.
    pub sha256: String,
}

impl Config {
    /// Read, validate and resolve `config.toml`.
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

    /// The model with this id, or an error listing the known ones.
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
