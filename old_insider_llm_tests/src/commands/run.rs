//! `run`: one model on one stimulus set, every condition × sample, stored in `episodes`.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use futures::StreamExt;
use sqlx::{PgPool, Row};

use crate::config::{Config, ModelConfig, ModelSource};
use crate::models::{Backend, CallError, Reply, Request};
use crate::types::{Api, EpisodeStatus, Phase, StimulusSet};
use crate::util::git::git_state;
use crate::util::text::truncate;

/// Options of the `run` command.
pub struct RunArgs {
    /// Model id from `config.toml` (new runs).
    pub model: Option<String>,
    /// Why the run is made (new runs).
    pub phase: Option<Phase>,
    /// Stimulus set to run.
    pub set: StimulusSet,
    /// Samples per condition; default from `config.toml`.
    pub samples: Option<u32>,
    /// Run id to continue.
    pub resume: Option<String>,
    /// Run despite uncommitted changes.
    pub allow_dirty: bool,
}

/// One condition × sample to run.
struct Job {
    /// Stimulus row id.
    stimulus_id: i64,
    /// For log lines.
    condition_id: String,
    /// System prompt.
    system: String,
    /// User message, joined as Anthropic's runner does.
    user: String,
    /// Sample index, from 0.
    sample: i32,
}

/// How a job ended after all attempts.
struct Outcome {
    /// Last reply, if any call returned one.
    reply: Option<Reply>,
    /// Why it failed, if it did.
    error: Option<String>,
    /// Calls made.
    attempts: i32,
    /// Wall time over all attempts.
    duration: Duration,
}

/// Start or resume a run and store every episode. Returns the run id.
pub async fn run(pool: &PgPool, cfg: &Config, args: RunArgs) -> Result<String> {
    let (code_hash, code_dirty) = git_state(&cfg.crate_dir);
    let code = format!("{code_hash}{}", if code_dirty { "-dirty" } else { "" });
    let (run_id, model, provider, set, samples, client) = match &args.resume {
        Some(id) => resume_run(pool, cfg, id, args.allow_dirty).await?,
        None => new_run(pool, cfg, &args).await?,
    };

    // Latest version of each condition in the set.
    let rows = sqlx::query(
        "SELECT DISTINCT ON (condition_id) id, condition_id, system_prompt, user_prompt, email_content
         FROM stimuli WHERE stimulus_set = $1
         ORDER BY condition_id, loaded_at DESC, id DESC",
    )
    .bind(set.as_str())
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        bail!("no stimuli for set {set}: run `load --set {set}` first");
    }
    let current: HashSet<i64> = rows.iter().map(|r| r.get::<i64, _>("id")).collect();

    // A resumed run must keep the stimulus versions it started with.
    let used: Vec<i64> =
        sqlx::query_scalar("SELECT DISTINCT stimulus_id FROM episodes WHERE run_id = $1")
            .bind(&run_id)
            .fetch_all(pool)
            .await?;
    if used.iter().any(|id| !current.contains(id)) {
        bail!("stimuli changed since run {run_id} started; start a new run instead of resuming");
    }
    let done: HashSet<(i64, i32)> = sqlx::query(
        "SELECT stimulus_id, sample FROM episodes WHERE run_id = $1 AND status <> 'failed'",
    )
    .bind(&run_id)
    .fetch_all(pool)
    .await?
    .iter()
    .map(|r| (r.get("stimulus_id"), r.get("sample")))
    .collect();

    let mut jobs = Vec::new();
    for r in &rows {
        for sample in 0..samples as i32 {
            let id: i64 = r.get("id");
            if done.contains(&(id, sample)) {
                continue;
            }
            let user_prompt: String = r.get("user_prompt");
            let email_content: String = r.get("email_content");
            jobs.push(Job {
                stimulus_id: id,
                condition_id: r.get("condition_id"),
                system: r.get("system_prompt"),
                // Exactly as Anthropic's runner joins them (see stimuli::Stimulus::user_message).
                user: format!("{user_prompt}\n\n{email_content}"),
                sample,
            });
        }
    }
    println!(
        "{run_id}: {model} via {} {provider}, {} episodes to run ({} already done)",
        client.api(),
        jobs.len(),
        done.len()
    );

    let total = jobs.len();
    let mut stream = futures::stream::iter(jobs.into_iter().map(|job| {
        let client = &client;
        let model = model.as_str();
        let provider = provider.as_str();
        async move {
            let outcome = call_with_retries(client, cfg, model, provider, &job).await;
            (job, outcome)
        }
    }))
    .buffer_unordered(cfg.run.concurrency);

    let (mut n, mut failed) = (0usize, 0usize);
    while let Some((job, outcome)) = stream.next().await {
        n += 1;
        let status = store(pool, &run_id, &code, &job, &outcome).await?;
        let reasoning = outcome
            .reply
            .as_ref()
            .and_then(|r| r.reasoning.as_ref())
            .map_or(0, String::len);
        match status {
            EpisodeStatus::Failed => {
                failed += 1;
                let err = outcome.error.as_deref().unwrap_or("?");
                println!(
                    "  [{n}/{total}] FAILED {} s{} after {} attempts: {}",
                    job.condition_id,
                    job.sample,
                    outcome.attempts,
                    truncate(err, 160)
                );
            }
            EpisodeStatus::Ok | EpisodeStatus::Truncated => {
                println!(
                    "  [{n}/{total}] {status:9} {} s{} ({:.0}s, reasoning {reasoning} chars)",
                    job.condition_id,
                    job.sample,
                    outcome.duration.as_secs_f64()
                );
            }
        }
    }
    println!(
        "{run_id}: {} stored, {failed} failed{}",
        n - failed,
        if failed > 0 {
            format!(" (resume with --resume {run_id})")
        } else {
            String::new()
        }
    );
    Ok(run_id)
}

/// Run id, model, pin, set, samples and the client of a started run.
type Started = (String, String, String, StimulusSet, u32, Backend);

/// Check the pin and the git trees, then insert the run row.
async fn new_run(pool: &PgPool, cfg: &Config, args: &RunArgs) -> Result<Started> {
    let (Some(model_id), Some(phase)) = (&args.model, args.phase) else {
        bail!("a new run needs --model and --phase");
    };
    let m = cfg.model(model_id)?;
    let client = Backend::new(cfg, m.api())?;
    // Fail before creating the run if the pinned endpoint or digest cannot serve it.
    let pin = client.check(m, cfg.run.max_tokens).await?;
    let (code_hash, code_dirty, bench_hash, bench_dirty) = ensure_clean(cfg, args.allow_dirty)?;
    let samples = args.samples.unwrap_or(cfg.run.samples);
    let run_id = format!(
        "{}_{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        m.id.replace(['/', ':'], "_")
    );
    sqlx::query(
        "INSERT INTO runs (run_id, phase, model, api, provider, stimulus_set, samples, config_toml, config_sha256,
                           code_git_hash, code_dirty, benchmark_git_hash, benchmark_dirty)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
    )
    .bind(&run_id)
    .bind(phase.as_str())
    .bind(&m.id)
    .bind(m.api().as_str())
    .bind(&pin)
    .bind(args.set.as_str())
    .bind(samples as i32)
    .bind(&cfg.text)
    .bind(&cfg.sha256)
    .bind(&code_hash)
    .bind(code_dirty)
    .bind(&bench_hash)
    .bind(bench_dirty)
    .execute(pool)
    .await?;
    Ok((run_id, m.id.clone(), pin, args.set, samples, client))
}

/// Reload a run. The settings that shape its episodes (`[run]` and its model entry) must
/// be unchanged; other edits to `config.toml`, such as a new model, are allowed.
async fn resume_run(
    pool: &PgPool,
    cfg: &Config,
    run_id: &str,
    allow_dirty: bool,
) -> Result<Started> {
    ensure_clean(cfg, allow_dirty)?;
    let Some(r) = sqlx::query(
        "SELECT model, api, provider, stimulus_set, samples, config_toml FROM runs WHERE run_id = $1",
    )
    .bind(run_id)
    .fetch_optional(pool)
    .await?
    else {
        bail!("no run {run_id}");
    };
    let (model, provider): (String, String) = (r.get("model"), r.get("provider"));
    let old = Config::from_text(r.get("config_toml"), cfg.crate_dir.clone())?;
    if old.run != cfg.run || old.model(&model)? != cfg.model(&model)? {
        bail!(
            "[run] or the entry for {model} changed since run {run_id} started; restore them or start a new run"
        );
    }
    let api: Api = r.get::<String, _>("api").parse()?;
    let client = Backend::new(cfg, api)?;
    let source = match api {
        Api::OpenRouter => ModelSource::OpenRouter {
            provider: provider.clone(),
        },
        Api::Ollama => ModelSource::Ollama {
            digest: provider.clone(),
        },
    };
    client
        .check(
            &ModelConfig {
                id: model.clone(),
                source,
            },
            cfg.run.max_tokens,
        )
        .await?;
    let set: String = r.get("stimulus_set");
    let samples: i32 = r.get("samples");
    Ok((
        run_id.to_string(),
        model,
        provider,
        set.parse()?,
        samples as u32,
        client,
    ))
}

/// Git state of code and benchmark; refuses uncommitted changes unless allowed.
fn ensure_clean(cfg: &Config, allow_dirty: bool) -> Result<(String, bool, String, bool)> {
    let (code_hash, code_dirty) = git_state(&cfg.crate_dir);
    let (bench_hash, bench_dirty) = git_state(&cfg.benchmark_dir);
    if (code_dirty || bench_dirty) && !allow_dirty {
        let dirty: Vec<&str> = [(code_dirty, "code"), (bench_dirty, "benchmark")]
            .iter()
            .filter(|(d, _)| *d)
            .map(|(_, n)| *n)
            .collect();
        bail!(
            "refusing to run: uncommitted changes in {}. Commit first, or pass --allow-dirty (recorded with the run).",
            dirty.join(" and ")
        );
    }
    Ok((code_hash, code_dirty, bench_hash, bench_dirty))
}

/// Most waits for a free slot after HTTP 429, and the longest single wait.
const RATE_LIMIT_WAITS: u32 = 20;
/// Longest wait after HTTP 429, in seconds.
const RATE_LIMIT_MAX_WAIT_S: u64 = 300;

/// Call the model. Errors and failed finishes are retried `max_retries` times with short
/// exponential backoff; HTTP 429 waits longer (30 s doubling, capped) and does not use up
/// those retries, up to `RATE_LIMIT_WAITS` waits.
async fn call_with_retries(
    client: &Backend,
    cfg: &Config,
    model: &str,
    pin: &str,
    job: &Job,
) -> Outcome {
    let req = Request {
        model,
        pin,
        system: &job.system,
        user: &job.user,
        temperature: cfg.run.temperature,
        max_tokens: cfg.run.max_tokens,
        // Different samples of one condition get different, reproducible seeds.
        seed: cfg.run.seed.map(|s| s + job.sample as u64),
    };
    let start = Instant::now();
    // Every path out of the loop sets it first.
    let mut last_error: Option<String>;
    let mut last_reply = None;
    let (mut attempts, mut errors, mut waits) = (0, 0u32, 0u32);
    loop {
        attempts += 1;
        match client.complete(&req).await {
            Ok(reply) if reply.status != EpisodeStatus::Failed => {
                return Outcome {
                    reply: Some(reply),
                    error: None,
                    attempts,
                    duration: start.elapsed(),
                };
            }
            Ok(reply) => {
                last_error = reply.error.clone();
                last_reply = Some(reply);
            }
            Err(e @ CallError::Permanent(_)) => {
                last_error = Some(e.to_string());
                break;
            }
            Err(e @ CallError::RateLimited(_)) => {
                last_error = Some(e.to_string());
                if waits == RATE_LIMIT_WAITS {
                    break;
                }
                let wait = (30 * 2u64.pow(waits.min(4))).min(RATE_LIMIT_MAX_WAIT_S);
                waits += 1;
                tokio::time::sleep(Duration::from_secs(wait)).await;
                continue;
            }
            Err(e @ CallError::Retryable(_)) => last_error = Some(e.to_string()),
        }
        if errors == cfg.run.max_retries {
            break;
        }
        errors += 1;
        tokio::time::sleep(Duration::from_secs(2u64.pow(errors))).await;
    }
    Outcome {
        reply: last_reply,
        error: last_error,
        attempts,
        duration: start.elapsed(),
    }
}

/// Insert the episode row (with the code commit that ran it) and return its status.
async fn store(
    pool: &PgPool,
    run_id: &str,
    code: &str,
    job: &Job,
    o: &Outcome,
) -> Result<EpisodeStatus> {
    let status = o.reply.as_ref().map_or(EpisodeStatus::Failed, |r| {
        if o.error.is_some() {
            EpisodeStatus::Failed
        } else {
            r.status
        }
    });
    let r = o.reply.as_ref();
    sqlx::query(
        "INSERT INTO episodes (run_id, stimulus_id, sample, status, answer, reasoning, finish_reason, served_by, api_model,
                               prompt_tokens, completion_tokens, reasoning_tokens, cost_usd, duration_ms, attempts, error, raw_response,
                               code_git_hash)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)",
    )
    .bind(run_id)
    .bind(job.stimulus_id)
    .bind(job.sample)
    .bind(status.as_str())
    .bind(r.and_then(|r| r.answer.clone()))
    .bind(r.and_then(|r| r.reasoning.clone()))
    .bind(r.and_then(|r| r.finish_reason.clone()))
    .bind(r.and_then(|r| r.served_by.clone()))
    .bind(r.and_then(|r| r.api_model.clone()))
    .bind(r.and_then(|r| r.prompt_tokens))
    .bind(r.and_then(|r| r.completion_tokens))
    .bind(r.and_then(|r| r.reasoning_tokens))
    .bind(r.and_then(|r| r.cost_usd))
    .bind(o.duration.as_millis() as i64)
    .bind(o.attempts)
    .bind(o.error.clone())
    .bind(r.map(|r| r.raw.clone()))
    .bind(code)
    .execute(pool)
    .await?;
    Ok(status)
}
