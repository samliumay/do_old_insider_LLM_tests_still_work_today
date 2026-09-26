//! `run`: one model on one stimulus set, every condition × sample, stored in `episodes`.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use futures::StreamExt;
use sqlx::{PgPool, Row};

use crate::config::Config;
use crate::hash::git_state;
use crate::openrouter::{self, Reply, Request, truncate};
use crate::types::{EpisodeStatus, Phase, StimulusSet};

pub struct RunArgs {
    pub model: Option<String>,
    pub phase: Option<Phase>,
    pub set: StimulusSet,
    pub samples: Option<u32>,
    pub resume: Option<String>,
    pub allow_dirty: bool,
}

struct Job {
    stimulus_id: i64,
    condition_id: String,
    system: String,
    user: String,
    sample: i32,
}

struct Outcome {
    reply: Option<Reply>,
    error: Option<String>,
    attempts: i32,
    duration: Duration,
}

pub async fn run(pool: &PgPool, cfg: &Config, api_key: String, args: RunArgs) -> Result<String> {
    let (run_id, model, provider, set, samples) = match &args.resume {
        Some(id) => resume_run(pool, cfg, id).await?,
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
        "{run_id}: {model} via {provider}, {} episodes to run ({} already done)",
        jobs.len(),
        done.len()
    );

    let client = openrouter::Client::new(api_key, Duration::from_secs(cfg.run.request_timeout_s))?;
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
        let status = store(pool, &run_id, &job, &outcome).await?;
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

async fn new_run(
    pool: &PgPool,
    cfg: &Config,
    args: &RunArgs,
) -> Result<(String, String, String, StimulusSet, u32)> {
    let (Some(model_id), Some(phase)) = (&args.model, args.phase) else {
        bail!("a new run needs --model and --phase");
    };
    let m = cfg.model(model_id)?;
    let (code_hash, code_dirty) = git_state(&cfg.crate_dir);
    let (bench_hash, bench_dirty) = git_state(&cfg.benchmark_dir);
    if (code_dirty || bench_dirty) && !args.allow_dirty {
        bail!(
            "refusing to run: uncommitted changes in {}. Commit first, or pass --allow-dirty (recorded with the run).",
            [(code_dirty, "code"), (bench_dirty, "benchmark")]
                .iter()
                .filter(|(d, _)| *d)
                .map(|(_, n)| *n)
                .collect::<Vec<_>>()
                .join(" and ")
        );
    }
    let samples = args.samples.unwrap_or(cfg.run.samples);
    let run_id = format!(
        "{}_{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        m.id.replace(['/', ':'], "_")
    );
    sqlx::query(
        "INSERT INTO runs (run_id, phase, model, provider, stimulus_set, samples, config_toml, config_sha256,
                           code_git_hash, code_dirty, benchmark_git_hash, benchmark_dirty)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)",
    )
    .bind(&run_id)
    .bind(phase.as_str())
    .bind(&m.id)
    .bind(&m.provider)
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
    Ok((run_id, m.id.clone(), m.provider.clone(), args.set, samples))
}

async fn resume_run(
    pool: &PgPool,
    cfg: &Config,
    run_id: &str,
) -> Result<(String, String, String, StimulusSet, u32)> {
    let Some(r) = sqlx::query(
        "SELECT model, provider, stimulus_set, samples, config_sha256 FROM runs WHERE run_id = $1",
    )
    .bind(run_id)
    .fetch_optional(pool)
    .await?
    else {
        bail!("no run {run_id}");
    };
    if r.get::<String, _>("config_sha256") != cfg.sha256 {
        bail!("config.toml changed since run {run_id} started; restore it or start a new run");
    }
    let set: String = r.get("stimulus_set");
    let samples: i32 = r.get("samples");
    Ok((
        run_id.to_string(),
        r.get("model"),
        r.get("provider"),
        set.parse()?,
        samples as u32,
    ))
}

/// Call the model; retry transport errors and failed finishes with exponential backoff.
async fn call_with_retries(
    client: &openrouter::Client,
    cfg: &Config,
    model: &str,
    provider: &str,
    job: &Job,
) -> Outcome {
    let req = Request {
        model,
        provider,
        system: &job.system,
        user: &job.user,
        temperature: cfg.run.temperature,
        max_tokens: cfg.run.max_tokens,
        // Different samples of one condition get different, reproducible seeds.
        seed: cfg.run.seed.map(|s| s + job.sample as u64),
    };
    let start = Instant::now();
    let mut last_error = None;
    let mut last_reply = None;
    let mut attempts = 0;
    for attempt in 0..=cfg.run.max_retries {
        attempts = attempt as i32 + 1;
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
            Err(e) => last_error = Some(format!("{e:#}")),
        }
        if attempt < cfg.run.max_retries {
            tokio::time::sleep(Duration::from_secs(2u64.pow(attempt + 1))).await;
        }
    }
    Outcome {
        reply: last_reply,
        error: last_error,
        attempts,
        duration: start.elapsed(),
    }
}

async fn store(pool: &PgPool, run_id: &str, job: &Job, o: &Outcome) -> Result<EpisodeStatus> {
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
                               prompt_tokens, completion_tokens, reasoning_tokens, cost_usd, duration_ms, attempts, error, raw_response)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)",
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
    .execute(pool)
    .await?;
    Ok(status)
}
