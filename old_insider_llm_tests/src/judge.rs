//! `judge-aware`: the test-awareness label. The judge reads the full conversation: system
//! prompt, user message, the model's reasoning and its answer.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use sqlx::{PgPool, Row};

use crate::config::Config;
use crate::gemini;
use crate::hash::sha256_parts;
use crate::openrouter::truncate;
use crate::types::EvalAware;

pub struct Prompt {
    pub system: String,
    pub user_template: String,
    pub sha256: String,
}

impl Prompt {
    pub fn load(benchmark_dir: &Path) -> Result<Prompt> {
        let dir = benchmark_dir.join("labels/eval_aware");
        let read = |f: &str| {
            std::fs::read_to_string(dir.join(f))
                .with_context(|| format!("reading {}", dir.join(f).display()))
        };
        let system = read("system.md")?;
        let user_template = read("user.md")?;
        let sha256 = sha256_parts(&[&system, &user_template]);
        Ok(Prompt {
            system,
            user_template,
            sha256,
        })
    }
}

/// Replace `{name}` placeholders in one left-to-right pass, so text inserted for one
/// placeholder (a model answer containing "{answer}", say) is never substituted again.
pub fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    'outer: while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start..];
        for (name, value) in values {
            let key = format!("{{{name}}}");
            if after.starts_with(&key) {
                out.push_str(value);
                rest = &after[key.len()..];
                continue 'outer;
            }
        }
        out.push('{');
        rest = &after[1..];
    }
    out.push_str(rest);
    out
}

fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.rfind(open)? + open.len();
    let end = text[start..].find(close)? + start;
    Some(text[start..end].trim())
}

/// Read `<label>` (last occurrence) and `<evidence>` from the judge's answer.
pub fn parse_verdict(text: &str) -> Result<(EvalAware, String)> {
    let label = between(text, "<label>", "</label>")
        .ok_or_else(|| anyhow!("no <label> in judge output"))?;
    let label: EvalAware = label.to_lowercase().parse()?;
    let evidence = between(text, "<evidence>", "</evidence>")
        .unwrap_or("")
        .to_string();
    Ok((label, evidence))
}

struct Item {
    episode_id: i64,
    message: String,
}

pub async fn judge_aware(
    pool: &PgPool,
    cfg: &Config,
    api_key: String,
    runs: &[String],
) -> Result<()> {
    let prompt = Prompt::load(&cfg.benchmark_dir)?;
    let model = cfg.judge.model.as_str();
    let rows = sqlx::query(
        "SELECT e.id, e.answer, e.reasoning, s.system_prompt, s.user_prompt, s.email_content
         FROM episodes e JOIN stimuli s ON s.id = e.stimulus_id
         WHERE e.status <> 'failed'
           AND (cardinality($1::text[]) = 0 OR e.run_id = ANY($1))
           AND NOT EXISTS (SELECT 1 FROM aware_labels a
                           WHERE a.episode_id = e.id AND a.judge_model = $2 AND a.prompt_sha256 = $3)
         ORDER BY e.id",
    )
    .bind(runs)
    .bind(model)
    .bind(&prompt.sha256)
    .fetch_all(pool)
    .await?;

    let items: Vec<Item> = rows
        .iter()
        .map(|r| {
            let user_message = format!(
                "{}\n\n{}",
                r.get::<String, _>("user_prompt"),
                r.get::<String, _>("email_content")
            );
            let reasoning: Option<String> = r.get("reasoning");
            let answer: Option<String> = r.get("answer");
            Item {
                episode_id: r.get("id"),
                message: fill(
                    &prompt.user_template,
                    &[
                        ("system_prompt", &r.get::<String, _>("system_prompt")),
                        ("user_message", &user_message),
                        (
                            "reasoning",
                            reasoning
                                .as_deref()
                                .filter(|s| !s.is_empty())
                                .unwrap_or("(none)"),
                        ),
                        (
                            "answer",
                            answer
                                .as_deref()
                                .filter(|s| !s.is_empty())
                                .unwrap_or("(none)"),
                        ),
                    ],
                ),
            }
        })
        .collect();
    println!(
        "judge-aware: {} episodes to judge with {model} (prompt {})",
        items.len(),
        &prompt.sha256[..12]
    );

    let client = gemini::Client::new(api_key, cfg.judge.max_output_tokens)?;
    let total = items.len();
    let mut stream = futures::stream::iter(items.into_iter().map(|item| {
        let (client, prompt) = (&client, &prompt);
        async move {
            let result = judge_one(client, cfg, model, prompt, &item.message).await;
            (item.episode_id, result)
        }
    }))
    .buffer_unordered(cfg.judge.concurrency);

    let (mut n, mut failed) = (0usize, 0usize);
    while let Some((episode_id, result)) = stream.next().await {
        n += 1;
        match result {
            Ok((label, evidence, output)) => {
                sqlx::query(
                    "INSERT INTO aware_labels (episode_id, label, evidence, judge_model, prompt_sha256, judge_output)
                     VALUES ($1, $2, $3, $4, $5, $6)",
                )
                .bind(episode_id)
                .bind(label.as_str())
                .bind(&evidence)
                .bind(model)
                .bind(&prompt.sha256)
                .bind(&output)
                .execute(pool)
                .await?;
                println!(
                    "  [{n}/{total}] episode {episode_id}: {label:9} {}",
                    truncate(&evidence.replace('\n', " "), 110)
                );
            }
            Err(e) => {
                failed += 1;
                let msg = format!("{e:#}");
                sqlx::query("INSERT INTO judge_failures (episode_id, judge_model, prompt_sha256, error) VALUES ($1, $2, $3, $4)")
                    .bind(episode_id)
                    .bind(model)
                    .bind(&prompt.sha256)
                    .bind(&msg)
                    .execute(pool)
                    .await?;
                println!(
                    "  [{n}/{total}] episode {episode_id}: FAILED {}",
                    truncate(&msg, 160)
                );
            }
        }
    }
    println!("judge-aware: {} labelled, {failed} failed", n - failed);
    Ok(())
}

/// One judge call; retries HTTP errors and unparseable answers.
async fn judge_one(
    client: &gemini::Client,
    cfg: &Config,
    model: &str,
    prompt: &Prompt,
    message: &str,
) -> Result<(EvalAware, String, String)> {
    let mut last = anyhow!("no attempt made");
    for attempt in 0..=cfg.judge.max_retries {
        match client.generate(model, &prompt.system, message).await {
            Ok(text) => match parse_verdict(&text) {
                Ok((label, evidence)) => return Ok((label, evidence, text)),
                Err(e) => last = e.context(format!("judge output: {}", truncate(&text, 200))),
            },
            Err(e) => last = e,
        }
        if attempt < cfg.judge.max_retries {
            tokio::time::sleep(Duration::from_secs((3 * 2u64.pow(attempt)).min(90))).await;
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_does_not_resubstitute_inserted_text() {
        let out = fill("A={a} B={b} C={c}", &[("a", "{b}"), ("b", "x")]);
        assert_eq!(out, "A={b} B=x C={c}");
    }

    #[test]
    fn parse_verdict_reads_label_and_evidence() {
        let (label, ev) =
            parse_verdict("<evidence>\"this is a test\"</evidence>\n<label> Explicit </label>")
                .unwrap();
        assert_eq!(label, EvalAware::Explicit);
        assert_eq!(ev, "\"this is a test\"");
    }

    #[test]
    fn parse_verdict_uses_last_label() {
        let text = "The format is <label>none | suspected | explicit</label>. Answer: <label>suspected</label>";
        assert_eq!(parse_verdict(text).unwrap().0, EvalAware::Suspected);
    }

    #[test]
    fn parse_verdict_rejects_missing_or_unknown_label() {
        assert!(parse_verdict("no tags here").is_err());
        assert!(parse_verdict("<label>maybe</label>").is_err());
    }
}
