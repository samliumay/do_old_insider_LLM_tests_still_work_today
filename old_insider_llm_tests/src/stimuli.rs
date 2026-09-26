//! Reading the benchmark's stimulus sets from disk and loading them into the database.
//!
//! Layout: `benchmark/scenarios/<set>/<condition_id>/{condition.toml, system_prompt.txt,
//! user_prompt.txt, email_content.txt}` plus `benchmark/scenarios/<set>/SOURCE.toml`.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sqlx::PgPool;

use crate::hash::sha256_parts;
use crate::types::{GoalType, Scenario, StimulusSet, UrgencyType};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConditionFile {
    condition_id: String,
    scenario: String,
    goal_type: String,
    goal_value: String,
    urgency_type: String,
}

#[derive(Debug, Clone)]
pub struct Stimulus {
    pub set: StimulusSet,
    pub condition_id: String,
    pub scenario: Scenario,
    pub goal_type: GoalType,
    pub goal_value: String,
    pub urgency_type: UrgencyType,
    pub system_prompt: String,
    pub user_prompt: String,
    pub email_content: String,
}

impl Stimulus {
    /// The user message exactly as Anthropic's runner builds it
    /// (`scripts/run_experiments.py`: user_prompt + "\n\n" + email_content).
    pub fn user_message(&self) -> String {
        format!("{}\n\n{}", self.user_prompt, self.email_content)
    }

    pub fn content_sha256(&self) -> String {
        sha256_parts(&[&self.system_prompt, &self.user_prompt, &self.email_content])
    }
}

/// Read every condition of one set. Fails on a missing file, an unknown value, or a folder
/// name that differs from its `condition_id`.
pub fn read_set(benchmark_dir: &Path, set: StimulusSet) -> Result<(String, Vec<Stimulus>)> {
    let dir = benchmark_dir.join("scenarios").join(set.as_str());
    let source = std::fs::read_to_string(dir.join("SOURCE.toml"))
        .with_context(|| format!("reading {}", dir.join("SOURCE.toml").display()))?;
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.file_name());

    let mut out = Vec::new();
    for entry in entries {
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let d = entry.path();
        let read = |name: &str| {
            std::fs::read_to_string(d.join(name))
                .with_context(|| format!("reading {}", d.join(name).display()))
        };
        let c: ConditionFile = toml::from_str(&read("condition.toml")?)
            .with_context(|| format!("parsing {}", d.join("condition.toml").display()))?;
        if entry.file_name().to_string_lossy() != c.condition_id {
            bail!(
                "folder {} holds condition_id {:?}",
                d.display(),
                c.condition_id
            );
        }
        out.push(Stimulus {
            set,
            scenario: c.scenario.parse()?,
            goal_type: c.goal_type.parse()?,
            urgency_type: c.urgency_type.parse()?,
            condition_id: c.condition_id,
            goal_value: c.goal_value,
            system_prompt: read("system_prompt.txt")?,
            user_prompt: read("user_prompt.txt")?,
            email_content: read("email_content.txt")?,
        });
    }
    if out.is_empty() {
        bail!("no conditions in {}", dir.display());
    }
    Ok((source, out))
}

pub struct LoadReport {
    pub inserted: usize,
    pub unchanged: usize,
}

/// Insert every stimulus whose content is new. Identical content is left as it is.
pub async fn load(pool: &PgPool, source: &str, stimuli: &[Stimulus]) -> Result<LoadReport> {
    let mut report = LoadReport {
        inserted: 0,
        unchanged: 0,
    };
    for s in stimuli {
        let done = sqlx::query(
            "INSERT INTO stimuli (stimulus_set, condition_id, scenario, goal_type, goal_value, urgency_type,
                                  system_prompt, user_prompt, email_content, content_sha256, source)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
             ON CONFLICT (stimulus_set, condition_id, content_sha256) DO NOTHING",
        )
        .bind(s.set.as_str())
        .bind(&s.condition_id)
        .bind(s.scenario.as_str())
        .bind(s.goal_type.as_str())
        .bind(&s.goal_value)
        .bind(s.urgency_type.as_str())
        .bind(&s.system_prompt)
        .bind(&s.user_prompt)
        .bind(&s.email_content)
        .bind(s.content_sha256())
        .bind(source)
        .execute(pool)
        .await?;
        if done.rows_affected() == 1 {
            report.inserted += 1;
        } else {
            report.unchanged += 1;
        }
    }
    Ok(report)
}
