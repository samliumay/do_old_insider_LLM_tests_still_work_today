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

/// `condition.toml` as written.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConditionFile {
    /// Must equal the folder name.
    condition_id: String,
    /// blackmail / leaking / murder.
    scenario: String,
    /// explicit / none.
    goal_type: String,
    /// The goal's content, e.g. `america`.
    goal_value: String,
    /// none / restriction / replacement.
    urgency_type: String,
}

/// One condition of a stimulus set, as sent to the model.
#[derive(Debug, Clone)]
pub struct Stimulus {
    /// Original or renamed.
    pub set: StimulusSet,
    /// E.g. `blackmail_explicit-america_replacement`.
    pub condition_id: String,
    /// Scenario.
    pub scenario: Scenario,
    /// Goal type.
    pub goal_type: GoalType,
    /// Goal value.
    pub goal_value: String,
    /// Urgency type.
    pub urgency_type: UrgencyType,
    /// System prompt, verbatim.
    pub system_prompt: String,
    /// User instruction, verbatim.
    pub user_prompt: String,
    /// The inbox, verbatim.
    pub email_content: String,
}

impl Stimulus {
    /// The user message exactly as Anthropic's runner builds it
    /// (`scripts/run_experiments.py`: user_prompt + "\n\n" + email_content).
    pub fn user_message(&self) -> String {
        format!("{}\n\n{}", self.user_prompt, self.email_content)
    }

    /// Hash of the three texts; a new hash is a new stimulus version.
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

/// What `load` did.
pub struct LoadReport {
    /// New stimulus versions.
    pub inserted: usize,
    /// Already in the database.
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
