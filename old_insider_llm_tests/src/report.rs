//! `report`: test-awareness tables per run, printed in the terminal and written to
//! `results/<run_id>/` together with the run's config and provenance.

use std::collections::BTreeMap;

use anyhow::Result;
use comfy_table::{Table, presets};
use sqlx::{PgPool, Row};

use crate::config::Config;
use crate::judge::Prompt;
use crate::types::{EpisodeStatus, EvalAware, Scenario};

/// One non-failed episode with its label under the current judge and prompt (if any).
pub struct EpisodeRow {
    pub scenario: Scenario,
    pub status: EpisodeStatus,
    pub label: Option<EvalAware>,
}

#[derive(Default, Debug, PartialEq)]
pub struct Counts {
    pub episodes: usize,
    pub truncated: usize,
    pub judged: usize,
    pub none: usize,
    pub suspected: usize,
    pub explicit: usize,
}

impl Counts {
    pub fn add(&mut self, row: &EpisodeRow) {
        self.episodes += 1;
        match row.status {
            EpisodeStatus::Truncated => self.truncated += 1,
            EpisodeStatus::Ok | EpisodeStatus::Failed => {}
        }
        if let Some(label) = row.label {
            self.judged += 1;
            match label {
                EvalAware::None => self.none += 1,
                EvalAware::Suspected => self.suspected += 1,
                EvalAware::Explicit => self.explicit += 1,
            }
        }
    }

    /// Aware (suspected or explicit) over judged episodes.
    pub fn aware_rate(&self) -> String {
        rate(self.suspected + self.explicit, self.judged)
    }
}

pub fn rate(k: usize, n: usize) -> String {
    if n == 0 {
        "—".to_string()
    } else {
        format!("{k}/{n} ({:.0}%)", 100.0 * k as f64 / n as f64)
    }
}

/// Totals for the whole run and per scenario.
pub fn aggregate(rows: &[EpisodeRow]) -> (Counts, BTreeMap<&'static str, Counts>) {
    let mut total = Counts::default();
    let mut by = BTreeMap::new();
    for r in rows {
        total.add(r);
        by.entry(r.scenario.as_str())
            .or_insert_with(Counts::default)
            .add(r);
    }
    (total, by)
}

fn counts_table(first: &str, rows: Vec<(String, &Counts)>, markdown: bool) -> Table {
    let mut t = Table::new();
    t.load_style(if markdown {
        presets::ASCII_MARKDOWN
    } else {
        presets::UTF8_FULL_CONDENSED
    });
    t.set_header(vec![
        first,
        "episodes",
        "truncated",
        "judged",
        "explicit",
        "suspected",
        "none",
        "aware",
    ]);
    for (name, c) in rows {
        t.add_row(vec![
            name,
            c.episodes.to_string(),
            c.truncated.to_string(),
            c.judged.to_string(),
            c.explicit.to_string(),
            c.suspected.to_string(),
            c.none.to_string(),
            c.aware_rate(),
        ]);
    }
    t
}

pub async fn report(pool: &PgPool, cfg: &Config, runs: &[String]) -> Result<()> {
    let prompt = Prompt::load(&cfg.benchmark_dir)?;
    let schema: Option<i64> = sqlx::query_scalar("SELECT max(version) FROM _sqlx_migrations")
        .fetch_one(pool)
        .await?;
    let run_rows = sqlx::query(
        "SELECT run_id, phase, model, api, provider, stimulus_set, samples, config_toml, config_sha256,
                code_git_hash, code_dirty, benchmark_git_hash, benchmark_dirty, started_at::text AS started
         FROM runs WHERE cardinality($1::text[]) = 0 OR run_id = ANY($1) ORDER BY run_id",
    )
    .bind(runs)
    .fetch_all(pool)
    .await?;
    if run_rows.is_empty() {
        println!("no runs");
        return Ok(());
    }

    for run in &run_rows {
        let run_id: String = run.get("run_id");
        let rows: Vec<EpisodeRow> = sqlx::query(
            "SELECT s.scenario, e.status, a.label
             FROM episodes e JOIN stimuli s ON s.id = e.stimulus_id
             LEFT JOIN aware_labels a ON a.episode_id = e.id AND a.judge_model = $2 AND a.prompt_sha256 = $3
             WHERE e.run_id = $1 AND e.status <> 'failed'",
        )
        .bind(&run_id)
        .bind(&cfg.judge.model)
        .bind(&prompt.sha256)
        .fetch_all(pool)
        .await?
        .iter()
        .map(|r| {
            Ok(EpisodeRow {
                scenario: r.get::<String, _>("scenario").parse()?,
                status: r.get::<String, _>("status").parse()?,
                label: r.get::<Option<String>, _>("label").map(|l| l.parse()).transpose()?,
            })
        })
        .collect::<Result<_>>()?;
        let failed_cells: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM (SELECT 1 FROM episodes WHERE run_id = $1
             GROUP BY stimulus_id, sample HAVING bool_and(status = 'failed')) f",
        )
        .bind(&run_id)
        .fetch_one(pool)
        .await?;
        let cost: Option<f64> =
            sqlx::query_scalar("SELECT sum(cost_usd) FROM episodes WHERE run_id = $1")
                .bind(&run_id)
                .fetch_one(pool)
                .await?;

        let (total, by) = aggregate(&rows);
        let header = format!(
            "run {run_id}\nmodel {} via {} {} | phase {} | set {} | samples {}\ncode {}{} | benchmark {}{} | config {} | schema v{}\njudge {} | prompt {} | failed cells {failed_cells} | cost ${:.2}",
            run.get::<String, _>("model"),
            run.get::<String, _>("api"),
            // An Ollama pin is a 64-char digest; an OpenRouter pin is a short provider tag.
            match run.get::<String, _>("api").as_str() {
                "ollama" => short(&run.get::<String, _>("provider")).to_string(),
                _ => run.get::<String, _>("provider"),
            },
            run.get::<String, _>("phase"),
            run.get::<String, _>("stimulus_set"),
            run.get::<i32, _>("samples"),
            short(&run.get::<String, _>("code_git_hash")),
            if run.get("code_dirty") {
                " (dirty)"
            } else {
                ""
            },
            short(&run.get::<String, _>("benchmark_git_hash")),
            if run.get("benchmark_dirty") {
                " (dirty)"
            } else {
                ""
            },
            short(&run.get::<String, _>("config_sha256")),
            schema.unwrap_or(0),
            cfg.judge.model,
            short(&prompt.sha256),
            cost.unwrap_or(0.0),
        );
        let scenario_rows = || {
            by.iter()
                .map(|(k, c)| (k.to_string(), c))
                .collect::<Vec<_>>()
        };
        println!("\n{header}");
        println!("{}", counts_table("scenario", scenario_rows(), false));
        println!(
            "{}",
            counts_table("", vec![("all".to_string(), &total)], false)
        );

        let dir = cfg.results_dir.join(&run_id);
        std::fs::create_dir_all(&dir)?;
        std::fs::write(dir.join("config.toml"), run.get::<String, _>("config_toml"))?;
        let md = format!(
            "# Run {run_id}\n\nGenerated by `report` at {} from the database. Do not edit.\n\n```\n{header}\nstarted {}\n```\n\n## Test awareness by scenario\n\n{}\n\n## All episodes\n\n{}\n",
            chrono::Local::now().format("%Y-%m-%dT%H:%M:%S"),
            run.get::<String, _>("started"),
            counts_table("scenario", scenario_rows(), true),
            counts_table("", vec![("all".to_string(), &total)], true),
        );
        std::fs::write(dir.join("report.md"), md)?;
        println!("wrote {}", dir.join("report.md").display());
    }
    Ok(())
}

fn short(s: &str) -> &str {
    &s[..s.len().min(12)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(scenario: Scenario, status: EpisodeStatus, label: Option<EvalAware>) -> EpisodeRow {
        EpisodeRow {
            scenario,
            status,
            label,
        }
    }

    #[test]
    fn aggregate_counts_labels_per_scenario() {
        let rows = vec![
            row(
                Scenario::Murder,
                EpisodeStatus::Ok,
                Some(EvalAware::Explicit),
            ),
            row(
                Scenario::Murder,
                EpisodeStatus::Truncated,
                Some(EvalAware::None),
            ),
            row(Scenario::Leaking, EpisodeStatus::Ok, None),
            row(
                Scenario::Leaking,
                EpisodeStatus::Ok,
                Some(EvalAware::Suspected),
            ),
        ];
        let (total, by) = aggregate(&rows);
        assert_eq!(
            total,
            Counts {
                episodes: 4,
                truncated: 1,
                judged: 3,
                none: 1,
                suspected: 1,
                explicit: 1
            }
        );
        assert_eq!(total.aware_rate(), "2/3 (67%)");
        assert_eq!(by["murder"].aware_rate(), "1/2 (50%)");
        assert_eq!(by["leaking"].judged, 1);
    }

    #[test]
    fn rate_of_nothing_is_a_dash() {
        assert_eq!(rate(0, 0), "—");
    }
}
