//! Command-line entry point: parses the subcommand and calls the library.

#![warn(missing_docs, clippy::missing_docs_in_private_items)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use old_insider_llm_tests::commands::{judge_aware, load, report, run};
use old_insider_llm_tests::config::Config;
use old_insider_llm_tests::db;
use old_insider_llm_tests::types::Phase;

/// Command-line arguments.
#[derive(Parser)]
#[command(about = "Do published insider-LLM tests still work today?")]
struct Cli {
    /// The experiment config.
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,
    /// What to do.
    #[command(subcommand)]
    command: Command,
}

/// The pipeline steps.
#[derive(Subcommand)]
enum Command {
    /// Apply database migrations.
    Migrate,
    /// Load a stimulus set from the benchmark into the database.
    Load {
        /// Stimulus set: `original` or `renamed`.
        #[arg(long, default_value = "original")]
        set: String,
    },
    /// Run one model on one stimulus set.
    Run {
        /// Model id from config.toml.
        #[arg(long)]
        model: Option<String>,
        /// smoke | pilot | study
        #[arg(long)]
        phase: Option<String>,
        /// Stimulus set: `original` or `renamed`.
        #[arg(long, default_value = "original")]
        set: String,
        /// Samples per condition (default: config.toml).
        #[arg(long)]
        samples: Option<u32>,
        /// Continue an existing run; only missing or failed cells are run.
        #[arg(long)]
        resume: Option<String>,
        /// Run despite uncommitted changes; recorded with the run.
        #[arg(long)]
        allow_dirty: bool,
    },
    /// Label test awareness of every unjudged episode (all runs, or the ones given).
    JudgeAware {
        /// Run ids (repeatable); none means all runs.
        #[arg(long = "run")]
        runs: Vec<String>,
    },
    /// Print test-awareness tables and write results/<run_id>/.
    Report {
        /// Run ids (repeatable); none means all runs.
        #[arg(long = "run")]
        runs: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(&cli.config)?;
    match dotenvy::from_path(&cfg.env_file) {
        Ok(()) => {}
        Err(e) if e.not_found() => {}
        Err(e) => return Err(e).context("reading .env"),
    }
    let pool = db::connect().await?;
    match cli.command {
        Command::Migrate => println!("migrations applied"),
        Command::Load { set } => load::load(&pool, &cfg, &set).await?,
        Command::Run {
            model,
            phase,
            set,
            samples,
            resume,
            allow_dirty,
        } => {
            let args = run::RunArgs {
                model,
                phase: phase.map(|p| p.parse::<Phase>()).transpose()?,
                set: set.parse()?,
                samples,
                resume,
                allow_dirty,
            };
            run::run(&pool, &cfg, args).await?;
        }
        Command::JudgeAware { runs } => judge_aware::judge_aware(&pool, &cfg, &runs).await?,
        Command::Report { runs } => report::report(&pool, &cfg, &runs).await?,
    }
    Ok(())
}
