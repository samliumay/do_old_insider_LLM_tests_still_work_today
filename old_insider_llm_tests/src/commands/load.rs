//! `load`: read a stimulus set from the benchmark and insert its new content versions.

use anyhow::Result;
use sqlx::PgPool;

use crate::config::Config;
use crate::stimuli;
use crate::types::StimulusSet;

/// Load the stimulus set named `set` and print what was inserted.
pub async fn load(pool: &PgPool, cfg: &Config, set: &str) -> Result<()> {
    let set: StimulusSet = set.parse()?;
    let (source, list) = stimuli::read_set(&cfg.benchmark_dir, set)?;
    let r = stimuli::load(pool, &source, &list).await?;
    println!(
        "load {set}: {} conditions, {} new, {} unchanged",
        list.len(),
        r.inserted,
        r.unchanged
    );
    Ok(())
}
