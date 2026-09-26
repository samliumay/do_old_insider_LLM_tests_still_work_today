//! Database connection. Migrations are embedded at compile time and applied on connect.

use anyhow::{Context, Result};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

/// Connect to `DATABASE_URL` and apply pending migrations.
pub async fn connect() -> Result<PgPool> {
    let url =
        std::env::var("DATABASE_URL").context("DATABASE_URL is not set (see ../.env.example)")?;
    let pool = PgPoolOptions::new()
        .max_connections(8)
        .connect(&url)
        .await
        .context("connecting to PostgreSQL (is it running? `make db`)")?;
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .context("applying migrations")?;
    Ok(pool)
}
