//! Do published insider-LLM tests still work today? Runs Anthropic's Agentic Misalignment
//! scenarios as released on open reasoning models and labels verbalised test awareness.
//!
//! Layout: a parent file holds the shared types and the dispatch, its folder holds one unit
//! per file (`models/` one model API each, `judges/` one judge each, `commands/` one CLI
//! subcommand each, `util/` small helpers).

#![warn(missing_docs, clippy::missing_docs_in_private_items)]

pub mod commands;
pub mod config;
pub mod db;
pub mod judges;
pub mod models;
pub mod stimuli;
pub mod types;
pub mod util;
