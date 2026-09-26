//! Do published insider-LLM tests still work today? Runs Anthropic's Agentic Misalignment
//! scenarios as released on open reasoning models and labels verbalised test awareness.

pub mod config;
pub mod db;
pub mod gemini;
pub mod hash;
pub mod judge;
pub mod openrouter;
pub mod report;
pub mod run;
pub mod stimuli;
pub mod types;
