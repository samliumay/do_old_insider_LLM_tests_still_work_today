# Do Old Insider-LLM Tests Still Work Today?

Do published insider-LLM tests, such as Anthropic's Agentic Misalignment scenarios, still elicit insider behaviour in current models, or do the models recognise them as tests?

The tests are run exactly as released on open-weight models whose reasoning is returned, and a judge labels every episode for verbalised test awareness.

| | |
|---|---|
| Benchmark | [Hugging Face](https://huggingface.co/datasets/logicBombExe/do_old_insider_LLM_tests_still_work_today) |
| Stack | Rust (`cargo`), PostgreSQL in Docker |
| Earlier study | [INSIDER_LLM_DETECTION_CODE](https://github.com/samliumay/INSIDER_LLM_DETECTION_CODE) (archived) |

## Quick start

```bash
cp .env.example .env          # add OPENROUTER_API_KEY and GEMINI_API_KEY
make db                       # PostgreSQL in Docker + migrations
make data                     # load the 18 original conditions from ../benchmark
make run MODEL=qwen/qwen3.8-27b PHASE=pilot
make judge                    # test-awareness labels
make report                   # tables in the terminal, files in ../results/<run_id>/
```

Needs Docker, Rust (edition 2024), and the benchmark repo cloned next to this one as `../benchmark`. Models with `api = "ollama"` need a local Ollama server (`http://localhost:11434`, signed in for `:cloud` tags) with the model pulled.

## Make targets

| Target | Does |
|---|---|
| `make db` / `make db-down` | start / stop PostgreSQL (port 55434) |
| `make data [SET=original]` | load a stimulus set into the database |
| `make run MODEL=… PHASE=smoke\|pilot\|study [ARGS=…]` | run one model; `ARGS="--samples 1"`, `"--resume <run_id>"`, `"--allow-dirty"` |
| `make judge` (= `make eval`) | label test awareness of unjudged episodes |
| `make report` | per-run tables, overall and per scenario |
| `make check` | fmt, clippy `-D warnings`, tests |
| `make baseline` | not implemented yet |

## Configuration

One file, `old_insider_llm_tests/config.toml`, for every model: temperature 1.0, token budget, samples, retries, seed, the judge model, and the model list. Each model is pinned: an OpenRouter model to one provider (fallbacks off), an Ollama model to its manifest digest. The runner checks the pin before a run starts.

## Layout

```
old_insider_llm_tests/src/
├── main.rs            # CLI; each subcommand calls one function in commands/
├── lib.rs             # module list, crate docs, doc lints
├── config.rs  types.rs  db.rs  stimuli.rs
├── models.rs          # Request / Reply / CallError, Backend enum + dispatch
├── models/            # one file per model API: openrouter.rs, ollama.rs
├── judges.rs          # Prompt, fill, tag parsing, retry loop
├── judges/            # gemini.rs (client), eval_aware.rs (test-awareness judge)
├── commands.rs
├── commands/          # load.rs, run.rs, judge_aware.rs, report.rs
├── util.rs
└── util/              # hash.rs, git.rs, text.rs
```

A parent file holds the shared types and the dispatch; its folder holds one unit per file (no `mod.rs`). A new model API or judge is one new file plus a line in the parent. Details: `../docs/code_documentation.md`.

## Status

Milestone 1 (database, runner, awareness judge, report). Harm classifiers and the renamed stimulus set are next.

## License

Not chosen yet for this code. The scenarios are Anthropic's Agentic Misalignment prompts (MIT).
