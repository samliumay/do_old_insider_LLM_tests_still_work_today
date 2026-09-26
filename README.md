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

Needs Docker, Rust (edition 2024), and the benchmark repo cloned next to this one as `../benchmark`.

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

One file, `old_insider_llm_tests/config.toml`, for every model: temperature 1.0, token budget, samples, retries, seed, the judge model, and the model list. Each model has one pinned OpenRouter provider with fallbacks off.

## Status

Milestone 1 (database, runner, awareness judge, report). Harm classifiers and the renamed stimulus set are next.

## License

Not chosen yet for this code. The scenarios are Anthropic's Agentic Misalignment prompts (MIT).
